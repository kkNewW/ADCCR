import argparse
from transformers import AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import pickle as pk
import time

from models import ADCCRModel
from datasets.coco import COCODataset, transform_preds
from datasets.constants import (
    COCO_KEYPOINT_NAME,
    KeypointLocationDescription,
    DESCRIPTION_BANK,
    CROP_SIZE_MAP
)
from datasets.conversation import conv_keypoint, conv_llama2, conv_simple
from datasets.desc_bank import DescriptionSampler
from utils.inference import (
    generation_sequence_confidence,
    parse_normalized_coordinates,
)
from utils.metrics import coco_per_joint_pck
from utils.prompt_variants import (
    PROMPT_VARIANTS,
    PromptVariantBank,
    prompt_file_sha256,
)
from utils.reproducibility import set_global_seed
from utils.refinement_policy import (
    merge_refined_coordinates,
    select_refinement_indices,
    validate_refinement_policy,
)
from dataclasses import dataclass
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "

@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(
        self,
        image_token_len,
        conv_format,
        use_dynamic_desc=False,
        eval_desc_mode="fixed",
        prompt_variant_file=None,
        prompt_variant="canonical",
    ):
        self.image_token_len = image_token_len
        self.conv_format = conv_format
        self.use_dynamic_desc = use_dynamic_desc
        self.eval_desc_mode = eval_desc_mode
        self.prompt_variant = prompt_variant
        self.desc_sampler = DescriptionSampler(
            DESCRIPTION_BANK
        )

        self.prompt_bank = None
        if prompt_variant_file:
            self.prompt_bank = PromptVariantBank(
                prompt_variant_file,
                expected_keypoints=COCO_KEYPOINT_NAME,
            )

    def _get_description(self, kpt_name):
        if (
            not self.use_dynamic_desc
            or self.eval_desc_mode == "fixed"
        ):
            return KeypointLocationDescription[kpt_name]

        desc_text, _ = self.desc_sampler.build_description(
            kpt_name,
            mode=self.eval_desc_mode,
        )
        return desc_text

    def _get_prompt_parts(self, kpt_name):
        if self.prompt_bank is not None:
            return self.prompt_bank.get(
                self.prompt_variant,
                kpt_name,
            )

        description = self._get_description(kpt_name)
        question = (
            f"Where is the {kpt_name} of this person in this "
            "image? Please provide its coordinates."
        )
        return description, question

    def __call__(self, instances):
        """Collate examples for supervised fine-tuning."""
        batch_prompts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        if self.conv_format == 'simple':
            conv = conv_simple.copy()
        elif self.conv_format == 'keypoint':
            conv = conv_keypoint.copy()
        else:
            conv = conv_llama2.copy()

        for i, line in enumerate(instances):
            images = line['images'].unsqueeze(0)
            image_id = line['image_id']
            c = line['c']
            s = line['s']
            for kpt_id, kpt_name in enumerate(
                    COCO_KEYPOINT_NAME
            ):
                # Each keypoint needs an independent record. Reusing one
                # dictionary here aliases all 17 entries to the final item.
                result_dict = {}
                kpt_des, question = self._get_prompt_parts(
                    kpt_name
                )

                conv.messages = []
                if self.conv_format == "keypoint":
                    conv.append_message(
                        conv.roles[0],
                        kpt_des,
                    )
                    conv.append_message(
                        conv.roles[1],
                        question,
                    )
                    conv.append_message(
                        conv.roles[2],
                        None,
                    )
                elif self.conv_format == "simple":
                    conv.append_message(
                        conv.roles[0],
                        question,
                    )
                    conv.append_message(
                        conv.roles[1],
                        None,
                    )
                else:
                    conv.append_message(
                        conv.roles[0],
                        question,
                    )
                    conv.append_message(
                        conv.roles[1],
                        None,
                    )
                
                if self.conv_format == 'llama2':
                    conv.system = "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n".format(system_message=PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
                    cur_prompt = conv.get_prompt()
                else:
                    text_inputs = conv.get_prompt()
                    cur_prompt = PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs

                has_images = True

                result_dict['initial_prompt'] = cur_prompt
                result_dict['image_id'] = image_id
                result_dict['c'] = c
                result_dict['s'] = s
                result_dict["annotation_id"] = line[
                    "annotation_id"
                ]
                result_dict["bbox"] = line["bbox"]
                result_dict["joints_orig"] = line[
                    "joints_orig"
                ]
                result_dict["joints_vis_orig"] = line[
                    "joints_vis_orig"
                ]
                result_dict["joints_224"] = line["joints"]
                result_dict["joints_vis_224"] = line[
                    "joints_vis"
                ]
                result_dict["kpt_name"] = kpt_name
                result_dict["description"] = kpt_des
                result_dict["question"] = question
                result_dict["prompt_variant"] = (
                    self.prompt_variant
                )
                batch_prompts.append(cur_prompt)
                batch_images.append(images)
                batch_has_images.append(has_images)
                result_dicts.append(result_dict)

        return result_dicts, batch_prompts, batch_images, batch_has_images



@torch.no_grad()
def worker(model, tokenizer, dataset, args, output_dir):
    crop_size = model.config.crop_size
    image_token_len = model.config.num_patches

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    indices = list(range(rank, len(dataset), world_size))
    print("==>" + " Worker {} Started, responsible for {} images".format(rank, len(indices)))

    sub_dataset = torch.utils.data.Subset(dataset, indices)
    batch_size = 1
    data_loader = DataLoader(
        sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=DataCollatorForSupervisedDataset(
            image_token_len=image_token_len,
            conv_format=args.conv_format,
            use_dynamic_desc=args.use_dynamic_desc,
            eval_desc_mode=args.eval_desc_mode,
            prompt_variant_file=args.prompt_variant_file,
            prompt_variant=args.prompt_variant,
        )
    )


    all_preds = []
    for result_dicts, batch_prompts, batch_images, batch_has_images in tqdm(data_loader):
        assert len(result_dicts) == 17
        # inputs = tokenizer()
        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        batch_images = torch.cat(batch_images, dim=0).cuda()
        assert batch_images.shape[0] == 17

        input_ids = torch.as_tensor(tokenized_output.input_ids).cuda()
        attention_mask = torch.as_tensor(tokenized_output.attention_mask).cuda()

        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=batch_images,
                has_images=batch_has_images,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True
            )
            output_ids = output_dict['sequences']
            sequence_confidence = (
                generation_sequence_confidence(
                    output_dict,
                    input_ids.shape[1],
                )
                .float()
                .cpu()
                .numpy()
            )

        outputs = []
        for output_index, (
            input_id,
            output_id,
        ) in enumerate(zip(input_ids, output_ids)):
            input_token_len = input_id.shape[0]
            n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f"[Warning] Sample {output_index}: "
                    f"{n_diff_input_output} output_ids differ "
                    "from the input prefix."
                )
            output = tokenizer.batch_decode(output_id[input_token_len:].unsqueeze(0), skip_special_tokens=True)[0]
            output = output.strip()
            outputs.append(output)

        assert len(outputs) == 17
        decoded_kpt = np.zeros((17, 3))
        image_id = result_dicts[0]['image_id']
        c = result_dicts[0]['c']
        s = result_dicts[0]['s']

        for i in range(len(outputs)):
            # decode coordinates from token
            pred_kpt = outputs[i]
            coordinates = parse_normalized_coordinates(
                pred_kpt
            )
            if coordinates is None:
                print('Format error', pred_kpt)
                continue
            x, y = coordinates
            x, y = x * crop_size, y * crop_size
            x_s = float(sequence_confidence[i])
            y_s = x_s

            decoded_kpt[i, 0] = x
            decoded_kpt[i, 1] = y
            decoded_kpt[i, 2] = (x_s + y_s) / 2.0
        coarse_xy_224 = decoded_kpt[:, :2].copy()
        effective_crop_sizes = np.asarray(
            [
                CROP_SIZE_MAP[result["kpt_name"]]
                for result in result_dicts
            ],
            dtype=np.float32,
        )
        effective_crop_sizes *= getattr(
            model.config,
            "refiner_crop_scale",
            1.0,
        )
        refinement_indices = np.empty(0, dtype=np.int64)

        if args.use_local_refiner:
            if not getattr(
                    model.config,
                    "use_local_refiner",
                    False,
            ):
                raise RuntimeError(
                    "Evaluation requested the local refiner, "
                    "but the checkpoint configuration does "
                    "not contain one."
                )

            refinement_indices = select_refinement_indices(
                sequence_confidence,
                use_confidence_gate=(
                    args.use_refinement_confidence_gate
                ),
                confidence_threshold=(
                    args.refinement_confidence_threshold
                ),
            )
            if refinement_indices.size:
                descriptions = [
                    result_dicts[index]["description"]
                    for index in refinement_indices
                ]
                description_tokens = tokenizer(
                    descriptions,
                    padding=True,
                    truncation=True,
                    max_length=96,
                    return_tensors="pt",
                )

                index_tensor = torch.as_tensor(
                    refinement_indices,
                    dtype=torch.long,
                    device=batch_images.device,
                )
                refined_xy, _ = model.refine_coordinates(
                    images=batch_images.index_select(
                        0,
                        index_tensor,
                    ),
                    coarse_xy=torch.as_tensor(
                        coarse_xy_224[refinement_indices],
                        dtype=torch.float32,
                        device=batch_images.device,
                    ),
                    crop_sizes=torch.as_tensor(
                        effective_crop_sizes[
                            refinement_indices
                        ],
                        dtype=torch.float32,
                        device=batch_images.device,
                    ),
                    desc_input_ids=(
                        description_tokens.input_ids.cuda()
                    ),
                    desc_attention_mask=(
                        description_tokens.attention_mask.cuda()
                    ),
                )

                decoded_kpt[:, :2] = merge_refined_coordinates(
                    coarse_xy_224,
                    refined_xy.float().cpu().numpy(),
                    refinement_indices,
                )

        final_xy_224 = decoded_kpt[:, :2].copy()

        decoded_kpt[:, :2] = transform_preds(
            decoded_kpt[:, :2], c, s, (crop_size, crop_size)
        )

        data = dict()
        data['image_id'] = image_id
        data["annotation_id"] = int(
            result_dicts[0]["annotation_id"]
        )
        data["bbox"] = np.asarray(
            result_dicts[0]["bbox"],
        ).tolist()
        ground_truth = np.asarray(
            result_dicts[0]["joints_orig"],
        ).copy()
        visibility = np.asarray(
            result_dicts[0]["joints_vis_orig"],
        )
        ground_truth[:, 2] = visibility[:, 0]
        data["gt_keypoints"] = ground_truth.reshape(
            -1
        ).tolist()
        ground_truth_224 = np.asarray(
            result_dicts[0]["joints_224"],
        ).copy()
        visibility_224 = np.asarray(
            result_dicts[0]["joints_vis_224"],
        )
        ground_truth_224[:, 2] = visibility_224[:, 0]
        refinement_applied = np.zeros(17, dtype=bool)
        refinement_applied[refinement_indices] = True
        data["gt_keypoints_224"] = ground_truth_224.reshape(
            -1
        ).tolist()
        data["coarse_keypoints_224"] = coarse_xy_224.reshape(
            -1
        ).tolist()
        data["final_keypoints_224"] = final_xy_224.reshape(
            -1
        ).tolist()
        data["coarse_confidence"] = sequence_confidence.tolist()
        data["refinement_crop_sizes"] = (
            effective_crop_sizes.tolist()
        )
        data["refinement_applied"] = (
            refinement_applied.tolist()
        )
        data["refinement_confidence_gate"] = (
            args.use_refinement_confidence_gate
        )
        data["refinement_confidence_threshold"] = (
            args.refinement_confidence_threshold
        )
        data['score'] = float(np.mean(decoded_kpt[:, 2]))
        data['keypoints'] = decoded_kpt.reshape(-1).tolist()
        data['category_id'] = 1
        
        all_preds.append(data)
    
    with open(os.path.join(output_dir, f'test_gt_kpt_rank_{rank}.pkl'), 'wb') as fid:
        pk.dump(all_preds, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if rank == 0:
        # manually sleep to wait all file are saved
        while True:
            ready = True
            for r in range(world_size):
                if not os.path.exists(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl')):
                    ready = False
            if ready: 
                break
            else:
                time.sleep(20)
        # sleep 30s to make sure all files are saved
        time.sleep(20)
        kpt_all_pred = []
        for r in range(world_size):
            with open(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            # os.remove(os.path.join(output_dir, f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.extend(kpt_pred)

        ann_file = args.question_file
        detailed_file = os.path.join(
            output_dir,
            "predictions_detailed.json",
        )
        with open(detailed_file, "w") as fid:
            json.dump(kpt_all_pred, fid)

        coco_results = [
            {
                key: item[key]
                for key in (
                    "image_id",
                    "score",
                    "keypoints",
                    "category_id",
                )
            }
            for item in kpt_all_pred
        ]
        res_file = os.path.join(output_dir, 'pred_kpt.json')
        with open(res_file, 'w') as fid:
            json.dump(coco_results, fid)

        cocoGt = COCO(ann_file)
        cocoDt = cocoGt.loadRes(res_file)

        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        metric_names = (
            "AP",
            "AP50",
            "AP75",
            "APM",
            "APL",
            "AR",
            "AR50",
            "AR75",
            "ARM",
            "ARL",
        )
        metrics = {
            name: float(value)
            for name, value in zip(
                metric_names,
                cocoEval.stats.tolist(),
            )
        }
        metrics["difficult_joint_metric"] = (
            coco_per_joint_pck(kpt_all_pred)
        )
        metrics["protocol"] = {
            "person_boxes": "ground-truth",
            "flip_test": False,
            "description_mode": args.eval_desc_mode,
            "prompt_variant": args.prompt_variant,
            "prompt_variant_file": (
                args.prompt_variant_file
            ),
            "prompt_variant_sha256": prompt_file_sha256(
                args.prompt_variant_file
            ),
            "local_refiner": args.use_local_refiner,
            "refinement_confidence_gate": (
                args.use_refinement_confidence_gate
            ),
            "refinement_confidence_threshold": (
                args.refinement_confidence_threshold
            ),
            "seed": args.seed,
        }
        with open(
            os.path.join(output_dir, "metrics.json"),
            "w",
        ) as fid:
            json.dump(metrics, fid, indent=2)

        return True
    else:
        return False

def eval_model(args):
    validate_refinement_policy(
        args.use_local_refiner,
        args.use_refinement_confidence_gate,
        args.refinement_confidence_threshold,
    )
    torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print('Init process group: world_size: {}, rank: {}'.format(world_size, rank))
    torch.cuda.set_device(rank)
    set_global_seed(args.seed)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

    model = ADCCRModel.from_pretrained(model_name, use_cache=True)
    for name, param in model.model.named_parameters():
        if "lora_" not in name:
            param.data = param.data.bfloat16()
    model.lm_head.to(torch.bfloat16)
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    dataset = COCODataset(tokenizer=None,
                        data_path=os.path.join(args.question_file),
                        multimodal_cfg=dict(
                            image_folder=args.image_folder,
                            image_size=224,
                            crop_size=224,
                            conv_format=args.conv_format),
                            is_train=False)

    worker(model, tokenizer, dataset, args, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-format", type=str, default="keypoint")
    parser.add_argument("--output-dir", type=str, default="")
    # ===== Auto Description Evaluation =====
    parser.add_argument("--use-dynamic-desc", action="store_true")
    parser.add_argument("--eval-desc-mode", type=str, default="fixed",
                        choices=["fixed", "name_only", "name_anatomy", "name_relation", "name_anatomy_relation", "all"])
    parser.add_argument(
        "--prompt-variant-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="canonical",
        choices=PROMPT_VARIANTS,
    )
    parser.add_argument(
        "--use-local-refiner",
        action="store_true",
    )
    parser.add_argument(
        "--use-refinement-confidence-gate",
        action="store_true",
        help=(
            "Refine only predictions whose generated-token "
            "confidence reaches the configured threshold."
        ),
    )
    parser.add_argument(
        "--refinement-confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    eval_model(args)
