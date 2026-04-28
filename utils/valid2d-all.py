import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from transformers import StoppingCriteria
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import pickle as pk
import time

import math
from models import ADCCRMModel
from datasets.coco import COCODataset, transform_preds
from datasets.constants import (
    COCO_KEYPOINT_NAME,
    KeypointLocationDescription,
    KeypointLocationQuestion,
    DESCRIPTION_BANK,
)
from datasets.convsersation import conv_keypoint, conv_llama2, conv_simple
from datasets.desc_bank import DescriptionSampler
from dataclasses import dataclass
import re
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
    def __init__(self, image_token_len, conv_format, use_dynamic_desc=False, eval_desc_mode="fixed"):
        self.image_token_len = image_token_len
        self.conv_format = conv_format
        self.use_dynamic_desc = use_dynamic_desc
        self.eval_desc_mode = eval_desc_mode
        self.desc_sampler = DescriptionSampler(DESCRIPTION_BANK)
    def _get_description(self, kpt_name):
        if not self.use_dynamic_desc or self.eval_desc_mode == "fixed":
            return KeypointLocationDescription[kpt_name]

        # 支持指定某种描述模式，例如:
        # name_only / name_anatomy / name_relation / name_anatomy_relation / all
        desc_text, _ = self.desc_sampler.build_description(kpt_name, mode=self.eval_desc_mode)
        return desc_text

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
            result_dict = {}
            images = line['images'].unsqueeze(0)
            image_id = line['image_id']
            c = line['c']
            s = line['s']
            for kpt_id, kpt_name in enumerate(COCO_KEYPOINT_NAME):
                question = KeypointLocationQuestion[kpt_name][0]
                kpt_des = self._get_description(kpt_name)

                conv.messages = []
                if self.conv_format == 'keypoint':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name.replace("_", " "))
                    conv.append_message(conv.roles[0], kpt_des)
                    conv.append_message(conv.roles[1], q1)
                    conv.append_message(conv.roles[2], None)
                elif self.conv_format == 'simple':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name.replace("_", " "))
                    conv.append_message(conv.roles[0], q1)
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                
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
                result_dict["kpt_name"] = kpt_name
                result_dict["description"] = kpt_des
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
        num_workers=8,
        collate_fn=DataCollatorForSupervisedDataset(
            image_token_len=image_token_len,
            conv_format=args.conv_format,
            use_dynamic_desc=args.use_dynamic_desc,
            eval_desc_mode=args.eval_desc_mode,
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

        input_ids_all = torch.as_tensor(tokenized_output.input_ids).cuda()
        attention_mask_all = torch.as_tensor(tokenized_output.attention_mask).cuda()

        outputs = []
        score_pairs = []   # [(x_s, y_s), ...] aligned with outputs
        chunk_size = args.chunk_size
        pattern = re.compile(r'0\.\d+')

        for start in range(0, input_ids_all.shape[0], chunk_size):
            end = min(start + chunk_size, input_ids_all.shape[0])

            input_ids = input_ids_all[start:end]
            attention_mask = attention_mask_all[start:end]
            images_chunk = batch_images[start:end]
            has_images_chunk = batch_has_images[start:end]

            with torch.inference_mode():
                output_dict = model.generate(
                    input_ids,
                    images=images_chunk,
                    has_images=has_images_chunk,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=16,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            output_ids = output_dict["sequences"]
            output_scores_list = output_dict["scores"]

            # [gen_len-1, chunk, vocab]
            if len(output_scores_list) > 0:
                output_scores = torch.stack(output_scores_list[:-1], dim=0) if len(output_scores_list) > 1 else torch.stack(output_scores_list, dim=0)
            else:
                output_scores = None

            for j, (input_id, output_id) in enumerate(zip(input_ids, output_ids)):
                input_token_len = input_id.shape[0]
                n_diff_input_output = (input_id != output_id[:input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

                output = tokenizer.batch_decode(
                    output_id[input_token_len:].unsqueeze(0),
                    skip_special_tokens=True
                )[0].strip()

                outputs.append(output)

                # default confidence
                x_s = 0.0
                y_s = 0.0

                if output_scores is not None:
                    res = pattern.findall(output)
                    if len(res) >= 1:
                        x_pos = output.find(res[0])
                        if x_pos >= 0 and x_pos + len(res[0]) <= output_scores.shape[0]:
                            xs = output_scores[x_pos:x_pos+len(res[0]), j, :].cpu()
                            xs = F.softmax(xs, dim=1)
                            x_s = torch.max(xs, dim=1)[0].mean().float().item()

                    if len(res) >= 2:
                        y_pos = output.find(res[1])
                        if y_pos >= 0 and y_pos + len(res[1]) <= output_scores.shape[0]:
                            ys = output_scores[y_pos:y_pos+len(res[1]), j, :].cpu()
                            ys = F.softmax(ys, dim=1)
                            y_s = torch.max(ys, dim=1)[0].mean().float().item()

                score_pairs.append((x_s, y_s))

            del output_dict, output_ids, output_scores_list
            if output_scores is not None:
                del output_scores
            del input_ids, attention_mask, images_chunk
            # torch.cuda.empty_cache()

        assert len(outputs) == 17

        decoded_kpt = np.zeros((17, 3))
        image_id = result_dicts[0]['image_id']
        c = result_dicts[0]['c']
        s = result_dicts[0]['s']

        for i in range(len(outputs)):
            pred_kpt = outputs[i]
            res = pattern.findall(pred_kpt)

            if len(res) == 0:
                print('Format error', pred_kpt)
                continue

            if len(res) == 1:
                x = float(res[0]) * crop_size
                y = 0.0
            else:
                x, y = float(res[0]), float(res[1])
                x, y = x * crop_size, y * crop_size

            x_s, y_s = score_pairs[i]

            decoded_kpt[i, 0] = x
            decoded_kpt[i, 1] = y
            decoded_kpt[i, 2] = (x_s + y_s) / 2.0


        decoded_kpt[:, :2] = transform_preds(
            decoded_kpt[:, :2], c, s, (crop_size, crop_size)
        )

        data = dict()
        data['image_id'] = image_id
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
        res_file = os.path.join(output_dir, 'pred_kpt.json')
        with open(res_file, 'w') as fid:
            json.dump(kpt_all_pred, fid)

        cocoGt = COCO(ann_file)
        cocoDt = cocoGt.loadRes(res_file)

        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return True
    else:
        return False

def eval_model(args):
    torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print('Init process group: world_size: {}, rank: {}'.format(world_size, rank))
    torch.cuda.set_device(rank)

    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')

    model = ADCCRModel.from_pretrained(model_name, use_cache=False)
    model.config.use_cache = False
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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-format", type=str, default="keypoint")
    parser.add_argument("--output-dir", type=str, default="")
    # ===== Auto Description Evaluation =====
    parser.add_argument("--use-dynamic-desc", action="store_true")
    parser.add_argument("--eval-desc-mode", type=str, default="fixed",
                        choices=["fixed", "name_only", "name_anatomy", "name_relation", "name_anatomy_relation", "all"])
    # reduce eval memory
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
