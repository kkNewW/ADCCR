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
import cv2
from scipy.io import loadmat
from collections import OrderedDict
from torch.utils.data import Dataset

from models import ADCCRModel
from datasets.coco import (
    affine_transform,
    get_affine_transform,
    transform_preds,
)
from datasets.constants import (
    CROP_SIZE_MAP,
    DESCRIPTION_BANK,
    KeypointLocationDescription,
    KeypointLocationQuestion,
)
from datasets.conversation import conv_keypoint, conv_llama2, conv_simple
from datasets.desc_bank import DescriptionSampler
from utils.inference import (
    generation_sequence_confidence,
    parse_normalized_coordinates,
)
from utils.reproducibility import set_global_seed
from utils.refinement_policy import (
    merge_refined_coordinates,
    select_refinement_indices,
    validate_refinement_policy,
)
from dataclasses import dataclass

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

MPII_KEYPOINT_NAME = ['right ankle', 'right knee', 'right hip', 'left hip', 'left knee', 'left ankle', 'pelvis', 'thorax', 'neck', 'head_top', 'right wrist', 'right elbow', 'right shoulder', 'left shoulder', 'left elbow', 'left wrist']

# Only the 12 MPII joints shared with COCO have defined
# category-aware crop sizes. Indices 6--9 remain coarse-only.
MPII_REFINER_KEYPOINTS = {
    'right ankle',
    'right knee',
    'right hip',
    'left hip',
    'left knee',
    'left ankle',
    'right wrist',
    'right elbow',
    'right shoulder',
    'left shoulder',
    'left elbow',
    'left wrist',
}

class MPIIDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, annot_file, data_path):
        super(MPIIDataset, self).__init__()
        print("Loading data...")
        self.data_path = data_path
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.aspect_ratio = 224 * 1.0 / 224
        self.pixel_std = 200
        self.num_joints = 16
        self.size = 224
        from torchvision import transforms
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        file_name = annot_file
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        ins_id = 0
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float32)
            s = np.array([a['scale'], a['scale']], dtype=np.float32)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float32)
            joints = np.array(a['joints'])
            joints[:, 0:2] = joints[:, 0:2] - 1
            joints_vis = np.array(a['joints_vis'])
            assert len(joints) == self.num_joints, \
                'joint num diff: {} vs {}'.format(len(joints),
                                                    self.num_joints)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints_vis[:]
            joints_3d_vis[:, 1] = joints_vis[:]
            visible = joints_vis

            gt_db.append(
                {
                    'image': os.path.join(data_path, 'images', image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'visible': visible,
                    'ins_id': ins_id
                }
            )
            ins_id += 1
        
        list_data_dict = gt_db
        
        print("The number of training samples is {}".format(len(list_data_dict)))
        assert len(list_data_dict) == ins_id
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        return self._parse_data_item_val(i)

    def _parse_data_item_val(self, i):
        sources = self.list_data_dict[i]
        result_dict = {}
        image, _, _, c, s = self._get_pose_item(sources)
        instance_id = sources['ins_id']
        result_dict['images'] = image
        result_dict['c'] = c
        result_dict['s'] = s
        result_dict['instance_id'] = instance_id
        return result_dict
 
    def _get_pose_item(self, sources):
        image_file = sources['image']
        image = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # process image
        joints = sources['joints_3d'].copy()
        joints_vis = sources['joints_3d_vis'].copy()
        c = sources['center'].copy()
        s = sources['scale'].copy()
        r = 0

        trans = get_affine_transform(c, s, r, (int(self.size), int(self.size)))
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.size), int(self.size)),
            flags=cv2.INTER_LINEAR)
        image = self.transforms(image)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        
        return image, joints, joints_vis, c, s
    
    def evaluate(self, preds, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.data_path, 'annot/gt_valid.mat')
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        pelv = np.where(dataset_joints == 'pelv')[1][0]
        neck = np.where(dataset_joints == 'neck')[1][0]
        thor = np.where(dataset_joints == 'thor')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:10] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:10] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Pelv', PCKh[pelv]),
            ('Neck', PCKh[neck]),
            ('Thor', PCKh[thor]),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(
        self,
        image_token_len,
        conv_format,
        use_dynamic_desc=False,
        eval_desc_mode="fixed",
    ):
        self.image_token_len = image_token_len
        self.conv_format = conv_format
        self.use_dynamic_desc = use_dynamic_desc
        self.eval_desc_mode = eval_desc_mode
        self.desc_sampler = DescriptionSampler(
            DESCRIPTION_BANK
        )

    def _get_description(self, kpt_name):
        if (
            not self.use_dynamic_desc
            or self.eval_desc_mode == "fixed"
            or kpt_name not in DESCRIPTION_BANK
        ):
            return KeypointLocationDescription[kpt_name]

        description, _ = (
            self.desc_sampler.build_description(
                kpt_name,
                mode=self.eval_desc_mode,
            )
        )
        return description

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
            ins_id = line['instance_id']
            c = line['c']
            s = line['s']
            for kpt_id, kpt_name in enumerate(MPII_KEYPOINT_NAME):
                result_dict = {}
                question = KeypointLocationQuestion[kpt_name][0]
                kpt_des = self._get_description(kpt_name)

                conv.messages = []
                if self.conv_format == 'keypoint':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name)
                    conv.append_message(conv.roles[0], kpt_des)
                    conv.append_message(conv.roles[1], q1)
                    conv.append_message(conv.roles[2], None)
                elif self.conv_format == 'simple':
                    q1 = "Where is the {} of this person in this image? Please provide its coordinates.".format(kpt_name)
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
                result_dict['instance_id'] = ins_id
                result_dict['c'] = c
                result_dict['s'] = s
                result_dict['kpt_name'] = kpt_name
                result_dict['description'] = kpt_des
                batch_prompts.append(cur_prompt)
                batch_images.append(images)
                batch_has_images.append(has_images)
                result_dicts.append(result_dict)

        return result_dicts, batch_prompts, batch_images, batch_has_images


DEFAULT_IMAGE_TOKEN = "<image>"


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
        ),
    )

    all_preds = []
    for result_dicts, batch_prompts, batch_images, batch_has_images in tqdm(data_loader):
        assert len(result_dicts) == 16
        # inputs = tokenizer()
        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        batch_images = torch.cat(batch_images, dim=0).cuda()
        assert batch_images.shape[0] == 16

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

        assert len(outputs) == 16
        decoded_kpt = np.zeros((16, 3))
        ins_id = result_dicts[0]['instance_id']
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

            candidate_indices = [
                index
                for index, result in enumerate(result_dicts)
                if result["kpt_name"]
                in MPII_REFINER_KEYPOINTS
            ]

            if len(candidate_indices) != 12:
                raise RuntimeError(
                    "Expected 12 overlapping COCO/MPII "
                    "keypoints for local refinement, got "
                    f"{len(candidate_indices)}."
                )

            refinement_indices = select_refinement_indices(
                sequence_confidence,
                use_confidence_gate=(
                    args.use_refinement_confidence_gate
                ),
                confidence_threshold=(
                    args.refinement_confidence_threshold
                ),
                candidate_indices=candidate_indices,
            )

            descriptions = [
                result_dicts[index]["description"]
                for index in refinement_indices
            ]
            if refinement_indices.size:
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
                crop_sizes = torch.as_tensor(
                    [
                        CROP_SIZE_MAP[
                            result_dicts[index]["kpt_name"]
                        ]
                        for index in refinement_indices
                    ],
                    dtype=torch.float32,
                    device=batch_images.device,
                )
                crop_sizes *= getattr(
                    model.config,
                    "refiner_crop_scale",
                    1.0,
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
                    crop_sizes=crop_sizes,
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

        decoded_kpt[:, :2] = transform_preds(
            decoded_kpt[:, :2], c, s, (crop_size, crop_size)
        )

        data = dict()
        data['ins_id'] = ins_id
        data['score'] = float(np.mean(decoded_kpt[:, 2]))
        data['keypoints'] = decoded_kpt.reshape(-1).tolist()
        refinement_applied = np.zeros(16, dtype=bool)
        refinement_applied[refinement_indices] = True
        data["coarse_confidence"] = sequence_confidence.tolist()
        data["refinement_applied"] = (
            refinement_applied.tolist()
        )
        
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

        res_file = os.path.join(output_dir, 'pred_kpt.json')
        with open(res_file, 'w') as fid:
            json.dump(kpt_all_pred, fid)

        num_instance = len(kpt_all_pred)
        preds = np.zeros((num_instance, 16, 2))
        for item in kpt_all_pred:
            ins_id = item['ins_id']
            kpt = item['keypoints']
            preds[ins_id] = np.array(kpt).reshape(16, 3)[:, :2]
        
        res_all, res_mean = dataset.evaluate(preds)
        print(res_all)
        print(res_mean)
        metrics = {
            key: float(value)
            for key, value in res_all.items()
        }
        metrics["protocol"] = {
            "person_instances": "MPII center/scale",
            "description_mode": args.eval_desc_mode,
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

    dataset = MPIIDataset(args.question_file, args.image_folder)

    worker(model, tokenizer, dataset, args, args.output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument('--gpus', help='gpu ids for eval', default='0', type=str)
    parser.add_argument("--conv-format", type=str, default="keypoint")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--use-dynamic-desc",
        action="store_true",
    )
    parser.add_argument(
        "--eval-desc-mode",
        type=str,
        default="fixed",
        choices=[
            "fixed",
            "name_only",
            "name_anatomy",
            "name_relation",
            "name_anatomy_relation",
            "all",
        ],
    )
    parser.add_argument(
        "--use-local-refiner",
        action="store_true",
    )
    parser.add_argument(
        "--use-refinement-confidence-gate",
        action="store_true",
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
