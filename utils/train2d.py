import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import random

import torch

import transformers

from models import ADCCRModel
from datasets import COCODataset
from datasets.constants import CROP_SIZE_MAP
from utils.llavasimple_trainer import (
    LLaVASimpleTrainer
)

from PIL import Image
import torch.nn as nn
import io


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
# FIXME: seems wrong?
# DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m"
    )
    llama_path: Optional[str] = field(default="")
    dino_path: Optional[str] = field(default=None)

    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None
    )
    tune_mm_mlp_adapter: bool = field(default=True)

    mm_projector_type: str = field(
        default="mlp"
    )
    mm_projector_depth: int = field(
        default=2
    )

    use_local_refiner: bool = field(
        default=True
    )
    refiner_input_size: int = field(
        default=128
    )
    refiner_heatmap_size: int = field(
        default=64
    )
    refiner_text_dim: int = field(
        default=768
    )
    refiner_feat_dim: int = field(
        default=256
    )
    refiner_sigma: float = field(
        default=2.0
    )
    refiner_noise_ratio: float = field(
        default=0.25
    )
    lambda_hm: float = field(
        default=0.5
    )

    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_size: int = field(default=224)
    crop_size: int = field(default=224)
    data_augmentation: bool = field(default=False)
    conv_format: str = field(default="keypoint")
    # ===== Auto Description =====
    use_dynamic_desc: bool = field(default=False)
    desc_mode: str = field(default="dynamic")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class LoRAArguments:
    lora_vision_r: int = field(default=8)
    lora_vision_alpha: float = field(default=16)
    lora_vision_dropout: float = field(default=0.05)
    lora_vision_enable: bool = field(default=False)
    lora_llm_r: int = field(default=8)
    lora_llm_alpha: float = field(default=16)
    lora_llm_dropout: float = field(default=0.05)
    lora_llm_enable: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    use_local_refiner: bool = False
    desc_max_length: int = 96

    def __call__(
        self,
        instances: Sequence[Dict],
    ) -> Dict[str, torch.Tensor]:
        input_ids = [
            instance["input_ids"]
            for instance in instances
        ]
        labels = [
            instance["labels"]
            for instance in instances
        ]

        input_ids = (
            torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=(
                    self.tokenizer.pad_token_id
                ),
            )
        )

        labels = (
            torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(
                self.tokenizer.pad_token_id
            ),
        }

        images = [
            instance["image"]
            for instance in instances
        ]

        if not all(
            image is not None
            and image.shape == images[0].shape
            for image in images
        ):
            raise ValueError(
                "All images in a training batch "
                "must have the same shape."
            )

        batch["images"] = torch.stack(
            images,
            dim=0,
        )

        batch["has_images"] = [
            instance["has_image"]
            for instance in instances
        ]

        if self.use_local_refiner:
            descriptions = []
            target_coordinates = []
            crop_sizes = []
            image_indices = []

            for image_index, instance in enumerate(
                instances
            ):
                for (
                    keypoint_name,
                    description,
                    target_xy,
                ) in zip(
                    instance["kpt_name"],
                    instance["description"],
                    instance["target_xy_224"],
                ):
                    descriptions.append(description)
                    target_coordinates.append(
                        target_xy
                    )
                    crop_sizes.append(
                        CROP_SIZE_MAP[
                            keypoint_name
                        ]
                    )
                    image_indices.append(
                        image_index
                    )

            if not descriptions:
                raise ValueError(
                    "Local refiner is enabled but "
                    "the batch has no descriptions."
                )

            description_tokens = self.tokenizer(
                descriptions,
                padding=True,
                truncation=True,
                max_length=self.desc_max_length,
                return_tensors="pt",
            )

            batch["desc_input_ids"] = (
                description_tokens.input_ids
            )
            batch["desc_attention_mask"] = (
                description_tokens.attention_mask
            )
            batch["refine_target_xy"] = (
                torch.stack(
                    target_coordinates,
                    dim=0,
                ).float()
            )
            batch["refine_crop_sizes"] = (
                torch.tensor(
                    crop_sizes,
                    dtype=torch.float32,
                )
            )
            batch["refine_image_indices"] = (
                torch.tensor(
                    image_indices,
                    dtype=torch.long,
                )
            )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,use_local_refiner,) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = COCODataset
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    image_folder=data_args.image_folder,
                                    data_augmentation=data_args.data_augmentation,
                                    image_size=data_args.image_size,
                                    crop_size=data_args.crop_size,
                                    conv_format=data_args.conv_format,
                                    use_dynamic_desc=data_args.use_dynamic_desc,
                                    desc_mode=data_args.desc_mode,))
    data_collator = (
        DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            use_local_refiner=use_local_refiner,
            desc_max_length=96,
        )
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    print("===== Data Config =====")
    print("use_dynamic_desc:", data_args.use_dynamic_desc)
    print("desc_mode:", data_args.desc_mode)
    print("conv_format:", data_args.conv_format)
    print("=======================")


    model = ADCCRModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        llama_path=model_args.llama_path,
        dino_path=model_args.dino_path,
        lora_vision_r=lora_args.lora_vision_r,
        lora_vision_alpha=lora_args.lora_vision_alpha,
        lora_vision_dropout=lora_args.lora_vision_dropout,
        lora_vision_enable=lora_args.lora_vision_enable,
        lora_llm_enable=lora_args.lora_llm_enable,
        lora_llm_r=lora_args.lora_llm_r,
        lora_llm_alpha=lora_args.lora_llm_alpha,
        lora_llm_dropout=(
            lora_args.lora_llm_dropout
        ),
        crop_size=data_args.crop_size,

        mm_projector_type=(
            model_args.mm_projector_type
        ),
        mm_projector_depth=(
            model_args.mm_projector_depth
        ),

        use_local_refiner=(
            model_args.use_local_refiner
        ),
        refiner_input_size=(
            model_args.refiner_input_size
        ),
        refiner_heatmap_size=(
            model_args.refiner_heatmap_size
        ),
        refiner_text_dim=(
            model_args.refiner_text_dim
        ),
        refiner_feat_dim=(
            model_args.refiner_feat_dim
        ),
        refiner_sigma=(
            model_args.refiner_sigma
        ),
        refiner_noise_ratio=(
            model_args.refiner_noise_ratio
        ),
        lambda_hm=model_args.lambda_hm,
    )
    
    # load mm projector weights
    if model_args.pretrain_mm_mlp_adapter is not None:
        checkpoint = torch.load(
            model_args.pretrain_mm_mlp_adapter,
            map_location="cpu",
        )

        state_dict = checkpoint

        for container_key in (
                "state_dict",
                "model",
        ):
            if (
                    isinstance(state_dict, dict)
                    and container_key in state_dict
                    and isinstance(
                state_dict[container_key],
                dict,
            )
            ):
                state_dict = state_dict[
                    container_key
                ]

        projector_state = {}

        for key, value in state_dict.items():
            if "mm_projector." in key:
                projector_key = key.split(
                    "mm_projector.",
                    1,
                )[1]
                projector_state[
                    projector_key
                ] = value

        if not projector_state:
            raise RuntimeError(
                "No mm_projector parameters were "
                "found in the adapter checkpoint."
            )

        model.mm_projector.load_state_dict(
            projector_state,
            strict=True,
        )

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    model.initialize_vision_tokenizer(tokenizer=tokenizer)

    dtype = torch.bfloat16
    model.model.to(dtype)
    model.lm_head.to(dtype)

    for param in model.parameters():
        param.requires_grad_(False)

    if model_args.tune_mm_mlp_adapter:
        for parameter in (
                model.mm_projector.parameters()
        ):
            parameter.requires_grad = True

    if model_args.use_local_refiner:
        for parameter in (
                model.local_refiner.parameters()
        ):
            parameter.requires_grad = True

        for parameter in (
                model.description_projection.parameters()
        ):
            parameter.requires_grad = True


    data_args.image_token_len = model.config.num_patches
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        use_local_refiner=(
            model_args.use_local_refiner
        ),
    )

    if not model_args.freeze_vit:
        assert model.config.lora_vision_enable
        for name, param in model.vision_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True
    else:
        model.vision_model.train = disabled_train
        model.vision_model.eval()

    if not model_args.freeze_llm:
        assert model.config.lora_llm_enable
        for name, param in model.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True

    params_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print("param_grad: {}".format(params_grad))
    # NOTE: enable grad on embedding for gradient checkpoint
    # for p in model.get_input_embeddings().parameters():
    #     p.requires_grad = True
    trainer = LLaVASimpleTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()