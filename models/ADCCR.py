from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
from copy import deepcopy
logger = logging.get_logger(__name__)

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForPreTraining, \
                         LlamaConfig, LlamaForCausalLM, LlamaModel, CLIPVisionModel, \
                         CLIPImageProcessor, CLIPModel, PretrainedConfig, PreTrainedModel

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .dino import vit_large
from .lora import lora, lora_dino
from .local_refiner import (
    LocalRefiner,
    crop_and_resize,
    build_gaussian_heatmaps,
    heatmaps_to_global,
)

import math


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
BEGIN_DESCRIPTION = "<des>"
END_DESCRIPTION = "</des>"



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def convert_weights_to_dtype(model: nn.Module, dtype):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_dtype(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype=dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype=dtype)

    model.apply(_convert_weights_to_dtype)


class ADCCRConfig(LlamaConfig):
    model_type = "ADCCR"
    def __init__(
        self,
        llama_path=None,
        dino_path=None,
        lora_vision_r=8,
        lora_vision_alpha=16,
        lora_vision_dropout=0.05,
        lora_vision_enable=False,
        lora_llm_enable=False,
        lora_llm_r=8,
        lora_llm_alpha=16,
        lora_llm_dropout=0.05,
        crop_size=224,
        # visual-language connector
        mm_projector_type="mlp",
        mm_projector_depth=2,

        # local refiner
        use_local_refiner=True,
        refiner_input_size=128,
        refiner_heatmap_size=64,
        refiner_text_dim=768,
        refiner_feat_dim=256,
        refiner_sigma=2.0,
        refiner_noise_ratio=0.25,
        refiner_crop_scale=1.0,
        refiner_use_text=True,
        lambda_hm=0.5,
        **kwargs,
    ):
        self.llama_path = llama_path
        self.dino_path = dino_path
        self.lora_vision_r = lora_vision_r
        self.lora_vision_alpha = lora_vision_alpha
        self.lora_vision_dropout = lora_vision_dropout
        self.lora_vision_enable = lora_vision_enable
        self.lora_llm_enable = lora_llm_enable
        self.lora_llm_r = lora_llm_r
        self.lora_llm_alpha = lora_llm_alpha
        self.lora_llm_dropout = lora_llm_dropout
        self.crop_size = crop_size
        self.mm_projector_type = mm_projector_type
        self.mm_projector_depth = mm_projector_depth

        self.use_local_refiner = use_local_refiner
        self.refiner_input_size = refiner_input_size
        self.refiner_heatmap_size = refiner_heatmap_size
        self.refiner_text_dim = refiner_text_dim
        self.refiner_feat_dim = refiner_feat_dim
        self.refiner_sigma = refiner_sigma
        self.refiner_noise_ratio = refiner_noise_ratio
        self.refiner_crop_scale = refiner_crop_scale
        self.refiner_use_text = refiner_use_text
        self.lambda_hm = lambda_hm

        super().__init__(**kwargs)


class ADCCRModel(LlamaForCausalLM):
    config_class = ADCCRConfig

    def __init__(self, config: ADCCRConfig):
        with lora(r=config.lora_llm_r, alpha=config.lora_llm_alpha, dropout=config.lora_llm_dropout, enabled=config.lora_llm_enable):
            super().__init__(config)
        # Initialize weights and apply final processing
        with lora_dino(r=config.lora_vision_r, alpha=config.lora_vision_alpha, dropout=config.lora_vision_dropout, enabled=config.lora_vision_enable):
            # from transformers import CLIPVisionModel
            vision_model = vit_large(patch_size=14, img_size=518, drop_path_rate=0.4, drop_path_uniform=True, init_values=1.0, block_chunks=0)
            state_dict = torch.load(config.dino_path)
            msg = vision_model.load_state_dict(state_dict, strict=False)
            print("dino init: {}".format(msg))
            self.vision_model = vision_model
        for module_name, module in self.vision_model.named_modules():
            module._is_hf_initialized = True

        num_features = self.vision_model.num_features
        if config.mm_projector_type == "linear":
            self.mm_projector = nn.Linear(
                num_features,
                config.hidden_size
            )

        elif config.mm_projector_type == "mlp":
            if config.mm_projector_depth != 2:
                raise ValueError(
                    "The manuscript defines a two-layer MLP connector; "
                    "mm_projector_depth must be 2."
                )

            self.mm_projector = nn.Sequential(
                nn.Linear(num_features, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

        else:
            raise ValueError(
                f"Unknown mm_projector_type: "
                f"{config.mm_projector_type}"
            )

        num_patches = (config.crop_size // self.vision_model.patch_size) ** 2
        self.config.num_patches = num_patches
        self.local_refiner = None
        self.description_projection = None

        if (
            config.use_local_refiner
            and config.refiner_use_text
        ):
            self.description_projection = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(
                    config.hidden_size,
                    config.refiner_text_dim
                ),
            )

            self.local_refiner = LocalRefiner(
                text_dim=config.refiner_text_dim,
                feat_dim=config.refiner_feat_dim,
                hm_size=config.refiner_heatmap_size,
                use_text=config.refiner_use_text,
            )
        elif config.use_local_refiner:
            self.local_refiner = LocalRefiner(
                text_dim=config.refiner_text_dim,
                feat_dim=config.refiner_feat_dim,
                hm_size=config.refiner_heatmap_size,
                use_text=False,
            )

    def get_model(self):
        return self.model
    
    def encode_image(self, images):
        image_forward_out = self.vision_model(images)
        image_features = image_forward_out['x_norm_patchtokens']
        image_features = self.mm_projector(image_features)
        return image_features

    def encode_descriptions(
            self,
            desc_input_ids,
            desc_attention_mask,
    ):
        """
        Encode the complete textual descriptions using
        Vicuna token embeddings, masked mean pooling,
        LayerNorm and a trainable projection.
        """
        if self.description_projection is None:
            raise RuntimeError(
                "Description encoder is not initialized."
            )

        token_features = self.model.embed_tokens(
            desc_input_ids
        )

        mask = desc_attention_mask.unsqueeze(-1)
        mask = mask.to(token_features.dtype)

        text_features = (
                token_features * mask
        ).sum(dim=1)

        denominator = mask.sum(dim=1).clamp_min(1.0)
        text_features = text_features / denominator

        projection_dtype = next(
            self.description_projection.parameters()
        ).dtype

        text_features = text_features.to(
            projection_dtype
        )

        return self.description_projection(
            text_features
        )

    def compute_refiner_loss(
            self,
            images,
            desc_input_ids,
            desc_attention_mask,
            refine_target_xy,
            refine_crop_sizes,
            refine_image_indices,
    ):
        if self.local_refiner is None:
            raise RuntimeError(
                "Local refiner is not initialized."
            )

        refiner_dtype = next(
            self.local_refiner.parameters()
        ).dtype

        refine_images = images.index_select(
            0,
            refine_image_indices
        ).to(refiner_dtype)

        refine_target_xy = refine_target_xy.to(
            device=refine_images.device,
            dtype=refiner_dtype,
        )

        refine_crop_sizes = refine_crop_sizes.to(
            device=refine_images.device,
            dtype=refiner_dtype,
        )

        # U(-0.25s, 0.25s) for each coordinate.
        perturbation = (
                torch.rand_like(refine_target_xy) * 2.0
                - 1.0
        )
        perturbation = (
                perturbation
                * self.config.refiner_noise_ratio
                * refine_crop_sizes.unsqueeze(-1)
        )

        noisy_centers = (
                refine_target_xy + perturbation
        )

        patches, crop_boxes = crop_and_resize(
            images=refine_images,
            centers=noisy_centers,
            crop_sizes=refine_crop_sizes,
            output_size=self.config.refiner_input_size,
        )

        text_features = None
        if self.config.refiner_use_text:
            text_features = self.encode_descriptions(
                desc_input_ids=desc_input_ids,
                desc_attention_mask=desc_attention_mask,
            )

        predicted_heatmaps = self.local_refiner(
            patches,
            text_features,
        )

        target_heatmaps = build_gaussian_heatmaps(
            target_xy=refine_target_xy,
            crop_boxes=crop_boxes,
            heatmap_size=(
                self.config.refiner_heatmap_size
            ),
            sigma=self.config.refiner_sigma,
        )

        return F.mse_loss(
            predicted_heatmaps,
            target_heatmaps,
        )

    @torch.no_grad()
    def refine_coordinates(
            self,
            images,
            coarse_xy,
            crop_sizes,
            desc_input_ids,
            desc_attention_mask,
    ):
        """
        Args:
            images: [N, 3, 224, 224]
            coarse_xy: [N, 2], in 224x224 coordinates
            crop_sizes: [N]
        """
        if self.local_refiner is None:
            raise RuntimeError(
                "Checkpoint does not contain a local refiner."
            )

        refiner_dtype = next(
            self.local_refiner.parameters()
        ).dtype

        images = images.to(refiner_dtype)
        coarse_xy = coarse_xy.to(
            device=images.device,
            dtype=refiner_dtype,
        )
        crop_sizes = crop_sizes.to(
            device=images.device,
            dtype=refiner_dtype,
        )

        patches, crop_boxes = crop_and_resize(
            images=images,
            centers=coarse_xy,
            crop_sizes=crop_sizes,
            output_size=self.config.refiner_input_size,
        )

        text_features = None
        if self.config.refiner_use_text:
            text_features = self.encode_descriptions(
                desc_input_ids,
                desc_attention_mask,
            )

        heatmaps = self.local_refiner(
            patches,
            text_features,
        )

        refined_xy = heatmaps_to_global(
            heatmaps,
            crop_boxes,
        )

        return refined_xy, heatmaps

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        has_images: Optional[bool] = None,
        desc_input_ids: Optional[torch.LongTensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        refine_target_xy: Optional[torch.Tensor] = None,
        refine_crop_sizes: Optional[torch.Tensor] = None,
        refine_image_indices: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        assert inputs_embeds is None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        useful_images = []
        for image, has_image in zip(images, has_images):
            if has_image:
                useful_images.append(image)
        if len(useful_images) > 0:
            useful_images = torch.stack(useful_images, dim=0)
            image_features = self.encode_image(useful_images)

        new_inputs_embeds = []
        cur_image_index = 0
        batch_id = 0
        for input_id, has_image in zip(input_ids, has_images):
            if has_image and (input_id.shape[0] != 1 or self.training):
                image_feature = image_features[cur_image_index]
                cur_image_index += 1
                num_patches = image_feature.shape[0]
                if (input_id == self.config.im_patch_token).sum() != num_patches:
                    raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(input_id == self.config.im_patch_token)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                pre_input_embed = self.model.embed_tokens(input_ids[batch_id:batch_id+1, :mask_index_start])[0]
                nxt_input_embed = self.model.embed_tokens(input_ids[batch_id:batch_id+1, mask_index_start+num_patches:])[0]

                image_feature = image_feature.to(pre_input_embed.dtype)

                new_inputs_embed = torch.cat((pre_input_embed, image_feature, nxt_input_embed), dim=0)
                new_inputs_embeds.append(new_inputs_embed)
            else:
                inputs_embed = self.model.embed_tokens(input_ids[batch_id:batch_id+1])[0]
                new_inputs_embeds.append(inputs_embed)
            batch_id += 1

        new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        llama_output = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = llama_output[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if self.training and self.config.use_local_refiner:
            required_refiner_inputs = [
                refine_target_xy,
                refine_crop_sizes,
                refine_image_indices,
            ]
            if self.config.refiner_use_text:
                required_refiner_inputs.extend(
                    [
                        desc_input_ids,
                        desc_attention_mask,
                    ]
                )

            if any(
                    item is None
                    for item in required_refiner_inputs
            ):
                raise ValueError(
                    "Local refiner is enabled, but the training "
                    "batch does not contain all refiner inputs."
                )

            heatmap_loss = self.compute_refiner_loss(
                images=images,
                desc_input_ids=desc_input_ids,
                desc_attention_mask=desc_attention_mask,
                refine_target_xy=refine_target_xy,
                refine_crop_sizes=refine_crop_sizes,
                refine_image_indices=refine_image_indices,
            )

            if loss is None:
                loss = (
                        self.config.lambda_hm
                        * heatmap_loss
                )
            else:
                loss = (
                        loss
                        + self.config.lambda_hm
                        * heatmap_loss
                )

        if not return_dict:
            output = (logits,) + llama_output[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "has_images": kwargs.get("has_images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, tokenizer):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]


AutoConfig.register("ADCCR", ADCCRConfig)
AutoModelForCausalLM.register(ADCCRConfig, ADCCRModel)
