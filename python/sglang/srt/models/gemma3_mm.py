# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3_mm.py

import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypedDict

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.multi_modality_padding import (
    MultiModalityDataPaddingPatternTokenPairs,
)
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3 import Gemma3Model
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Gemma3ImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class SiglipVisionModel(nn.Module):
    """Vision model for Gemma3 multimodal."""

    def __init__(
        self,
        vision_config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vision_config
        self.embed_dim = vision_config.hidden_size
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim)
        )

        self.pre_layernorm = nn.LayerNorm(self.embed_dim)

        self.layers = nn.ModuleList(
            [
                SiglipVisionLayer(
                    vision_config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(vision_config.num_hidden_layers)
            ]
        )

        self.post_layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]

        # Extract patches
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        # Add class token
        class_embedding = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat([class_embedding, x], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding

        # Apply pre-layernorm
        x = self.pre_layernorm(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Apply post-layernorm
        x = self.post_layernorm(x)

        return x


class SiglipVisionLayer(nn.Module):
    """Vision transformer layer for Gemma3 multimodal."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=False,
            use_context_forward=False,
            softmax_in_single_precision=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.attention_layernorm = nn.LayerNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp.0", prefix),
            ),
            nn.GELU(),
            RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp.2", prefix),
            ),
        )
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_layernorm(hidden_states)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        hidden_states_parallel, _ = mlp_fc1(hidden_states)
        hidden_states_parallel = mlp_act(hidden_states_parallel)
        hidden_states, _ = mlp_fc2(hidden_states_parallel)

        hidden_states = residual + hidden_states
        return hidden_states


class Gemma3MultiModalProjector(nn.Module):
    """Projector for Gemma3 multimodal."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image ** 0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        print("multi modal embeding...")
        batch_size, seq_length, hidden_size = vision_outputs.shape

        # Reshape for pooling
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        # Apply pooling
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        # Apply normalization
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        # Project to text embedding space
        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )

        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3ForConditionalGeneration(nn.Module):
    """Gemma3 multimodal model for conditional generation."""

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Vision components
        self.vision_tower = SiglipVisionModel(
            config.vision_config,
            quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        self.multi_modal_projector = Gemma3MultiModalProjector(config)

        # Text model
        self.model = Gemma3Model(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def calculate_num_image_tokens(self, image_grid_thw: Tuple[int, int, int]) -> int:
        """Calculate the number of image tokens for a given image grid."""
        processor = cached_get_processor(self.config._name_or_path)
        return self.config.mm_tokens_per_image

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: ImageInputs
    ) -> List[int]:
        """Pad input IDs with image tokens."""
        # Get special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)
        return pattern.pad_input_tokens(input_ids, image_inputs)

    def _process_image_input(self, image_input: Gemma3ImagePixelInputs) -> torch.Tensor:
        """Process image input to get embeddings."""
        pixel_values = image_input["pixel_values"]
        vision_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(vision_outputs)

    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs,
    ) -> Dict:
        """Prepare attention masks for multimodal inputs."""
        kwargs["has_images"] = True

        # Distinguish sequences by position id 0
        start_indices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_indices)
        seq_lens = []

        for i in range(num_seqs):
            start_idx = start_indices[i].item()
            if i < num_seqs - 1:
                end_idx = start_indices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)

        kwargs["seq_lens"] = seq_lens

        # Create attention masks
        global_attn_masks = []
        local_attn_masks = []
        sliding_window = self.config.text_config.interleaved_sliding_window

        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx

            # Create global causal mask
            global_attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            global_attn_mask.fill_(float("-inf"))
            global_attn_mask = global_attn_mask.triu(diagonal=1)

            # Consider bidirectional attention between image tokens
            img_mask = torch.zeros_like(global_attn_mask)
            img_pos = input_token_ids == self.config.image_token_index
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            global_attn_mask = torch.where(img_mask == 2, 0, global_attn_mask)
            global_attn_masks.append(global_attn_mask)

            # Create local causal mask with sliding window
            local_attn_mask = torch.ones_like(global_attn_mask)
            local_attn_mask = torch.tril(local_attn_mask, diagonal=-sliding_window)
            local_attn_mask = torch.where(
                local_attn_mask == 0, global_attn_mask, float("-inf")
            )
            local_attn_masks.append(local_attn_mask)

        kwargs["global_attn_masks"] = global_attn_masks
        kwargs["local_attn_masks"] = local_attn_masks
        return kwargs

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for Gemma3 multimodal model."""
        image_inputs = None
        if forward_batch.image_inputs is not None:
            image_inputs = [
                img for img in forward_batch.image_inputs if img is not None
            ]

        if (
            forward_batch.forward_mode.is_decode()
            or image_inputs is None
            or len(image_inputs) == 0
        ):
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            # Clamp input ids to valid range
            input_ids.clamp_(min=0, max=self.config.text_config.vocab_size - 1)

            inputs_embeds = self.model.embed_tokens(input_ids)
            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

            for i, image in enumerate(forward_batch.image_inputs):
                if image is None or image.pixel_values is None:
                    continue

                start_idx = extend_start_loc_cpu[i]
                prefix_len = prefix_lens_cpu[i]
                pixel_values = image.pixel_values.clone()

                # Process image
                image_input = Gemma3ImagePixelInputs(pixel_values=pixel_values)
                image_embeds = self._process_image_input(image_input)

                # Replace token embeddings with image embeddings
                image_embeds_offset = 0
                for idx, image_offset in enumerate(image.image_offsets):
                    if image_offset < prefix_len:
                        continue

                    num_image_tokens = self.config.mm_tokens_per_image

                    left_idx = start_idx + (image_offset - prefix_len + 1)
                    right_idx = left_idx + num_image_tokens
                    inputs_embeds[left_idx:right_idx] = image_embeds[
                                                        image_embeds_offset: image_embeds_offset + num_image_tokens
                                                        ]
                    image_embeds_offset += num_image_tokens

            # Prepare attention masks for multimodal inputs
            kwargs = self.prepare_attn_masks(
                input_ids, positions, mask_dtype=inputs_embeds.dtype, **kwargs
            )
            input_ids = None

        # Forward through the language model
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
            **kwargs,
        )

        # Process logits
        return self.logits_processor(
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue

                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip lm_head.weight as it's tied with embed_tokens
                if "lm_head.weight" in name:
                    continue

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)


EntryClass = Gemma3ForConditionalGeneration
