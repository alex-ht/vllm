# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
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
"""PyTorch Mllama model."""
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union, cast)
import math

import numpy as np
import torch
import torch.utils.checkpoint
from transformers import WhisperConfig
from transformers.models.whisper import WhisperFeatureExtractor
import transformers.models.mllama.configuration_mllama as config_mllama
from PIL import Image
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mllama.image_processing_mllama import (
    get_optimal_tiled_canvas)
from transformers.models.mllama.processing_mllama import (
    get_cross_attention_token_mask)

import vllm.distributed.parallel_state as ps
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DummyData, EncoderDecoderInputs,
                         InputContext, TokenInputs, token_inputs, DecoderOnlyInputs)
from vllm.logger import init_logger
from .interfaces import SupportsMultiModal
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from .llama import LlamaDecoderLayer
from .mllama import (
    _get_num_image_in_last_group,
    convert_dense_cross_attention_mask_to_tensor,
    convert_sparse_cross_attention_mask_to_dense, skip_attention_mask,
    MllamaVisionModel, MllamaCrossAttentionDecoderLayer)
from .ultravox import (cached_feature_extractor,
    ModifiedWhisperEncoder,
    StackAudioFrames, FlippedSiluAndMul, 
    UltravoxAudioInputs, UltravoxAudioFeatureInputs, UltravoxAudioEmbeddingInputs)
from .utils import (maybe_prefix, flatten_bn, merge_multimodal_embeddings_from_map)

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.utils import (consecutive_placeholder_ranges, cached_get_tokenizer,
    repeat_and_pad_placeholder_tokens)
from vllm.sequence import SequenceData
from vllm.utils import is_list_of
from vllm.transformers_utils.configs.ocismllama import OcisMllamaConfig, MllamaAudioConfig

logger = init_logger(__name__)
MLLAMA_IMAGE_TOKEN_ID = 128256
MLLAMA_IMAGE_TOKEN = "<|image|>"
_AUDIO_PLACEHOLDER_TOKEN = 128002
_AUDIO_TOKENS_PER_SECOND = 6.25

def whisper_feature_extractor(ctx: InputContext) -> WhisperFeatureExtractor:
    return cached_feature_extractor(
        ctx.get_hf_config(OcisMllamaConfig).audio_config.audio_model_id)

def get_ultravox_max_audio_tokens(ctx: InputContext):
    feature_extractor = whisper_feature_extractor(ctx)
    return math.ceil(feature_extractor.chunk_length * _AUDIO_TOKENS_PER_SECOND)

def input_mapper_for_ultravox(ctx: InputContext, data: object):
    if not isinstance(data, list):
        data = [data]

    if len(data) == 0:
        return MultiModalKwargs()

    # If the audio inputs are embeddings, no need for preprocessing
    if is_list_of(data, torch.Tensor, check="all"):
        return MultiModalKwargs({"audio_embeds": data})

    audio_features = []
    for audio_input in data:
        if not isinstance(audio_input, tuple):
            raise NotImplementedError(
                f"Unsupported data type: {type(audio_input)}")

        (audio, sr) = cast(Tuple[np.ndarray, Union[float, int]], audio_input)
        feature_extractor = whisper_feature_extractor(ctx)

        if sr != feature_extractor.sampling_rate:
            try:
                import librosa
            except ImportError as exc:
                raise ImportError(
                    "Please install vllm[audio] for audio support.") from exc
            audio = librosa.resample(audio,
                                     orig_sr=sr,
                                     target_sr=feature_extractor.sampling_rate)
            sr = feature_extractor.sampling_rate

        minimum_audio_length = feature_extractor.n_fft // 2 + 1
        if len(audio) < minimum_audio_length:
            # Not enough audio; pad it.
            audio = np.pad(audio, (0, minimum_audio_length - len(audio)))

        single_audio_features = feature_extractor(
            audio, sampling_rate=sr, padding="longest",
            return_tensors="pt")["input_features"]

        # Remove the batch dimension because we're wrapping it in a list.
        audio_features.append(single_audio_features.squeeze(0))

    return MultiModalKwargs({"audio_features": audio_features})

def input_processor_for_ultravox(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return inputs

    if "multi_modal_placeholders" in inputs and "audio" in inputs[
            "multi_modal_placeholders"]:
        # The inputs already have placeholders.
        return inputs

    feature_extractor = whisper_feature_extractor(ctx)
    audios = multi_modal_data["audio"]
    if not isinstance(audios, list):
        audios = [audios]

    audio_token_counts = []
    for audio in audios:
        if isinstance(audio, torch.Tensor):
            audio_num_tokens = audio.shape[1]
            audio_token_counts.append(audio_num_tokens)
        else:
            audio_data, sample_rate = audio
            audio_length = audio_data.shape[0]
            if sample_rate != feature_extractor.sampling_rate:
                # Account for resampling.
                adjustment = feature_extractor.sampling_rate / sample_rate
                audio_length = math.ceil(adjustment * audio_length)

            feature_extractor_output_length = math.ceil(
                (audio_length - (feature_extractor.hop_length - 1)) /
                feature_extractor.hop_length)

            uv_config = ctx.get_hf_config(OcisMllamaConfig)
            audio_num_tokens = min(
                max(
                    1,
                    math.ceil(feature_extractor_output_length /
                              (uv_config.audio_config.stack_factor * 2))),
                get_ultravox_max_audio_tokens(ctx))
            audio_token_counts.append(audio_num_tokens)

    tokenizer = cached_get_tokenizer(ctx.model_config.tokenizer)

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        inputs.get("prompt"),
        inputs["prompt_token_ids"],
        placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN,
        repeat_count=audio_token_counts,
    )

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"audio": ranges})
    
def dummy_audio_for_ultravox(
    ctx: InputContext,
    audio_count: int,
):
    feature_extractor = whisper_feature_extractor(ctx)
    audio_and_sr = (np.array([0.0] * feature_extractor.chunk_length), 1)
    return {"audio": [audio_and_sr] * audio_count}

class UltravoxProjector(nn.Module):

    def __init__(self, config: MllamaAudioConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim = config.input_hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim)
        self.linear_1 = nn.Linear(dim, self.hidden_dim, bias=False)
        dim = self.hidden_dim

        self.act = FlippedSiluAndMul()
        dim = dim // 2

        self.linear_2 = nn.Linear(dim,
                                  config.output_hidden_size,
                                  bias=False)
        self.ln_post = RMSNorm(config.output_hidden_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class MllamaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: """
    """(batch_size, max_num_image, max_num_chunk, num_channel, height, width)"""
    aspect_ratio_ids: torch.Tensor
    """Shape: `(batch_size, max_num_image)`"""
    aspect_ratio_mask: torch.Tensor
    """Shape: `(batch_size, max_num_image, max_num_tiles)`"""


# TODO: support LlamaImageEmbeddingInputs


def input_processor_for_mllama(
    ctx: InputContext,
    inputs: EncoderDecoderInputs,
) -> EncoderDecoderInputs:
    # Example input to processor:
    # {
    #     'encoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000, 128256, 128000, 3923, 374, 279, 2262, 315, 420, 2217, 30],  # noqa: E501
    #         'prompt': '<|image|><|begin_of_text|>What is the content of this image?',  # noqa: E501
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    #     'decoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000],
    #     },
    # }

    # move encoder prompt to decoder
    dec_inputs = TokenInputs(**inputs["encoder"])

    multi_modal_data = dec_inputs.get("multi_modal_data")
    
    dec_inputs = input_processor_for_ultravox(ctx, dec_inputs)
        
    if multi_modal_data is None or "image" not in multi_modal_data:
        # decoder-only
        return EncoderDecoderInputs(
            encoder=token_inputs([]),
            decoder=dec_inputs,
        )

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        image_data = [image_data]

    assert is_list_of(image_data, Image.Image)

    # Since only the last group of consecutive images
    # are attended by the decoded tokens, we only need to
    # get the number of tiles for those images.
    num_decode_images = _get_num_image_in_last_group(
        dec_inputs["prompt_token_ids"])

    hf_config = ctx.model_config.hf_config
    vision_config = hf_config.vision_config

    num_tiles = 0
    for image in image_data[::-1]:
        width, height = image.size
        tile_size = vision_config.image_size
        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=height,
            image_width=width,
            max_image_tiles=vision_config.max_num_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size
        num_tiles += num_tiles_height * num_tiles_width
        num_decode_images -= 1
        if num_decode_images == 0:
            break

    # Set encoder prompt length based on the number of tiles.
    # This tells the block manager to allocate correct number
    # of slots for encoder tokens.
    assert vision_config.image_size % 14 == 0, \
        "chunk size should be multiple of 14"
    token_per_chunk = (vision_config.image_size // 14)**2 + 1
    num_tokens = num_tiles * token_per_chunk

    # Example output from processor:
    # {
    #     'encoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128256, 128256, ..., 128256],
    #         'prompt': '<|image|><|image|>...<|image|>',
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    #     'decoder': {
    #         'type': 'token',
    #         'prompt_token_ids': [128000, 128256, 128000, 3923, 374, 279, 2262, 315, 420, 2217, 30],  # noqa: E501
    #         'prompt': '<|image|><|begin_of_text|>What is the content of this image?',  # noqa: E501
    #         'multi_modal_data': {'image': <PIL.Image.Image image mode=RGB size=1770x1180 at 0x7FDE2C624880>},  # noqa: E501
    #     },
    # }
    return EncoderDecoderInputs(
        encoder=token_inputs(
            prompt_token_ids=[MLLAMA_IMAGE_TOKEN_ID] * num_tokens,
            prompt=MLLAMA_IMAGE_TOKEN * num_tokens,
            multi_modal_data=multi_modal_data,
        ),
        decoder=dec_inputs,
    )


def get_max_mllama_image_tokens(ctx: InputContext) -> int:
    hf_config = ctx.model_config.hf_config
    token_per_chunk = (hf_config.vision_config.image_size // 14)**2 + 1
    return hf_config.vision_config.max_num_tiles * token_per_chunk


def dummy_encoder_seq_data(ctx: InputContext, num_images: int):
    num_tokens = get_max_mllama_image_tokens(ctx) * num_images

    return SequenceData.from_prompt_token_counts(
        (MLLAMA_IMAGE_TOKEN_ID, num_tokens))


def dummy_image(num_images: int, ):
    width = height = 1024
    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}

        
def dummy_decoder_data_for_mllama(ctx: InputContext, seq_len: int,
                                  mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    audio_count = mm_counts["audio"]
    
    # <|image|> * num_images + 0 * (seq_len - num_images)
    assert seq_len >= num_images, \
        "seq_len should be greater than or equal to num_images"
    audio_length = min(get_ultravox_max_audio_tokens(ctx),
                       seq_len // audio_count)
    
    seq_data = SequenceData.from_prompt_token_counts(
        (MLLAMA_IMAGE_TOKEN_ID, num_images),
        (_AUDIO_PLACEHOLDER_TOKEN, audio_length * audio_count),
        (0, seq_len - num_images - audio_length * audio_count),
    )
    ranges = dict(
        audio=consecutive_placeholder_ranges(num_items=audio_count,
                                             item_size=audio_length))
    mm_audio = dummy_audio_for_ultravox(ctx, audio_count)
    mm_image = dummy_image(num_images)
    return DummyData(seq_data, mm_audio|mm_image, ranges)


def dummy_encoder_data_for_mllama(ctx: InputContext, seq_len: int,
                                  mm_counts: Mapping[str, int]):
    num_images = mm_counts["image"]
    return DummyData(dummy_encoder_seq_data(ctx, num_images))


class MllamaTextModel(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "model"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size + 8,
                                                   config.hidden_size)
        self.cross_attention_layers = config.cross_attention_layers

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(
                    MllamaCrossAttentionDecoderLayer(
                        config,
                        layer_idx,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    ))
            else:
                # TODO: force LlamaDecoderLayer to config.attention_bias=False
                layers.append(
                    LlamaDecoderLayer(
                        config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    ))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        kv_range_for_decode: Optional[List[Tuple[int, int]]],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor,
                                                      torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        skip_cross_attention: bool,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if isinstance(decoder_layer, MllamaCrossAttentionDecoderLayer):
                if not skip_cross_attention:
                    hidden_states = decoder_layer(
                        hidden_states=hidden_states,
                        cross_attention_states=cross_attention_states,
                        cross_attention_mask=cross_attention_mask,
                        kv_range_for_decode=kv_range_for_decode,
                        full_text_row_masked_out_mask=
                        full_text_row_masked_out_mask,
                        kv_cache=kv_caches[idx],
                        attn_metadata=attn_metadata,
                    )
            elif isinstance(decoder_layer, LlamaDecoderLayer):
                hidden_states, residual = decoder_layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_caches[idx],
                    attn_metadata=attn_metadata,
                    residual=None,
                )
                hidden_states = hidden_states + residual
            else:
                raise ValueError(
                    f"Unknown decoder layer type {type(decoder_layer)}")
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MllamaForCausalLM(nn.Module):
    config_class = config_mllama.MllamaTextConfig
    base_model_prefix = "language_model"
    _no_split_modules = [
        "MllamaCrossAttentionDecoderLayer", "MllamaSelfAttentionDecoderLayer"
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.model = MllamaTextModel(vllm_config=vllm_config,
                                     prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )
        
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        cross_attention_mask: Optional[torch.LongTensor],
        kv_range_for_decode: Optional[List[Tuple[int, int]]],
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor,
                                                      torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        skip_cross_attention: bool,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            kv_range_for_decode=kv_range_for_decode,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            skip_cross_attention=skip_cross_attention,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_mllama_image_tokens)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_ultravox)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_ultravox_max_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_decoder_data_for_mllama)
@INPUT_REGISTRY.register_dummy_encoder_data(dummy_encoder_data_for_mllama)
@INPUT_REGISTRY.register_input_processor(input_processor_for_mllama)
class OcisMllamaForConditionalGeneration(nn.Module, SupportsMultiModal):
    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: OcisMllamaConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = \
            config.pad_token_id if config.pad_token_id is not None else -1
        self.image_size = config.vision_config.image_size

        self.vision_model = MllamaVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_model"))
        self.language_model = MllamaForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.multi_modal_projector = ColumnParallelLinear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            gather_output=True,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )
        self.audio_projector = UltravoxProjector(config.audio_config)
        whisper_config = WhisperConfig.from_pretrained(config.audio_config.audio_model_id)
        self.audio_model = ModifiedWhisperEncoder(whisper_config)
        self.logits_processor = LogitsProcessor(config.output_hidden_states,
                                                config.text_config.vocab_size)
        self.sampler = get_sampler()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.language_model.lm_head,
                                       hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def _parse_and_validate_image_input(self, **kwargs: object):
        # tensor with the same shape will be batched together by
        # MultiModalKwargs.batch, so pixel_values here can be:
        #   - List[List[torch.Tensor]]:
        #       with shape (num_tiles, 3, image_res, image_res)
        #   - List[torch.Tensor]:
        #       with shape (num_image, num_tiles, 3, image_res, image_res)
        #   - torch.Tensor:
        #       with shape (bs, num_image, num_tiles, 3, image_res, image_res)
        pixel_values: Optional[Union[List[List[torch.Tensor]],
                                     List[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "pixel_values", None)
        image_embeds: Optional[Union[List[List[torch.Tensor]],
                                     List[torch.Tensor],
                                     torch.Tensor]] = kwargs.pop(
                                         "image_embeds", None)
        aspect_ratio_ids: Optional[Union[List[List[torch.Tensor]],
                                         List[torch.Tensor],
                                         torch.Tensor]] = kwargs.pop(
                                             "aspect_ratio_ids", None)
        aspect_ratio_mask: Optional[Union[List[List[torch.Tensor]],
                                          List[torch.Tensor],
                                          torch.Tensor]] = kwargs.pop(
                                              "aspect_ratio_mask", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError(
                "Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            assert aspect_ratio_ids is not None
            assert aspect_ratio_mask is not None
            max_num_images = max([len(x[0]) for x in pixel_values])
            if max_num_images == 0:
                raise ValueError("No images provided.")
            max_num_tiles = max(
                max([len(x) for x in y[0]]) for y in pixel_values)
            device = next(self.multi_modal_projector.parameters()).device
            bsz = len(pixel_values)
            out_num_tiles = []
            out_images = torch.zeros(
                bsz,
                max_num_images,
                max_num_tiles,
                3,
                self.image_size,
                self.image_size,
                dtype=torch.float32,
                device=device,
            )
            out_ar_ids = torch.ones(bsz,
                                    max_num_images,
                                    dtype=torch.int64,
                                    device=device)
            out_ar_mask = torch.zeros(bsz,
                                      max_num_images,
                                      max_num_tiles,
                                      dtype=torch.int64,
                                      device=device)
            for b in range(len(pixel_values)):
                _num_tiles = []
                for i in range(len(pixel_values[b][0])):
                    img = pixel_values[b][0][i]
                    out_images[b, i, :img.shape[0]] = img
                    out_ar_ids[b, i] = aspect_ratio_ids[b][0][i]
                    out_ar_mask[b, i] = aspect_ratio_mask[b][0][i]
                    _num_tiles.append(img.shape[0])
                out_num_tiles.append(_num_tiles)

            return MllamaImagePixelInputs(
                type="pixel_values",
                data=out_images,
                aspect_ratio_ids=out_ar_ids,
                aspect_ratio_mask=out_ar_mask,
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def flat_encoder_result(self, cross_attention_states: torch.Tensor,
                            attn_metadata: AttentionMetadata,
                            actual_encoder_seq_lens: List[int]):

        cross_attention_states_flat = torch.zeros(
            sum(actual_encoder_seq_lens),
            cross_attention_states.shape[-1],
            device=cross_attention_states.device,
            dtype=cross_attention_states.dtype)
        start_pos = 0
        for seq_len, vision_token_in_batch in zip(actual_encoder_seq_lens,
                                                  cross_attention_states):
            end_pos = start_pos + seq_len
            cross_attention_states_flat[
                start_pos:end_pos] = vision_token_in_batch[:seq_len]
            start_pos = end_pos
        cross_attention_states = cross_attention_states_flat
        return cross_attention_states

    def get_cross_attention_states(
        self,
        image_inputs: MllamaImagePixelInputs,
        attn_metadata: AttentionMetadata,
        actual_encoder_seq_lens: List[int],
    ) -> Tuple[torch.Tensor]:
        # NOTE: llama's reference implementation runs vision model on CPU
        pixel_values = image_inputs['data']
        aspect_ratio_ids = image_inputs['aspect_ratio_ids']
        aspect_ratio_mask = image_inputs['aspect_ratio_mask']
        cross_attention_states = self.vision_model(pixel_values,
                                                   aspect_ratio_ids,
                                                   aspect_ratio_mask)
        cross_attention_states, _ = self.multi_modal_projector(
            cross_attention_states)

        bsz, _, _, _, image_token_dim = tuple(cross_attention_states.shape)
        cross_attention_states = cross_attention_states.view(
            bsz, -1, image_token_dim)

        cross_attention_states = self.flat_encoder_result(
            cross_attention_states, attn_metadata, actual_encoder_seq_lens)

        return cross_attention_states

    def get_cross_attention_mask(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_tiles: List[List[int]],
        num_tokens_per_tile: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids = input_ids.tolist()
        start = 0
        batch_token_ids = []
        for seq_len in attn_metadata.seq_lens:
            batch_token_ids.append(token_ids[start:start + seq_len])
            start += seq_len
        sparse_mask = [
            get_cross_attention_token_mask(t, MLLAMA_IMAGE_TOKEN_ID)
            for t in batch_token_ids
        ]

        # Skip generating cross-attention mask if all samples
        # are text-only or have only 1 leading image.
        if skip_attention_mask(sparse_mask):
            return None, None

        dense_mask, tile_range_for_decode = \
            convert_sparse_cross_attention_mask_to_dense(
                sparse_mask, num_tiles, attn_metadata.seq_lens)
        cross_attention_mask = \
            convert_dense_cross_attention_mask_to_tensor(
                dense_mask, num_tokens_per_tile, input_ids.device, dtype)
        kv_range_for_decode = [[
            t[0] * num_tokens_per_tile, t[1] * num_tokens_per_tile
        ] for t in tile_range_for_decode]

        return cross_attention_mask, kv_range_for_decode

    def get_full_text_row_masked_out_mask(
        self,
        attn_metadata: AttentionMetadata,
        device: torch.device,
    ) -> torch.Tensor:
        full_text_row_masked_out_mask = torch.ones(
            (attn_metadata.num_prefill_tokens, 1), dtype=torch.bool)
        start_pos = 0
        for seq_len, encoder_seq_len in zip(attn_metadata.seq_lens,
                                            attn_metadata.encoder_seq_lens):
            if encoder_seq_len == 0:
                full_text_row_masked_out_mask[start_pos:start_pos +
                                              seq_len] = False
            start_pos += seq_len
        full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(
            device)
        return full_text_row_masked_out_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if attn_metadata.num_prefill_tokens > 0 and \
            attn_metadata.num_decode_tokens > 0:
            raise ValueError("Chunk prefill not supported")
        image_inputs = self._parse_and_validate_image_input(**kwargs)
        cross_attention_states = None
        cross_attention_mask = None
        kv_range_for_decode = None

        # For 1) text-only prefill and decode, 2) image-present decode.
        if image_inputs is None:
            full_text_row_masked_out_mask = (
                attn_metadata.encoder_seq_lens_tensor != 0).reshape(-1, 1).to(
                    input_ids.device)
            skip_cross_attention = max(attn_metadata.encoder_seq_lens) == 0

        # For image-present prefill.
        else:
            skip_cross_attention = False

            # Get the actual number of encoder tokens for each sample.
            # Because attn_metadata.encoder_seq_lens only counts the last
            # group of images for each sample, which is used to cheat the
            # block manager to allocate blocks for those images only.
            # See input_processor_for_mllama() for more details.
            num_tiles_tensor = kwargs.pop("num_tiles")
            num_tiles = [t[0].tolist() for t in num_tiles_tensor]
            num_tokens_per_tile = (self.image_size // 14)**2 + 1
            actual_encoder_seq_lens = [
                sum(num_tile) * num_tokens_per_tile for num_tile in num_tiles
            ]
            for actual_len, last_group_len in zip(
                    actual_encoder_seq_lens, attn_metadata.encoder_seq_lens):
                assert actual_len >= last_group_len

            cross_attention_states = self.get_cross_attention_states(
                image_inputs, attn_metadata, actual_encoder_seq_lens)

            full_text_row_masked_out_mask = \
                self.get_full_text_row_masked_out_mask(
                    attn_metadata, input_ids.device)

            cross_attention_mask, kv_range_for_decode = \
                self.get_cross_attention_mask(
                    input_ids, attn_metadata, num_tiles,
                    num_tokens_per_tile, cross_attention_states.dtype)

        multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
        inputs_embeds = self.get_input_embeddings(input_ids,
                                                  multimodal_embeddings,
                                                  attn_metadata)

        outputs = self.language_model(
            input_ids=None,
            positions=positions,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            kv_range_for_decode=kv_range_for_decode,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            skip_cross_attention=skip_cross_attention,
            inputs_embeds=inputs_embeds,
        )

        return outputs

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: Set[str] = set()
        for name, loaded_weight in weights:
            if 'patch_embedding.weight' in name:
                name = name.replace('patch_embedding.weight',
                                    'patch_embedding._linear.weight')
                loaded_weight = loaded_weight.view(loaded_weight.shape[0], -1)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'audio_model' in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.pop(name)
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                updated_params.add(name)
        return updated_params

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        audio_embeddings = self._process_audio_input(audio_input)
        return audio_embeddings
    
    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[UltravoxAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return UltravoxAudioFeatureInputs(type="audio_features",
                                              data=audio_features)

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return UltravoxAudioEmbeddingInputs(type="audio_embeds",
                                                data=audio_embeds)

        raise AssertionError("This line should be unreachable.")
    
    def _process_audio_input(
            self, audio_input: UltravoxAudioInputs) -> NestedTensors:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        if isinstance(audio_features, torch.Tensor):
            # Combine the B and N dimensions for the encoder/projector
            flattened = flatten_bn(audio_features)
            flattened_embeddings = self._audio_features_to_embeddings(
                flattened)

            # Restore the original dimensions
            embeddings = flattened_embeddings.unflatten(
                0, audio_features.shape[:2])
            return embeddings

        result = []
        # TODO: Batch heterogeneous tensors through the encoder/projector
        for audio_features_item in audio_features:
            if isinstance(audio_features_item, torch.Tensor):
                result.append(
                    self._audio_features_to_embeddings(audio_features_item))
            else:
                embeddings = [
                    # Add a batch dimension to embed it, then remove it.
                    self._audio_features_to_embeddings(tensor.unsqueeze(0)
                                                       ).squeeze(0)
                    for tensor in audio_features_item
                ]
                result.append(embeddings)

        return result
    
    def _audio_features_to_embeddings(
            self, input_features: torch.Tensor) -> torch.Tensor:
        audio_input = input_features.to(self.audio_model.dtype)
        audio_features = self.audio_model(audio_input)
        audio_features = audio_features.to(self.audio_model.dtype)
        audio_embeddings = self.audio_projector(audio_features)
        return audio_embeddings
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:

            # TODO(ywang96): use merge_multimodal_embeddings after
            # v0 is deprecated
            merge_multimodal_embeddings_from_map(
                inputs_embeds, multimodal_embeddings,
                attn_metadata.multi_modal_placeholder_index_maps["audio"])
        return inputs_embeds