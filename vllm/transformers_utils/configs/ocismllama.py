# coding=utf-8
"""OcisMllama model configuration"""
from transformers import WhisperConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mllama.configuration_mllama import (
    MllamaTextConfig, MllamaVisionConfig)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MllamaAudioConfig(PretrainedConfig):
    model_type = "ocismllama"

    def __init__(
        self,
        output_hidden_size: int = 4096,
        hidden_size: int = 4096,
        audio_model_id: str = 'AlexHung29629/whisper-large-v3-turbo-encoder',
        stack_factor: int = 8,
        norm_init: float = 0.4,
        **kwargs,
    ):
        self.output_hidden_size = output_hidden_size
        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.audio_model_id = audio_model_id
        whisper_config = WhisperConfig.from_pretrained(audio_model_id)
        self.input_hidden_size = whisper_config.hidden_size
        super().__init__(**kwargs)

class OcisMllamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.
    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vision_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaTextConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 128256):
            The image token index to encode the image prompt.
    Example:
    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig
    >>> # Initializing a CLIP-vision config
    >>> vision_config = MllamaVisionConfig()
    >>> # Initializing a Llama config
    >>> text_config = MllamaTextConfig()
    >>> # Initializing a mllama-11b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)
    >>> # Initializing a model from the mllama-11b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ocismllama"
    sub_configs = {"vision_config": MllamaVisionConfig,
                   "text_config": MllamaTextConfig,
                   "audio_config": MllamaAudioConfig}
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        audio_config=None,
        image_token_index=128256,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaVisionConfig):
            self.vision_config = vision_config

        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = MllamaTextConfig(**text_config)
        elif isinstance(text_config, MllamaTextConfig):
            self.text_config = text_config
        
        if audio_config is None:
            self.audio_config = MllamaAudioConfig(output_hidden_size=self.text_config.hidden_size)
            logger.info("audio_config is None, using default mllama audio config")
        elif isinstance(audio_config, dict):
            self.audio_config = MllamaAudioConfig(**audio_config)
        elif isinstance(audio_config, MllamaAudioConfig):
            self.audio_config = audio_config
        
        self.text_config.is_encoder_decoder = True

        super().__init__(**kwargs)