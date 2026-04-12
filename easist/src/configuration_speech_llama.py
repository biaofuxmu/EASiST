"""BLSP config"""

from transformers import PretrainedConfig, LlamaConfig
from transformers import logging
try:
    from .modeling_speech_model import SpeechConfigs
except:
    from modeling_speech_model import SpeechConfigs
# import pdb;pdb.set_trace()
logger = logging.get_logger(__name__)

class SpeechLlamaConfig(PretrainedConfig):
    def __init__(
        self, 
        speech_config=None, 
        llama_config=None,
        speech_model_type="wav2vec_s",
        adapter_type="ffn",
        adapter_inner_dim=512,
        conv_kernel_sizes="5,5",
        pad_id=128009,
        speech_label=False,
        text_label=True,
        cfd_weight=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        if speech_config is None:
            speech_config = {}
            logger.info("speech config is None. Initializing the SpeechConfig with default values")
        
        if llama_config is None:
            llama_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")
        
        self.speech_config = SpeechConfigs[speech_model_type](**speech_config).to_dict()
        
        self.llama_config = LlamaConfig(**llama_config).to_dict()

        self.speech_model_type = speech_model_type
        # self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_type = adapter_type
        self.adapter_inner_dim = adapter_inner_dim
        self.conv_kernel_sizes = conv_kernel_sizes if isinstance(conv_kernel_sizes, list) else[int(k) for k in conv_kernel_sizes.split(",")] 
        self.pad_id = pad_id
        self.speech_label = speech_label
        self.text_label = text_label
        self.cfd_weight = cfd_weight