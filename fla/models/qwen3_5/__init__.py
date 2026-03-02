
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from fla.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextModel

AutoConfig.register(Qwen3_5TextConfig.model_type, Qwen3_5TextConfig, exist_ok=True)
AutoModel.register(Qwen3_5TextConfig, Qwen3_5TextModel, exist_ok=True)
AutoModelForCausalLM.register(Qwen3_5TextConfig, Qwen3_5ForCausalLM, exist_ok=True)

__all__ = ['Qwen3_5ForCausalLM', 'Qwen3_5TextConfig', 'Qwen3_5TextModel']
