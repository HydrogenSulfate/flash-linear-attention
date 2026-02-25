
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.kimi.configuration_kimi import KimiLinearConfig
from fla.models.kimi.modeling_kimi import KimiLinearForCausalLM, KimiLinearModel

AutoConfig.register(KimiLinearConfig.model_type, KimiLinearConfig, exist_ok=True)
AutoModel.register(KimiLinearConfig, KimiLinearModel, exist_ok=True)
AutoModelForCausalLM.register(KimiLinearConfig, KimiLinearForCausalLM, exist_ok=True)

__all__ = ['KimiLinearConfig', 'KimiLinearForCausalLM', 'KimiLinearModel']
