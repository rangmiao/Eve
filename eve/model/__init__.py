from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .language_model.llava_llama_moe import EveLLaVALlamaForCausalLM, EveLLaVALlamaConfig
# from .language_model.llava_qwen import LlavaQWenForCausalLM, LlavaQWenConfig
# from .language_model.llava_qwen_moe import EveLLaVAQWenForCausalLM, EveLLaVAQWenConfig
from .language_model.llava_zhizi import LlavaZhiziForCausalLM, LlavaZhiziConfig
from .language_model.llava_zhizi_moe import EveLlavaZhiziForCausalLM, EveLlavaZhiziConfig
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
    from .language_model.llava_mistral_moe import EveLLaVAMistralForCausalLM, EveLLaVAMistralConfig
if a == '4' and int(b) >= 36:
    from .language_model.llava_minicpm import LlavaMiniCPMForCausalLM, LlavaMiniCPMConfig
    from .language_model.llava_minicpm_moe import EveLLaVAMiniCPMForCausalLM, EveLLaVAMiniCPMConfig
    from .language_model.llava_phi import LlavaPhiForCausalLM, LlavaPhiConfig
    from .language_model.llava_phi_moe import EveLLaVAPhiForCausalLM, EveLLaVAPhiConfig
    from .language_model.llava_stablelm import LlavaStablelmForCausalLM, LlavaStablelmConfig
    from .language_model.llava_stablelm_moe import EveLLaVAStablelmForCausalLM, EveLLaVAStablelmConfig
if a == '4' and int(b) >= 37:
    from .language_model.llava_qwen1_5 import LlavaQwen1_5ForCausalLM, LlavaQwen1_5Config
    from .language_model.llava_qwen1_5_moe import EveLLaVAQwen1_5ForCausalLM, EveLLaVAQwen1_5Config
if a == '4' and int(b) <= 31:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
