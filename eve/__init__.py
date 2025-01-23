from .model import LlavaLlamaForCausalLM
from .model import EveLLaVALlamaForCausalLM
# from .model import LlavaQWenForCausalLM
from .model import EveLLaVALlamaForCausalLM
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 34:
    from .model import LlavaMistralForCausalLM
    from .model import EveLLaVAMistralForCausalLM
if a == '4' and int(b) >= 36:
    from .model import LlavaMiniCPMForCausalLM
    from .model import EveLLaVAMiniCPMForCausalLM
    from .model import LlavaPhiForCausalLM
    from .model import EveLLaVAPhiForCausalLM
    from .model import LlavaStablelmForCausalLM
    from .model import EveLLaVAStablelmForCausalLM
if a == '4' and int(b) >= 37:
    from .model import LlavaQwen1_5ForCausalLM
    from .model import EveLLaVAQwen1_5ForCausalLM
