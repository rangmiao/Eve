import os
import torch
import transformers
from moellava.model.language_model.llava_zhizi import LlavaZhiziForCausalLM
# ckpt
model_path = "/home/ma-user/work/project/LMM/result/moe-llava/stage2_models/zhizi_new_siglip_from_deepseek_chat/mobilevlm-2.finetune"
base_model_path = '/home/ma-user/work/project/LMM/result/moe-llava/stage2_models/llavaZhizi-1.5b-2.finetune_new'
output_path = "/home/ma-user/work/project/LMM/result/moe-llava/stage2_models/llavaZhizi-1.5b-2.finetune_new_siglip_ap63"

# os.system(f"cp -r {model_path} {output_path}")
# os.system(f"cp -r {base_model_path}/config.json {output_path}/config.json")

ckpt_path = os.path.join(model_path, "pytorch_model.bin")
ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt = {k.replace('vision_tower', 'image_tower'):v for k,v in ckpt.items()}
ckpt = {k.replace('mm_projector', 'mm_projector.image_spatial_proj'):v for k,v in ckpt.items()}


kwargs = {}
kwargs["device_map"] = 'cuda'
kwargs['torch_dtype'] = torch.float16
tokenizer = transformers.LlamaTokenizer.from_pretrained(output_path, use_fast=False)
model = LlavaZhiziForCausalLM.from_pretrained(output_path, low_cpu_mem_usage=True, **kwargs)
model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
model.config.eos_token_id = tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('</s>')



# output
print("start load ckpt")
model.load_state_dict(ckpt)
print("✅ The model is loaded successful!")

# model
out_ckpt_path = os.path.join(output_path, "pytorch_model.bin")
torch.save(model.state_dict(), out_ckpt_path)
print("reload model finish!")
