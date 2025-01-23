
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, Cache
from .zhizi.configuration_internlm import InternLMConfig
from .zhizi.modeling_internlm import InternLMModel, InternLMForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from deepspeed.moe.layer import Eve
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class EveLlavaZhiziConfig(InternLMConfig):
    model_type = "moe_llava_zhizi"
    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[
                # 'up_proj', 'down_proj', 'gate_proj', 'wg',
                # 'embed_tokens', 'lm_head'
            ]
        )
        super(EveLlavaZhiziConfig, self).__init__(**kwargs)

class EveLlavaZhiziModel(LlavaMetaModel, InternLMModel):
    config_class = EveLlavaZhiziConfig

    def __init__(self, config: InternLMConfig):
        super(EveLlavaZhiziModel, self).__init__(config)

@dataclass
class EveBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class EveCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None

def EveInternLMDecoderLayerAug_forward(self):
    def forward(
            # self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # import pdb;pdb.set_trace()

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )


        hidden_states = residual + hidden_states + self.attn_linear(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        moe_losses = []
        # TODO 补充moe_loss的填充方式
        if len(hidden_states) == 3:
            moe_losses.append(hidden_states[1])
            hidden_states = hidden_states[0]

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward

def EveInternLMDecoderLayerVanilla_forward(self):
    def forward(
            # self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # import pdb;pdb.set_trace()

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )


        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        moe_losses = []
        # TODO 补充moe_loss的填充方式
        if len(hidden_states) == 3:
            moe_losses.append(hidden_states[1])
            hidden_states = hidden_states[0]

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward

def EveInternLMModel_forward(self):

    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, EveBaseModelOutputWithPast]:
        # import pdb;pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_moe_loss = [] if output_moe_loss else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if v is not None)
        return EveBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_loss,
        )

    return forward

class EveLlavaZhiziForCausalLM(InternLMForCausalLM, LlavaMetaForCausalLM):
    config_class = EveLlavaZhiziConfig

    def __init__(self, config):
        super(InternLMForCausalLM, self).__init__(config)
        self.model = EveLlavaZhiziModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, EveCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLMForCausalLM

        >>> model = InternLMForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb;pdb.set_trace()
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, audios)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        #TODO 需要检查 moe_loss 实现方式是否存在问题
        moe_loss, moe_losses = None, []
        if len(outputs[-1]) > 0:
            moe_loss_list = outputs[-1]
            # import ipdb
            # ipdb.set_trace()
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            # print("moe_losses", moe_losses)
            moe_loss = self.router_aux_loss_coef * sum(moe_losses)
            if labels is not None:
                print(loss, sum(moe_losses), loss + moe_loss)
                loss += moe_loss



        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output
        return EveCausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=outputs.moe_loss_list,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # import pdb;pdb.set_trace()
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
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_moe_modules(self, model_args):
        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        #self.config.moe['use_l1loss'] = model_args.use_l1loss
        #self.config.moe['hard_top1'] = model_args.hard_top1
        self.config.moe['moe_fix_other'] = model_args.moe_fix_other
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size']= model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef

        # if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
        #     for n, p in self.named_parameters():
        #         if any(name in n for name in self.config.moe['train_modules']):
        #             continue
        #         else:
        #             p.requires_grad = False


        num_layers = self.config.num_hidden_layers
        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            elif model_args.moe_mode == "one_debug":
                moe_layers_idx = list(range(0, 1))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense", "one_debug"], but found {model_args.moe_mode}')

        self.config.moe['moe_layers_idx'] = moe_layers_idx
        if len(model_args.num_experts) == 1:
            self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        assert len(self.config.moe['num_experts']) == len(moe_layers_idx)
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            pretrained_state_dict = self.model.layers[layer_num].mlp.state_dict()
            self.model.layers[layer_num].mlp = Eve(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=model_args.ep_size,
                k=model_args.top_k_experts,
                capacity_factor=model_args.capacity_factor,
                eval_capacity_factor=model_args.eval_capacity_factor,
                min_capacity=model_args.min_capacity,
                use_residual=model_args.use_residual,
                # use_l1loss=model_args.use_l1loss,
                # hard_top1=model_args.hard_top1,
            )
            for e in self.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts:  # check weight
                loaded_state_dict = e.state_dict()
                assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()]), layer_num
                assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()]), layer_num
        rank0_print(f"LLM num_layers: {num_layers}, Eve num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])
        split_dict = {29: 19, 12: 8, 22: 15}
        for idx, m in enumerate(self.model.layers):

            if idx < split_dict[self.config.num_hidden_layers]:
                m.forward = EveInternLMDecoderLayerVanilla_forward(m)
            else:
                m.forward = EveInternLMDecoderLayerAug_forward(m)
        rank0_print(f'replace InternLMDecoderLayerVanilla.forward to EveInternLMDecoderLayerVanilla.forward')
        rank0_print(f'replace InternLMDecoderLayerAug.forward to EveInternLMDecoderLayerAug.forward')
        self.model.forward = EveInternLMModel_forward(self.model)
        rank0_print(f'replace InternLMModel.forward to EveInternLMModel.forward')
        # freeze model
        # model_args.num_experts
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if self.config.moe['moe_fix_other']:
                    train_exp_name = f"deepspeed_experts.{model_args.num_experts[0]-1}"
                    if any(name in n for name in self.config.moe['train_modules']) and (train_exp_name in n) and ("deepspeed_moe" in n): #
                        continue
                    else:
                        p.requires_grad = False
                else:
                    if any(name in n for name in self.config.moe['train_modules']):
                        continue
                    else:
                        p.requires_grad = False

class EvalEveLlavaZhiziForCausalLM(EveLlavaZhiziForCausalLM):
    config_class = EveLlavaZhiziConfig
    def __init__(self, config):
        super(EvalEveLlavaZhiziForCausalLM, self).__init__(config)
        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']
        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            self.model.layers[layer_num].mlp = Eve(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=self.config.moe['ep_size'],
                k=self.config.moe['top_k_experts'],
                capacity_factor=self.config.moe['capacity_factor'],
                eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                min_capacity=self.config.moe['min_capacity'],
                use_residual=self.config.moe['use_residual'],
                # use_l1loss=self.config.moe['use_l1loss'],
                # hard_top1=self.config.moe['hard_top1'],
            )
        rank0_print(f"LLM num_layers: {num_layers}, Eve num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        split_dict = {29: 19, 12: 8, 22: 15}
        for idx, m in enumerate(self.model.layers):

            if idx < split_dict[self.config.num_hidden_layers]:
                m.forward = EveInternLMDecoderLayerVanilla_forward(m)
            else:
                m.forward = EveInternLMDecoderLayerAug_forward(m)
        rank0_print(f'replace InternLMDecoderLayerVanilla.forward to EveInternLMDecoderLayerVanilla.forward')
        rank0_print(f'replace InternLMDecoderLayerAug.forward to EveInternLMDecoderLayerAug.forward')
        self.model.forward = EveInternLMModel_forward(self.model)
        rank0_print(f'replace InternLMModel.forward to EveInternLMModel.forward')

AutoConfig.register("moe_llava_zhizi", EveLlavaZhiziConfig)
# AutoTokenizer.register(LlavaPhiConfig, PhiTokenizer)
AutoModelForCausalLM.register(EveLlavaZhiziConfig, EveLlavaZhiziForCausalLM)
AutoModelForCausalLM.register(EveLlavaZhiziConfig, EvalEveLlavaZhiziForCausalLM)
