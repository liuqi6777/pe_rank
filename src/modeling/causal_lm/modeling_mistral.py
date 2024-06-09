from typing import Union, Optional, List, Tuple
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from modeling.model import ELMMetaModel
from modeling.meta import MetaLM


class EmbedMistralConfig(MistralConfig):
    model_type = "embed_mistral"


class EmbedMistralModel(ELMMetaModel, MistralModel):
    config_class = EmbedMistralConfig

    def __init__(self, config: MistralConfig):
        super().__init__(config)


class EmbedMistralForCausalLM(MetaLM, MistralForCausalLM):
    config_class = EmbedMistralConfig

    def __init__(self, config: MistralConfig):
        super(MistralForCausalLM, self).__init__(config)
        self.model = EmbedMistralModel(config)

        self.vocab_size = self.original_vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        return_dict: Optional[bool] = None,
        **extra_texts_inputs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _,
                _
            ) = self.prepare_inputs_labels_embeddings(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                **extra_texts_inputs
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, **kwargs) -> dict[str, Tensor]:
        extra_text_input_ids = kwargs.pop("extra_text_input_ids", None)
        extra_text_attention_mask = kwargs.pop("extra_text_attention_mask", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, **kwargs
        )
        if extra_text_input_ids is not None and extra_text_attention_mask is not None:
            inputs["extra_text_input_ids"] = extra_text_input_ids
            inputs["extra_text_attention_mask"] = extra_text_attention_mask
        return inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        extra_text_input_ids = kwargs.pop("extra_text_input_ids", None)
        extra_text_attention_mask = kwargs.pop("extra_text_attention_mask", None)

        if extra_text_input_ids is not None and extra_text_attention_mask is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                extra_text_embeddings,
                _,
            ) = self.prepare_inputs_labels_embeddings(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                extra_text_input_ids=extra_text_input_ids,
                extra_text_attention_mask=extra_text_attention_mask
            )

            n = extra_text_embeddings.shape[1]
            if self.vocab_size != self.original_vocab_size + n:
                self.resize_token_embeddings(self.original_vocab_size + n)
            assert self.vocab_size == self.original_vocab_size + n
            self.get_input_embeddings().weight.data[self.original_vocab_size:] = extra_text_embeddings
            self.get_output_embeddings().weight.data[self.original_vocab_size:] = extra_text_embeddings
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


AutoConfig.register("embed_mistral", EmbedMistralConfig)
AutoModelForCausalLM.register(EmbedMistralConfig, EmbedMistralForCausalLM)
