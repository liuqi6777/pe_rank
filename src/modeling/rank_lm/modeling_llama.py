from typing import Union, Optional, List, Tuple
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from modeling.model import ELMMetaModel
from modeling.meta import MetaLM
from modeling.rank_lm.loss import basic_rank_loss


class EmbedLlamaConfig(LlamaConfig):
    model_type = "embed_llama"


class EmbedLlamaModel(ELMMetaModel, LlamaModel):
    config_class = EmbedLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class EmbedLlamaForRankLM(MetaLM, LlamaPreTrainedModel):
    config_class = EmbedLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = EmbedLlamaModel(config)
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
                extra_embeddings,
                _,
            ) = self.prepare_inputs_labels_embeddings(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                **extra_texts_inputs
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        ranking = extra_texts_inputs["ranking"]
        loss = basic_rank_loss(outputs[0], extra_embeddings, labels, ranking)
        return {"loss": loss}

    @torch.no_grad()
    def rank(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        extra_text_input_ids = kwargs.pop("extra_text_input_ids", None)
        extra_text_attention_mask = kwargs.pop(
            "extra_text_attention_mask", None)

        if extra_text_input_ids is not None and extra_text_attention_mask is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                extra_text_embeddings,
                extra_text_positions,
            ) = self.prepare_inputs_labels_embeddings(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                extra_text_input_ids=extra_text_input_ids,
                extra_text_attention_mask=extra_text_attention_mask
            )
        else:
            raise NotImplementedError(
                "`extra_text_input_ids` and `extra_text_attention_mask` are required")
        past_key_values = None

        # now only support one input
        assert extra_text_embeddings.shape[0] == 1, extra_text_embeddings.shape
        num_extra_texts = extra_text_embeddings.shape[1]

        extra_text_positions = extra_text_positions[0]

        rankings = []
        ranking_mask = torch.zeros(extra_text_embeddings.shape[1], 
                                   dtype=torch.long, device=extra_text_embeddings.device)
        for _ in range(num_extra_texts):
            outputs = self.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = outputs[0].to(extra_text_embeddings.device)
            hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
            extra_text_output_embeddings = hidden_states[0, extra_text_positions]
            logits = (hidden_states[0, -1] @ extra_text_embeddings[0].T).flatten()
            # logits = (hidden_states[0, -1] @ extra_text_output_embeddings.T).flatten()
            
            # rankings = (torch.argsort(logits, descending=True) + 1).detach().cpu().tolist()
            # break
            
            logits[ranking_mask == 1] = -float("inf")
            ranking = torch.argmax(logits).item()
            ranking_mask[ranking] = 1
            rankings.append(ranking + 1)
            inputs_embeds = torch.cat([inputs_embeds, extra_text_embeddings[:, [ranking]].to(inputs_embeds.device)], dim=1)
            extra_text_positions = torch.cat([extra_text_positions, 
                                              torch.tensor([False], device=extra_text_positions.device)])
            past_key_values = outputs[1]

        return rankings
