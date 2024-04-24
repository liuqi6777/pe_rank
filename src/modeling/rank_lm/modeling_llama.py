import torch
from dataclasses import dataclass
from typing import Optional
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.file_utils import ModelOutput
from modeling.model import ELMMetaModel
from modeling.meta import MetaLM
from modeling.rank_lm.loss import ListMLELoss


class EmbedLlamaConfig(LlamaConfig):
    model_type = "embed_llama"


class EmbedLlamaModel(ELMMetaModel, LlamaModel):
    config_class = EmbedLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


@dataclass
class RankingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    ranking: Optional[torch.LongTensor] = None


class EmbedLlamaForRankLM(MetaLM, LlamaForCausalLM):
    config_class = EmbedLlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = EmbedLlamaModel(config)
        self.config = config
        self.oringinal_vocab_size = config.vocab_size
        self.post_init()

        self.loss_function = ListMLELoss(weighted="weighted_1")
        self.normalize_embeddings = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **extra_texts_inputs
    ) -> RankingOutput:

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
        hidden_states = outputs[0]
        if self.normalize_embeddings:
            hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
            extra_embeddings = torch.nn.functional.normalize(extra_embeddings, p=2, dim=-1)
        loss, logits = self.loss_function(hidden_states, extra_embeddings, labels, ranking)
        return RankingOutput(
            loss=loss,
            logits=logits,
            ranking=ranking,
        )

    @torch.no_grad()
    def rank(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> list[int]:
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
        else:
            raise NotImplementedError(
                "`extra_text_input_ids` and `extra_text_attention_mask` are required")

        self.resize_token_embeddings(self.oringinal_vocab_size + extra_text_embeddings.shape[1])
        self.get_input_embeddings().weight.data[self.oringinal_vocab_size:] = extra_text_embeddings.to(self.lm_head.weight.device)
        self.get_output_embeddings().weight.data[self.oringinal_vocab_size:] = extra_text_embeddings.to(self.lm_head.weight.device)

        self.forward = super().forward
        rankings = self.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=extra_text_embeddings.shape[1],
            do_sample=False,
            num_beams=20,
            prefix_allowed_tokens_fn=lambda _, prev_ids: list(set([x + self.oringinal_vocab_size for x in range(extra_text_embeddings.shape[1])]) - set(prev_ids.tolist())),
            pad_token_id=self.config.eos_token_id,
        )
        rankings = rankings[0].cpu().tolist()[-extra_text_embeddings.shape[1]:]
        rankings = [x - self.oringinal_vocab_size + 1 for x in rankings]
        return rankings
