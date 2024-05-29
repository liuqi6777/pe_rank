import torch
import transformers
from torch import Tensor
from typing import Optional, Union

from modeling.model import ELMMetaModel
from constants import PLACEHOLDER, RANK_TOKEN


class MetaLM:
    
    model: ELMMetaModel
    
    def get_model(self) -> ELMMetaModel:
        return self.model

    def get_encoder(self):
        return self.get_model().get_encoder()

    def get_projector(self):
        return self.get_model().get_projector()

    def get_encoder_head(self):
        return self.get_model().get_encoder_head()

    def prepare_inputs_labels_embeddings(
        self,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[tuple],
        labels: Optional[Tensor],
        **extra_texts_inputs: dict[str, Tensor]
    ):
        if self.get_model().get_encoder() is None or "extra_text_input_ids" not in extra_texts_inputs:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        assert "extra_text_input_ids" in extra_texts_inputs and "extra_text_attention_mask" in extra_texts_inputs, extra_texts_inputs.keys()

        input_embeddings = []
        all_text_embeddings = []
        extra_text_positions = []

        # TODO: allow one text corresponding to multiple placeholders, now it's 1 to 1
        for extra_text_input_ids, extra_text_attention_masks, cur_input_ids in \
                zip(extra_texts_inputs["extra_text_input_ids"], extra_texts_inputs["extra_text_attention_mask"], input_ids):

            num_extra_texts = (cur_input_ids == PLACEHOLDER_ID).sum() + (cur_input_ids == RANK_TOKEN_ID).sum()
            assert num_extra_texts <= extra_text_input_ids.shape[0]
            extra_text_input_ids = extra_text_input_ids[:num_extra_texts]
            extra_text_attention_masks = extra_text_attention_masks[:num_extra_texts]
            project_text_embeddings = self.get_model().encode_texts(
                input_ids=extra_text_input_ids, attention_mask=extra_text_attention_masks)

            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids.to(self.get_model().device))
            new_input_embeds = cur_input_embeds.clone()
            project_text_embeddings = project_text_embeddings.to(new_input_embeds.device)

            text_as_token_indices = (cur_input_ids == PLACEHOLDER_ID) | (cur_input_ids == RANK_TOKEN_ID)
            new_input_embeds[text_as_token_indices] = project_text_embeddings.to(cur_input_embeds.dtype)
            input_embeddings.append(new_input_embeds)

            all_text_embeddings.append(
                project_text_embeddings[:(cur_input_ids == PLACEHOLDER_ID).sum().item(), :])

            extra_text_position = (cur_input_ids == PLACEHOLDER_ID)
            extra_text_positions.append(extra_text_position)

        input_embeddings = torch.stack(input_embeddings)
        all_text_embeddings = torch.stack(all_text_embeddings)
        extra_text_positions = torch.stack(extra_text_positions)

        return None, position_ids, attention_mask, past_key_values, input_embeddings, labels, \
               all_text_embeddings, extra_text_positions

    def initialize_tokenizer(self, tokenizer: transformers.PreTrainedTokenizer):
        global PLACEHOLDER_ID
        global RANK_TOKEN_ID
        tokenizer.add_tokens([PLACEHOLDER], special_tokens=True)
        tokenizer.add_tokens([RANK_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        PLACEHOLDER_ID = tokenizer.convert_tokens_to_ids(PLACEHOLDER)
        RANK_TOKEN_ID = tokenizer.convert_tokens_to_ids(RANK_TOKEN)
