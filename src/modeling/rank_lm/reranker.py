import copy
import transformers
import torch
from typing import Optional
from fastchat.model.model_adapter import get_conversation_template

from modeling.rank_lm.modeling_llama import EmbedLlamaForRankLM


PLACEHOLDER = "<PLACEHOLDER>"
INSTRUCTION = """Providing with {n} passages, each is enclosed in the identifier []. Rank the passages based on their relevance to the search query: {query}. 
{inputs}
Search Query: {query}. Rank the {n} passages above based on their relevance to the search query in descending order.
"""


class RerankLLM:
    def __init__(
        self,
        model: EmbedLlamaForRankLM,
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._encoder_tokenizer = encoder_tokenizer

    def permutation_pipeline(
        self,
        result: dict,
        rank_start: int,
        rank_end: int,
        record_num_processed_tokens: bool = False,
    ) -> list[dict]:
        model_inputs = self.create_inputs(result, rank_start, rank_end)
        permutation = self._model.rank(**model_inputs)
        result = self.receive_permutation(
            result, permutation, rank_start, rank_end)
        if record_num_processed_tokens:
            return result, model_inputs["inputs"].shape[1]
        return result

    @torch.inference_mode()
    def rerank(
        self,
        retrieved_result: dict,
        window_size: Optional[int] = None,
        step: Optional[int] = None,
        record_num_processed_tokens: bool = False,
    ) -> list[dict]:
        rerank_result = copy.deepcopy(retrieved_result)
        rank_start, rank_end = 0, len(rerank_result["hits"])
        if not window_size:
            window_size = len(rerank_result["hits"])
            step = window_size
        else:
            if not step:
                step = window_size // 2
        end_pos = rank_end
        start_pos = rank_end - window_size

        num_processed_tokens_per_query = 0

        # end_pos > rank_start ensures that the list is non-empty while allowing last window to be smaller than window_size
        # start_pos + step != rank_start prevents processing of redundant windows (e.g. 0-20, followed by 0-10)
        sliding_steps = 0
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            if record_num_processed_tokens:
                rerank_result, num_processed_tokens = self.permutation_pipeline(
                    rerank_result, start_pos, end_pos, record_num_processed_tokens
                )
                num_processed_tokens_per_query += num_processed_tokens
            else:
                rerank_result = self.permutation_pipeline(rerank_result, start_pos, end_pos)
            end_pos = end_pos - step
            start_pos = start_pos - step
            sliding_steps += 1
        if record_num_processed_tokens:
            return rerank_result, sliding_steps, num_processed_tokens_per_query
        return rerank_result

    def receive_permutation(
        self,
        result: dict,
        permutation: list[int],
        rank_start: int,
        rank_end: int
    ):
        cut_range = copy.deepcopy(result["hits"][rank_start:rank_end])
        if min(permutation) != 0:
            permutation = [x - min(permutation) for x in permutation]
        for j, x in enumerate(permutation):
            result["hits"][j + rank_start] = copy.deepcopy(cut_range[x])
            if "rank" in result["hits"][j + rank_start]:
                result["hits"][j + rank_start]["rank"] = cut_range[j]["rank"]
            if "score" in result["hits"][j + rank_start]:
                result["hits"][j + rank_start]["score"] = cut_range[j]["score"]
        return result

    def create_inputs(self, retrieval_results: dict, rank_start: int, rank_end: int) -> dict[str, torch.Tensor]:
        retrieved_passages = retrieval_results["hits"][rank_start:rank_end]
        model_inputs = {
            "conversations": [
                {
                    "from": "human",
                    "value": INSTRUCTION.format(
                        n=len(retrieved_passages),
                        query=retrieval_results["query"],
                        inputs="\n".join(["[<PLACEHOLDER>]"] * len(retrieved_passages))
                    )
                },
            ],
            "extra_texts": [p["content"] for p in retrieved_passages]
        }

        # process data
        conv = get_conversation_template("vicuna")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversation = model_inputs["conversations"]
        if roles[conversation[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            conversation = conversation[1:]
        conv.messages = []
        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        conversation = conv.get_prompt()

        input_ids = self._tokenizer(
            [conversation],
            return_tensors="pt",
            padding="longest",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        extra_text_input_ids = self._encoder_tokenizer(
            model_inputs["extra_texts"],
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        ).input_ids.unsqueeze(0)

        return {
            "inputs": input_ids,
            "extra_text_input_ids": extra_text_input_ids,
            "extra_text_attention_mask": extra_text_input_ids.ne(self._encoder_tokenizer.pad_token_id)
        }
