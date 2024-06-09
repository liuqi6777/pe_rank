import re
import torch
from transformers import AutoTokenizer

from modeling.builder import load_pretrained_model


PLACEHOLDER = "<PLACEHOLDER>"


class Ranker:
    def __init__(self, model_path, model_base, model_name="embed_mistral"):
        self._tokenizer, self._model, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device_map="cuda",
        )
        self.model_name = model_path
        self._model.to(torch.float16)
        self._model.config.use_cache = True
        self._model.eval()
        if getattr(self._model.config, "encoder_name", None):
            self._encoder_tokenizer = AutoTokenizer.from_pretrained(self._model.config.encoder_name)
        else:
            self._encoder_tokenizer = None
        self.oringinal_vocab_size = self._model.config.vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_encoder_inputs(self, candidates: list[str]) -> torch.Tensor:
        input_ids = self._encoder_tokenizer(
            [p["content"] for p in candidates],
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ).input_ids.unsqueeze(0).to(self.device)
        return input_ids


class ListwiseTextEmbeddingRanker(Ranker):

    def _add_prefix_prompt(self, query: str, num: int, query_max_length: int = 180) -> str:
        # TODO: make max_length configurable
        if len(self._tokenizer.tokenize(query)) > query_max_length:
            query = " ".join(self._tokenizer.tokenize(query)[:query_max_length])
        return f"""I will provide you with {num} passages, each with a special token representing the passage enclosed in [], followed by original text.
Rank the passages based on their relevance to the search query: {query}.
"""

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"""Search Query: {query}.
Rank the {num} relatively ordered passages above based on their relevance to the search query, output the ranking in descending order. Only output the {num} unique special token in the ranking.

"""

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def _get_message(self, query: str, candidates: list[str]) -> str:
        num = len(candidates)
        candidates = [p["content"] for p in candidates]
        messages = []
        input_context = self._add_prefix_prompt(query, num)
        for i, content in enumerate(candidates):
            content = self._replace_number(content.strip())
            input_context += self._get_input_for_one_passage(content, i + 1)
        input_context += self._add_post_prompt(query, num)
        messages.append({"role": "user", "content": input_context})
        return messages

    def _get_input_for_one_passage(self, content: str, i: int, passage_max_length: int = 180) -> str:
        if len(self._tokenizer.tokenize(content)) > passage_max_length:
            content = " ".join(self._tokenizer.tokenize(content)[:passage_max_length])
        return f"Passage {i}: [<PLACEHOLDER>] {content}\n\n"

    def _get_llm_inputs(self, query: str, candidates: list[str]) -> torch.Tensor:
        messages = self._get_message(query, candidates)
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="longest",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
        ).to(self.device)
        return input_ids

    @torch.no_grad()
    def __call__(self, query: str, candidates: list[str]) -> dict[str]:
        input_ids = self._get_llm_inputs(query, candidates)
        extra_text_input_ids = self._get_encoder_inputs(candidates)

        def prefix_allowed_tokens_fn(batch_id, prev_ids):
            allowed_tokens = list(
                set([x + self.oringinal_vocab_size for x in range(len(candidates))]) \
                - set(prev_ids.tolist())
            )
            if len(allowed_tokens) == 0:
                return [self._tokenizer.eos_token_id]
            elif len(allowed_tokens) == len(candidates):
                if prev_ids[-1] != self._tokenizer.bos_token_id:
                    return [self._tokenizer.bos_token_id]
                return allowed_tokens
            else:
                return allowed_tokens

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self._model.config.eos_token_id,
            extra_text_input_ids=extra_text_input_ids,
            extra_text_attention_mask=extra_text_input_ids.ne(self._encoder_tokenizer.pad_token_id),
        )

        rankings = outputs[0].cpu().tolist()
        rankings = [x - self.oringinal_vocab_size for x in rankings if x >= self.oringinal_vocab_size]
        return rankings


class ListwiseEmbeddingRanker(ListwiseTextEmbeddingRanker):
    def _get_input_for_one_passage(self, content: str, i: int) -> str:
        return f"Passage {i}: [<PLACEHOLDER>]\n\n"


class ListwiseTextRanker(ListwiseTextEmbeddingRanker):
    def _get_input_for_one_passage(self, content: str, i: int, passage_max_length: int = 180) -> str:
        if len(self._tokenizer.tokenize(content)) > passage_max_length:
            content = " ".join(self._tokenizer.tokenize(content)[:passage_max_length])
        return f"[{i}]: {content}\n\n"

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"""I will provide you with {num} passages.
Rank the passages based on their relevance to the search query: {query}.
"""

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"""Search Query: {query}.
Rank the {num} relatively ordered passages above based on their relevance to the search query, output the ranking in descending order. The output format should be [] > [] > ..., e.g., [4] > [2] > ..., Only respond with the ranking results with {num} unique numbers, do not say anything else or explain.

"""

    def parse_output(self, output: str) -> list[int]:
        response = self._clean_response(output)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        return response

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: list[int]) -> list[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def __call__(self, query: str, candidates: list[str]) -> dict[str]:
        input_ids = self._get_llm_inputs(query, candidates)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self._model.config.eos_token_id,
        )
        outputs = self._tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

        permutation = self.parse_output(outputs)
        original_rank = [tt for tt in range(len(candidates))]
        permutation = [ss for ss in permutation if ss in original_rank]
        permutation = permutation + [tt for tt in original_rank if tt not in permutation]
        return permutation
