from dataclasses import dataclass
import ujson as json
from typing import Sequence, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
import transformers

from utils import *
from constants import RANK_TOKEN, IGNORE_TOKEN_ID


def preprocess_messages(
    tokenizer: transformers.PreTrainedTokenizer,
    messages: list[dict[str, str]],
    mask_targets_func: Callable[[list[dict[str, str]], Tensor], Tensor],
) -> dict[str, Tensor]:
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    targets = input_ids.clone()
    targets = mask_targets_func(tokenizer, messages, targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def _get_messages_length(
    messages: list[dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        return_tensors='pt',
        max_length=tokenizer.model_max_length,
        truncation=True
    ).shape[1]


def _mask_targets_for_causal_lm(
    tokenizer: transformers.PreTrainedTokenizer,
    messages: list[dict[str, str]],
    targets: Tensor,
) -> Tensor:
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            message_start_idx = _get_messages_length(messages[:message_idx], tokenizer) if message_idx > 0 else 0
            message_end_idx = _get_messages_length(messages[:message_idx+1], tokenizer)         
            targets[:, message_start_idx:message_end_idx] = IGNORE_TOKEN_ID
            if message_end_idx >= tokenizer.model_max_length:
                break
    return targets


def _mask_targets_for_ranking(
    tokenizer: transformers.PreTrainedTokenizer,
    messages: list[list[dict[str, str]]],
    targets: Tensor,
) -> Tensor:
    if RANK_TOKEN not in tokenizer.all_special_tokens:
        tokenizer.add_tokens([RANK_TOKEN], special_tokens=True)
    for target in targets:
        target[target != tokenizer.convert_tokens_to_ids(RANK_TOKEN)] = IGNORE_TOKEN_ID
    return targets


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.encoder_tokenizer = encoder_tokenizer
        if self.encoder_tokenizer and self.encoder_tokenizer.eos_token:
            print("WARNING: will add eos token to the end of extra texts")
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
        self.raw_data = self.load_data(data_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        raise NotImplementedError

    def load_data(self, data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                raw_data = json.load(f)
                assert isinstance(raw_data, list)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        print(f"Loaded {len(raw_data)} examples from {data_path}")
        return raw_data


class DatasetForCausalLM(SFTDataset):

    def __getitem__(self, i) -> dict[str, Tensor]:
        ret = preprocess_messages(
            self.tokenizer,
            self.raw_data[i]["messages"],
            _mask_targets_for_causal_lm,
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        if "extra_texts" in self.raw_data[i]:
            if self.encoder_tokenizer.eos_token:
                self.raw_data[i]["extra_texts"] = [
                    f"{text} {self.encoder_tokenizer.eos_token}"
                    for text in self.raw_data[i]["extra_texts"]
                ]
            extra_text_inputs = self.encoder_tokenizer(
                self.raw_data[i]["extra_texts"],
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True,
            )
            ret["extra_text_inputs"] = dict(
                input_ids=extra_text_inputs["input_ids"],
            )

        return ret


class DatasetForRanking(SFTDataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
        use_embedding_with_content: bool = True,
        use_embedding_without_content: bool = False,
    ):
        super().__init__(data_path, tokenizer, encoder_tokenizer)
        self.use_embedding_with_content = use_embedding_with_content
        self.use_embedding_without_content = use_embedding_without_content

    def __getitem__(self, i) -> dict[str, Tensor]:
        ranking = torch.tensor(self.raw_data[i]["ranking"], dtype=torch.long)

        if self.use_embedding_with_content:
            messages_w_content = self.raw_data[i]["messages_w_content"]
            if messages_w_content[-1]["role"] == "assistant":
                messages_w_content[-1]["content"] = f"{RANK_TOKEN}" * len(ranking)
            else:
                messages_w_content.append({"role": "assistant", "content": f"{RANK_TOKEN}" * len(ranking)})
            inputs_w_content = preprocess_messages(
                self.tokenizer,
                messages_w_content,
                _mask_targets_for_ranking
            )
            inputs_w_content = dict(
                input_ids=inputs_w_content["input_ids"][0],
                labels=inputs_w_content["labels"][0],
                attention_mask=inputs_w_content["attention_mask"][0],
            )
        else:
            inputs_w_content = None
        if self.use_embedding_without_content:
            messages_wo_content = self.raw_data[i]["messages_wo_content"]
            if messages_wo_content[-1]["role"] == "assistant":
                messages_wo_content[-1]["content"] = f"{RANK_TOKEN}" * len(ranking)
            else:
                messages_wo_content.append({"role": "assistant", "content": f"{RANK_TOKEN}" * len(ranking)})
            inputs_wo_content = preprocess_messages(
                self.tokenizer,
                messages_wo_content,
                _mask_targets_for_ranking
            )
            inputs_wo_content = dict(
                input_ids=inputs_wo_content["input_ids"][0],
                labels=inputs_wo_content["labels"][0],
                attention_mask=inputs_wo_content["attention_mask"][0],
            )
        else:
            inputs_wo_content = None
        if "extra_texts" in self.raw_data[i]:
            if self.encoder_tokenizer.eos_token:
                self.raw_data[i]["extra_texts"] = [
                    f"{text}{self.encoder_tokenizer.eos_token}" for text in self.raw_data[i]["extra_texts"]
                ]
            extra_text_inputs = self.encoder_tokenizer(
                self.raw_data[i]["extra_texts"],
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True,
            )
            extra_text_inputs = dict(
                input_ids=extra_text_inputs["input_ids"],
            )
        else:
            extra_text_inputs = None
        ret = dict(
            inputs_w_content=inputs_w_content,
            inputs_wo_content=inputs_wo_content,
            extra_text_inputs=extra_text_inputs,
            ranking=ranking,
        )

        return ret


@dataclass
class DataCollatorForCausalLM:
    tokenizer: transformers.PreTrainedTokenizer
    encoder_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if instances[0].get("extra_text_inputs", None) is not None:
            extra_text_input_ids = [instance["extra_text_inputs"]["input_ids"] for instance in instances]
            extra_text_input_ids = torch.nn.utils.rnn.pad_sequence(
                extra_text_input_ids,
                batch_first=True,
                padding_value=self.encoder_tokenizer.pad_token_id
            )
            batch["extra_text_input_ids"] = extra_text_input_ids
            batch["extra_text_attention_mask"] = extra_text_input_ids.ne(self.encoder_tokenizer.pad_token_id)
        return batch


@dataclass
class DataCollatorForRanking:
    tokenizer: transformers.PreTrainedTokenizer
    encoder_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        batch = dict()

        if instances[0].get("inputs_w_content", None) is not None:
            input_ids, labels = tuple(
                [instance["inputs_w_content"][key] for instance in instances] for key in ("input_ids", "labels")
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_TOKEN_ID
            )
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
            batch["inputs_w_content"] = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

        if instances[0].get("inputs_wo_content", None) is not None:
            input_ids, labels = tuple(
                [instance["inputs_wo_content"][key] for instance in instances] for key in ("input_ids", "labels")
            )
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_TOKEN_ID
            )
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
            batch["inputs_wo_content"] = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

        if instances[0].get("extra_text_inputs", None) is not None:
            extra_text_input_ids = [instance["extra_text_inputs"]["input_ids"] for instance in instances]
            extra_text_input_ids = torch.nn.utils.rnn.pad_sequence(
                extra_text_input_ids,
                batch_first=True,
                padding_value=self.encoder_tokenizer.pad_token_id
            )
            batch["extra_text_inputs"] = dict(
                extra_text_input_ids=extra_text_input_ids,
                extra_text_attention_mask=extra_text_input_ids.ne(self.encoder_tokenizer.pad_token_id)
            )

        batch["ranking"] = torch.stack([instance["ranking"] for instance in instances])

        return batch


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_type: str,
) -> dict:
    if model_type == "causal_lm":
        train_dataset = DatasetForCausalLM(
            data_args.data_path,
            tokenizer=tokenizer,
            encoder_tokenizer=encoder_tokenizer,
        )
    else:
        train_dataset = DatasetForRanking(
            data_args.data_path,
            tokenizer=tokenizer,
            encoder_tokenizer=encoder_tokenizer,
            use_embedding_with_content=data_args.use_embedding_with_content,
            use_embedding_without_content=data_args.use_embedding_without_content,
        )
    data_collator_cls = DataCollatorForCausalLM if model_type == "causal_lm" else DataCollatorForRanking
    data_collator = data_collator_cls(
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer
    )

    # TODO: make eval_dataset available
    # if data_args.eval_data_path:
    #     eval_dataset = dataset_cls(
    #         data_args.eval_data_path,
    #         tokenizer=tokenizer,
    #         encoder_tokenizer=encoder_tokenizer
    #     )
    # else:
    #     eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
