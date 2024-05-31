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
            ret["extra_text_input_ids"] = extra_text_inputs["input_ids"]

        return ret


class DatasetForRanking(SFTDataset):

    def __getitem__(self, i) -> dict[str, Tensor]:
        messages = self.raw_data[i]["messages"]
        ranking = torch.tensor(self.raw_data[i]["ranking"], dtype=torch.long)
        if messages[-1]["role"] == "assistant":
            messages[-1]["content"] = f"{RANK_TOKEN}" * len(ranking)
        else:
            messages.append({"role": "assistant", "content": f"{RANK_TOKEN}" * len(ranking)})
        ret = preprocess_messages(
            self.tokenizer,
            messages,
            _mask_targets_for_ranking
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            ranking=ranking,
        )
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
            ret["extra_text_input_ids"] = extra_text_inputs["input_ids"]

        return ret


@dataclass
class DataCollator:
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

        if instances[0].get("ranking", None) is not None:
            batch["ranking"] = torch.stack([instance["ranking"] for instance in instances])

        if instances[0].get("extra_text_input_ids", None) is not None:
            extra_text_input_ids = [
                instance["extra_text_input_ids"] for instance in instances]
            extra_text_input_ids = torch.nn.utils.rnn.pad_sequence(
                extra_text_input_ids,
                batch_first=True,
                padding_value=self.encoder_tokenizer.pad_token_id
            )
            batch["extra_text_input_ids"] = extra_text_input_ids
            batch["extra_text_attention_mask"] = extra_text_input_ids.ne(self.encoder_tokenizer.pad_token_id)
        return batch


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_type: str,
) -> dict:
    dataset_cls = DatasetForCausalLM if model_type == "causal_lm" else DatasetForRanking
    train_dataset = dataset_cls(
        data_args.data_path,
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer,
    )
    data_collator = DataCollator(
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer
    )
    if data_args.eval_data_path:
        eval_dataset = dataset_cls(
            data_args.eval_data_path,
            tokenizer=tokenizer,
            encoder_tokenizer=encoder_tokenizer
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
