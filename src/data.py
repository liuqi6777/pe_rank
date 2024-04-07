import copy
from dataclasses import dataclass
import ujson as json
from typing import Sequence

import torch
from torch.utils.data import Dataset, IterableDataset
import transformers

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from utils import *
from constants import RANK_TOKEN, IGNORE_TOKEN_ID


# FIXME: mask placeholders
def preprocess_conversations_for_causal_lm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conversation_template: str = "vicuna",
) -> dict[str, torch.Tensor]:
    conv = get_conversation_template(conversation_template)
    # conv.set_system_message()  # TODO: decide whether to modify system message
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_plain_for_causal_lm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    inputs = [s["input"] for s in sources]
    targets = [s["output"] + tokenizer.eos_token for s in sources]
    examples = [s + t for s, t in zip(inputs, targets)]
    examples_tokenized, inputs_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, inputs)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, input_len in zip(labels, inputs_tokenized["input_ids_lens"]):
        label[:input_len] = IGNORE_TOKEN_ID
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_TOKEN_ID
    )
    return dict(input_ids=input_ids, labels=labels,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


def preprocess_conversations_for_ranking(
    sources,
    rankings,
    tokenizer: transformers.PreTrainedTokenizer,
    conversation_template: str = "vicuna",
) -> dict[str, torch.Tensor]:
    conv = get_conversation_template(conversation_template)
    # conv.set_system_message()  # TODO: decide whether to modify system message
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, (source, ranking) in enumerate(zip(sources, rankings)):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source[:-1]):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        assert roles[source[-1]["from"]] == conv.roles[1]
        conv.append_message(conv.roles[1], f"{RANK_TOKEN}" * len(ranking))
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    
    # Mask targets. Only compute loss on the rankings.
    if RANK_TOKEN not in tokenizer.all_special_tokens:
        tokenizer.add_tokens([RANK_TOKEN], special_tokens=True)
    for target in targets:
        target[target != tokenizer.convert_tokens_to_ids(RANK_TOKEN)] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_plain_for_rank_lm(
    sources,
    rankings,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    inputs = [s["input"] for s in sources]
    targets = [f"{RANK_TOKEN}" * len(ranking) + tokenizer.eos_token for ranking in rankings]
    examples = [s + t for s, t in zip(inputs, targets)]
    examples_tokenized, inputs_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, inputs)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label in labels:
        label[label != tokenizer.convert_tokens_to_ids(RANK_TOKEN)] = IGNORE_TOKEN_ID
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=IGNORE_TOKEN_ID
    )
    return dict(input_ids=input_ids, labels=labels,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
        conversation_template: str = "vicuna",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.encoder_tokenizer = encoder_tokenizer
        if self.encoder_tokenizer.eos_token:
            print("WARNING: will add eos token to the end of extra texts")
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
            self.append_eos = True
        else:
            self.append_eos = False
        self.conversation_template = conversation_template
        if data_path.endswith(".json"):
            # directly load the data
            with open(data_path, "r") as f:
                self.raw_data = json.load(f)
                assert isinstance(self.raw_data, list)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                self.raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        return self.raw_data[i]


class LazyDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
        conversation_template: str = "vicuna",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.encoder_tokenizer = encoder_tokenizer
        if self.encoder_tokenizer.eos_token:
            print("WARNING: will add eos token to the end of extra texts")
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
            self.append_eos = True
        else:
            self.append_eos = False
        self.conversation_template = conversation_template
        if data_path.endswith(".json"):
            # directly load the data
            with open(data_path, "r") as f:
                self.raw_data = json.load(f)
                assert isinstance(self.raw_data, list)
        elif data_path.endswith(".jsonl"):
            self.raw_data = open(data_path, "r", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def __iter__(self):
        for item in self.raw_data:
            if isinstance(item, str):
                item = json.loads(item)
            yield item


class DatasetForCausalLM(SFTDataset):

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if "conversations" in self.raw_data[i]:
            ret = preprocess_conversations_for_causal_lm(
                [self.raw_data[i]["conversations"]],
                self.tokenizer,
                self.conversation_template
            )
        else:
            ret = preprocess_plain_for_causal_lm(
                [self.raw_data[i]],
                self.tokenizer
            )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        if "extra_texts" in self.raw_data[i]:
            if self.append_eos:
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


class DatasetForRanking(SFTDataset):

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if "conversations" in self.raw_data[i]:
            ret = preprocess_conversations_for_ranking(
                [self.raw_data[i]["conversations"]],
                [self.raw_data[i]["ranking"]],
                self.tokenizer,
                self.conversation_template
            )
        else:
            ret = preprocess_plain_for_rank_lm(
                [self.raw_data[i]],
                [self.raw_data[i]["ranking"]],
                self.tokenizer
            )
        ranking = torch.tensor(self.raw_data[i]["ranking"])
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            ranking=ranking,
        )
        if "extra_texts" in self.raw_data[i]:
            if self.append_eos:
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

class LazyDatasetForCausalLM(LazyDataset):

    def __iter__(self):
        for item in self.raw_data:
            if isinstance(item, str):
                item = json.loads(item)
            if "conversations" in item:
                ret = preprocess_conversations_for_causal_lm(
                    [item["conversations"]],
                    self.tokenizer,
                    self.conversation_template
                )
            else:
                ret = preprocess_plain_for_causal_lm(
                    [item],
                    self.tokenizer
                )
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
            )
            if "extra_texts" in item:
                if self.append_eos:
                    item["extra_texts"] = [
                        f"{text}{self.encoder_tokenizer.eos_token}" for text in item["extra_texts"]
                    ]
                extra_text_inputs = self.encoder_tokenizer(
                    item["extra_texts"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
                ret["extra_text_input_ids"] = extra_text_inputs["input_ids"]

            yield ret


class LazyDatasetForRanking(LazyDataset):

    def __iter__(self):
        for item in self.raw_data:
            if isinstance(item, str):
                item = json.loads(item)
            if "conversations" in item:
                ret = preprocess_conversations_for_ranking(
                    [item["conversations"]],
                    [item["ranking"]],
                    self.tokenizer,
                    self.conversation_template
                )
            else:
                ret = preprocess_plain_for_rank_lm(
                    [item],
                    [item["ranking"]],
                    self.tokenizer
                )
            ranking = torch.tensor(item["ranking"])
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
                ranking=ranking,
            )
            if "extra_texts" in item:
                if self.append_eos:
                    item["extra_texts"] = [
                        f"{text}{self.encoder_tokenizer.eos_token}" for text in item["extra_texts"]
                    ]
                extra_text_inputs = self.encoder_tokenizer(
                    item["extra_texts"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                )
                ret["extra_text_input_ids"] = extra_text_inputs["input_ids"]

            yield ret


@dataclass
class DataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    encoder_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
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
    if not data_args.lazy_loading:
        dataset_cls = DatasetForCausalLM if model_type == "causal_lm" else DatasetForRanking
    else:
        dataset_cls = LazyDatasetForCausalLM if model_type == "causal_lm" else LazyDatasetForRanking
    train_dataset = dataset_cls(
        data_args.data_path,
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        conversation_template=data_args.conversation_template,
    )

    data_collator = DataCollator(
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer)

    if data_args.eval_data_path:
        eval_dataset = dataset_cls(
            data_args.eval_data_path, tokenizer=tokenizer, encoder_tokenizer=encoder_tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
