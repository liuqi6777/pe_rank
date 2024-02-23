import argparse
import os
import random
import ujson as json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from fastchat.model.model_adapter import get_conversation_template

from builder import load_pretrained_model


EVAL_DATASETS = [
    "data/pretrain_eval/query_reconstruction_eval.jsonl",
    "data/pretrain_eval/doc_reconstruction_eval.jsonl",
    "data/pretrain_eval/pair_relevance_comparision_eval.jsonl",
]


def evaluate_one_dataset(
    data: list[dict],
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    args: argparse.Namespace,
) -> list[str]:
    conv = get_conversation_template(args.conv_mode)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    extra_texts = []
    for source in data:
        conversation = source["conversations"]
        if roles[conversation[0]["from"]] != conv.roles[0]:
            conversation = conversation[1:]

        conv.messages = []
        conv.append_message(conv.roles[0], conversation[0]["value"])
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        
        extra_texts.append(source["extra_texts"])
    
    outputs = []

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    extra_text_input_ids = []
    for texts in extra_texts:
        extra_text_input_ids.append(
            encoder_tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                max_length=encoder_tokenizer.model_max_length,
                truncation=True,
            ).input_ids
        )
    
    for one_input_ids, one_extra_text_input_ids in tqdm(zip(input_ids, extra_text_input_ids),
                                                        total=len(input_ids)):
        one_input_ids = one_input_ids.unsqueeze(0)
        one_extra_text_input_ids = one_extra_text_input_ids.unsqueeze(0)
        with torch.inference_mode():
            one_output_ids = model.generate(
                one_input_ids,
                extra_text_input_ids=one_extra_text_input_ids,
                extra_text_attention_mask=one_extra_text_input_ids.ne(encoder_tokenizer.pad_token_id),
                do_sample=False if args.temperature == 0 else True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            output = tokenizer.decode(one_output_ids[0], skip_special_tokens=True)
            outputs.append(output)
    
    return outputs


def eval_model(args):

    tokenizer, model, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name="ellama",
        projector_path=args.model_path,
        use_flash_attn=True,
    )
    encoder_tokenizer = AutoTokenizer.from_pretrained(model.config.encoder_name)

    for eval_set in EVAL_DATASETS:
        with open(eval_set, "r") as f:
            data = [json.loads(line) for line in f]
        outputs = evaluate_one_dataset(data, model, tokenizer, encoder_tokenizer, args)
        
        dataset_name = eval_set.split("/")[-1].replace(".jsonl", "")
        model_name = args.model_path.split("/")[-1]
        
        with open(os.path.join("results/pretrain_eval", f"{model_name}--{dataset_name}.txt"), "w") as f:
            for output in outputs:
                f.write(output + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/vicuna.jina.construct200k.pretrain")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--conv-mode", type=str, default="vicuna")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

