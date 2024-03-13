import argparse
import json
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from fastchat.model.model_adapter import get_conversation_template

from modeling.builder import load_pretrained_model


PLACEHOLDER = "<PLACEHOLDER>"
INSTRUCTION = """Providing with {n} passages, each is enclosed in the identifier []. Rank the passages based on their relevance to the search query: {query}. 
{inputs}
Search Query: {query}. Rank the {n} passages above based on their relevance to the search query in descending order.
"""


def eval_model(args):

    tokenizer, model, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name="embed_llama",
        device_map="cuda",
    )
    config = model.config
    model.to(torch.float16)

    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    # load data
    data = []
    input_file = args.input_file or f"results/{args.dataset}_retrival_{args.retriever}_top{args.topk}.jsonl"
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if "conversations" not in data[0]:
        new_data = []
        for d in data:
            retrieved_passages = d["hits"]
            new_data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": INSTRUCTION.format(
                            n=len(retrieved_passages),
                            query=d["query"],
                            inputs="\n".join(["[<PLACEHOLDER>]"] * len(retrieved_passages))
                        )
                    },
                ],
                "extra_texts": [p["content"] for p in retrieved_passages]
            })
    else:
        new_data = []
        for d in data:
            d["conversations"] = d["conversations"][:-1]
            new_data.append(d)
        new_data = random.sample(new_data, 200)

    # process data
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    extra_texts = []
    for i, source in enumerate(new_data):
        conversation = source["conversations"]
        if roles[conversation[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            conversation = conversation[1:]

        conv.messages = []
        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        
        extra_texts.append(source["extra_texts"])

    print(conversations[0])

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
                padding="max_length",
                max_length=200,
                truncation=True,
            ).input_ids
        )
    extra_text_input_ids = torch.nn.utils.rnn.pad_sequence(
        extra_text_input_ids,
        batch_first=True,
        padding_value=encoder_tokenizer.pad_token_id
    )

    ranks = []
    with torch.inference_mode():
        for i in tqdm(range(input_ids.shape[0])):
            rank = model.rank(
                input_ids[[i]],
                extra_text_input_ids=extra_text_input_ids[[i]],
                extra_text_attention_mask=extra_text_input_ids.ne(encoder_tokenizer.pad_token_id)[[i]],
                use_cache=True,
            )
            ranks.append(rank)

    # format output
    results = []
    if "hits" in data[0]:
        for i, rank in enumerate(ranks):
            result = []
            hits = data[i]["hits"]
            for j, r in enumerate(rank):
                result.append(f"{hits[0]['qid']} Q{i} {hits[r - 1]['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
            results.append(result)

        reranker = (args.model_path or args.model_base).split("/")[-1]
        if not args.alias:
            output_file = f"results/eval_{args.dataset}_{args.retriever}_{reranker}_top{args.topk}.txt"
        else:
            output_file = f"results/eval_{args.dataset}_{args.retriever}_{reranker}_top{args.topk}_{args.alias}.txt"
        with open(output_file, "w") as f:
            for result in results:
                f.write("\n".join(result))
                f.write("\n")
    else:
        for i, rank in enumerate(ranks):
            results.append((data[i]["ranking"], rank))
        with open("results/ranks.txt", "w") as f:
            for result in results:
                f.write("gpt-labeled: " + json.dumps(result[0]) + "\n")
                f.write("our outputs: " + json.dumps(result[1]) + "\n")
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "splade++ed"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--alias", type=str, default=None)
    args = parser.parse_args()
    eval_model(args)
