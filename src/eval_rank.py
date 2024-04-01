import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from modeling.builder import load_pretrained_model
from modeling.rank_lm.reranker import RerankLLM


def write_results(rerank_results, output_file):
    with open(output_file, "w") as f:
        for i, result in enumerate(rerank_results):
            hits = result["hits"]
            for j, hit in enumerate(hits):
                f.write(f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
                f.write("\n")


def eval_model(args):

    tokenizer, model, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name="embed_llama",
        device_map="cuda",
    )
    config = model.config
    model.to(torch.float16)
    model.eval()
    if not args.model_path.split("/")[-1].split(".")[-1].endswith("no"):  # hotfix
        setattr(model, "normalize_embeddings", True)

    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    reranker_llm = RerankLLM(
        model=model,
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer,
    )

    for dataset in args.datasets:

        reranker = (args.model_path or args.model_base).split("/")[-1]
        output_file = os.path.join(
            "results", "rerank_results", args.retriever,
            f"eval_{dataset}_{reranker}_{args.rerank_mode}_top{args.topk}.txt"
        )
        if os.path.exists(output_file) and not args.overwrite:
            print(f"{output_file} exists, skipping")
            continue

        input_file = os.path.join(
            "results", "retrieval_results", args.retriever,
            f"{dataset}_top{args.topk}.jsonl"
        )
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        if args.rerank_mode == "direct":
            window_size, step = None, None
        elif args.rerank_mode.startswith("sliding"):
            _, window_size, step = args.rerank_mode.split("-")
            window_size, step = int(window_size), int(step)

        rerank_results = []
        for i in tqdm(range(len(data))):
            rerank_result = reranker_llm.rerank(
                retrieved_result=data[i],
                window_size=window_size,
                step=step,
            )
            rerank_results.append(rerank_result)
        write_results(rerank_results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["dl19"])
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--rerank-mode", type=str, default="direct")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_model(args)
