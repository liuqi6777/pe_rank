import argparse
import json
import os
from tqdm import tqdm

from llm4ranking.ranker.base import ListwiseSilidingWindowReranker
from ranker import *


def write_results(rerank_results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for i, hits in enumerate(rerank_results):
            for j, hit in enumerate(hits):
                f.write(f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
                f.write("\n")


def eval_model(args):
    from scripts.trec_eval import trec_eval
    from scripts.indexes_and_topics import TOPICS

    reranker = ListwiseSilidingWindowReranker()
    
    if args.ranker == "listwise-text-embedding":
        ranking_model = ListwiseTextEmbeddingRanker(
            model_path=args.model_path,
            model_base=args.model_base,
        )
    elif args.ranker == "listwise-embedding":
        ranking_model = ListwiseEmbeddingRanker(
            model_path=args.model_path,
            model_base=args.model_base,
        )
    else:
        raise ValueError(f"Ranker {args.ranker} not supported")

    for dataset in args.datasets:

        output_file = os.path.join(
            "results", "rerank_results", args.retriever,
            f"eval_{dataset}_{ranking_model.model_name.split('/')[-1]}_{args.ranker}_top{args.topk}.txt"
        )
        if os.path.exists(output_file) and not args.overwrite:
            print(f"{output_file} exists, skipping")
            trec_eval(TOPICS[dataset], output_file)
            continue

        input_file = os.path.join(
            "results", "retrieval_results", args.retriever,
            f"{dataset}_top{args.topk}.jsonl"
        )
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        rerank_results = []
        for i in tqdm(range(len(data))):
            rerank_result = reranker.rerank(
                query=data[i]["query"],
                candidates=data[i]["hits"],
                ranking_func=ranking_model,
                window_size=20,
                step=10,
            )
            rerank_results.append(rerank_result)
        write_results(rerank_results, output_file)
        trec_eval(TOPICS[dataset], output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["dl19", "dl20"])
    parser.add_argument(
        "--retriever", type=str, default="bm25",
        choices=["bm25", "jina-embeddings-v2-base-en", "e5-mistral", "splade++ed"]
    )
    parser.add_argument(
        "--ranker", type=str, default="listwise-text-embedding",
        choices=["listwise-text-embedding", "listwise-embedding"]
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_model(args)
