import argparse
import tempfile
import os
import json
import shutil
import torch
import numpy as np
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from trec_eval import EvalFunction
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


INDEX = {
    'dl19': 'msmarco-v1-passage',  # msmarco-v1-passage-splade-pp-ed-text
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',
    'nq': 'beir-v1.0.0-nq.flat',
    'cfever': 'beir-v1.0.0-climate-fever.flat',
    'hotpotqa': 'beir-v1.0.0-hotpotqa.flat',
}

TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',
    'nq': 'beir-v1.0.0-nq-test',
    'cfever': 'beir-v1.0.0-climate-fever-test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-test',
}


def run_retriever(topics, searcher, qrels=None, topk=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=topk)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + \
                    content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=topk)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + \
                        content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def write_retrival_results(rank_results, file):
    with open(file, 'w') as f:
        for item in rank_results:
            f.write((json.dumps(item) + '\n'))
    return True


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


@torch.no_grad()
def run_cross_rerank(retrieval_results, model, tokenizer):
    model.eval()
    model.to('cuda')
    rerank_results = []
    all_queries = [hit['query'] for hit in retrieval_results]
    for i in tqdm(range(len(retrieval_results))):
        all_passages = [hit['content'] for hit in retrieval_results[i]['hits']]
        if len(all_passages) == 0:
            continue
        inputs = tokenizer(
            [(all_queries[i], passage) for passage in all_passages],
            return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        scores = model(**inputs).logits.flatten().cpu().numpy().tolist()
        ranking = np.argsort(scores)[::-1]
        rerank_results.append({'query': retrieval_results[i]['query'], 'hits': []})
        for j in range(0, len(ranking)):
            hit = retrieval_results[i]['hits'][ranking[j]]
            hit['score'] = scores[ranking[j]]
            rerank_results[-1]['hits'].append(hit)
    return rerank_results


@torch.no_grad()
def run_embedding_rerank(retrieval_results, model):
    model.eval()
    model.to('cuda')
    rerank_results = []
    all_queries = [hit['query'] for hit in retrieval_results]
    queries_embeddings = model.encode(all_queries, convert_to_tensor=True)
    queries_embeddings = torch.nn.functional.normalize(queries_embeddings, p=2, dim=-1)
    for i in tqdm(range(len(retrieval_results))):
        all_passages = [hit['content'] for hit in retrieval_results[i]['hits']]
        if len(all_passages) == 0:
            continue
        passages_embeddings = model.encode(all_passages, convert_to_tensor=True)
        passages_embeddings = torch.nn.functional.normalize(passages_embeddings, p=2, dim=-1)
        scores = (queries_embeddings[i] @ passages_embeddings.T).flatten().cpu().numpy()
        ranking = np.argsort(scores)[::-1]
        rerank_results.append({'query': retrieval_results[i]['query'], 'hits': []})
        for j in range(0, len(ranking)):
            hit = retrieval_results[i]['hits'][ranking[j]]
            hit['score'] = scores[ranking[j]]
            rerank_results[-1]['hits'].append(hit)
    return rerank_results


def eval_dataset(dataset, retriver, reranker, reranker_type=None, topk=100):
    print('#' * 20)
    print(f'Evaluation on {dataset}')
    print('#' * 20)

    retrieval_results_file = f'results/{dataset}_retrival_{retriver}_top{topk}.jsonl'
    if os.path.exists(retrieval_results_file):
        with open(retrieval_results_file) as f:
            retrieval_results = [json.loads(line) for line in f]
    else:
        # Retrieve passages using pyserini BM25.
        try:
            searcher = LuceneSearcher.from_prebuilt_index(INDEX[dataset])
            topics = get_topics(TOPICS[dataset] if dataset != 'dl20' else 'dl20')
            qrels = get_qrels(TOPICS[dataset])
            retrieval_results = run_retriever(topics, searcher, qrels, topk=topk)
            write_retrival_results(retrieval_results, f'results/{dataset}_retrival_bm25_top{topk}.jsonl')
        except:
            print(f'Failed to retrieve passages for {dataset}')
            return

    # Rerank
    if reranker is None or reranker_type is None:
        rerank_results = retrieval_results
    elif reranker and reranker_type == "embedding":
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = SentenceTransformer(reranker, trust_remote_code=True)
        rerank_results = run_embedding_rerank(retrieval_results, model)
    elif reranker and reranker_type == "cross":
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker, num_labels=1, trust_remote_code=True)
        rerank_results = run_cross_rerank(retrieval_results, model, tokenizer)
    else:
        raise NotImplementedError(f"Reranker type {reranker_type} is not supported")
    # write_retrival_results(rerank_results, f'results/{dataset}_rerank_{reranker}_top{topk}.jsonl')
    

    # Evaluate nDCG@10
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rerank_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'results/eval_{dataset}_{retriver}_{reranker.split("/")[-1]}_top{topk}.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--retriver', type=str, default='bm25', choices=['bm25', 'splade++ed'])
    parser.add_argument('--reranker', type=str, default=None)
    parser.add_argument('--reranker-type', type=str, default=None)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()
    eval_dataset(args.dataset, args.retriver, args.reranker, args.reranker_type, args.topk)
