import argparse
import os
import json
import torch
import numpy as np
from pyserini.index import IndexReader
from pyserini.search import LuceneSearcher, LuceneImpactSearcher, FaissSearcher, get_topics, get_qrels
from pyserini.search.faiss import AutoQueryEncoder
from trec_eval import trec_eval
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from indexes_and_topics import INDEX, TOPICS


def run_retriever(topics, searcher, index_reader, qrels=None, topk=100, qid=None):
    ranks = []
    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=topk)
            rank = 0
            for hit in hits:
                rank += 1
                if index_reader.doc(hit.docid):
                    content = json.loads(index_reader.doc(hit.docid).raw())
                else:
                    continue
                if "title" in content:
                    content = (
                        "Title: " + content["title"] +
                        " " + "Content: " + content["text"]
                    )
                elif "contents" in content:
                    content = content["contents"]
                else:
                    content = content["passage"]
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid,
                    'docid': hit.docid,
                    'rank': rank,
                    'score': hit.score if isinstance(hit.score, float) else hit.score.item()
                })
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
        rerank_results.append(
            {'query': retrieval_results[i]['query'], 'hits': []})
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
    queries_embeddings = torch.nn.functional.normalize(
        queries_embeddings, p=2, dim=-1)
    for i in tqdm(range(len(retrieval_results))):
        all_passages = [hit['content'] for hit in retrieval_results[i]['hits']]
        if len(all_passages) == 0:
            continue
        passages_embeddings = model.encode(all_passages, convert_to_tensor=True)
        passages_embeddings = torch.nn.functional.normalize(passages_embeddings, p=2, dim=-1)
        scores = (queries_embeddings[i] @ passages_embeddings.T).flatten().cpu().numpy()
        ranking = np.argsort(scores)[::-1]
        rerank_results.append(
            {'query': retrieval_results[i]['query'], 'hits': []})
        for j in range(0, len(ranking)):
            hit = retrieval_results[i]['hits'][ranking[j]]
            hit['score'] = scores[ranking[j]]
            rerank_results[-1]['hits'].append(hit)
    return rerank_results


def eval_dataset(args):

    dataset, retriever, reranker, topk = args.dataset, args.retriever, args.reranker, args.topk

    print('#' * 20)
    print(f'Evaluation on {dataset}')
    print('#' * 20)

    retrieval_results_path = os.path.join('results', 'retrieval_results', retriever.split('/')[-1])
    retrieval_results_file = os.path.join(retrieval_results_path, f'{dataset}_top{topk}.jsonl')
    if os.path.exists(retrieval_results_file):
        with open(retrieval_results_file) as f:
            retrieval_results = [json.loads(line) for line in f]
    else:
        if retriever == 'bm25':
            searcher = LuceneSearcher.from_prebuilt_index(INDEX[retriever][dataset])
        elif retriever == 'splade++ed':
            searcher = LuceneImpactSearcher.from_prebuilt_index(
                INDEX[retriever][dataset],
                query_encoder='SpladePlusPlusEnsembleDistil',
                min_idf=0,
                encoder_type='onnx'
            )
        else:
            encoder = AutoQueryEncoder(retriever, pooling=args.dense_encoder_pooling, l2_norm=True)
            retriever = retriever.split('/')[-1]  # maybe hf model
            index_dir = os.path.join(
                'indexes', f'{INDEX["dense"][dataset]}.{retriever}')
            searcher = FaissSearcher(
                index_dir=index_dir,
                query_encoder=encoder
            )

        index_reader = IndexReader.from_prebuilt_index(INDEX["bm25"][dataset])
        topics = get_topics(TOPICS[dataset] if dataset != 'dl20' else 'dl20')
        qrels = get_qrels(TOPICS[dataset])
        retrieval_results = run_retriever(topics, searcher, index_reader, qrels, topk=topk)
        os.makedirs(retrieval_results_path, exist_ok=True)
        write_retrival_results(
            retrieval_results,
            retrieval_results_file
        )

    if reranker is None or args.reranker_type is None:
        output_file = os.path.join(retrieval_results_path, f'eval_{dataset}_top{topk}.txt')
        write_eval_file(retrieval_results, output_file)
        trec_eval(TOPICS[dataset], output_file)
        return

    # Rerank
    if reranker and args.reranker_type == 'embedding':
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = SentenceTransformer(reranker, trust_remote_code=True)
        rerank_results = run_embedding_rerank(retrieval_results, model)
    elif reranker and args.reranker_type == 'cross':
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker, num_labels=1, trust_remote_code=True
        )
        rerank_results = run_cross_rerank(retrieval_results, model, tokenizer)
    else:
        raise NotImplementedError(f'Reranker type {args.reranker_type} is not supported')

    # Evaluate nDCG@10
    rerank_results_path = os.path.join('results', 'rerank_results', retriever)
    os.makedirs(rerank_results_path, exist_ok=True)
    output_file = os.path.join(rerank_results_path,
                               f'eval_{dataset}_{reranker.split("/")[-1]}_top{topk}.txt')
    write_eval_file(rerank_results, output_file)
    trec_eval(TOPICS[dataset], output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    retriever = parser.add_argument_group('retriever')
    retriever.add_argument('--retriever', type=str, default='bm25')
    retriever.add_argument('--dense-encoder-pooling', type=str, default='mean')
    retriever.add_argument('--topk', type=int, default=100)
    reranker = parser.add_argument_group('reranker')
    reranker.add_argument('--reranker', type=str, default=None)
    reranker.add_argument('--reranker-type', type=str, default=None)
    args = parser.parse_args()
    eval_dataset(args)
