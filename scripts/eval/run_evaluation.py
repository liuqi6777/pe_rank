import argparse
import tempfile
import os
import json
import shutil
import torch
import numpy as np
from pyserini.index import IndexReader
from pyserini.search import LuceneSearcher, LuceneImpactSearcher, FaissSearcher, get_topics, get_qrels
from pyserini.search.faiss import AutoQueryEncoder
from trec_eval import EvalFunction
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


INDEX = {
    'bm25': {
        'dl19': 'msmarco-v1-passage',
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
        'fever': 'beir-v1.0.0-fever.flat',
        'robust04': 'beir-v1.0.0-robust04.flat',
        'signal': 'beir-v1.0.0-signal1m.flat',
        'nq': 'beir-v1.0.0-nq.flat',
        'cfever': 'beir-v1.0.0-climate-fever.flat',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.flat',
    },
    'splade++ed': {
        'dl19': 'msmarco-v1-passage-splade-pp-ed-text',
        'dl20': 'msmarco-v1-passage-splade-pp-ed-text',
        'covid': 'beir-v1.0.0-trec-covid.splade-pp-ed',
        'arguana': 'beir-v1.0.0-arguana.splade-pp-ed',
        'touche': 'beir-v1.0.0-webis-touche2020.splade-pp-ed',
        'news': 'beir-v1.0.0-trec-news.splade-pp-ed',
        'scifact': 'beir-v1.0.0-scifact.splade-pp-ed',
        'fiqa': 'beir-v1.0.0-fiqa.splade-pp-ed',
        'scidocs': 'beir-v1.0.0-scidocs.splade-pp-ed',
        'nfc': 'beir-v1.0.0-nfcorpus.splade-pp-ed',
        'quora': 'beir-v1.0.0-quora.splade-pp-ed',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.splade-pp-ed',
        'fever': 'beir-v1.0.0-fever.splade-pp-ed',
        'robust04': 'beir-v1.0.0-robust04.splade-pp-ed',
        'signal': 'beir-v1.0.0-signal1m.splade-pp-ed',
        'nq': 'beir-v1.0.0-nq.splade-pp-ed',
        'cfever': 'beir-v1.0.0-climate-fever.splade-pp-ed',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.splade-pp-ed'
    },
    'dense': {
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
        'covid': 'beir-v1.0.0-trec-covid',
        'arguana': 'beir-v1.0.0-arguana',
        'touche': 'beir-v1.0.0-webis-touche2020',
        'news': 'beir-v1.0.0-trec-news',
        'scifact': 'beir-v1.0.0-scifact',
        'fiqa': 'beir-v1.0.0-fiqa',
        'scidocs': 'beir-v1.0.0-scidocs',
        'nfc': 'beir-v1.0.0-nfcorpus',
        'quora': 'beir-v1.0.0-quora',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity',
        'fever': 'beir-v1.0.0-fever',
        'robust04': 'beir-v1.0.0-robust04',
        'signal': 'beir-v1.0.0-signal1m',
        'nq': 'beir-v1.0.0-nq',
        'cfever': 'beir-v1.0.0-climate-fever',
        'hotpotqa': 'beir-v1.0.0-hotpotqa'
    }
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
                content = json.loads(index_reader.doc(hit.docid).raw())
                if "title" in content:
                    content = (
                        "Title: " + content["title"] + " " + "Content: " + content["text"]
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
                    'score': hit.score
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


def eval_dataset(args):
    
    dataset, retriever, reranker, topk = args.dataset, args.retriever, args.reranker, args.topk
    
    print('#' * 20)
    print(f'Evaluation on {dataset}')
    print('#' * 20)

    retrieval_results_file = f'results/{dataset}_retrival_{retriever.split("/")}_top{topk}.jsonl'
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
            retriever = retriever.split('/')[-1]  # maybe hf model
            index_dir = os.path.join('indexes', f'{INDEX["dense"][dataset]}.{retriever}')
            searcher = FaissSearcher(
                index_dir=index_dir,
                query_encoder=AutoQueryEncoder(retriever, pooling=args.dense_encoder_pooling, l2_norm=True)
            )

        index_reader = IndexReader.from_prebuilt_index(INDEX["bm25"][dataset])
        topics = get_topics(TOPICS[dataset] if dataset != 'dl20' else 'dl20')
        qrels = get_qrels(TOPICS[dataset])
        retrieval_results = run_retriever(topics, searcher, index_reader, qrels, topk=topk)
        write_retrival_results(
            retrieval_results, f'results/{dataset}_retrival_{retriever}_top{topk}.jsonl')

    # Rerank
    if reranker is None or args.reranker_type is None:
        rerank_results = retrieval_results
    elif reranker and args.reranker_type == 'embedding':
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = SentenceTransformer(reranker, trust_remote_code=True)
        rerank_results = run_embedding_rerank(retrieval_results, model)
    elif reranker and args.reranker_type == 'cross':
        tokenizer = AutoTokenizer.from_pretrained(reranker)
        model = AutoModelForSequenceClassification.from_pretrained(
            reranker, num_labels=1, trust_remote_code=True)
        rerank_results = run_cross_rerank(retrieval_results, model, tokenizer)
    else:
        raise NotImplementedError(f'Reranker type {args.reranker_type} is not supported')

    # Evaluate nDCG@10
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rerank_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], output_file])
    # Rename the output file to a better name
    if reranker:
        reranker = reranker.split('/')[-1]
    shutil.move(output_file, f'results/eval_{dataset}_{retriever}_{reranker}_top{topk}.txt')


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
