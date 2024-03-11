import argparse
import tempfile
import os
import json
import shutil
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from trec_eval import EvalFunction
from tqdm import tqdm


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


def eval_dataset(dataset, retriver, reranker, topk=100):
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
    rerank_results = retrieval_results  # TODO
    # write_retrival_results(rerank_results, f'results/{dataset}_rerank_{reranker}_top{topk}.jsonl')

    # Evaluate nDCG@10
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rerank_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'results/eval_{dataset}_{retriver}_{reranker}_top{topk}.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--retriver', type=str, default='bm25', choices=['bm25', 'splade++ed'])
    parser.add_argument('--reranker', type=str, default=None)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()
    eval_dataset(args.dataset, args.retriver, args.reranker, args.topk)
