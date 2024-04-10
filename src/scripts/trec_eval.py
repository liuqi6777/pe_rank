import argparse
import pytrec_eval
from pyserini.search import get_qrels_file


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: tuple[int] = (10, 50, 100, 200, 1000)
) -> dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        _map[f"MAP@{k}"] = 0.0
        ndcg[f"NDCG@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 4) for k, v in m.items()}

    _map = _normalize(_map)
    ndcg = _normalize(ndcg)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [_map, ndcg, recall]:
        all_metrics.update(mt)

    return all_metrics


def pretty_print_metrics(metrics: dict[str, float]):
    for metric, value in metrics.items():
        print(f"{metric:<12}\t{value}")


def trec_eval(dataset, ranking):
    with open(ranking, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    with open(get_qrels_file(dataset), 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)
    all_metrics = compute_metrics(qrels, run, k_values=(1, 5, 10, 20, 100))
    pretty_print_metrics(all_metrics)
    return all_metrics


if __name__ == '__main__':
    from indexes_and_topics import TOPICS

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dl19')
    parser.add_argument('--ranking', type=str, required=True)
    args = parser.parse_args()
    trec_eval(TOPICS[args.dataset], args.ranking)
