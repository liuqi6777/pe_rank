import argparse
import tempfile
import pandas as pd
import pytrec_eval
from pyserini.search import get_qrels_file


def trec_eval(
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
        if len(metric) < 8:
            print(f"{metric}\t\t{value}")
        else:
            print(f"{metric}\t{value}")


class EvalFunction:

    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        # print(qrels)
        run = pd.read_csv(run, sep='\s+', header=None)
        qrels = pd.read_csv(qrels, sep='\s+', header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)
        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def main(args_qrel, args_run):
        args_qrel = EvalFunction.trunc(args_qrel, args_run)
        with open(args_qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
        with open(args_run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)
        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10, 20, 100))
        pretty_print_metrics(all_metrics)
        return all_metrics


if __name__ == '__main__':
    from run_evaluation import TOPICS

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dl19')
    parser.add_argument('--ranking', type=str, required=True)
    args = parser.parse_args()
    EvalFunction.main(TOPICS[args.dataset], args.ranking)
