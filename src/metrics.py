import pytrec_eval

from transformers.trainer_utils import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction):
    logits, rankings = eval_pred.predictions
    rankings -= 1
    first_step_logits = logits[:, 0, :]
    first_step_acc = (first_step_logits.argmax(-1) == rankings[:, 0]).mean()
    metrics = {"first_step_acc": first_step_acc}

    qrels, run = {}, {}
    for i, (logit, ranking) in enumerate(zip(logits, rankings)):
        qrels[str(i)] = {str(j): 20 - int(ranking[j]) for j in range(len(ranking))}
        run[str(i)] = {str(j): float(logit[0, j]) for j in range(len(logit))}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.20"})
    scores = evaluator.evaluate(run)
    metrics["first_step_ndcg_at_20"] = sum([scores[query_id]["ndcg_cut_20"] for query_id in scores]) / len(scores)

    return metrics
