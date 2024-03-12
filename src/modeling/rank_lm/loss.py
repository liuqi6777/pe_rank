import torch
from itertools import product
from loguru import logger
from torch import nn, Tensor, LongTensor

from constants import IGNORE_TOKEN_ID


def rank_minus_one(ranking: LongTensor) -> LongTensor:
    if ranking.min() == 1:
        ranking = ranking - 1
    return ranking


def make_ranking_mask(ranking: LongTensor) -> LongTensor:
    ranking = rank_minus_one(ranking)
    n = ranking.shape[-1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.long, device=ranking.device))
    mask = mask[:, torch.sort(ranking, dim=-1).indices].contiguous()
    return mask


def make_label_mask(ranking: LongTensor) -> LongTensor:
    ranking = rank_minus_one(ranking)
    n = ranking.shape[-1]
    mask = torch.zeros(n, n, dtype=torch.long, device=ranking.device)
    mask[torch.arange(n), ranking] = 1
    return mask


def make_mask_with_labels(
    labels: LongTensor, 
    ranking: LongTensor
) -> tuple[LongTensor, LongTensor, Tensor]:
    assert labels.shape[0] == ranking.shape[0]
    assert labels.shape[1] >= ranking.shape[1]
    label_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    ranking_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    weights = torch.zeros(
        labels.shape[0], labels.shape[1], dtype=torch.float, device=labels.device)
    for i in range(ranking.shape[0]):
        assert (labels[i] != IGNORE_TOKEN_ID).sum() == ranking.shape[1]
        label_mask[i, labels[i] != IGNORE_TOKEN_ID] = make_label_mask(ranking[i])
        ranking_mask[i, labels[i] != IGNORE_TOKEN_ID] = make_ranking_mask(ranking[i])
        weights[i, labels[i] != IGNORE_TOKEN_ID] = 1 / torch.arange(
            1, ranking.shape[1] + 1, device=labels.device, dtype=torch.float)
    return label_mask.contiguous(), ranking_mask.contiguous(), weights.contiguous()


class RankingLoss(nn.Module):
    def __init__(self, weighted: bool = False):
        super().__init__()
        self.weighted = weighted

    def forward(
        self,
        hidden_states:  Tensor,
        text_embeddings: Tensor,
        labels: LongTensor,
        ranking: LongTensor
    ) -> tuple[Tensor, Tensor]:
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        logits = hidden_states @ text_embeddings.permute(0, 2, 1)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        label_mask, ranking_mask, weights = make_mask_with_labels(shift_labels, ranking)
        target = (shift_logits * label_mask).sum(-1)
        shift_logits[ranking_mask == 0] = float("-inf")
        z = torch.logsumexp(shift_logits, dim=-1)
        prob = torch.where(z == float("-inf"), torch.zeros_like(target), target - z)
        # logger.debug(f"prob: {torch.exp(prob[0][prob[0] != 0]).detach().cpu().tolist()}")
        if not self.weighted:
            loss = -torch.sum(prob, dim=-1).mean()
        else:
            loss = -torch.sum(prob * weights, dim=-1).mean()
        return loss, logits
