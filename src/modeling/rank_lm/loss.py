import torch
from torch import nn, Tensor, LongTensor

from constants import IGNORE_TOKEN_ID


def basic_rank_loss(hidden_states: Tensor, text_embeddings: Tensor, labels: LongTensor, ranking: LongTensor):
    hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
    logits = hidden_states @ text_embeddings.permute(0, 2, 1)
    label_mask, ranking_mask = make_mask_with_labels(labels, ranking)
    target = (logits * label_mask).sum(-1)  # (batch_size, sequence_length)
    z = torch.clamp((logits * ranking_mask).sum(-1), min=1e-9)
    loss = -torch.where((target / z) != 0, torch.log(target / z), 0).sum(-1).mean()
    return loss


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


def make_mask_with_labels(labels: LongTensor, ranking: LongTensor) -> tuple[LongTensor, LongTensor]:
    assert labels.shape[0] == ranking.shape[0]
    assert labels.shape[1] >= ranking.shape[1]
    label_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    ranking_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    for i in range(ranking.shape[0]):
        assert (labels[i] != IGNORE_TOKEN_ID).sum() == ranking.shape[1]
        label_mask[i, labels[i] !=
                   IGNORE_TOKEN_ID] = make_label_mask(ranking[i])
        ranking_mask[i, labels[i] !=
                     IGNORE_TOKEN_ID] = make_ranking_mask(ranking[i])
    return label_mask.contiguous(), ranking_mask.contiguous()