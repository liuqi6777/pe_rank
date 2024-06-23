import torch
from torch import nn, Tensor, LongTensor

from constants import IGNORE_TOKEN_ID


class ListMLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _rank_minus_one(self, ranking: LongTensor) -> LongTensor:
        if ranking.min() == 1:
            ranking = ranking - 1
        return ranking

    def _make_ranking_mask(self, ranking: LongTensor) -> LongTensor:
        ranking = self._rank_minus_one(ranking)
        n = ranking.shape[-1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.long, device=ranking.device))
        mask = mask[:, torch.sort(ranking, dim=-1).indices].contiguous()
        return mask

    def _make_label_mask(self, ranking: LongTensor) -> LongTensor:
        ranking = self._rank_minus_one(ranking)
        n = ranking.shape[-1]
        mask = torch.zeros(n, n, dtype=torch.long, device=ranking.device)
        mask[torch.arange(n), ranking] = 1
        return mask

    def _make_mask_with_labels(
        self,
        labels: LongTensor,
        ranking: LongTensor,
    ) -> tuple[LongTensor, LongTensor, Tensor]:
        assert labels.shape[0] == ranking.shape[0]
        assert labels.shape[1] >= ranking.shape[1]
        label_mask = torch.zeros(
            labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
        ranking_mask = torch.zeros(
            labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
        
        for i in range(ranking.shape[0]):
            assert (labels[i] != IGNORE_TOKEN_ID).sum() == ranking.shape[1]
            label_mask[i, labels[i] != IGNORE_TOKEN_ID] = self._make_label_mask(ranking[i])
            ranking_mask[i, labels[i] != IGNORE_TOKEN_ID] = self._make_ranking_mask(ranking[i])

        return label_mask.contiguous(), ranking_mask.contiguous()

    def forward(
        self,
        hidden_states:  Tensor,
        text_embeddings: Tensor,
        labels: LongTensor,
        ranking: LongTensor
    ) -> tuple[Tensor, Tensor]:
        logits = (hidden_states @ text_embeddings.permute(0, 2, 1)).contiguous()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        label_mask, ranking_mask = self._make_mask_with_labels(shift_labels, ranking)
        labels = self._rank_minus_one(ranking).view(-1)
        shift_logits[ranking_mask == 0] = -1e9  # don't set to -inf, otherwise it will cause NaN
        shift_logits = shift_logits[label_mask.sum(-1).bool()]
        logprob = torch.nn.functional.cross_entropy(shift_logits, labels, reduce=False).view(ranking.shape[0], -1)
        loss = torch.sum(logprob, dim=-1).mean()
        return loss, shift_logits.reshape(ranking.shape[0], ranking.shape[1], -1)
