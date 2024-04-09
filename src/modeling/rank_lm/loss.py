import torch
from typing import Optional
from torch import nn, Tensor, LongTensor

from constants import IGNORE_TOKEN_ID


def set_loss_function(model: nn.Module, loss_type: str):
    print(f"Setting loss function: {loss_type}")
    loss_type = loss_type.split("+")
    if len(loss_type) == 1:
        loss_type, use_ib, temperature = loss_type[0], False, "no"
    elif len(loss_type) == 2:
        loss_type, use_ib_or_temperature = loss_type
        if use_ib_or_temperature == "ib":
            use_ib, temperature = True, "no"
        else:
            use_ib, temperature = None, use_ib_or_temperature
    elif len(loss_type) == 3:
        loss_type, use_ib, temperature = loss_type
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    if temperature == "no":
        # use dot product as similarity
        setattr(model, "normalize_embeddings", False)
        temperature = 1.0
    else:
        # use cosine similarity
        try:
            temperature = float(temperature)
        except ValueError:
            raise ValueError(f"Invalid temperature: {temperature}")
        setattr(model, "normalize_embeddings", True)

    if loss_type == "listnet":
        loss_function = ListNetLoss(temperature)
    elif loss_type == "listmle":
        if use_ib:
            loss_function = ListMLELossWithIBNegs(weighted=None, temperature=temperature)
        else:
            loss_function = ListMLELoss(weighted=None, temperature=temperature)
    elif loss_type == "plistmle":
        if use_ib:
            loss_function = ListMLELossWithIBNegs(weighted="weighted_4", temperature=temperature)
        else:
            loss_function = ListMLELoss(weighted="weighted_4", temperature=temperature)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    setattr(model, "loss_function", loss_function)


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
    ranking: LongTensor,
    weighted: Optional[str] = None
) -> tuple[LongTensor, LongTensor, Tensor]:
    assert labels.shape[0] == ranking.shape[0]
    assert labels.shape[1] >= ranking.shape[1]
    label_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    ranking_mask = torch.zeros(
        labels.shape[0], labels.shape[1], ranking.shape[1], dtype=torch.long, device=labels.device)
    
    for i in range(ranking.shape[0]):
        assert (labels[i] != IGNORE_TOKEN_ID).sum() == ranking.shape[1]
        label_mask[i, labels[i] != IGNORE_TOKEN_ID] = make_label_mask(ranking[i])
        ranking_mask[i, labels[i] != IGNORE_TOKEN_ID] = make_ranking_mask(ranking[i])
    
    weights = torch.zeros(ranking.shape[1], dtype=torch.float, device=labels.device)
    if weighted is None:
        weights[:] = 1
    elif weighted == "listnet":
        weights[0] = 1
    elif weighted == "weighted_1":
        weights = 1 / torch.arange(1, ranking.shape[1] + 1, device=labels.device, dtype=torch.float)
    elif weighted == "weighted_2":
        weights = torch.arange(1, ranking.shape[1] + 1, device=labels.device, dtype=torch.float).flip(0) / ranking.shape[1]
    elif weighted == "weighted_3":
        weights = 1 / torch.log(torch.arange(2, ranking.shape[1] + 2, device=labels.device, dtype=torch.float))
    elif weighted == "weighted_4":
        weights = 1 / torch.pow(2, torch.arange(ranking.shape[1], device=labels.device, dtype=torch.float))

    return label_mask.contiguous(), ranking_mask.contiguous(), weights.unsqueeze(0)


class ListNetLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        hidden_states:  Tensor,
        text_embeddings: Tensor,
        labels: LongTensor,
        ranking: LongTensor
    ) -> tuple[Tensor, Tensor]:
        logits = (hidden_states @ text_embeddings.permute(0, 2, 1)) / self.temperature

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        label_mask, ranking_mask, weights = make_mask_with_labels(shift_labels, ranking, weighted="listnet")
        labels = rank_minus_one(ranking).view(-1)
        shift_logits[ranking_mask == 0] = float("-inf")
        shift_logits = shift_logits[label_mask.sum(-1).bool()]
        logprob = torch.nn.functional.cross_entropy(shift_logits, labels, reduce=False).view(ranking.shape[0], -1)
        loss = torch.sum(logprob * weights, dim=-1).mean()
        return loss, shift_logits.reshape(ranking.shape[0], ranking.shape[1], -1)


class ListMLELoss(nn.Module):
    def __init__(self, weighted: Optional[str] = None, temperature: float = 1.0):
        super().__init__()
        self.weighted = weighted
        self.temperature = temperature

    def forward(
        self,
        hidden_states:  Tensor,
        text_embeddings: Tensor,
        labels: LongTensor,
        ranking: LongTensor
    ) -> tuple[Tensor, Tensor]:
        logits = (hidden_states @ text_embeddings.permute(0, 2, 1)) / self.temperature

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        label_mask, ranking_mask, weights = make_mask_with_labels(shift_labels, ranking, weighted=self.weighted)
        labels = rank_minus_one(ranking).view(-1)
        shift_logits[ranking_mask == 0] = float("-inf")
        shift_logits = shift_logits[label_mask.sum(-1).bool()]
        logprob = torch.nn.functional.cross_entropy(shift_logits, labels, reduce=False).view(ranking.shape[0], -1)
        loss = torch.sum(logprob * weights, dim=-1).mean()
        return loss, shift_logits.reshape(ranking.shape[0], ranking.shape[1], -1)


class ListMLELossWithIBNegs(ListMLELoss):

    def forward(
        self,
        hidden_states:  Tensor,
        text_embeddings: Tensor,
        labels: LongTensor,
        ranking: LongTensor
    ):
        loss1, shift_logits = super().forward(hidden_states, text_embeddings, labels, ranking)

        label_mask, *_ = make_mask_with_labels(labels[..., 1:], ranking, weighted=self.weighted)        
        hidden_states = hidden_states[..., :-1, :][label_mask.sum(-1).bool()]
        hidden_states = hidden_states.view(ranking.size(0), -1, hidden_states.size(-1))[:, 0]
        ib_labels = rank_minus_one(ranking)[:, 0] + torch.arange(
            hidden_states.shape[0], device=ranking.device) * ranking.shape[-1]
        all_logits = (hidden_states @ text_embeddings.view(-1, text_embeddings.shape[-1]).T) / self.temperature
        loss2 = torch.nn.functional.cross_entropy(all_logits, ib_labels)
        return loss1 + loss2, shift_logits
