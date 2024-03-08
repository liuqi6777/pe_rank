import re
import torch
from torch import nn, Tensor
from transformers import AutoModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"projector_type": 'identity'}


def build_projector(config):
    projector_type = getattr(config, 'projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.embedding_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.embedding_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class Encoder(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.config = self.encoder.config
        self.pooling = config.encoder_pooling
        self.requires_grad_(False)
    
    @staticmethod
    @torch.no_grad()
    def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        return (torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) \
            / torch.clamp(torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9)).to(embeddings.dtype)

    @torch.no_grad()    
    def forward(self, **inputs) -> dict[str, Tensor]:
        for key in inputs:
            inputs[key] = inputs[key].to(self.encoder.device)
        outputs = self.encoder(**inputs)
        if self.pooling == 'mean':
            embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings