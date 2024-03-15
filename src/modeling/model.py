import torch
from copy import deepcopy

from modeling.encoder import Encoder, build_projector


class ELMMetaModel:
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if hasattr(config, 'encoder_name'):
            self.encoder = Encoder(config.encoder_name, config)
            self.projector = build_projector(config)
            self.set_encoder_head()

    def get_encoder(self):
        return getattr(self, 'encoder', None)

    def get_projector(self):
        return getattr(self, 'projector', None)

    def get_encoder_head(self):
        return getattr(self, 'encoder_head', None)

    def set_encoder_head(self):
        self.encoder_head = deepcopy(self.projector)

    def initialize_modules(self, model_args):
        encoder_name = model_args.encoder_name
        pretrain_mlp_adapter = model_args.pretrain_mlp_adapter

        self.config.encoder_name = encoder_name
        self.config.encoder_pooling = model_args.encoder_pooling
        if self.get_encoder() is None:
            self.encoder = Encoder(self.config.encoder_name, self.config)

        self.config.use_proj = True
        self.config.projector_type = getattr(
            model_args, 'projector_type', 'linear')
        self.config.embedding_size = self.encoder.config.hidden_size

        if self.get_projector() is None:
            self.projector = build_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.projector.parameters():
                p.requires_grad = True

        if pretrain_mlp_adapter is not None:
            projector_weights = torch.load(pretrain_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.projector.load_state_dict(get_w(projector_weights, 'projector'))

            print("Initialized encoder head with pre-trained projector weights")
            self.set_encoder_head()

    def encode_texts(self, **inputs: dict):
        embeddings = self.get_encoder()(**inputs)
        project_as_token_embeddings = self.get_projector()(embeddings)
        # no need to normalize to align with the original token embeddings
        project_text_embeddings = self.get_encoder_head()(embeddings)
        project_text_embeddings = torch.nn.functional.normalize(project_text_embeddings, p=2, dim=-1)
        return project_as_token_embeddings, project_text_embeddings
