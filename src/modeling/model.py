import torch

from modeling.encoder import Encoder, build_projector


class ELMMetaModel:
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if hasattr(config, 'encoder_name') and not getattr(config, 'freeze_backbone', False):
            self.encoder = Encoder(config.encoder_name, config)
            self.projector = build_projector(config)

    def get_encoder(self):
        return getattr(self, 'encoder', None)

    def get_projector(self):
        return getattr(self, 'projector', None)

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
            projector_weights = torch.load(
                pretrain_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.projector.load_state_dict(
                get_w(projector_weights, 'projector'))

    def encode_texts(self, **inputs: dict):
        embeddings = self.get_encoder()(**inputs)
        embeddings = self.get_projector()(embeddings)
        return embeddings
