# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple
import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import  LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


PLACEHOLDER = '<PLACEHOLDER>'  # FIXME: move it to constants
PLACEHOLDER_ID = None


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


# Encoder

def mean_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    return torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) \
        / torch.clamp(torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9)


class Encoder(nn.Module):
    def __init__(self, model_name, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = self.encoder.config
        
        self.encoder.requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, texts: list[str], **kwargs: dict) -> dict[str, Tensor]:
        inputs = self._tokenize(texts, **kwargs)
        for key in inputs:
            inputs[key] = inputs[key].to(self.encoder.device)
        outputs = self.encoder(**inputs)
        return mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    
    def _tokenize(self, texts: list[str], **kwargs: dict) -> dict[str, Tensor]:
        tokenizer_options = {
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt"
        }
        tokenizer_options.update(kwargs)
        return self.tokenizer(texts, **tokenizer_options)


def build_encoder(config):
    return Encoder(config.encoder_name, config)
    
    
# Full Model

class ELMMetaModel:
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        if hasattr(config, 'encoder_name'):
            self.encoder = build_encoder(config)
            self.projector = build_projector(config)
            
    def get_encoder(self):
        return getattr(self, 'encoder', None)
    
    def get_projector(self):
        return getattr(self, 'projector', None)

    def initialize_modules(self, model_args):
        encoder_name = model_args.encoder_name
        pretrain_mlp_adapter = model_args.pretrain_mlp_adapter

        self.config.encoder_name = encoder_name
        if self.get_encoder() is None:
            self.encoder = build_encoder(self.config)

        self.config.use_proj = True
        self.config.projector_type = getattr(model_args, 'projector_type', 'linear')
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


class ELMMetaForCausalLM(ABC):
    
    @abstractmethod
    def get_model(self):
        pass
    
    def get_encoder(self):
        return self.get_model().get_encoder()
    
    def get_projector(self):
        return self.get_model().get_projector()
    
    def encode_texts(self, texts: list[str], **kwargs: dict) -> Tensor:
        embeddings = self.get_encoder()(texts, **kwargs)
        embeddings = self.get_projector()(embeddings)
        return embeddings
    
    def prepare_inputs_labels_for_elm(
        self, 
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        texts
    ):
        if self.get_encoder() is None or texts is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        input_embeddings = []
        
        assert len(texts) == input_ids.shape[0]
        
        for idx, (cur_texts, cur_input_ids) in enumerate(zip(texts, input_ids)):
            
            assert isinstance(cur_input_ids, Tensor), type(cur_input_ids)
            assert (cur_input_ids == PLACEHOLDER_ID).sum() == len(cur_texts)
            
            text_embeddings = self.encode_texts(cur_texts)
            
            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds[cur_input_ids == PLACEHOLDER_ID] = text_embeddings.to(cur_input_embeds.dtype)
            input_embeddings.append(cur_input_embeds)
            
        input_embeddings = torch.stack(input_embeddings)
        
        return None, position_ids, attention_mask, past_key_values, input_embeddings, labels
    
    def initialize_tokenizer(self, tokenizer):
        global PLACEHOLDER_ID
        tokenizer.add_tokens([PLACEHOLDER], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        PLACEHOLDER_ID = tokenizer.convert_tokens_to_ids(PLACEHOLDER)
    

class ELlamaCofig(LlamaConfig):
    model_type = "ellama"
    

class ELlamaModel(ELMMetaModel, LlamaModel):
    config_class = ELlamaCofig
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        LlamaModel.__init__(self, config)
        
        
class ELlamaForCausalLM(ELMMetaForCausalLM, LlamaForCausalLM):
    config_class = ELlamaCofig
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        LlamaForCausalLM.__init__(self, config)
        self.model = ELlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        texts: Optional[list[list[str]]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_elm(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                texts
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        texts: Optional[list[list[str]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if texts is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_elm(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                texts
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        texts = kwargs.pop("texts", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if texts is not None:
            inputs['texts'] = texts
        return inputs
    
    
AutoConfig.register("ellama", ELlamaCofig)
AutoModelForCausalLM.register(ELlamaCofig, ELlamaForCausalLM)


if __name__ == "__main__":
    from train import ModelArguments
    
    model_args = ModelArguments(
        model_name_or_path="JackFram/llama-68m",
        encoder_name="jinaai/jina-embeddings-v2-base-en",
        version="dev",
        freeze_backbone=True,
        tune_mlp_adapter=True,
        projector_type="mlp1x_gelu",
    )
    
    model: ELlamaForCausalLM = ELlamaForCausalLM.from_pretrained("JackFram/llama-68m")
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")
    tokenizer.pad_token = tokenizer.eos_token
    
    model.get_model().initialize_modules(model_args)
    model.initialize_tokenizer(tokenizer)
    
    print(model)
    
    texts = [
        ["This is a test sentence <PLACEHOLDER>"],
        ["This is another <PLACEHOLDER> test sentence"]
    ]
    inputs = tokenizer([t[0] for t in texts], return_tensors="pt", padding=True, truncation=True)
    inputs["texts"] = texts
    model(**inputs)
    model.generate(inputs["input_ids"], texts=texts, max_length=20)

