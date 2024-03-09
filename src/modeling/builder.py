import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from constants import PLACEHOLDER
from modeling.encoder import Encoder, build_projector
from modeling.causal_lm import EmbedLlamaForCausalLM
from modeling.rank_lm import EmbedLlamaForRankLM


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'embed' in model_name.lower():
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            model = EmbedLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(
                    model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith(
                'base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith(
                    'model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be projector only
            print('Loading model from base model...')
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            cfg_pretrained.vocab_size -= 2  # FIXME
            model = EmbedLlamaForRankLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            model.initialize_tokenizer(tokenizer)

            model.get_model().encoder = Encoder(cfg_pretrained.encoder_name, cfg_pretrained)
            model.get_model().projector = build_projector(cfg_pretrained)
            model.get_encoder().to("cuda")   # FIXME
            model.get_projector().to("cuda")

            projector_weights = torch.load(os.path.join(model_path, 'projector.bin'), map_location='cpu')
            projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
            model.load_state_dict(projector_weights, strict=False)
        else:
            print(f'Loading model from {model_path}...')
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            config = AutoConfig.from_pretrained(model_path)
            model = EmbedLlamaForRankLM.from_pretrained(
                model_path,
                config=config,
                low_cpu_mem_usage=True,
                **kwargs
            )
            model.initialize_tokenizer(tokenizer)

        model.resize_token_embeddings(len(tokenizer))

        global PLACEHOLDER_ID
        PLACEHOLDER_ID = tokenizer.convert_tokens_to_ids(PLACEHOLDER)

        encoder = model.get_encoder()
        if device_map != 'auto':
            encoder.to(device=device_map, dtype=torch.float16)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
