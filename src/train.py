import math
import os
import pathlib

from peft import LoraConfig, get_peft_model
import torch
import transformers

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from data import make_data_module
from modeling.causal_lm import EmbedLlamaForCausalLM
from modeling.rank_lm import EmbedLlamaForRankLM
from trainer import Trainer
from utils import *


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Set up model
    if model_args.encoder_name:
        if model_args.model_type == "causal_lm":
            model = EmbedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            model.generation_config.do_sample = True
        elif model_args.model_type == "rank_lm":
            model = EmbedLlamaForRankLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            raise ValueError(f"Invalid model type: {model_args.model_type}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model.generation_config.do_sample = True

    # Set RoPE scaling factor
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        model.config.rope_scaling = {
            "type": "linear", "factor": scaling_factor}
    model.config.use_cache = False

    # Load Lora
    if lora_args.lora_enable:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.encoder_name:
        model.get_model().initialize_modules(model_args)

        encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.encoder_name,
            cache_dir=training_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mlp_adapter = training_args.tune_mlp_adapter = model_args.tune_mlp_adapter
        model.config.freeze_backbone = training_args.freeze_backbone = model_args.freeze_backbone
        if model_args.freeze_backbone:
            model.requires_grad_(False)
        for p in model.get_model().projector.parameters():
            p.requires_grad = training_args.tune_mlp_adapter

        model.initialize_tokenizer(tokenizer)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load data
    data_module = make_data_module(
        tokenizer=tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        data_args=data_args,
        model_type=model_args.model_type,
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True

    if lora_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
