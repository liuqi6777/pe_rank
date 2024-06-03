from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_type: str = field(
        default="causal_lm",
        metadata={
            "help": "The type of model to use. Can be 'causal_lm' or 'rank_lm'."
        },
    )
    model_name_or_path: Optional[str] = field(default="JackFram/llama-68m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    freeze_embedding_layer: bool = field(default=False)
    tune_mlp_adapter: bool = field(default=False)
    encoder_name: Optional[str] = field(default=None)
    encoder_pooling: Optional[str] = field(
        default="mean", metadata={"help": "mean or cls"}
    )
    pretrain_mlp_adapter: Optional[str] = field(default=None)
    projector_type: Optional[str] = field(default='linear')
    loss_type: Optional[str] = field(default='plistmle')


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    use_embedding_with_content: bool = field(default=True)
    use_embedding_without_content: bool = field(default=False)


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "The implementation of attention. Can be 'flash_attention_2'."
        },
    )

    loss1_weight: float = field(default=1.0)
    loss2_weight: float = field(default=1.0)
    kl_loss_weight: float = field(default=0.0)


@dataclass
class LoraArguments:
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
