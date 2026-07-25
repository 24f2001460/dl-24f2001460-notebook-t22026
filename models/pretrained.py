import torch
from typing import Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

LORA_CONFIG: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}
# ──────────────────────────────────────────────────────────────────────────────


def load_tokenizer(model_name: str = MODEL_NAME):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_bnb_config() -> BitsAndBytesConfig:
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_name: str = MODEL_NAME):
  
    bnb_config = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, lora_cfg: dict[str, Any] | None = None):

    cfg = lora_cfg or LORA_CONFIG
    lora_config = LoraConfig(**cfg)
    return get_peft_model(model, lora_config)


def build_model_and_tokenizer(model_name: str = MODEL_NAME):
    
    tokenizer = load_tokenizer(model_name)
    base      = load_base_model(model_name)
    model     = apply_lora(base)
    return tokenizer, model