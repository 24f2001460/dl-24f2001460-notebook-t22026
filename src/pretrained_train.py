import os
import pandas as pd
import wandb

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from models.pretrained   import build_model_and_tokenizer, MODEL_NAME
from .pretrained_preprocess import make_hf_datasets, MAX_LEN
from .pretrained_utils import build_compute_metrics, build_preprocess_logits_for_metrics

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH    = os.getenv("TRAIN_PATH",   "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",   "/kaggle/working/mcq_qwen_output")
WANDB_PROJECT = "24f2001460-t22026"
WANDB_RUN     = "qwen2.5-7b-qlora-mc__new1"

EPOCHS        = 3
LR            = 1e-4      
BATCH_SIZE    = 1
GRAD_ACCUM    = 16
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1       
VAL_SIZE      = 0.1
SEED          = 42

EVAL_STEPS    = 20        



def get_wandb_key() -> str:

    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("wandb_key")
    except Exception:
        return os.environ.get("wandb_key", "")


def main():
    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_key = get_wandb_key()
    if wandb_key:
        wandb.login(key=wandb_key)
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN)

    # ── Model + Tokenizer ─────────────────────────────────────────────────────
    print("Loading tokenizer and model...")
    tokenizer, model = build_model_and_tokenizer(MODEL_NAME)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded {len(train_df)} training rows")

    train_ds, val_ds = make_hf_datasets(
        train_df, tokenizer,
        test_size=VAL_SIZE,
        seed=SEED,
        max_len=MAX_LEN,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Collator ──────────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Training Args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",          
        eval_steps=EVAL_STEPS,
        save_strategy="steps",          
        save_steps=EVAL_STEPS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,              
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="wandb",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="map@3",  
        greater_is_better=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset={"val": val_ds, "train": train_ds},
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=build_preprocess_logits_for_metrics(tokenizer),
    )

    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training stopped early: {e}")
        raise
    finally:
        print(f"Training complete (or halted). Adapter saved to: {OUTPUT_DIR}")
        wandb.finish()


if __name__ == "__main__":
    main()