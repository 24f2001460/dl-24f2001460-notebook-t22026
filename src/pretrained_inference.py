import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from models.pretrained      import load_tokenizer, MODEL_NAME
from .pretrained_preprocess import build_prompt, OPTIONS
from .pretrained_utils      import get_option_ids_tensor



def load_model_for_inference(adapter_path: str, model_name: str = MODEL_NAME, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def predict_top3(row: dict, tokenizer, model, option_ids_tensor: torch.Tensor) -> str:
    prompt = build_prompt(row) + " "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]   # (vocab_size,)

    option_logits = logits[option_ids_tensor.to(model.device)]  # (5,)
    probs         = torch.softmax(option_logits, dim=0).float().cpu().numpy()
    top3_idx      = np.argsort(-probs)[:3]
    return " ".join(OPTIONS[i] for i in top3_idx)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 QLoRA inference")
    parser.add_argument("--test_path",    default="/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv")
    parser.add_argument("--adapter_path", default="/kaggle/input/<your-dataset>/mcq_qwen_output")
    parser.add_argument("--output_path",  default="/kaggle/working/submission.csv")
    parser.add_argument("--model_name",   default=MODEL_NAME)
    args = parser.parse_args()

    print("Loading model and adapter...")
    tokenizer, model   = load_model_for_inference(args.adapter_path, args.model_name)
    option_ids_tensor  = get_option_ids_tensor(tokenizer)

    test_df = pd.read_csv(args.test_path)
    print(f"Predicting on {len(test_df)} test rows...")

    predictions = [
        predict_top3(row, tokenizer, model, option_ids_tensor)
        for _, row in test_df.iterrows()
    ]

    submission = pd.DataFrame({
        "ID":         test_df["id"].tolist(),
        "Prediction": predictions,
    })
    submission.to_csv(args.output_path, index=False)

    print(f"\nSaved {len(submission)} predictions → {args.output_path}")
    print(submission.head(10))

    # ── Validation ────────────────────────────────────────────────────────────
    assert list(submission.columns) == ["ID", "Prediction"], "Column name mismatch!"
    assert submission["Prediction"].apply(lambda x: len(x.split()) == 3).all(), \
        "Some rows don't have exactly 3 predictions!"
    print("\n✓ Submission format validated.")


if __name__ == "__main__":
    main()