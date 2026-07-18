import numpy as np
import torch

from pretrained_preprocess import OPTIONS


def get_option_token_ids(tokenizer) -> dict:
    return {
        opt: tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in OPTIONS
    }


def get_option_ids_tensor(tokenizer) -> torch.Tensor:
    token_ids = get_option_token_ids(tokenizer)
    return torch.tensor([token_ids[o] for o in OPTIONS])


# ── MAP@K ─────────────────────────────────────────────────────────────────────

def apk(actual, predicted, k: int = 3) -> float:
    predicted = [predicted][:k]
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def mapk(actuals, predictions, k: int = 3) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))