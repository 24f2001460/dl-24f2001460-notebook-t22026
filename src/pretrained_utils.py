import numpy as np
import torch

from .pretrained_preprocess import OPTIONS


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
    predicted = list(predicted)[:k]
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def mapk(actuals, predictions, k: int = 3) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))




def build_preprocess_logits_for_metrics(tokenizer):
    option_ids_tensor = get_option_ids_tensor(tokenizer)

    def preprocess_logits_for_metrics(logits, labels):
        option_ids = option_ids_tensor.to(logits.device)

        mask = labels != -100                      
        first_idx = mask.float().argmax(dim=1)      
        pred_pos = (first_idx - 1).clamp(min=0)      

        batch_idx = torch.arange(logits.size(0), device=logits.device)
        step_logits = logits[batch_idx, pred_pos]   
        option_logits = step_logits[:, option_ids]   

        return option_logits

    return preprocess_logits_for_metrics


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        option_logits, labels = eval_pred
        option_logits = np.asarray(option_logits)   
        labels = np.asarray(labels)                 

        mask = labels != -100
        has_answer = mask.any(axis=1)
        first_idx = mask.argmax(axis=1)
        true_token_ids = labels[np.arange(len(labels)), first_idx]

        
        true_options = [
            tokenizer.decode([tid]).strip() if ok else None
            for tid, ok in zip(true_token_ids, has_answer)
        ]

        top3_idx = np.argsort(-option_logits, axis=-1)[:, :3]
        pred_lists = [[OPTIONS[i] for i in row] for row in top3_idx]

        valid = [(p, a) for p, a in zip(pred_lists, true_options) if a in OPTIONS]
        if not valid:
            return {"accuracy": 0.0, "map@3": 0.0}

        accuracy = float(np.mean([p[0] == a for p, a in valid]))
        map3 = mapk([a for _, a in valid], [p for p, _ in valid], k=3)

        return {"accuracy": accuracy, "map@3": map3}

    return compute_metrics