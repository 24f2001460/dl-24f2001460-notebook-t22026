import numpy as np


def apk(actual, predicted, k: int = 3) -> float:
    predicted = [predicted]
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    for i, p in enumerate(predicted):
        if p == actual:
            score = 1.0 / (i + 1)
            break
    return score


def mapk(actuals, predictions, k: int = 3) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))
