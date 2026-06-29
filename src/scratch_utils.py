import numpy as np


def apk(actual, predicted, k: int = 3) -> float:
    predicted = [predicted][:k]
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0


def mapk(actuals, predictions, k: int = 3) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))
