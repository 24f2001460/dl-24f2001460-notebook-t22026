import re
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
OPTIONS   = ['A', 'B', 'C', 'D', 'E']
MAX_WORDS = 30000
MAX_LEN   = 128
# ──────────────────────────────────────────────────────────────────────────────


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Row expansion ─────────────────────────────────────────────────────────────
def expand_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        prompt  = clean_text(row['prompt'])
        correct = row['answer']
        for opt in OPTIONS:
            option_text = clean_text(row[opt])
            text  = f"question {prompt} answer {option_text}"
            label = 1 if opt == correct else 0
            rows.append({'text': text, 'label': label})
    return pd.DataFrame(rows)


def expand_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        prompt = clean_text(row['prompt'])
        for opt in OPTIONS:
            option_text = clean_text(row[opt])
            text = f"question {prompt} answer {option_text}"
            rows.append({'id': row['id'], 'option': opt, 'text': text})
    return pd.DataFrame(rows)


# ── Vocabulary ────────────────────────────────────────────────────────────────
def build_vocab(texts, max_words: int = MAX_WORDS) -> dict:
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    most_common = counter.most_common(max_words - 2)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
    return word2idx


# ── Encoding ─────────────────────────────────────────────────────────────────
def encode_text(text: str, word2idx: dict, max_len: int = MAX_LEN) -> list:
    tokens = text.split()
    seq    = [word2idx.get(token, 1) for token in tokens]
    seq    = seq[:max_len]
    seq   += [0] * (max_len - len(seq))   # pad
    return seq


def encode_texts(texts, word2idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    return np.array([encode_text(t, word2idx, max_len) for t in texts])


# ── Dataset ───────────────────────────────────────────────────────────────────
class MCQDataset(Dataset):
    def __init__(self, X: np.ndarray, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y,
                              dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
