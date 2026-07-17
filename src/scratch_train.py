import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .scratch_utils import mapk , apk

from .scratch_preprocess import (
    expand_train_rows,
    build_vocab,
    encode_texts,
    MCQDataset,
    MAX_WORDS,
    MAX_LEN,
)
from models.scratch import build_model

# ── Config ────────────────────────────────────────────────────────────────────
# TRAIN_PATH = "..data/train.csv"
# TEST_PATH  = "..data/test.csv"
TRAIN_PATH   = os.getenv('TRAIN_PATH',  '/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv')
TEST_PATH    = os.getenv('TEST_PATH',   '/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv')

MODEL_OUT    = os.getenv('MODEL_OUT',   'model.pt')
VOCAB_OUT    = os.getenv('VOCAB_OUT',   'word2idx.pkl')
WANDB_KEY    = os.getenv('WANDB_API_KEY', '')
WANDB_PROJECT = '24f2001460-t22026'
WANDB_RUN     = 'scratch_lstm_v4'

# EPOCHS       = 15
# BATCH_SIZE   = 64
# LR           = 1e-3
# WEIGHT_DECAY = 1e-5
#TEST_SIZE    = 0.2
#RANDOM_STATE = 42


# EPOCHS = 20
# BATCH_SIZE = 32
# LR = 5e-4
# WEIGHT_DECAY = 1e-4
# TEST_SIZE = 0.1
# RANDOM_STATE = 42

# EPOCHS = 25
# BATCH_SIZE = 16
# LR = 2e-4
# WEIGHT_DECAY = 5e-5
# TEST_SIZE = 0.15
# RANDOM_STATE = 42

EPOCHS = 35
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 1e-4
MAX_LEN = 384
TEST_SIZE = 0.10
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds       = (torch.sigmoid(outputs) > 0.5).float()
        correct    += (preds == y_batch).sum().item()
        total      += y_batch.size(0)
    return total_loss, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Convert probabilities to predicted labels
    predicted_labels = [1 if s > 0.5 else 0 for s in all_scores]

    map3 = mapk(all_labels, predicted_labels, k=3)

    return total_loss, correct / total, map3


def main():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config={
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'max_words': MAX_WORDS, 'max_len': MAX_LEN,
        },
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_PATH)
    expanded  = expand_train_rows(train_df)
    print('Expanded train shape:', expanded.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(
        expanded['text'], expanded['label'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=expanded['label'],
    )

    # ── Vocab ─────────────────────────────────────────────────────────────────
    word2idx   = build_vocab(X_train, MAX_WORDS)
    vocab_size = len(word2idx)
    print('VOCAB SIZE:', vocab_size)

    # Save vocab so inference can reuse it
    with open(VOCAB_OUT, 'wb') as f:
        pickle.dump(word2idx, f)
    print('Vocab saved to', VOCAB_OUT)

    # ── Encode ────────────────────────────────────────────────────────────────
    X_train_seq = encode_texts(X_train, word2idx)
    X_valid_seq = encode_texts(X_valid, word2idx)

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_loader = DataLoader(MCQDataset(X_train_seq, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(MCQDataset(X_valid_seq, y_valid),
                              batch_size=BATCH_SIZE)

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────
    model     = build_model(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=1, factor=0.5
    )
    print(model)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_map3 = evaluate(
                                                model,
                                                valid_loader,
                                                criterion,
                                                device,
                                                )
        scheduler.step(val_acc)

        print(
                f"Epoch {epoch+1:02d}"
                f" | Train Loss: {train_loss:.4f}"
                f" | Train Acc: {train_acc:.4f}"
                f" | Val Loss: {val_loss:.4f}"
                f" | Val Acc: {val_acc:.4f}"
                f" | MAP@3: {val_map3:.4f}"
            )

        wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_map@3': val_map3,
                'lr': optimizer.param_groups[0]['lr'],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {'model_state': model.state_dict(), 'vocab_size': vocab_size},
                MODEL_OUT,
            )
            print(f" Best score (val_acc={val_acc:.4f})")

    print(f'\nBest val acc: {best_val_acc:.4f}')
    wandb.finish()


if __name__ == '__main__':
    main()
