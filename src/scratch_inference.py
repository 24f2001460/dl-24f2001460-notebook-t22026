import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .scratch_preprocess import expand_test_rows, encode_texts, MAX_LEN
from models.scratch import build_model


def load_artefacts(model_path: str, vocab_path: str, device):
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    checkpoint  = torch.load(model_path, map_location=device)
    vocab_size  = checkpoint['vocab_size']
    model       = build_model(vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return word2idx, model


def predict(test_df: pd.DataFrame, word2idx: dict, model, device,
            batch_size: int = 64) -> pd.DataFrame:
    expanded     = expand_test_rows(test_df)
    X_test_seq   = encode_texts(expanded['text'], word2idx, MAX_LEN)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.long)
    loader       = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size)

    all_probs = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch   = X_batch.to(device)
            outputs   = model(X_batch)
            probs     = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())

    expanded['prob'] = all_probs

    results = []
    for qid in expanded['id'].unique():
        temp   = expanded[expanded['id'] == qid].sort_values('prob', ascending=False)
        top3   = ' '.join(temp['option'].values[:3])
        results.append({'ID': qid, 'Prediction': top3})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='DL MCQ inference')
    parser.add_argument('--test_path',   default='/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv')
    parser.add_argument('--model_path',  default='/kaggle/input/<your-dataset>/model.pt')
    parser.add_argument('--vocab_path',  default='/kaggle/input/<your-dataset>/word2idx.pkl')
    parser.add_argument('--output_path', default='/kaggle/working/submission.csv')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    test_df          = pd.read_csv(args.test_path)
    word2idx, model  = load_artefacts(args.model_path, args.vocab_path, device)

    submission = predict(test_df, word2idx, model, device)
    submission.to_csv(args.output_path, index=False)

    print(f'Saved {len(submission)} predictions to {args.output_path}')
    print(submission.head(10))

    # Quick validation
    assert list(submission.columns) == ['ID', 'Prediction'], "Column name mismatch!"
    assert submission['Prediction'].apply(lambda x: len(x.split()) == 3).all()

if __name__ == '__main__':
    main()
