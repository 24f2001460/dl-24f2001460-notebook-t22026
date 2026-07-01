import os
import joblib
from sklearn.model_selection import train_test_split

from src.tfidf_preprocess import load_data, expand_mcq_rows
from src.tfidf_utils import mapk as compute_mapk
from models.tfidf  import build_vectorizer, build_model

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH = os.getenv('TRAIN_PATH', 'data/train.csv')
TEST_PATH  = os.getenv('TEST_PATH',  'data/test.csv')
MODEL_OUT  = os.getenv('MODEL_OUT',  'tfidf.pkl')
WANDB_KEY  = os.getenv('WANDB_API_KEY', '')
WANDB_PROJECT = 'smart-mcq-solver'
WANDB_RUN     = 'tfidf-logreg-v2'
TEST_SIZE  = 0.2
RANDOM_STATE = 42
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # 1. W&B login & init


    # 2. Load & expand data
    train_df, _ = load_data(TRAIN_PATH, TEST_PATH)
    expanded = expand_mcq_rows(train_df)
    print(f"Expanded train shape: {expanded.shape}")

    # 3. Train / validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        expanded['text'],
        expanded['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 4. Vectorise
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # 5. Train
    model = build_model()
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluate
    preds = model.predict(X_valid_tfidf)
    score = compute_mapk(y_valid, preds)
    print(f"MAP@3 on validation set: {score:.4f}")

    # 7. Log metrics

    # 8. Save artefacts
    joblib.dump({'vectorizer': vectorizer, 'model': model}, MODEL_OUT)
    print(f"Artefacts saved to {MODEL_OUT}")


if __name__ == '__main__':
    main()
