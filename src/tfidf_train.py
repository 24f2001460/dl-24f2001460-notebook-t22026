import os
import joblib
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss
)

from src.tfidf_preprocess import load_data, expand_mcq_rows
from src.tfidf_utils import mapk as compute_mapk
from models.tfidf import build_vectorizer, build_model


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
TEST_PATH = os.getenv("TEST_PATH", "data/test.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "tfidf.pkl")

WANDB_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = "24f2001460-t22026"
WANDB_RUN = "tfidf-logreg-v4"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():

    # --------------------------------------------------------------------------
    # 1. W&B Login
    # --------------------------------------------------------------------------

    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    else:
        wandb.login()

    vectorizer = build_vectorizer()
    model = build_model()

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config={
            "model": "LogisticRegression",
            "vectorizer": "TF-IDF",
            "max_features": vectorizer.get_params().get("max_features"),
            "ngram_range": vectorizer.get_params().get("ngram_range"),
            "stop_words": vectorizer.get_params().get("stop_words"),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "class_weight": model.get_params().get("class_weight"),
            "C": model.get_params().get("C"),
            "solver": model.get_params().get("solver"),
            "max_iter": model.get_params().get("max_iter"),
        },
    )

    # --------------------------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------------------------

    train_df, _ = load_data(TRAIN_PATH, TEST_PATH)
    expanded = expand_mcq_rows(train_df)

    print(f"Expanded train shape: {expanded.shape}")

    # --------------------------------------------------------------------------
    # 3. Train / Validation split
    # --------------------------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        expanded["text"],
        expanded["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------------------------------

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # --------------------------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------------------------

    model.fit(X_train_tfidf, y_train)

    # --------------------------------------------------------------------------
    # 6. Predictions
    # --------------------------------------------------------------------------

    preds = model.predict(X_valid_tfidf)
    probs = model.predict_proba(X_valid_tfidf)

    # --------------------------------------------------------------------------
    # 7. Metrics
    # --------------------------------------------------------------------------

    accuracy = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    loss = log_loss(y_valid, probs)
    map3 = compute_mapk(y_valid, preds)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Loss     : {loss:.4f}")
    print(f"MAP@3    : {map3:.4f}")

    # --------------------------------------------------------------------------
    # 8. Log to W&B
    # --------------------------------------------------------------------------

    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1,
        "loss": loss,
        "map@3": map3,
    })

    # --------------------------------------------------------------------------
    # 9. Save model
    # --------------------------------------------------------------------------

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
        },
        MODEL_OUT,
    )

    artifact = wandb.Artifact(
        "tfidf-model",
        type="model"
    )

    artifact.add_file(MODEL_OUT)
    wandb.log_artifact(artifact)

    print(f"Model saved to {MODEL_OUT}")

    wandb.finish()


if __name__ == "__main__":
    main()
