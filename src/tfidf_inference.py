import argparse
import joblib
import pandas as pd

from src.tfidf_preprocess import OPTIONS


def load_artefacts(model_path: str):
    bundle = joblib.load(model_path)
    return bundle['vectorizer'], bundle['model']


def build_test_texts(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in test_df.iterrows():
        prompt = row['prompt']
        for opt in OPTIONS:
            text = prompt + ' [SEP] ' + str(row[opt])
            rows.append({'id': row.get('id', _), 'option': opt, 'text': text})
    return pd.DataFrame(rows)


def predict(test_df: pd.DataFrame, vectorizer, model) -> pd.DataFrame:
    expanded = build_test_texts(test_df)
    X_test_tfidf = vectorizer.transform(expanded['text'])

    # Probability of class 1 (correct answer)
    proba = model.predict_proba(X_test_tfidf)[:, 1]
    expanded['score'] = proba

    # Rank options by score per question, return top 3 space-separated
    def rank_options(group):
        ranked = group.sort_values('score', ascending=False)['option'].tolist()
        return ' '.join(ranked[:3])  # top 3 options ranked by confidence

    result = expanded.groupby('id').apply(rank_options).reset_index()
    result.columns = ['ID', 'Prediction']
    return result


def main():
    parser = argparse.ArgumentParser(description='TF-IDF MCQ inference')
    parser.add_argument('--test_path',   default='/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv')
    parser.add_argument('--model_path',  default='tfidf.pkl')
    parser.add_argument('--output_path', default='submission.csv')
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_path)
    vectorizer, model = load_artefacts(args.model_path)

    submission = predict(test_df, vectorizer, model)
    submission.to_csv(args.output_path, index=False)
    print(f"Saved {len(submission)} predictions to {args.output_path}")


if __name__ == '__main__':
    main()
