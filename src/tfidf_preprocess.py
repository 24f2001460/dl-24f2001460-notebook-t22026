import pandas as pd

OPTIONS = ['A', 'B', 'C', 'D', 'E']


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def expand_mcq_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        prompt = row['prompt']
        correct = row['answer']
        for opt in OPTIONS:
            text = prompt + ' [SEP] ' + str(row[opt])
            label = 1 if opt == correct else 0
            rows.append({'text': text, 'label': label, 'option': opt})
    return pd.DataFrame(rows)
