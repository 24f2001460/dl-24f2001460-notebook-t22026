from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_vectorizer(max_features: int = 20000, ngram_range=(1, 2), stop_words='english'):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )


def build_model():
    return LogisticRegression(class_weight='balanced')
