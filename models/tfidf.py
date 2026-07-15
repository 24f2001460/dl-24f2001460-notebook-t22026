from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# def build_vectorizer(max_features: int = 20000, ngram_range=(1, 2), stop_words='english'):
#     return TfidfVectorizer(
#         max_features=max_features,
#         ngram_range=ngram_range,
#         stop_words=stop_words,
#     )


# def build_model():
#     return LogisticRegression(class_weight='balanced')


# def build_vectorizer():
#     return TfidfVectorizer(
#         max_features=30000,
#         ngram_range=(1,2),
#         stop_words='english'
#     )

# def build_model():
#     return LogisticRegression(
#         class_weight='balanced',
#         C=0.5,
#         solver='liblinear',
#         max_iter=1000,
#         random_state=42
#     )


# def build_vectorizer():
#     return TfidfVectorizer(
#         max_features=40000,
#         ngram_range=(1,3),
#         stop_words='english'
#     )

# def build_model():
#     return LogisticRegression(
#         class_weight='balanced',
#         C=2.0,
#         solver='saga',
#         max_iter=2000,
#         random_state=42
#     )

def build_vectorizer():
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1,3),
        stop_words='english',
        sublinear_tf=True,
        min_df=2
    )

def build_model():
    return LogisticRegression(
        class_weight='balanced',
        C=3.0,
        solver='saga',
        max_iter=3000,
        random_state=42
    )