from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
import numpy as np
import pandas as pd
from rfphate.rfgap import RFGAP
from joblib import Parallel, delayed

def is_continuous(y: np.ndarray) -> bool:
    """Check if the target variable is continuous."""
    return pd.Series(y).dtype.kind == 'f'  # Faster check

def evaluate_fold(model, rf_model, x, y, embedding, train_idx, test_idx):
    """Evaluate models on train-test split."""
    x_train, x_test = np.take(x, train_idx, axis=0), np.take(x, test_idx, axis=0)
    y_train, y_test = np.take(y, train_idx, axis=0), np.take(y, test_idx, axis=0)
    emb_train, emb_test = np.take(embedding, train_idx, axis=0), np.take(embedding, test_idx, axis=0)

    # Clone models to avoid re-fitting the same instance
    knn = clone(model)
    rf = clone(rf_model)

    # KNN on original data
    knn.fit(x_train, y_train)
    knn_score_x = knn.score(x_test, y_test)

    # KNN on embedding
    knn.fit(emb_train, y_train)
    knn_score_emb = knn.score(emb_test, y_test)

    # RFGAP on original data
    rf.fit(x_train, y_train)
    rf_score_x = rf.score(x_test, y_test)

    # RFGAP on embedding
    rf.fit(emb_train, y_train)
    rf_score_emb = rf.score(emb_test, y_test)

    return knn_score_x, knn_score_emb, rf_score_x, rf_score_emb

def model_embedding_diff(
    x: np.ndarray,
    y: np.ndarray,
    embedding: np.ndarray,
    model=None,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = None,
    model_kwargs: dict = None,
    n_jobs: int = 1,
    **kwargs
) -> dict:
    """Evaluate KNN and RFGAP models on original and embedded data."""
    
    # Determine model type
    if model is None:
        model = KNeighborsRegressor if is_continuous(y) else KNeighborsClassifier
    model = model(**(model_kwargs or {}))

    rf_model = RFGAP(y=y, oob_score=True)

    # Precompute cross-validation splits to avoid redundant computation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(x))  # Store splits beforehand

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(model, rf_model, x, y, embedding, train_idx, test_idx)
        for _ in range(n_repeats) for train_idx, test_idx in splits
    )

    # Unpack results efficiently
    knn_scores_x, knn_scores_emb, rf_scores_x, rf_scores_emb = zip(*results)

    return {
        "knndiff": np.mean(knn_scores_emb) - np.mean(knn_scores_x),
        "rfdiff": np.mean(rf_scores_emb) - np.mean(rf_scores_x),
        "knn_scores_x": np.mean(knn_scores_x),
        "knn_scores_emb": np.mean(knn_scores_emb),
        "rf_scores_x": np.mean(rf_scores_x),
        "rf_scores_emb": np.mean(rf_scores_emb),
    }
