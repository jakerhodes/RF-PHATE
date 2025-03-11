from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from rfphate.rfgap import RFGAP
from joblib import Parallel, delayed

def is_continuous(y: np.ndarray) -> bool:
    """Check if the target variable is continuous."""
    return pd.api.types.is_float_dtype(y)

def evaluate_fold(model, rf_model, x, y, embedding, train_idx, test_idx):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    emb_train, emb_test = embedding[train_idx], embedding[test_idx]

    # KNN on original data
    model.fit(x_train, y_train)
    knn_score_x = model.score(x_test, y_test)

    # KNN on embedding
    model.fit(emb_train, y_train)
    knn_score_emb = model.score(emb_test, y_test)

    # RFGAP on original data
    rf_model.fit(x_train, y_train)
    rf_score_x = rf_model.score(x_test, y_test)

    # RFGAP on embedding
    rf_model.fit(emb_train, y_train)
    rf_score_emb = rf_model.score(emb_test, y_test)

    return knn_score_x, knn_score_emb, rf_score_x, rf_score_emb

def model_embedding_diff(
    x: np.ndarray,
    y: np.ndarray,
    embedding: np.ndarray,
    model=None,
    n_splits: int = 10,
    n_repeats: int = 10,
    random_state: int = None,
    model_kwargs: dict = None,
    n_jobs: int = 1,
    **kwargs
) -> dict:
    """
    Evaluate KNN and RFGAP models on original and embedded data.
    
    Parameters:
    - x: Feature matrix.
    - y: Target variable.
    - embedding: Embedded feature representation.
    - model: KNN model (default: inferred based on y type).
    - n_splits: Number of folds in cross-validation.
    - n_repeats: Number of repetitions for cross-validation.
    - random_state: Random seed for reproducibility.
    - model_kwargs: Additional arguments for KNN model.
    - n_jobs: Number of jobs to run in parallel.
    
    Returns:
    - Dictionary containing performance differences and mean scores.
    """
    np.random.seed(random_state)
    
    # Choose model based on target type
    if model is None:
        model = KNeighborsRegressor if is_continuous(y) else KNeighborsClassifier
    model = model(**(model_kwargs or {}))
    
    rf_model = RFGAP(y=y, oob_score=True)
    
    results = Parallel(n_jobs = n_jobs)(delayed(evaluate_fold)(
        model, rf_model, x, y, embedding, train_idx, test_idx
    ) for _ in range(n_repeats) for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(x))
    
    knn_scores_x, knn_scores_emb, rf_scores_x, rf_scores_emb = zip(*results)
    
    return {
        "knndiff": np.mean(knn_scores_emb) - np.mean(knn_scores_x),
        "rfdiff": np.mean(rf_scores_emb) - np.mean(rf_scores_x),
        "knn_scores_x": np.mean(knn_scores_x),
        "knn_scores_emb": np.mean(knn_scores_emb),
        "rf_scores_x": np.mean(rf_scores_x),
        "rf_scores_emb": np.mean(rf_scores_emb),
    }
