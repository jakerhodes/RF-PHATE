# Imports
import numpy as np
from scipy.spatial.distance import pdist

#sklearn imports
from sklearn.inspection import permutation_importance
from scipy.stats import (spearmanr, pearsonr)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def global_structure_preservation(x, y, embedding, model = None,
                           n_repeats = 10, n_neighbors = 10, n_jobs = 1,
                           type = 'spearman', prediction_type = 'classification',
                           random_state = None,
                           **kwargs):
    
    """
    Global structure preservation provides a quantitative metric to assess the level at which a low-dimensional embedding retains global variable information relative to the supervised problem.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,)
        Target values (labels).

    embedding : array-like of shape (n_samples, n_components)
        Low-dimensional embedding.

    model : object, optional
        The estimator to use for predictions for permutation importance. If None,
        KNeighborsClassifier or KNeighborsRegressor is used based on prediction_type.

    n_repeats : int, optional
        Number of times to permute the data.

    n_neighbors : int, optional
        Number of neighbors to use for kNN.

    n_jobs : int, optional
        The number of jobs to run in parallel.

    type : {'spearman', 'pearson'}, optional
        Type of correlation to compute.

    prediction_type : {'classification', 'regression'}, optional
        Type of prediction task. If None, it is inferred based on the target values.

    random_state : int or None, optional
        Seed for reproducibility.

    **kwargs
        Additional parameters passed to the kNN model.

    Returns
    -------
    tuple
        Mean and standard deviation of the correlations between kNN importances and univariate correlations.
    """
    
    if prediction_type is None and y is None:
        raise ValueError("prediction_type or y must be provided.")
    

    if prediction_type is None and y is not None:
        if np.dtype(y) == 'float64' or np.dtype(y) == 'float32':
            prediction_type = 'regression'
        else:
            prediction_type = 'classification'


    if model is None:
        if prediction_type == 'classification':
            model = KNeighborsClassifier(n_neighbors = n_neighbors, n_jobs = n_jobs, **kwargs)

        else:
            model = KNeighborsRegressor(n_neighbors = n_neighbors, n_jobs = n_jobs, **kwargs)

    model.fit(x, y)

    importance = permutation_importance(model, x, y, n_repeats = n_repeats, n_jobs = n_jobs,
            random_state = random_state, **kwargs)
    
    univar_correls = univariate_correlations(x, embedding, type)

    correls = np.zeros((n_repeats,))

    if type == 'spearman':
        for repeat in range(n_repeats):
            correl = spearmanr(importance.importances[:, repeat], univar_correls).correlation
            correls[repeat] = correl

    elif type == 'pearson':
        for repeat in range(n_repeats):
            correl = pearsonr(importance.importances[:, repeat], univar_correls).correlation
            correls[repeat] = correl

    return (correls.mean(), correls.std()) 


def univariate_distance(x):
    """
    Calculate pairwise distances between points in a vector.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    ndarray
        Pairwise distances.
    """
    return pdist(np.expand_dims(x, 1))


def univariate_correlations(x, embedding, type):
    """
    Calculate univariate correlations between each feature in x and the embedding.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input data.

    embedding : array-like of shape (n_samples, n_components)
        Low-dimensional embedding.

    type : {'spearman', 'pearson'}
        Type of correlation to compute.

    Returns
    -------
    ndarray
        Univariate correlations for each variable in x.
    """
    _, d = np.shape(x)
    univar_correls = np.zeros((d, 1))
    embedding_dist = pdist(embedding)

    if type == 'pearson':
        for i in range(d):
            univar_correls[i] = pearsonr(univariate_distance(x[:, i]), embedding_dist).correlation

    else:
        for i in range(d):
            univar_correls[i] = spearmanr(univariate_distance(x[:, i]), embedding_dist).correlation   

    return univar_correls   
