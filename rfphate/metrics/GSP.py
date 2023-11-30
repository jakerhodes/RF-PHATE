# Imports
import numpy as np

#sklearn imports
from sklearn.inspection import permutation_importance
from scipy.stats import (spearmanr, pearsonr)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score as cv

from scipy.spatial.distance import pdist


def global_structure_preservation(x, y, embedding, model = None,
                           n_repeats = 10, n_neighbors = 10, n_jobs = 1,
                           type = 'spearman', prediction_type = 'classification',
                           random_state = None,
                           **kwargs):
    
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
    
    # Original Implementation
    univar_correls = univariate_correlations(x, embedding, type)

    # Try: Univaritate kNN scores for embedding creation, correlated with full kNN importances
    correls = np.zeros((n_repeats,))

    if type == 'spearman':
        # TODO: replace loop
        for repeat in range(n_repeats):
            correl = spearmanr(importance.importances[:, repeat], univar_correls).correlation
            correls[repeat] = correl

    elif type == 'pearson':
        # TODO: replace loop
        for repeat in range(n_repeats):
            correl = pearsonr(importance.importances[:, repeat], univar_correls).correlation
            correls[repeat] = correl

    return (correls.mean(), correls.std()) 


def global_structure_knn(x, y, embedding, model = None,
                           n_repeats = 10, n_neighbors = 10, n_jobs = 1,
                           type = 'spearman', prediction_type = 'classification',
                           random_state = None,
                           **kwargs):
    
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
    
    univar_scores = univar_knn_scores(x, embedding)

    # Try: Univaritate kNN scores for embedding creation, correlated with full kNN importances
    correls = np.zeros((n_repeats,))

    if type == 'spearman':
        # TODO: replace loop
        for repeat in range(n_repeats):
            correl = spearmanr(importance.importances[:, repeat], univar_scores).correlation

            correls[repeat] = correl

    elif type == 'pearson':
        # TODO: replace loop
        for repeat in range(n_repeats):
            correl = pearsonr(importance.importances[:, repeat], univar_scores).correlation
            correls[repeat] = correl

    return (correls.mean(), correls.std()) 



def global_structure_rank_weighted(x, y, embedding, model = None,
                           n_repeats = 10, n_neighbors = 10, n_jobs = 1,
                           type = 'spearman', prediction_type = 'classification',
                           random_state = None,
                           **kwargs):
    
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
    avgs = np.zeros((n_repeats,))


    if type == 'spearman':
        for repeat in range(n_repeats):

            importances = importance.importances[:, repeat]           
            rank_weights = create_rank_weights_from_importances(importances)
            avgs[repeat] = compute_weighted_average(univar_correls, rank_weights)

    elif type == 'pearson':
        for repeat in range(n_repeats):

            importances = importance.importances[:, repeat]
            rank_weights = create_rank_weights_from_importances(importances)
            avgs[repeat] = avgs[repeat] = compute_weighted_average(univar_correls, rank_weights)
    
    return (avgs.mean(), avgs.std()) 







# Pairwise distances between points in a vector
def univariate_distance(x):
    return pdist(np.expand_dims(x, 1))


def univariate_correlations(x, embedding, type):
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




def univar_knn_scores(x, y):

    _, d = np.shape(x)
    univar_scores = np.zeros((d, 1))

    model = KNeighborsRegressor(n_neighbors = 10, n_jobs = -5)

    for i in range(d):
        scores = cv(model, x[:, i].reshape(-1, 1), y, cv = 5)
        univar_scores[i] = scores.mean()

    return univar_scores


# Updates from Adrien
def create_rank_weights_from_importances(importances):
    # Ensure the list of importances is not empty
    # if not importances:
    #     raise ValueError("The list of importances cannot be empty.")

    # Sort the importances in ascending order and get the ranks
    sorted_importances = sorted(importances, reverse = True) # Original implementation is reverse = False
    ranks = [sorted_importances.index(imp) + 1 for imp in importances]

    # Assign weights based on the rank
    rank_weights = [1 / rank for rank in ranks]

    # Normalize the rank weights to make them sum to 1
    sum_rank_weights = sum(rank_weights)
    normalized_weights = [w / sum_rank_weights for w in rank_weights]

    return normalized_weights

def compute_weighted_average(numbers, weights):
    # Check if the input lists have the same length
    # if len(numbers) != len(weights):
    #     raise ValueError("The number of numbers must be equal to the number of weights.")

    return sum(number * weight for number, weight in zip(numbers, weights)) / sum(weights)
