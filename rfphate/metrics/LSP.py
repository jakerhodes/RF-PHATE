# Imports
import numpy as np

#sklearn imports
from sklearn.inspection import permutation_importance
from scipy.stats import (spearmanr, pearsonr)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# TODO: Add some sort of verbose messaging
def local_structure_preservation(x, y, embedding, model = None, emb_model = None,
                           n_repeats = 10, n_neighbors = 10, n_jobs = 1,
                           type = 'spearman', prediction_type = 'classification',
                           random_state = None, keep_scores = False,
                           x_val = None, y_val = None,
                           **kwargs):
    """
    This code produces the correlation between the feature importances in the data's classification / 
    regression problem and the feature importance in determining the embedding.
    
    Parameters
    ----------
    x : numpy array, default: None
        an (n, d) data matrix.
       
    y : numpy array, default: None
        a (n, 1) array of data labels.
     
    emb : numpy array, default: None
        an (n, p) embedding of the data

    model : an sklearn model, default: KNeighborsClassifier or KNeighbors Regressor
        to fit to the dataset

    emb_model : an sklearn model, default: KNeighborsRegressor 
        to fit to the embedding. 
        
    prediction_type : string, default:'classification'
        only needed if proximity matrix is not included.  Needed to determine classification or regression forest type.
        options are 'classification' or 'regression'
        
    type: string, default: 'pearson'
        designation of whether Spearman or Pearson correlatoin should be used
        Please choose 'spearman' or 'pearson'
        
    n_repeats : int, default: 30
        the number of repetitions used in the permutation importance
        
    random_state : int, optional, default: None
        random seed for generator used in methods with random initialzations (['mds', 'tsne', 'phate', 'umap', 'kpca',
        'nca', 'lle', 'phate', 'umap'])

    keep_scores : bool, optional, default: False
        If true, the model and embedding model validation scores will be saved in addition the correlations
        Returned values are in a dictionary. Requires x_val and y_val.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used
    
    Returns
    -------
    correlation : numeric
        the mean Spearman or Pearson correlation between the embedding feature importance and the prediction feature importance
        along with the standard deviation
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

    if emb_model is None:
        emb_model = KNeighborsRegressor(n_neighbors = n_neighbors, n_jobs = n_jobs, **kwargs) 


    model.fit(x, y)

    # Get importances using validation data, if supplied
    if keep_scores:
        importance = permutation_importance(model, x_val, y_val, n_repeats = n_repeats, n_jobs = n_jobs,
            random_state = random_state, **kwargs)

    else:
        importance = permutation_importance(model, x, y, n_repeats = n_repeats, n_jobs = n_jobs,
            random_state = random_state, **kwargs)


    # TODO: need to work with x_val, y_val, keep_scores in this instance.
    if isinstance(embedding, dict):
        correl_dict = dict()
        for key in embedding:
   
            emb_model.fit(x, embedding[key])
            emb_importance = permutation_importance(emb_model, x, embedding[key], n_repeats = n_repeats, n_jobs = n_jobs, **kwargs,
                                                    scoring = 'r2', random_state = random_state)

            correls = np.zeros((n_repeats,))

            if type == 'spearman':
                for col in range(n_repeats):
                    correl = spearmanr(importance.importances[:, col], emb_importance.importances[:, col]).correlation
                    correls[col] = correl

            elif type == 'pearson':
                for col in range(n_repeats):
                    correl = pearsonr(importance.importances[:, col], emb_importance.importances[:, col])[0]
                    correls[col] = correl

            correl_dict[key] = (correls.mean(), correls.std())

        return(correl_dict)
    
    else:

        emb_model.fit(x, embedding)
        emb_importance = permutation_importance(emb_model, x, embedding, n_repeats = n_repeats, n_jobs = n_jobs, **kwargs,
                                                scoring = 'r2', random_state = random_state)

        correls = np.zeros((n_repeats,))

        if type == 'spearman':
            for col in range(n_repeats):
                correl = spearmanr(importance.importances[:, col], emb_importance.importances[:, col]).correlation
                correls[col] = correl

        elif type == 'pearson':
            for col in range(n_repeats):
                correl = pearsonr(importance.importances[:, col], emb_importance.importances[:, col])[0]
                correls[col] = correl

        if keep_scores:
            score = model.score(x_val, y_val)
            emb_score = emb_model.score(x, embedding)

            return {'correls':  (correls.mean(), correls.std()), 'model_score': score, 'emb_model_score': emb_score}

        else:
            return (correls.mean(), correls.std()) 