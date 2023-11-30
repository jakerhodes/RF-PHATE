
from sklearn.metrics import silhouette_score

def low_dimensional_group_separation(embedding, y, **kwargs):
    """
    Calculates the silhouette score, a metric for assessing the separation between groups in low-dimensional embeddings.

    Parameters
    ----------
    embedding : array-like, shape (n_samples, n_features)
        The low-dimensional embedding of the data.

    y : array-like, shape (n_samples,)
        True labels for each sample in the dataset.

    **kwargs : additional keyword arguments
        Additional parameters to be passed to the `sklearn.metrics.silhouette_score` function.

    Returns
    -------
    score : float
        The silhouette score, ranging from -1 to 1, where a higher score indicates better separation between clusters or groups.
    """
    score = silhouette_score(embedding, y, **kwargs)
    return score

