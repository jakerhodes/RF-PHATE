
from sklearn.metrics import silhouette_score

def low_dimensional_group_separation(embedding, y, **kwargs):
    score = silhouette_score(embedding, y, **kwargs)

    return score

