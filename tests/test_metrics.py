import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from rfphate import (
    global_structure_preservation, local_structure_preservation,
    low_dimensional_group_separation, model_embedding_diff,
)
from rfphate.metrics.GSP import univariate_correlations, univariate_distance
from rfphate.metrics.ModelDiff import evaluate_fold


@pytest.fixture
def metric_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(36, 3))
    embedding = np.column_stack((x[:, 0] + 0.2 * x[:, 1], x[:, 1] - 0.1 * x[:, 2]))
    continuous = 4 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.1, size=len(x))
    return x, continuous, embedding


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("correlation", ["spearman", "pearson"])
def test_global_metric_matches_independent_calculation(metric_data, kind, correlation):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int) if kind == "classification" else continuous
    estimator = KNeighborsClassifier if kind == "classification" else KNeighborsRegressor
    importance = permutation_importance(
        estimator(n_neighbors=3).fit(x, y), x, y, n_repeats=3, random_state=17,
    ).importances
    correlate = spearmanr if correlation == "spearman" else pearsonr
    distances = pdist(embedding)
    feature_correlations = [
        correlate(pdist(x[:, [i]]), distances).statistic for i in range(x.shape[1])
    ]
    values = [correlate(importance[:, i], feature_correlations).statistic for i in range(3)]
    actual = global_structure_preservation(
        x, y, embedding, n_neighbors=3, n_repeats=3, random_state=17,
        prediction_type=kind, type=correlation,
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [np.mean(values), np.std(values)])


@pytest.mark.parametrize("metric", [global_structure_preservation, local_structure_preservation])
@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("container", [np.asarray, list, pd.Series])
def test_metric_target_inference(metric_data, metric, kind, container):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int) if kind == "classification" else continuous
    options = {"n_neighbors": 3, "n_repeats": 3, "random_state": 17}
    expected = metric(x, y, embedding, prediction_type=kind, **options)
    actual = metric(x, container(y), embedding, prediction_type=None, **options)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("metric", [global_structure_preservation, local_structure_preservation])
def test_metric_requires_targets_for_inference(metric):
    with pytest.raises(ValueError, match="prediction_type or y"):
        metric(np.zeros((3, 2)), None, np.zeros((3, 2)), prediction_type=None)


@pytest.mark.parametrize("correlation", ["spearman", "pearson"])
def test_univariate_distances_and_correlations(metric_data, correlation):
    x, _, embedding = metric_data
    np.testing.assert_allclose(univariate_distance(x[:, 0]), pdist(x[:, [0]]))
    correlate = spearmanr if correlation == "spearman" else pearsonr
    expected = [
        correlate(pdist(x[:, [i]]), pdist(embedding)).statistic for i in range(x.shape[1])
    ]
    np.testing.assert_allclose(univariate_correlations(x, embedding, correlation).ravel(), expected)


@pytest.mark.parametrize("kind", ["classification", "regression"])
@pytest.mark.parametrize("correlation", ["spearman", "pearson"])
def test_local_metric_matches_independent_calculation(metric_data, kind, correlation):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int) if kind == "classification" else continuous
    estimator = KNeighborsClassifier if kind == "classification" else KNeighborsRegressor
    importance = permutation_importance(
        estimator(n_neighbors=3).fit(x, y), x, y, n_repeats=3, random_state=17,
    ).importances
    embedding_importance = permutation_importance(
        KNeighborsRegressor(n_neighbors=3).fit(x, embedding), x, embedding,
        n_repeats=3, random_state=17, scoring="r2",
    ).importances
    correlate = spearmanr if correlation == "spearman" else pearsonr
    values = [
        correlate(importance[:, i], embedding_importance[:, i]).statistic for i in range(3)
    ]
    options = dict(n_neighbors=3, n_repeats=3, random_state=17, type=correlation,
                   prediction_type=kind)
    actual = local_structure_preservation(x, y, embedding, **options)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [np.mean(values), np.std(values)])
    multiple = local_structure_preservation(x, y, {"first": embedding, "second": embedding}, **options)
    np.testing.assert_allclose(multiple["first"], actual)
    np.testing.assert_allclose(multiple["second"], actual)


def test_local_metric_validation_scores(metric_data):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int)
    result = local_structure_preservation(
        x[:24], y[:24], embedding[:24], n_neighbors=3, n_repeats=3,
        random_state=17, keep_scores=True, x_val=x[24:], y_val=y[24:],
    )
    expected_score = KNeighborsClassifier(n_neighbors=3).fit(x[:24], y[:24]).score(x[24:], y[24:])
    expected_embedding_score = KNeighborsRegressor(n_neighbors=3).fit(
        x[:24], embedding[:24]
    ).score(x[:24], embedding[:24])
    np.testing.assert_allclose(result["model_score"], expected_score)
    np.testing.assert_allclose(result["emb_model_score"], expected_embedding_score)


def test_group_separation_matches_silhouette(metric_data):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int)
    actual = low_dimensional_group_separation(embedding, y, random_state=17, sample_size=24)
    expected = silhouette_score(embedding, y, random_state=17, sample_size=24)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("regression", [False, True])
def test_model_comparison_matches_explicit_cross_validation(metric_data, regression):
    x, continuous, embedding = metric_data
    y = continuous if regression else (continuous > 0).astype(int)
    knn_type = KNeighborsRegressor if regression else KNeighborsClassifier
    forest_type = RandomForestRegressor if regression else RandomForestClassifier
    scores = []
    for train, test in KFold(n_splits=3, shuffle=True, random_state=17).split(x):
        row = []
        for estimator in (knn_type(n_neighbors=3), forest_type(random_state=17)):
            for features in (x, embedding):
                estimator.fit(features[train], y[train])
                row.append(estimator.score(features[test], y[test]))
        scores.append(row)
    mean = np.mean(scores, axis=0)
    expected = {
        "knndiff": mean[1] - mean[0], "rfdiff": mean[3] - mean[2],
        "knn_scores_x": mean[0], "knn_scores_emb": mean[1],
        "rf_scores_x": mean[2], "rf_scores_emb": mean[3],
    }
    actual = model_embedding_diff(
        x, y, embedding, n_splits=3, n_repeats=2, random_state=17,
        model_kwargs={"n_neighbors": 3},
    )
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(list(actual.values()), list(expected.values()))


def test_fold_evaluation_does_not_fit_caller_estimators(metric_data):
    x, continuous, embedding = metric_data
    y = (continuous > 0).astype(int)
    knn = KNeighborsClassifier(n_neighbors=3)
    forest = RandomForestClassifier(n_estimators=10, random_state=17)
    scores = evaluate_fold(knn, forest, x, y, embedding, np.arange(24), np.arange(24, 36))
    assert np.isfinite(scores).all()
    assert not hasattr(knn, "classes_") and not hasattr(forest, "estimators_")
