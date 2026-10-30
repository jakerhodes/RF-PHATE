from contextlib import nullcontext
from unittest.mock import patch

import graphtools
import numpy as np
import pytest
from forestgeom import Proximity
from scipy import sparse
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from rfphate import PageRankPHATE, RFPHATE


@pytest.mark.parametrize("force_symmetric", [False, True])
@pytest.mark.parametrize("adjust_diagonal", [False, True])
def test_training_kernel_matches_forestgeom(
    classification_data, make_model, force_symmetric, adjust_diagonal
):
    x, y = classification_data
    model = make_model()
    warning = (
        nullcontext() if force_symmetric and adjust_diagonal
        else pytest.warns(RuntimeWarning, match="K should be symmetric|K should have a non-zero diagonal")
    )
    with warning:
        assert model.fit(x, y, force_symmetric, adjust_diagonal) is model
    expected = model.proximity_model_.training_proximity(
        force_symmetric=force_symmetric, adjust_diagonal=adjust_diagonal
    )
    np.testing.assert_allclose(model.phate_op_.X.toarray(), expected.toarray())
    if force_symmetric:
        np.testing.assert_allclose(expected.toarray(), expected.T.toarray(), atol=1e-7)
    else:
        assert (expected - expected.T).nnz > 0
    if adjust_diagonal:
        assert (expected.diagonal() > 0).all()
    else:
        np.testing.assert_array_equal(expected.diagonal(), 0)


def test_default_asymmetric_embedding(classification_data, make_model):
    x, y = classification_data
    model = make_model()
    with pytest.warns(RuntimeWarning, match="K should be symmetric"):
        embedding = model.fit_transform(x, y)
    assert embedding.shape == (len(x), 2)
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize("force_symmetric,kernel_symm,expected_symm", [
    (False, None, None), (True, None, "+"), (True, "*", "*"),
])
def test_self_similarity_uses_query_geometry(
    classification_data, make_model, force_symmetric, kernel_symm, expected_symm
):
    x, y = classification_data
    model = make_model(self_similarity=True, phate_params={"kernel_symm": kernel_symm})
    model.proximity_model_ = model._make_proximity_model().fit(x, y)
    kernel, symm = model._get_training_kernel(x, force_symmetric, True)
    expected = model.proximity_model_.transform(x)
    training = model.proximity_model_.training_proximity()
    np.testing.assert_allclose(kernel.toarray(), expected.toarray())
    assert not np.allclose(kernel.toarray(), training.toarray())
    assert symm == expected_symm


def test_parameter_precedence_preserves_inputs():
    forest = RandomForestClassifier(n_estimators=40, random_state=7, n_jobs=1)
    proximity_params = {"forest": ExtraTreesClassifier(), "weight_scheme": "uniform"}
    phate_params = {
        "knn_dist": "euclidean", "random_state": 99, "n_jobs": 3,
        "kernel_symm": "+", "n_components": 3, "beta": 0.8,
    }
    original_proximity = proximity_params.copy()
    original_phate = phate_params.copy()
    model = RFPHATE(
        forest=forest, random_state=17, n_jobs=2,
        proximity_params=proximity_params, phate_params=phate_params,
    )
    proximity = model._make_proximity_model()
    operator = model._make_phate_operator("*")
    assert proximity.forest is forest
    assert proximity.weight_scheme == "uniform"
    assert forest.random_state == 7 and forest.n_jobs == 1
    assert operator.knn_dist == "precomputed_affinity"
    assert operator.random_state == 17 and operator.n_jobs == 2
    assert operator.n_components == 3 and operator.beta == 0.8
    assert operator.kwargs["kernel_symm"] == "*"
    assert proximity_params == original_proximity
    assert phate_params == original_phate


def test_fitted_forest_is_not_refit(classification_data, make_model):
    x, y = classification_data
    forest = RandomForestClassifier(n_estimators=40, random_state=42).fit(x, y)
    trees = tuple(forest.estimators_)
    model = make_model(forest=forest)
    with patch.object(forest, "fit", side_effect=AssertionError("Forest was refitted")):
        model.fit(x, y, force_symmetric=True)
    assert model.proximity_model_.forest_.estimator is forest
    assert tuple(forest.estimators_) == trees


def test_refit_replaces_learned_state(classification_data, make_model):
    x, y = classification_data
    model = make_model()
    model.fit_transform(x, y, force_symmetric=True)
    old_proximity, old_phate = model.proximity_model_, model.phate_op_
    embedding = model.fit_transform(x[:50], y[:50], force_symmetric=True)
    assert model.proximity_model_ is not old_proximity
    assert model.phate_op_ is not old_phate
    assert model.proximity_model_.training_proximity().shape == (50, 50)
    assert embedding.shape == (50, 2)
    assert not hasattr(model.forest, "estimators_")


@pytest.mark.parametrize("failure_stage", ["forest", "phate"])
def test_failed_refit_does_not_expose_stale_graph(classification_data, make_model, failure_stage):
    x, y = classification_data
    model = make_model().fit(x, y, force_symmetric=True)
    target = Proximity if failure_stage == "forest" else PageRankPHATE
    with patch.object(target, "fit", side_effect=ValueError("fit failed")):
        with pytest.raises(ValueError, match="fit failed"):
            model.fit(x, y, force_symmetric=True)
    with pytest.raises(NotFittedError, match="RFPHATE instance is not fitted"):
        model.transform(x[:2])
    assert model.phate_op_ is None
    # A subsequent successful fit recovers normally.
    assert model.fit_transform(x, y, force_symmetric=True).shape == (len(x), 2)


@pytest.mark.parametrize("method", ["transform", "extend_to_data"])
def test_unfitted_calls_raise(method):
    with pytest.raises(NotFittedError):
        getattr(RFPHATE(), method)(np.zeros((2, 6)))


@pytest.mark.parametrize("n_landmark", [None, 10])
@pytest.mark.parametrize("regression", [False, True])
def test_held_out_interpolation(classification_data, make_model, n_landmark, regression):
    if regression:
        x, y = make_regression(n_samples=80, n_features=6, noise=0.1, random_state=42)
        forest = RandomForestRegressor(n_estimators=40, max_depth=4, random_state=42)
    else:
        x, y = classification_data
        forest = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.25, random_state=17)
    model = make_model(forest=forest, phate_params={
        "n_landmark": n_landmark, "t": 2, "n_svd": 5, "mds": "classic", "verbose": 0,
    })
    training_embedding = model.fit_transform(x_train, y_train, force_symmetric=True)
    np.testing.assert_allclose(model.transform(), training_embedding)
    transitions = model.extend_to_data(x_test)
    raw_kernel = model.proximity_model_.transform(x_test).toarray()
    if n_landmark is None:
        assert not isinstance(model.phate_op_.graph, graphtools.graphs.LandmarkGraph)
        expected = raw_kernel
    else:
        assert isinstance(model.phate_op_.graph, graphtools.graphs.LandmarkGraph)
        clusters = model.phate_op_.graph.clusters
        expected = np.column_stack([
            raw_kernel[:, clusters == cluster].sum(axis=1)
            for cluster in np.unique(clusters)
        ])
    expected /= expected.sum(axis=1, keepdims=True)
    assert sparse.issparse(transitions)
    assert transitions.shape == expected.shape
    assert (transitions.data >= 0).all()
    np.testing.assert_allclose(transitions.toarray(), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(transitions.sum(axis=1)).ravel(), 1, atol=1e-6)
    projected = model.transform(x_test)
    assert projected.shape == (len(x_test), 2)
    assert np.isfinite(projected).all()
    np.testing.assert_allclose(projected, transitions @ model.phate_op_.embedding)


def test_cluster_map_preserves_noncontiguous_cluster_order():
    clusters = np.array([9, 2, 9, 15, 2])
    mapping = RFPHATE()._build_cluster_map(clusters, len(clusters))
    assert sparse.issparse(mapping)
    np.testing.assert_array_equal(mapping.toarray(), [
        [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
    ])


@pytest.mark.parametrize("scheme", ["gap", "uniform"])
@pytest.mark.parametrize("self_similarity", [False, True])
def test_sparse_features(classification_data, make_model, scheme, self_similarity):
    x, y = classification_data
    sparse_x = sparse.csr_matrix(x)
    model = make_model(
        proximity_params={"weight_scheme": scheme}, self_similarity=self_similarity
    )
    embedding = model.fit_transform(sparse_x, y, force_symmetric=True)
    assert embedding.shape == (len(x), 2)
    assert np.isfinite(embedding).all()
    assert np.isfinite(model.transform(sparse_x[:3])).all()


def test_zero_proximity_rows_stay_zero(classification_data, make_model):
    x, y = classification_data
    model = make_model().fit(x, y, force_symmetric=True)
    kernel = sparse.csr_matrix(([2.0, 3.0], ([1, 1], [0, 1])), shape=(2, len(x)))
    with patch.object(model.proximity_model_, "transform", return_value=kernel):
        transitions = model.extend_to_data(x[:2])
        projected = model.transform(x[:2])
    assert sparse.issparse(transitions)
    assert np.isfinite(transitions.data).all() and np.isfinite(projected).all()
    np.testing.assert_array_equal(transitions.toarray()[0], 0)
    np.testing.assert_allclose(transitions.toarray()[1, :2], [0.4, 0.6])
    np.testing.assert_array_equal(projected[0], 0)


@pytest.mark.parametrize("forest,scheme", [
    (RandomForestClassifier(n_estimators=10, bootstrap=False), "gap"),
    (RandomForestClassifier(n_estimators=10, bootstrap=False), "oob"),
    (ExtraTreesClassifier(n_estimators=10, bootstrap=False), "gap"),
    (GradientBoostingClassifier(n_estimators=10), "gap"),
    (RandomForestClassifier(n_estimators=10), "boosted"),
    (RandomForestClassifier(n_estimators=10), "invalid"),
])
def test_invalid_schemes_preserve_upstream_errors(classification_data, forest, scheme):
    x, y = classification_data
    with pytest.raises((ValueError, TypeError, AttributeError)) as upstream:
        Proximity(forest=forest, weight_scheme=scheme).fit(x, y)
    with pytest.raises(type(upstream.value)) as wrapped:
        RFPHATE(forest=forest, proximity_params={"weight_scheme": scheme}).fit(x, y)
    assert str(wrapped.value) == str(upstream.value)


def test_obsolete_matrix_type_is_rejected():
    with pytest.raises(TypeError, match="matrix_type"):
        RFPHATE(proximity_params={"matrix_type": "dense"})._make_proximity_model()


def test_pagerank_operator_requires_fit():
    with pytest.raises(NotFittedError):
        _ = PageRankPHATE().diff_op
