from forestgeom import ForestProximity
from .pagerank_phate import PageRankPHATE

import numpy as np
from scipy import sparse
import graphtools

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


_FOREST_ESTIMATORS = {
    ("classification", "rf"): RandomForestClassifier,
    ("regression", "rf"): RandomForestRegressor,
    ("classification", "et"): ExtraTreesClassifier,
    ("regression", "et"): ExtraTreesRegressor,
    ("classification", "gbt"): GradientBoostingClassifier,
    ("regression", "gbt"): GradientBoostingRegressor,
}


_FOREST_MODELS_WITH_N_JOBS = {"rf", "et"}


class RFPHATE:
    """An RF-PHATE class which is used to fit a random forest, generate
    RF-proximities, and create RF-PHATE embeddings.

    Parameters
    ----------
    prediction_type : {'classification', 'regression'}
        Prediction type used to choose the underlying forest estimator.
        Default is 'classification'.

    model_type : str
        Base forest model to use for generating proximities.
        Options include 'rf' for Random Forests, 'et' for ExtraTrees, and 'gbt'
        for Gradient Boosted Trees (default is 'rf')

    random_state : int or None
        Random seed passed to both the forest estimator and PageRankPHATE.
        (default is None)

    n_jobs : int
        Number of jobs passed to PageRankPHATE and to forest estimators that
        support it. Random forests and ExtraTrees support this directly;
        GradientBoosting estimators do not. (default is 1)

    self_similarity : bool
        Only used if `proximity_params["weight_scheme"] == "gap"`. All points
        are passed down as if OOB. Increases similarity between an observation
        and itself as well as other points of the same class. NOTE: This
        partially disrupts the geometry learned by the RF-GAP proximities, but
        can be useful for exploring particularly noisy data. If True,
        ForestProximity.transform is employed on the training data rather than
        training_proximity.

    forest_params : dict or None
        Extra keyword arguments passed to the underlying scikit-learn ensemble
        constructor. Common keys include `n_estimators`, `max_depth`,
        `max_features`, and `verbose`. The top-level `random_state` and
        `n_jobs` arguments are passed through by RF-PHATE and take precedence.
        The supported keys depend on `prediction_type` and `model_type` and
        follow scikit-learn estimator APIs:
        RandomForestClassifier/Regressor, ExtraTreesClassifier/Regressor, and
        GradientBoostingClassifier/Regressor.

        Reference:
        https://scikit-learn.org/stable/modules/ensemble.html

    proximity_params : dict or None
        Extra keyword arguments passed to ForestProximity. Common keys include
        `weight_scheme` (`'gap'` by default), `matrix_type`, and OOB/proximity
        controls supported by forestgeom. The `forest` key is controlled by
        RFPHATE and always overrides values supplied in this dictionary.

        `force_symmetric` and `adjust_diagonal` are training-kernel options
        passed to fit or fit_transform, not ForestProximity constructor
        options.

        Reference:
        https://github.com/JakeSRhodesLab/forestgeom

    phate_params : dict or None
        Extra keyword arguments passed to PageRankPHATE. Parameters controlled
        through this dictionary include PHATE and PageRankPHATE options such
        as `n_components`, `t`, `n_landmark`, `verbose`, `kernel_symm`, `mds`,
        `mds_solver`, `gamma`, and `beta`. The top-level `random_state` and
        `n_jobs` arguments are passed through by RF-PHATE and take precedence.

        RF-PHATE always enforces `knn_dist='precomputed_affinity'` and may
        override `kernel_symm` during fitting when `force_symmetric=True` and
        a non-symmetric training kernel is constructed.

        References:
        https://github.com/KrishnaswamyLab/PHATE
        https://github.com/KrishnaswamyLab/graphtools
    """

    def __init__(
        self,
        prediction_type="classification",
        model_type="rf",
        random_state=None,
        n_jobs=1,
        self_similarity=False,
        forest_params=None,
        proximity_params=None,
        phate_params=None,
    ):
        # Forest-proximity parameters
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.random_state = random_state
        self.n_jobs = n_jobs

        # RF-PHATE-specific parameter
        self.self_similarity = self_similarity

        # Explicit kwargs routing
        self.forest_params = dict(forest_params or {})
        self.proximity_params = {
            "weight_scheme": "gap",
            **dict(proximity_params or {}),
        }
        self.phate_params = {
            "kernel_symm": None,
            **dict(phate_params or {}),
        }

        # Learned objects
        self.proximity_model_ = None
        self.phate_op_ = None

    def _check_is_fitted(self):
        if self.proximity_model_ is None or self.phate_op_ is None:
            raise NotFittedError(
                "This RFPHATE instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' first."
            )

    def _make_forest(self):
        """Instantiate the configured scikit-learn forest estimator."""
        key = (self.prediction_type, self.model_type)
        if key not in _FOREST_ESTIMATORS:
            raise ValueError(
                "Invalid combination of prediction_type and model_type. "
                "Use prediction_type in {'classification', 'regression'} and "
                "model_type in {'rf', 'et', 'gbt'}."
            )

        Estimator = _FOREST_ESTIMATORS[key]
        forest_params = {
            **self.forest_params,
            "random_state": self.random_state,
        }
        if self.model_type in _FOREST_MODELS_WITH_N_JOBS:
            forest_params["n_jobs"] = self.n_jobs
        else:
            forest_params.pop("n_jobs", None)

        return Estimator(**forest_params)

    def _make_proximity_model(self):
        """Instantiate the underlying ForestProximity model around a forest.

        This builds a base ensemble estimator according to `model_type` and
        `prediction_type`, then wraps it with `ForestProximity` using the
        selected proximity weight scheme.
        """
        proximity_params = {
            **self.proximity_params,
            "forest": self._make_forest(),
        }

        return ForestProximity(**proximity_params)

    def _make_phate_operator(self, kernel_symm):
        """Instantiate the PageRankPHATE operator."""
        phate_params = {
            **self.phate_params,
            "knn_dist": "precomputed_affinity",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "kernel_symm": kernel_symm,
        }

        return PageRankPHATE(**phate_params)

    def _build_cluster_map(self, clusters, n_train_points):
        """Build sparse map from training points to active landmark ids."""
        unique_clusters, remapped_clusters = np.unique(
            clusters, return_inverse=True
        )
        n_landmarks = len(unique_clusters)
        row_idx = np.arange(n_train_points)
        col_idx = remapped_clusters
        data_ones = np.ones(n_train_points)
        return sparse.csc_matrix(
            (data_ones, (row_idx, col_idx)),
            shape=(n_train_points, n_landmarks),
        )

    def _get_training_kernel(self, x, force_symmetric, adjust_diagonal):
        """Build the forest-based kernel matrix used by PHATE."""
        if self.self_similarity:
            # Treat training points as out-of-sample (preserves self-sim behavior)
            kernel = self.proximity_model_.transform(x)
            kernel_symm = self.phate_params.get("kernel_symm")
            if force_symmetric and kernel_symm is None:
                # If force_symmetric but kernel_symm is None, force
                # symmetrization through PHATE because transform may not
                # be symmetric
                kernel_symm = "+"
        else:
            # Use the trained training-proximity matrix
            kernel = self.proximity_model_.training_proximity(
                force_symmetric=force_symmetric,
                adjust_diagonal=adjust_diagonal,
            )
            kernel_symm = self.phate_params.get("kernel_symm")

        return kernel, kernel_symm

    def _fit_phate(self, x, y, force_symmetric, adjust_diagonal):
        # ---------------------------------------------------------
        # STEP 1: fit forest proximity model
        # ---------------------------------------------------------
        self.proximity_model_ = self._make_proximity_model()
        self.proximity_model_.fit(x, y)

        # ---------------------------------------------------------
        # STEP 2: build training kernel for PHATE
        # ---------------------------------------------------------
        kernel, kernel_symm = self._get_training_kernel(
            x,
            force_symmetric=force_symmetric,
            adjust_diagonal=adjust_diagonal,
        )

        # ---------------------------------------------------------
        # STEP 3: fit PHATE on the kernel
        # ---------------------------------------------------------
        self.phate_op_ = self._make_phate_operator(kernel_symm)
        self.phate_op_.fit(kernel)
        return self

    def fit(self, x, y, force_symmetric=False, adjust_diagonal=True):
        """Fit the forest proximity model and the PHATE operator.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, the underlying estimators
            may convert the dtype as needed.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers
            in regression).

        force_symmetric : bool
            Force symmetry of the training proximity kernel. (default is False)

        adjust_diagonal : bool
            Whether to force the diagonal of the training proximity kernel to
            be nonzero. (default is True)

        Returns
        -------
        self : RFPHATE
            Fitted estimator.
        """
        self._fit_phate(
            x,
            y,
            force_symmetric=force_symmetric,
            adjust_diagonal=adjust_diagonal,
        )

        return self

    def fit_transform(self, x, y, force_symmetric=False, adjust_diagonal=True):
        """Fit RF-PHATE and return the embedding.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers
            in regression).

        force_symmetric : bool
            Force symmetry of the training proximity kernel. (default is False)

        adjust_diagonal : bool
            Whether to force the diagonal of the training proximity kernel to
            be nonzero. (default is True)

        Returns
        -------
        array-like of shape (n_samples, n_components)
            A lower-dimensional representation of the data following the
            RF-PHATE algorithm.
        """
        self.fit(
            x,
            y,
            force_symmetric=force_symmetric,
            adjust_diagonal=adjust_diagonal,
        )
        return self.transform()

    def extend_to_data(self, X):
        """Build transition matrix from new data to the training graph
        (full or landmark).

        Creates a transition matrix such that `X` can be approximated by
        a linear combination of landmarks. Any transformation of the
        landmarks can be trivially applied to `X` by performing

            transform_data = transitions.dot(transform)

        Parameters
        ----------
        X : array-like, shape (n_samples_new, n_features)
            New data for which a kernel block is calculated to the training
            data. `n_features` must match the ambient dimension of the fitted
            model.

        Returns
        -------
        transitions : array-like, shape (n_samples_new, n_train_samples)
            Transition matrix from `X` to the training graph, or to the
            active landmarks in landmark PHATE mode.
        """
        self._check_is_fitted()

        kernel = self.proximity_model_.transform(X)

        if isinstance(self.phate_op_.graph, graphtools.graphs.LandmarkGraph):
            clusters = self.phate_op_.graph.clusters
            n_train_points = kernel.shape[1]

            cluster_map = self._build_cluster_map(clusters, n_train_points)
            pnm = kernel @ cluster_map
            pnm = normalize(pnm, norm="l1", axis=1)
        else:
            pnm = normalize(kernel, norm="l1", axis=1)

        return pnm

    # NOTE: the output of fit(x, y) followed by transform(x) is NOT equivalent
    # to fit_transform(x, y) because transform() uses proximity extension
    # to build extended proximity blocks (even on training points)
    def transform(self, X=None):
        """Project data into the fitted RF-PHATE embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples_new, n_features), optional
            New data to project into the fitted RF-PHATE embedding.
            If None, return the training embedding.

        Returns
        -------
        array-like, shape (n_samples_new, n_components) or
        shape (n_samples, n_components)
            Embedded coordinates for new data, or training coordinates
            when X is None.
        """
        self._check_is_fitted()

        if X is None:
            return self.phate_op_.transform()

        self.phate_op_.transform()
        pnm = self.extend_to_data(X)
        return self.phate_op_.graph.interpolate(self.phate_op_.embedding, pnm)
