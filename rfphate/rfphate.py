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

    n_components : int
        The number of dimensions for the RF-PHATE embedding

    kernel_method : str
        The type of kernel to be constructed. Options are 'uniform', 'oob',
        and 'gap' (default is 'gap', highly recommended)

    model_type : str
        Base forest model to use for generating proximities.
        Options include 'rf' for Random Forests, 'et' for ExtraTrees, and 'gbt'
        for Gradient Boosted Trees (default is 'rf')

    n_landmark : int, optional
        number of landmarks to use in fast PHATE (default is 2000)

    t : int, optional
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator (default is 'auto')

    random_state : integer
        random seed state set for RF and MDS

    n_jobs : int
        Number of jobs to use where supported by the underlying forest and
        PHATE operators. Random forests and ExtraTrees support this directly;
        GradientBoosting estimators do not. (default is 1)

    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 1)

    adjust_diagonal : bool
        Whether to force the diagonal of the kernel matrix to be nonzero.
        (default is True)

    force_symmetric : bool
        Force symmetry of the kernel matrix. (default is False)

    kernel_symm : str, optional
        Selects which kernel symmetrization method is used for building the
        underlying PHATE diffusion operator.
        Note: Using force_symmetric is generally preferred over internal
        kernel_symm for memory and runtime savings. (default is None)

    beta : float
        The damping factor for the PageRank algorithm. The range is (0, 1).
        Values closer to 0 add more to the uniform teleporting probability.
        If 1, teleporting is not used.

    self_similarity : bool
        Only used if kernel_method == 'gap'. All points are passed down as if
        OOB. Increases similarity between an observation and itself as well as
        other points of the same class. NOTE: This partially disrupts the
        geometry learned by the RF-GAP proximities, but can be useful for
        exploring particularly noisy data. If True, ForestProximity.transform
        is employed on the training data rather than training_proximity.

    forest_params : dict or None
        Extra keyword arguments passed to the underlying scikit-learn ensemble
        constructor. The random_state and verbose arguments are controlled by
        RFPHATE and take precedence. The n_jobs argument is also controlled by
        RFPHATE for estimators that support it.

    proximity_params : dict or None
        Extra keyword arguments passed to ForestProximity. The forest and
        weight_scheme arguments are controlled by RFPHATE and take precedence.

    phate_params : dict or None
        Extra keyword arguments passed to PageRankPHATE. Parameters controlled
        directly by RFPHATE, such as n_components, t, n_landmark, knn_dist,
        n_jobs, random_state, verbose, beta, and kernel_symm, take precedence.
    """

    def __init__(
        self,
        prediction_type="classification",
        model_type="rf",
        kernel_method="gap",
        n_components=2,
        t="auto",
        n_landmark=2000,
        beta=0.9,
        n_jobs=1,
        random_state=None,
        verbose=1,
        self_similarity=False,
        force_symmetric=False,
        adjust_diagonal=True,
        kernel_symm=None,

        # Explicit extra kwargs routing for underlying objects
        forest_params=None,
        proximity_params=None,
        phate_params=None,
    ):
        # Forest-proximity parameters
        self.prediction_type = prediction_type
        self.kernel_method = kernel_method
        self.model_type = model_type
        self.adjust_diagonal = adjust_diagonal
        self.force_symmetric = force_symmetric

        # PHATE parameters
        self.n_components = n_components
        self.t = t
        self.n_landmark = n_landmark
        self.kernel_symm = kernel_symm
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.beta = beta

        # RF-PHATE-specific parameter
        self.self_similarity = self_similarity

        # Explicit kwargs routing
        self.forest_params = dict(forest_params or {})
        self.proximity_params = dict(proximity_params or {})
        self.phate_params = dict(phate_params or {})

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
            "verbose": self.verbose,
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
            "weight_scheme": self.kernel_method,
        }

        return ForestProximity(**proximity_params)

    def _make_phate_operator(self, kernel_symm):
        """Instantiate the PageRankPHATE operator."""
        phate_params = {
            **self.phate_params,
            "n_components": self.n_components,
            "t": self.t,
            "n_landmark": self.n_landmark,
            "knn_dist": "precomputed_affinity",
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "beta": self.beta,
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

    def _get_training_kernel(self, x):
        """Build the kernel matrix used by PHATE.

        This preserves the exact behavior of the previous implementation.
        """
        if self.self_similarity:
            # Treat training points as out-of-sample (preserves self-sim behavior)
            kernel = self.proximity_model_.transform(x)
            kernel_symm = self.kernel_symm
            if self.force_symmetric and kernel_symm is None:
                # If force_symmetric but kernel_symm is None, force
                # symmetrization through PHATE because transform may not
                # be symmetric
                kernel_symm = "+"
        else:
            # Use the trained training-proximity matrix
            kernel = self.proximity_model_.training_proximity(
                force_symmetric=self.force_symmetric,
                adjust_diagonal=self.adjust_diagonal,
            )
            kernel_symm = self.kernel_symm

        return kernel, kernel_symm

    def _fit_phate(self, x, y):
        # ---------------------------------------------------------
        # STEP 1: fit forest proximity model
        # ---------------------------------------------------------
        self.proximity_model_ = self._make_proximity_model()
        self.proximity_model_.fit(x, y)

        # ---------------------------------------------------------
        # STEP 2: build training kernel for PHATE
        # ---------------------------------------------------------
        kernel, kernel_symm = self._get_training_kernel(x)

        # ---------------------------------------------------------
        # STEP 3: fit PHATE on the kernel
        # ---------------------------------------------------------
        self.phate_op_ = self._make_phate_operator(kernel_symm)
        self.phate_op_.fit(kernel)
        return self

    def fit(self, x, y):
        """Fit the forest proximity model and the PHATE operator.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, the underlying estimators
            may convert the dtype as needed.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers
            in regression).

        Returns
        -------
        self : RFPHATE
            Fitted estimator.
        """
        self._fit_phate(x, y)

        return self

    def fit_transform(self, x, y):
        """Fit RF-PHATE and return the embedding.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers
            in regression).

        Returns
        -------
        array-like of shape (n_samples, n_components)
            A lower-dimensional representation of the data following the
            RF-PHATE algorithm.
        """
        self.fit(x, y)
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
        if getattr(self.phate_op_, "embedding", None) is None:
            self.phate_op_.transform()
        embedding = self.phate_op_.embedding

        if X is None:
            return embedding

        pnm = self.extend_to_data(X)
        return self.phate_op_.graph.interpolate(embedding, pnm)
