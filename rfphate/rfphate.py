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

    n_pca : int, optional
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time (default is 100)

    mds : string, optional
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction
        (default is 'metric')

    mds_solver : {'sgd', 'smacof'}
        which solver to use for metric MDS. SGD is substantially faster
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted
        (default is 'euclidean')

    random_state : integer
        random seed state set for RF and MDS

    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

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

    forest_kwargs : dict or None
        Extra keyword arguments passed only to ForestProximity / the underlying
        ensemble constructor.

    phate_kwargs : dict or None
        Extra keyword arguments passed only to PageRankPHATE.
    """

    def __init__(
        self,
        prediction_type="classification",
        n_components=2,
        kernel_method="gap",
        model_type="rf",
        n_landmark=2000,
        t="auto",
        n_pca=100,
        mds_solver="sgd",
        mds_dist="euclidean",
        mds="metric",
        random_state=None,
        verbose=0,
        adjust_diagonal=True,
        force_symmetric=False,
        kernel_symm=None,
        beta=0.9,
        self_similarity=False,
        forest_kwargs=None,
        phate_kwargs=None,
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
        self.n_pca = n_pca
        self.mds = mds
        self.knn_dist = "precomputed_affinity"
        self.kernel_symm = kernel_symm
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.verbose = verbose
        self.beta = beta

        # RF-PHATE-specific parameter
        self.self_similarity = self_similarity

        # Explicit kwargs routing
        self.forest_kwargs = {} if forest_kwargs is None else dict(forest_kwargs)
        self.phate_kwargs = {} if phate_kwargs is None else dict(phate_kwargs)

        # Learned objects
        self.proximity_model_ = None
        self.phate_op_ = None

    def _check_is_fitted(self):
        if self.proximity_model_ is None or self.phate_op_ is None:
            raise NotFittedError(
                "This RFPHATE instance is not fitted yet. "
                "Call 'fit' or 'fit_transform' first."
            )
        if getattr(self.phate_op_, "embedding", None) is None:
            raise NotFittedError(
                "The PHATE operator is missing its fitted embedding. "
                "Call 'fit' or 'fit_transform' first."
            )

    def _make_proximity_model(self):
        """Instantiate the underlying ForestProximity model around a forest.

        This builds a base ensemble estimator according to `model_type` and
        `prediction_type`, then wraps it with `ForestProximity` using the
        selected proximity weight scheme.
        """
        if self.prediction_type not in ("classification", "regression"):
            raise ValueError(
                f"Invalid prediction_type '{self.prediction_type}'. "
                "Choose 'classification' or 'regression'."
            )

        if self.model_type == "rf":
            Est = (
                RandomForestClassifier
                if self.prediction_type == "classification"
                else RandomForestRegressor
            )
        elif self.model_type == "et":
            Est = (
                ExtraTreesClassifier
                if self.prediction_type == "classification"
                else ExtraTreesRegressor
            )
        elif self.model_type == "gbt":
            Est = (
                GradientBoostingClassifier
                if self.prediction_type == "classification"
                else GradientBoostingRegressor
            )
        else:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                "Choose from 'rf', 'et', or 'gbt'."
            )

        # Build forest estimator with provided kwargs
        forest = Est(random_state=self.random_state, **self.forest_kwargs)

        return ForestProximity(forest=forest, weight_scheme=self.kernel_method)

    def _make_phate_operator(self, kernel_symm):
        """Instantiate the PageRankPHATE operator."""
        return PageRankPHATE(
            n_components=self.n_components,
            t=self.t,
            n_landmark=self.n_landmark,
            kernel_symm=kernel_symm,
            mds=self.mds,
            n_pca=self.n_pca,
            knn_dist=self.knn_dist,
            mds_dist=self.mds_dist,
            mds_solver=self.mds_solver,
            random_state=self.random_state,
            verbose=self.verbose,
            beta=self.beta,
            **self.phate_kwargs,
        )

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
        return self.phate_op_.fit_transform(kernel)

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
        return self._fit_phate(x, y)

    def extend_to_data(self, data):
        """Build transition matrix from new data to the training graph
        (full or landmark).

        Creates a transition matrix such that `data` can be approximated by
        a linear combination of landmarks. Any transformation of the
        landmarks can be trivially applied to `data` by performing

            transform_data = transitions.dot(transform)

        Parameters
        ----------
        data : array-like, shape (n_samples_new, n_features)
            New data for which a kernel block is calculated to the training
            data. `n_features` must match the ambient dimension of the fitted
            model.

        Returns
        -------
        transitions : array-like, shape (n_samples_new, n_train_samples)
            Transition matrix from `data` to the training graph, or to the
            active landmarks in landmark PHATE mode.
        """
        self._check_is_fitted()

        kernel = self.proximity_model_.transform(data)

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
    def transform(self, data):
        """Basic linear kernel extension for new points in the embedding space.

        Parameters
        ----------
        data : array-like, shape (n_samples_new, n_features)
            New data to project into the fitted RF-PHATE embedding.

        Returns
        -------
        array-like, shape (n_samples_new, n_components)
            Embedded coordinates for the new data.
        """
        self._check_is_fitted()
        pnm = self.extend_to_data(data)
        return self.phate_op_.graph.interpolate(self.phate_op_.embedding, pnm)
