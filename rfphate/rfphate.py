from forestkernel import ForestKernel

# For PHATE part
from phate import PHATE
import numpy as np
from scipy import sparse

import graphtools
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize


class PageRankPHATE(PHATE):
    """
    PageRankPHATE is an adaptation of PHATE which incorporates random jumps into
    the diffusion operator. This improvement is based on Google's PageRank
    algorithm and makes the PHATE algorithm more robust to parameter selection.
    """

    def __init__(self, beta=0.9, **kwargs):
        super(PageRankPHATE, self).__init__(**kwargs)
        self.beta = beta

    @property
    def diff_op(self):
        """diff_op : array-like, shape=[n_samples, n_samples]
        or [n_landmark, n_landmark]

        The diffusion operator built from the graph
        """
        if self.graph is None:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )

        if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
            diff_op = self.graph.landmark_op
        else:
            diff_op = self.graph.diff_op

        if sparse.issparse(diff_op):
            diff_op = diff_op.toarray()

        dim = diff_op.shape[0]
        diff_op_tele = (
            self.beta * diff_op
            + (1 - self.beta) * (1 / dim) * np.ones((dim, dim))
        )
        return diff_op_tele


def RFPHATE(
    prediction_type=None,
    y=None,
    n_components=2,
    kernel_method="gap",
    model_type="rf",
    matrix_type="sparse",
    n_landmark=2000,
    t="auto",
    n_pca=100,
    mds_solver="sgd",
    mds_dist="euclidean",
    mds="metric",
    n_jobs=1,
    random_state=None,
    verbose=0,
    force_nonzero_diag=True,
    force_symmetric=True,
    kernel_symm=None,
    beta=0.9,
    self_similarity=False,
    **kwargs
):
    """An RF-PHATE class which is used to fit a random forest, generate
    RF-proximities, and create RF-PHATE embeddings.

    Parameters
    ----------
    n_components : int
        The number of dimensions for the RF-PHATE embedding

    kernel_method : str
        The type of kernel to be constructed. Options are 'original', 'oob',
        and 'gap' (default is 'gap', highly recommended)

    matrix_type : str
        Whether the kernel type should be 'sparse' or 'dense'
        (default is sparse)

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

    n_jobs : integer, optional
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used (default is 1)

    random_state : integer
        random seed state set for RF and MDS

    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

    force_nonzero_diag : bool
        Whether to force the diagonal of the kernel matrix to be nonzero.
        (default is True)

    force_symmetric : str or None
        Enforce symmetry of proximities. (default is True)

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
        exploring particularly noisy data. If True, self.kernel_extend is
        employed to the training data rather than self.get_kernel.
    """

    if prediction_type is None and y is None:
        prediction_type = "classification"

    forest = ForestKernel(
        prediction_type=prediction_type,
        y=y,
        kernel_method=kernel_method,
        model_type=model_type,
        **kwargs,
    )

    class RFPHATE(forest.__class__, PageRankPHATE):
        def __init__(
            self,
            n_components=n_components,
            kernel_method=kernel_method,
            matrix_type=matrix_type,
            n_landmark=n_landmark,
            t=t,
            n_pca=n_pca,
            mds_solver=mds_solver,
            mds_dist=mds_dist,
            mds=mds,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            force_nonzero_diag=force_nonzero_diag,
            force_symmetric=force_symmetric,
            kernel_symm=kernel_symm,
            beta=beta,
            self_similarity=self_similarity,
            **kwargs
        ):
            super(RFPHATE, self).__init__(**kwargs)

            self.n_components = n_components
            self.t = t
            self.n_landmark = n_landmark
            self.mds = mds
            self.n_pca = n_pca
            self.knn_dist = "precomputed_affinity"
            self.kernel_symm = kernel_symm
            self.mds_dist = mds_dist
            self.mds_solver = mds_solver
            self.random_state = random_state
            self.n_jobs = n_jobs

            self.graph = None
            self._diff_potential = None
            self.embedding = None
            self.x = None
            self.optimal_t = None
            self.kernel_method = kernel_method
            self.matrix_type = matrix_type
            self.verbose = verbose
            self.force_nonzero_diag = force_nonzero_diag
            self.force_symmetric = force_symmetric
            self.beta = beta
            self.self_similarity = self_similarity

            # From https://www.geeksforgeeks.org/class-factories-a-powerful-pattern-in-python/
            for k, v in kwargs.items():
                setattr(self, k, v)

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

        def _get_training_proximity(self, x):
            """Build the kernel matrix used by PHATE.

            This preserves the exact behavior of the original implementation.
            """
            if self.self_similarity:
                kernel = self.kernel_extend(x)
                if self.force_symmetric and self.kernel_symm is None:
                    self.kernel_symm = "+"  # If force_symmetric but kernel_symm is None, force symmetrize through PHATE because kernel_extend may not be symmetric
            else:
                kernel = self.get_kernel()

            return kernel

        def _make_phate_operator(self):
            """Instantiate the PageRankPHATE operator."""
            return PageRankPHATE(
                n_components=self.n_components,
                t=self.t,
                n_landmark=self.n_landmark,
                kernel_symm=self.kernel_symm,
                mds=self.mds,
                n_pca=self.n_pca,
                knn_dist=self.knn_dist,
                mds_dist=self.mds_dist,
                mds_solver=self.mds_solver,
                random_state=self.random_state,
                verbose=self.verbose,
                beta=self.beta,
            )

        def extend_to_data(self, data):
            """Build transition matrix from new data to the training graph
            (Full or Landmark)

            Creates a transition matrix such that `Y` can be approximated by
            a linear combination of landmarks. Any transformation of the
            landmarks can be trivially applied to `Y` by performing

            `transform_Y = transitions.dot(transform)`

            Parameters
            ----------
            Y : array-like, [n_samples_y, n_features]
                new data for which an affinity matrix is calculated
                to the existing data. `n_features` must match the ambient
                dimensions

            Returns
            -------
            transitions : array-like, [n_samples_y, self.data.shape[0]]
                Transition matrix from `Y` to `self.data`
            """
            kernel = self.kernel_extend(data)

            if isinstance(self.phate_op.graph, graphtools.graphs.LandmarkGraph):
                clusters = self.phate_op.graph.clusters
                n_train_points = kernel.shape[1]

                # Remap cluster IDs to be contiguous (0 to N_active-1)
                # unique_clusters: the sorted unique IDs
                # remapped_clusters: the indices from 0 to len(unique)-1
                cluster_map = self._build_cluster_map(clusters, n_train_points)

                pnm = kernel @ cluster_map
                pnm = normalize(pnm, norm="l1", axis=1)
            else:
                pnm = normalize(kernel, norm="l1", axis=1)

            return pnm

        # NOTE: the output of fit(x,y) followed by transform(x) is NOT equivalent
        # to fit_transform(x,y) because transform() uses kernel_extend to build
        # extended proximities (even on training points)
        def transform(self, data):
            """Basic linear kernel extension for new points in the embedding space"""
            check_is_fitted(self)
            pnm = self.extend_to_data(data)
            return self.phate_op.graph.interpolate(self.phate_op.embedding, pnm)

        def _fit_transform(self, x, y):
            """Internal method for fitting and transforming the data

            Parameters
            ----------
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be
                converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a
                sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers
                in regression).
            """
            self.fit(x, y)

            kernel = self._get_training_proximity(x)

            phate_op = self._make_phate_operator()
            self.phate_op = phate_op
            self.embedding_ = phate_op.fit_transform(kernel)

        def fit_transform(self, x, y):
            """Applies _fit_transform to the data, x, y, and returns the
            RF-PHATE embedding

            Parameters
            ----------
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be
                converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a
                sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers
                in regression).

            Returns
            -------
            array-like (n_features, n_components)
                A lower-dimensional representation of the data following the
                RF-PHATE algorithm
            """
            self._fit_transform(x, y)
            return self.embedding_

    return RFPHATE(
        n_components=n_components,
        kernel_method=kernel_method,
        matrix_type=matrix_type,
        n_landmark=n_landmark,
        t=t,
        n_pca=n_pca,
        mds_solver=mds_solver,
        mds_dist=mds_dist,
        mds=mds,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        force_nonzero_diag=force_nonzero_diag,
        force_symmetric=force_symmetric,
        kernel_symm=kernel_symm,
        beta=beta,
        self_similarity=self_similarity,
        **kwargs,
    )