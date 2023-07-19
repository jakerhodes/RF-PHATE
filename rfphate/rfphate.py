# TODO: Import rfgap from RF-GAP repo. Make this a requirement.
from rfphate.rfgap import RFGAP

# For PHATE part
from phate import PHATE
import numpy as np
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import graphtools
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted

class PHATET(PHATE): 
    """
    PHATET is an adaptation of PHATE which incorporates random jumps into the diffusion operator.
    This improvement is based on Google's PageRank algorithm and makes the PHATE algorithm more
    robust to parameter selection.
    """

    def __init__(self, beta = 0.9, **kwargs):
        super(PHATET, self).__init__(**kwargs)

        self.beta = beta

    @property
    def diff_op(self):
        """diff_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The diffusion operator built from the graph
        """
        if self.graph is not None:
            if isinstance(self.graph, graphtools.graphs.LandmarkGraph):
                diff_op = self.graph.landmark_op
            else:
                diff_op = self.graph.diff_op
            if sparse.issparse(diff_op):
                diff_op = diff_op.toarray()

            dim = diff_op.shape[0]

            diff_op_tele = self.beta * diff_op + (1 - self.beta) * 1 / dim * np.ones((dim, dim))


            return diff_op_tele

        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )
            

def RFPHATE(prediction_type = None,
            y = None,           
            n_components = 2,
            prox_method = 'rfgap',
            matrix_type = 'sparse',
            n_landmark = 2000,
            t = "auto",
            n_pca = 100,
            mds_solver = "sgd",
            mds_dist = "euclidean",
            mds = "metric",
            n_jobs = 1,
            random_state = None,
            verbose = 0,
            non_zero_diagonal = True,
            beta = 0.9,
            self_similarity = True,
            **kwargs):
    
    
    """An RF-PHATE class which is used to fit a random forest, generate RF-proximities,
       and create RF-PHATE embeddings.

    Parameters
    ----------
    n_components : int
        The number of dimensions for the RF-PHATE embedding

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', and
        'rfgap' (default is 'oob')

    matrix_type : str
        Whether the proximity type should be 'sparse' or 'dense' (default is sparse)
    
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
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted (default is 'euclidean')

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

    non_zero_diagonal: bool
        Only used if prox_method == 'rfgap'.  Replaces the zero-diagonal entries
        of the rfgap proximities with ones (default is True)

    self_similarity: bool  
        Only used if prox_method == 'rfgap'. All points are passed down as if OOB. 
        Increases similarity between an observation and itself as well as other
        points of the same class. NOTE: This partially disrupts the geometry
        learned by the RF-GAP proximities, but can be useful for exploring
        particularly noisy data.
    """

    if prediction_type is None and y is None:
        prediction_type = 'classification'
        
    # In the rfgap module, rf is defined without arguements
    rf = RFGAP(prediction_type = prediction_type, y = y, **kwargs)

    class RFPHATE(rf.__class__, PHATET):
    # class RFPHATE(PHATET):
    
        def __init__(
            self,
            n_components = n_components,
            prox_method  = prox_method,
            matrix_type  = matrix_type,
            n_landmark   = n_landmark,
            t            = t,
            n_pca        = n_pca,
            mds_solver   = mds_solver,
            mds_dist     = mds_dist ,
            mds          = mds,
            n_jobs       = n_jobs,
            random_state = random_state,
            verbose      = verbose,
            non_zero_diagonal = non_zero_diagonal,
            beta         = beta,
            self_similarity = self_similarity,
            **kwargs
            ):

            super(RFPHATE, self).__init__(**kwargs)
            
            self.n_components = n_components
            self.t = t
            self.n_landmark = n_landmark
            self.mds = mds
            self.n_pca = n_pca
            self.knn_dist = 'precomputed_affinity'
            self.mds_dist = mds_dist
            self.mds_solver = mds_solver
            self.random_state = random_state
            self.n_jobs = n_jobs

            self.graph = None
            self._diff_potential = None
            self.embedding = None
            self.x = None
            self.optimal_t = None
            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.verbose = verbose
            self.non_zero_diagonal = non_zero_diagonal
            self.beta = beta
            self.self_similarity = self_similarity

        # From https://www.geeksforgeeks.org/class-factories-a-powerful-pattern-in-python/
            for k, v in kwargs.items():
                setattr(self, k, v)
                    
                
        def _transform(self, x):
            
            check_is_fitted(self)
            
            phate_op = PHATET(n_components = self.n_components,
                t = self.t,
                n_landmark = self.n_landmark,
                mds = self.mds,
                n_pca = self.n_pca,
                knn_dist = self.knn_dist,
                mds_dist = self.mds_dist,
                mds_solver = self.mds_solver,
                random_state = self.random_state,
                verbose = self.verbose, 
                beta = self.beta)
            
            self.phate_op = phate_op
            self.embedding_ = phate_op.fit_transform(self.proximity)
            
            return self.embedding_
            
            
        def _fit_transform(self, x, y, x_test = None, sample_weight = None):

            """Internal method for fitting and transforming the data
            
            Parameters
            ----------
            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).
                
            x_test : {array-like, sparse matrix} of shape (n__test_samples, n_features)
                An optional test set. The training set buildes the RF-PHATE model, but the 
                embedding can be extended to this test set.
            """

            n,  _= x.shape
            
            self.fit(x, y, x_test = x_test, sample_weight = sample_weight)

            if self.prox_method == 'rfgap' and self.self_similarity:
                if x_test is None:
                    proximity = self.prox_extend(x)
                else:
                    proximity = self.prox_extend(np.concatenate([x, x_test]))
            else:
                proximity = self.get_proximities()
                            
            phate_op = PHATET(n_components = self.n_components,
                t = self.t,
                n_landmark = self.n_landmark,
                mds = self.mds,
                n_pca = self.n_pca,
                knn_dist = self.knn_dist,
                mds_dist = self.mds_dist,
                mds_solver = self.mds_solver,
                random_state = self.random_state,
                verbose = self.verbose, 
                beta = self.beta)
            
            self.phate_op = phate_op

            self.embedding_ = phate_op.fit_transform(proximity)

        def fit_transform(self, x, y, x_test = None, sample_weight = None):

            """Applies _fit_tranform to the data, x, y, and returns the RF-PHATE embedding

            x : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).
                
            x_test : {array-like, sparse matrix} of shape (n__test_samples, n_features)
                An optional test set. The training set buildes the RF-PHATE model, but the 
                embedding can be extended to this test set.


            Returns
            -------
            array-like (n_features, n_components)
                A lower-dimensional representation of the data following the RF-PHATE algorithm
            """
            self._fit_transform(x, y, x_test, sample_weight = sample_weight)
            return self.embedding_

    return RFPHATE(    
                n_components = n_components,
                prox_method = prox_method,
                matrix_type = matrix_type,
                n_landmark = n_landmark,
                t = t,
                n_pca = n_pca,
                mds_solver = mds_solver,
                mds_dist = mds_dist,
                mds = mds,
                n_jobs = n_jobs,
                random_state = random_state,
                verbose = verbose,
                non_zero_diagonal = non_zero_diagonal,
                beta = beta,
                self_similarity = self_similarity,
                **kwargs)
