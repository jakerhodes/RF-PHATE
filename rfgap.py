# Imports
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

#sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn

from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble import _forest as forest
    from sklearn.ensemble._forest import _generate_sample_indices
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices # Remove underscore from _forest
    from sklearn.ensemble.forest import _generate_sample_indices # Remove underscore from _forest
    from sklearn.ensemble import forest

from sklearn.utils.validation import check_is_fitted


def RFGAP(prediction_type = None, y = None, prox_method = 'rfgap', matrix_type = 'sparse', triangular = True,
          non_zero_diagonal = True, **kwargs):
    """
    A factory method to conditionally create the RFGAP class based on RandomForestClassifier or RandomForestRegressor

    This class takes on a random forest predictors (sklearn) and adds methods to 
    construct proximities from the random forest object. 
        

    Parameters
    ----------

    prediction_type : str
        Options are 'regression' or 'classification'

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        This is an optional way to determine whether RandomForestClassifier or RandomForestRegressor
        should be used

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', 
        or 'rfgap' (default is 'oob')

    matrix_type : str
        Whether the matrix returned proximities whould be sparse or dense 
        (default is sparse)

    triangular : bool
        Should only the upper triangle of the proximity matrix be computed? This speeds up computation
        time. Not available for RF-GAP proximities (default is True)

    non_zero_diagonal : bool
        Only used for RF-GAP proximities. Should the diagonal entries be computed as non-zero? 
        If True, the proximities are also normalized to be between 0 (min) and 1 (max).
        (default is True)

    **kwargs
        Keyward arguements specific to the RandomForestClassifer or 
        RandomForestRegressor classes

        
    Returns
    -------
    self : object
        The RF object (unfitted)

    """


    if prediction_type is None and y is None:
        prediction_type = 'classification'
    

    if prediction_type is None and y is not None:
        if np.dtype(y) == 'float64' or np.dtype(y) == 'float32':
            prediction_type = 'regression'
        else:
            prediction_type = 'classification'


    if prediction_type == 'classification':
        rf = RandomForestClassifier
    elif prediction_type == 'regression':
        rf = RandomForestRegressor

    class RFGAP(rf):

        def __init__(self, prox_method = prox_method, matrix_type = matrix_type, triangular = triangular,
                     non_zero_diagonal = non_zero_diagonal, **kwargs):

            super(RFGAP, self).__init__(**kwargs)

            self.prox_method = prox_method
            self.matrix_type = matrix_type
            self.triangular  = triangular
            self.prediction_type = prediction_type
            self.non_zero_diagonal = non_zero_diagonal


        def fit(self, X, y, sample_weight = None, x_test = None):

            """Fits the random forest and generates necessary pieces to fit proximities

            Parameters
            ----------

            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted to dtype=np.float32.
                If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in regression).

            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. If None, then samples are equally weighted. Splits that would 
                create child nodes with net zero or negative weight are ignored while searching 
                for a split in each node. In the case of classification, splits are also ignored 
                if they would result in any single class carrying a negative weight in either child node.

            Returns
            -------
            self : object
                Fitted estimator.

            """
            super().fit(X, y, sample_weight)
            self.leaf_matrix = self.apply(X)

            
            if x_test is not None:
                n_test = np.shape(x_test)[0]
                
                self.leaf_matrix_test = self.apply(x_test)
                self.leaf_matrix = np.concatenate((self.leaf_matrix, self.leaf_matrix_test), axis = 0)
            
                        
            if self.prox_method == 'oob':
                self.oob_indices = self.get_oob_indices(X)
                
                if x_test is not None:
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                
                self.oob_leaves = self.oob_indices * self.leaf_matrix

            if self.prox_method == 'rfgap':

                self.oob_indices = self.get_oob_indices(X)
                self.in_bag_counts = self.get_in_bag_counts(X)

                
                if x_test is not None:
                    self.oob_indices = np.concatenate((self.oob_indices, np.ones((n_test, self.n_estimators))))
                    self.in_bag_counts = np.concatenate((self.in_bag_counts, np.zeros((n_test, self.n_estimators))))                
                                
                self.in_bag_indices = 1 - self.oob_indices

                self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix
                self.oob_leaves = self.oob_indices * self.leaf_matrix
            

        
        def _get_oob_samples(self, data):
            
            """This is a helper function for get_oob_indices. 

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)

            """
            n = len(data)
            oob_samples = []
            for tree in self.estimators_:
                # Here at each iteration we obtain out-of-bag samples for every tree.
                oob_indices = _generate_unsampled_indices(tree.random_state, n, n)
                oob_samples.append(oob_indices)

            return oob_samples



        def get_oob_indices(self, data):
            
            """This generates a matrix of out-of-bag samples for each decision tree in the forest

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)


            Returns
            -------
            oob_matrix : array_like (n_samples, n_estimators) 

            """
            n = len(data)
            num_trees = self.n_estimators
            oob_matrix = np.zeros((n, num_trees))
            oob_samples = self._get_oob_samples(data)

            for t in range(num_trees):
                matches = np.unique(oob_samples[t])
                oob_matrix[matches, t] = 1

            return oob_matrix.astype(int)

        def _get_in_bag_samples(self, data):

            """This is a helper function for get_in_bag_indices. 

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)

            """

            n = len(data)
            in_bag_samples = []
            for tree in self.estimators_:
            # Here at each iteration we obtain in-bag samples for every tree.
                in_bag_sample = _generate_sample_indices(tree.random_state, n, n)
                in_bag_samples.append(in_bag_sample)
            return in_bag_samples


        def get_in_bag_counts(self, data):
            
            """This generates a matrix of in-bag samples for each decision tree in the forest

            Parameters
            ----------
            data : array_like (numeric) of shape (n_samples, n_features)


            Returns
            -------
            in_bag_matrix : array_like (n_samples, n_estimators) 

            """
            n = len(data)
            num_trees = self.n_estimators
            in_bag_matrix = np.zeros((n, num_trees))
            in_bag_samples = self._get_in_bag_samples(data)

            for t in range(num_trees):
                matches, n_repeats = np.unique(in_bag_samples[t], return_counts = True)
                in_bag_matrix[matches, t] += n_repeats


            return in_bag_matrix

        def get_proximity_vector(self, ind):

            """This method produces a vector of proximity values for a given observation
            index. This is typically used in conjunction with get_proximities.
            
            Parameters
            ----------
            leaf_matrix : (n_samples, n_estimators) array_like
            oob_indices : (n_samples, n_estimators) array_like
            method      : string: methods may be 'original', 'oob', or 'rfgap (default is 'oob')
            
            Returns
            -------
            prox_vec : (n_samples, 1) array)_like: a vector of proximity values
            """
            n, num_trees = self.leaf_matrix.shape
            
            prox_vec = np.zeros((1, n))
            
            if self.prox_method == 'oob':

                if self.triangular:

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == self.oob_indices[ind:, ind_oob_leaves], axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts   = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == self.oob_leaves[ind:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.where(prox_vec != 0)[0] + ind
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols - ind]

                else:

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == self.oob_indices[:, ind_oob_leaves], axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts   = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == self.oob_leaves[:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols = np.nonzero(prox_vec)[0]
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols]

            elif self.prox_method == 'original':

                if self.triangular:

                    tree_inds = self.leaf_matrix[ind, :] # Only indices after selected index
                    prox_vec = np.sum(tree_inds == self.leaf_matrix[ind:, :], axis = 1)

                    cols = np.where(prox_vec != 0)[0] + ind
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols - ind] / num_trees

                else:

                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec = np.sum(tree_inds == self.leaf_matrix, axis = 1)

                    cols = np.nonzero(prox_vec)[0]
                    rows = np.ones(len(cols), dtype = int) * ind
                    data = prox_vec[cols] / num_trees


            elif self.prox_method == 'rfgap':

                oob_trees    = np.nonzero(self.oob_indices[ind, :])[0]
                in_bag_trees = np.nonzero(self.in_bag_indices[ind, :])[0]

                terminals = self.leaf_matrix[ind, :]

                matches = terminals == self.in_bag_leaves 

                match_counts = np.where(matches, self.in_bag_counts, 0)

                ks = np.sum(match_counts, axis = 0)
                ks[ks == 0] = 1
                ks_in  = ks[in_bag_trees]
                ks_out = ks[oob_trees]

                S_out = np.count_nonzero(self.oob_indices[ind, :])

                prox_vec = np.sum(np.divide(match_counts[:, oob_trees], ks_out), axis = 1) / S_out

                if self.non_zero_diagonal:
                    S_in  = np.count_nonzero(self.in_bag_indices[ind, :])
                    
                    # TODO: Check behavior here
                    if S_in > 0:
                        prox_vec[ind] = np.sum(np.divide(match_counts[ind, in_bag_trees], ks_in)) / S_in
                    else: 
                        prox_vec[ind] = np.sum(np.divide(match_counts[ind, in_bag_trees], ks_in))

                    prox_vec = prox_vec / np.max(prox_vec)
                    prox_vec[ind] = 1

                cols = np.nonzero(prox_vec)[0]
                rows = np.ones(len(cols), dtype = int) * ind
                data = prox_vec[cols]

            return data.tolist(), rows.tolist(), cols.tolist()
        
        
        def get_proximities(self):
            
            """This method produces a proximity matrix for the random forest object.
            
            
            Returns
            -------
            array-like
                (if self.matrix_type == 'dense') matrix of pair-wise proximities

            csr_matrix
                (if self.matrix_type == 'sparse') a sparse crs_matrix of pair-wise proximities
            
            """
            check_is_fitted(self)
            n, _ = self.leaf_matrix.shape

            for i in range(n):
                if i == 0:
                        prox_vals, rows, cols = self.get_proximity_vector(i)
                else:
                    if self.verbose:
                        if i % 100 == 0:
                            print('Finished with {} rows'.format(i))

                    prox_val_temp, rows_temp, cols_temp = self.get_proximity_vector(i)
                    prox_vals.extend(prox_val_temp)
                    rows.extend(rows_temp)
                    cols.extend(cols_temp)


            if self.triangular and self.prox_method != 'rfgap':
                prox_sparse = sparse.csr_matrix((np.array(prox_vals + prox_vals), (np.array(rows + cols), np.array(cols + rows))), shape = (n, n)) 
                prox_sparse.setdiag(1)

            else:
                prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(rows), np.array(cols))), shape = (n, n)) 

            if self.prox_method == 'rfgap':
                prox_sparse = (prox_sparse + prox_sparse.transpose()) / 2

            if self.matrix_type == 'dense':
                return np.array(prox_sparse.todense())
            
            else:
                return prox_sparse


        def prox_extend(self, data):
            """Method to compute proximities between the original training 
            observations and a set of new observations.

            Parameters
            ----------
            data : (n_samples, n_features) array_like (numeric)
            
            Returns
            -------
            array-like
                (if self.matrix_type == 'dense') matrix of pair-wise proximities between
                the training data and the new observations

            csr_matrix
                (if self.matrix_type == 'sparse') a sparse crs_matrix of pair-wise proximities
                between the training data and the new observations
            """
            check_is_fitted(self)
            n, num_trees = self.leaf_matrix.shape
            extended_leaf_matrix = self.apply(data)
            n_ext, _ = extended_leaf_matrix.shape

            prox_vals = []
            rows = []
            cols = []

            if self.prox_method == 'oob':

                for ind in range(n):

                    ind_oob_leaves = np.nonzero(self.oob_leaves[ind, :])[0]

                    tree_counts = np.sum(self.oob_indices[ind, ind_oob_leaves] == np.ones_like(extended_leaf_matrix[:, ind_oob_leaves]), axis = 1)
                    tree_counts[tree_counts == 0] = 1

                    prox_counts = np.sum(self.oob_leaves[ind, ind_oob_leaves]  == extended_leaf_matrix[:, ind_oob_leaves], axis = 1)
                    prox_vec = np.divide(prox_counts, tree_counts)

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp] 


                    cols.extend(cols_temp)
                    rows.extend(rows_temp)
                    prox_vals.extend(prox_temp)



            elif self.prox_method == 'original':

                for ind in range(n):

                    tree_inds = self.leaf_matrix[ind, :]
                    prox_vec  = np.sum(tree_inds == extended_leaf_matrix, axis = 1)

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp] / num_trees

                    cols.extend(cols_temp)
                    rows.extend(rows_temp)
                    prox_vals.extend(prox_temp)

            elif self.prox_method == 'rfgap':
  
                for ind in range(n_ext):

                    oob_terminals = extended_leaf_matrix[ind, :] 

                    matches = oob_terminals == self.in_bag_leaves
                    matched_counts = np.where(matches, self.in_bag_counts, 0)

                    ks = np.sum(matched_counts, axis = 0)
                    ks[ks == 0] = 1

                    prox_vec = np.sum(np.divide(matched_counts, ks), axis = 1) / num_trees

                    cols_temp = np.nonzero(prox_vec)[0]
                    rows_temp = np.ones(len(cols_temp), dtype = int) * ind
                    prox_temp = prox_vec[cols_temp]

                    cols.extend(rows_temp)
                    rows.extend(cols_temp)
                    prox_vals.extend(prox_temp)


            prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(cols), np.array(rows))), shape = (n_ext, n))

            if self.matrix_type == 'dense':
                return prox_sparse.todense() 
            else:
                return prox_sparse

    return RFGAP(prox_method = prox_method, matrix_type = matrix_type, triangular = triangular, **kwargs)