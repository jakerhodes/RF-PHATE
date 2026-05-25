import graphtools
from phate import PHATE
from sklearn.exceptions import NotFittedError
from scipy import sparse
import numpy as np


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
                "This PageRankPHATE instance is not fitted yet. Call "
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