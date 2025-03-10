"""
Classes and functions for different graph 
constructions.
- (symmetric) k-nearest neighbors (k-NN)
- epsilon
"""
import manifold_sampling as ms
from typing import (
    Union,
    Tuple,
    Callable
)
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse
from scipy.sparse import coo_array

# k-NN graph
from math import ceil
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import (
    identity,
    diags
)
from scipy.special import gamma
# scipy sparse matrix type hinting
from scipy.sparse.base import spmatrix

# epsilon graph
from sklearn.metrics.pairwise import euclidean_distances


"""
Shared graph construction functions
"""

def calc_LRWM(
    W: scipy.sparse.spmatrix | np.ndarray,
    normalize: bool = False,
) -> scipy.sparse.spmatrix | np.ndarray:
    r"""
    Calculates the (sparse) lazy random walk matrix P,
    either unnormalized (default) or normalized.

    Unnormalized LRWM:
    $P = \frac{1}{2}(I + WD^{-1})$, where $W$ is the
    (weighted) adjacency matrix, and $D$ is the
    diagonal degree matrix of a graph $G$.

    Normalized LRWM (i.e. cols sum to 1):
    $P_norm = \frac{1}{2}(I + D^{-1/2 W D^{-1/2}) 
    = D^{-1/2} P D^{1/2}$

    Args:
        W: un/weighted adjacency matrix of a graph.
        normalize: whether to return normalized LRWM.

    Returns:
        Scipy sparse un/normalized lazy random walk matrix P.
    """
    # for sparse matrix:
    if scipy.sparse.issparse(W):
        diagD = np.ravel(np.sum(W, axis=1))
        # diagD = np.sum(W, axis=1).A1
        D_inv = diags(1 / diagD, 0)
        # print(D_inv)
        P = 0.5 * (identity(W.shape[0]) + W @ D_inv)

        if normalize:
            diagD_sqrt = np.sqrt(diagD)
            P = diags(1 / diagD_sqrt, 0) @ P @ diags(diagD_sqrt, 0)

    # for numpy array:
    else:
        # W @ D_inv (= j-to-i probability matrix) is equiv. to 
        # col-wise mult of W by diag. entries of D_inv.
        jtoi_prob_matrix = np.einsum('nm,m->nm', W, 1 / diagD)
        P = 0.5 * (np.eye(W.shape[0]) + jtoi_prob_matrix)
    
        if normalize:
            diagD_sqrt = np.sqrt(diagD)
            P = np.einsum(
                'm,nm,m->nm',
                1 / diagD_sqrt,
                P,
                diagD_sqrt
            )
    return P


def get_eta_objects(
    d_manifold: int,
    eta_type: str = 'indicator',
) -> Tuple[Callable, float, float]:
    """
    For a string key, returns eta kernel function,
    c_d (n-dim unit ball volume), and corresponding 
    constant c_eta, used in scaling the graph Laplacians 
    of k-NN and epsilon graphs.
    
    Note that the calculation of e_eta for manifolds with
    dimension > 1 requires spherical integrals in general
    (and is not yet implemented in general here);
    the 'indicator' eta kernel is a special case:
    cf. Calder and Trillos, "Improved spectral convergence rates for 
    graph Laplacians on epsilon-graphs and k-NN graphs", Eq. (2.2).

    Args:
        eta_type: string key for eta kernel type.
        d_manifold: dimension of the manifold (e.g.,
            1 for a circle).

    Returns:
        3-tuple of eta function, c_eta, and c_d.
    """
    c_d = ms.n_dim_ball_vol(d_manifold, r=1.)
    
    if eta_type == 'indicator':
        eta = lambda y: np.ones_like(y)
        c_eta =  c_d / (d_manifold + 2.)
            
    elif eta_type == 'identity':
        eta = lambda y: y
        if d_manifold == 1:
            c_eta = 2.
        else:
            print('c_eta not implemented for eta = identity and d_manifold > 1!')
            c_eta = None
            
    elif eta_type == 'exp':
        eta = lambda y: np.exp(-y)
        if d_manifold == 1:
            c_eta = 1. / (4. - (10. / np.e))
        else:
            print('c_eta not implemented for eta = exp and d_manifold > 1!')
            c_eta = None
            
    return eta, c_eta, c_d


"""
Graph construction classes
"""

class KNNGraph:
    """
    Implementation for an un/weighted, symmetric 
    k-NN graph (symmetric: if x_i is a nearest 
    neighbor of x_j, OR vice versa, they get an
    edge).
    """
    
    def __init__(
        self, 
        x: np.ndarray,
        n: int,
        d_manifold: int,
        k: Union[int, str],
        k_constant: float = 1.,
        knn_algorithm: str = 'ball_tree',
        knn_logic: str = 'or',
        eta_type: str = 'indicator',
        verbosity: int = 0
    ) -> None:
        """
        Args:
            x: set of points on the manifold.
            n: number of data points/vertices.
            d_manifold: dimension of the manifold (integer > 0).
            k: max number of nearest neighbors: 'auto'
                calculates based on equation.
            k_constant: manual constant multiplier for k in 
                'auto' mode.
            distance: string key for distance metric
                between points in a dataset x.
            knn_algorithm: string key of the algorithm to be used
                by sklearn's NearestNeighbors method.
            knn_logic: 'or' for x_j is a nearest neighbor of x_i 
                OR vice versa; 'and' for x_j is a nearest neighbor 
                of x_i AND vice versa ('shared' or 'mutual' k-NN).
            eta_type: string key for the eta kernel function that 
                wraps the edge weights in the k-NN graph.
            verbosity: controls print output during 
                class' method executions.

        Ref:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.neighbors.NearestNeighbors.html
        #sklearn.neighbors.NearestNeighbors
        """
        self.x = x
        self.n = n
        self.d_manifold = d_manifold
        self.k_constant = k_constant
        self.knn_algorithm = knn_algorithm
        self.knn_logic = knn_logic
        self.eta_type = eta_type
        if self.eta_type not in ('indicator', 'identity'):
            raise Exception(
                f"No eta kernel yet implemented for eta_type={self.eta_type}."
            )
        else:
            self.eta_kernel = None
        self.c_d = None
        self.distances = None
        self.indices = None
        self.verbosity = verbosity

        # set k based on string key or int
        self.set_k(k)

        self.nbrs = None
        # self.G = None
        self.W = None
        self.diagD = None
        self.L = None

        
    def set_k(self, k) ->  None:
        if k == 'auto':
            # from MFCN paper, Example 2.7 
            k = self.k_constant * \
                np.log(self.n) ** (self.d_manifold / (self.d_manifold + 4)) \
                * self.n ** (4 / (self.d_manifold + 4))
            if k > self.n:
                print(
                    f'WARNING: k>n ({k}>{self.n}). The intrinsic'
                    f' dimension of the manifold may be too large.'
                    f' setting k = n - 1 = {self.n - 1} instead.'
                )
                self.k = self.n - 1
            else:
                self.k = ceil(k)

            # (almost) always print k when using 'auto' method
            if self.verbosity > -1:
                print(f'k = {self.k}')
        else:
            self.k = k
            if self.verbosity > 0:
                print(f'k = {self.k}')


    def calc_distances(self, x: np.ndarray | spmatrix) -> None:
        """
        Runs sklearn's NearestNeighbors method, setting
        `self.distances` and `self.indices` from the fit.

        Note that x can be dense or scipy sparse matrix:
            https://scikit-learn.org/0.15/modules/generated
            /sklearn.neighbors.NearestNeighbors.html
            #sklearn.neighbors.NearestNeighbors.fit
        """
        # exclude points from being their own nearest neighbors 
        # -> use k+1 in sklearn's implem.
        # note: scipy >v1.15 has a 'include_self=False' parameter in
        # `kneighbors_graph`, but I could only get v1.14...
        self.nbrs = NearestNeighbors(
            # includes self-neighborship, so do k+1, and remove 
            # self-neighbors in W
            n_neighbors=(self.k + 1), 
            algorithm=self.knn_algorithm
        ).fit(x)
        # returns numpy arrays of eucl. distances and nn indices for each pt
        self.distances, self.indices = self.nbrs.kneighbors(x)

    
    def calc_W(self, recalc_dists: bool = False) -> None:
        """
        Computes and sets self.W, the (un)weighted adjacency matrix
        (type: scipy.sparse._csr.csr_matrix).

        In the eta = indicator case, W is A, the unweighted adjacency 
        matrix.
        """

        # fit knn algorithm: calc nearest neighbors and distances
        if (not isinstance(self.distances, np.ndarray)) \
        or (not isinstance(self.indices, np.ndarray)) \
        or recalc_dists:
            self.calc_distances(self.x)

        if self.eta_type == 'indicator':
            self.W = self.nbrs.kneighbors_graph(
                self.x, 
                mode='connectivity'
            ).astype('int')
            
            # remove self-neighborship
            # note: scipy >v1.15 has a 'include_self=False' parameter in
            # `kneighbors_graph`, but I could only get v1.14...
            # print('type(self.W):', type(self.W))
            # print('self.n:', self.n)
            self.W = self.W - identity(self.n, dtype='int8')

            if self.knn_logic == 'or':
                # symmetrize W
                # add edges where x_i is a nearest neighbor of x_j, OR vice versa
                # (up to this point, we only have one-way neighborships / asymmetric W)
                # here, we add W to its transpose (getting entries of 0, 1, or 2)
                # and cast any entry >0 to 1
                # type: csr_array
                self.W = ((self.W + self.W.T) > 0).astype('int8')
            elif self.knn_logic == 'and':
                # (1) get non-symmetric entries (efficient in sparse matrices)
                W_nonsym = (self.W != self.W.T)
                # (2) set non-symmetric entries to 0 in W
                self.W = self.W.tolil()
                for i, j in np.stack(W_nonsym.nonzero()).T:
                    self.W[i, j] = 0
                    
        else:
            # NOTE: when NOT using an indicator for eta, we obtain an asymmetric
            # weighted k-NN graph!
            # (1) compute adjusted distances for nearest neighbors
            # proceed by x_i/rows of `knn_triu`; divide each |x_i - x_j| 
            # by r_k = max(d_ki, d_kj), where d_ki is the distance between 
            # x_i and its last (kth) nearest neighbor
            # divide distances by r_k and save in a new `adj_distances` array
            adj_distances = np.zeros((self.n, self.k))

            # ref: Calder and Trillos 2022, p. 6
            # ("Improved spectral convergence rates [...]")
            if self.knn_logic == 'or':
                denom_fn = max
            elif self.knn_logic == 'and':
                # use 'min' for mutual kNN graph
                denom_fn = min
                
            for i, d_row in enumerate(self.distances):
                d_ki = d_row[self.k]
                # print('d_ki =', d_ki)
                knns_js = self.indices[i, 1:]
                # print(knns_js)
                for j_i, j in enumerate(knns_js):
                    d_kj = self.distances[j, self.k]
                    # print('d_kj =', d_kj)
                    denom = denom_fn(d_ki, d_kj) 
                    # print('denom =', denom)
                    adj_distances[i, j_i] = self.distances[i, j_i + 1] / denom
            self.distances = np.concatenate(adj_distances)
            if self.eta_type != 'identity':
                self.distances = self.eta_kernel(self.distances)

            # (2) generate W ('lil' sparse type)
            row_idx = np.repeat(np.arange(self.n), self.k)
            col_idx = np.concatenate(self.indices[:, 1:]) # exclude self-neighborship
            self.W = coo_array(
                (self.distances, (row_idx, col_idx)),
                shape=(self.n, self.n)
            ).tolil()

            if self.knn_logic == 'or':
                # (3) symmetrize W: here, we 
                # (a) add W to its transpose and subtract 2*W
                # (b) then, any element >0 was non-symmetric in W 
                #     (works because all entries are distances >0)
                # (c) insert previously-missing symmetrical entries back into W
                miss_entries_W_sym = (self.W + self.W.T - 2 * self.W) > 0.
                miss_row_idx, miss_col_idx = miss_entries_W_sym.nonzero()
                for i, row_i in enumerate(miss_row_idx):
                    col_j = miss_col_idx[i]
                    self.W[row_i, col_j] = self.W[col_j, row_i]
                    
            elif self.knn_logic == 'and':
                # set non-symmetric entries to 0.
                W_nonsym = (self.W != self.W.T)
                for i, j in np.stack(W_nonsym.nonzero()).T:
                    self.W[i, j] = 0.

        # convert to csr sparse type (lil sparse type has bugs in
        # .data attribute (returns lists...), as of Feb 2025)
        self.W = self.W.tocsr()
                
        # calc graph sparsity
        nonzero_ct = self.W.count_nonzero()
        total_ct = self.n ** 2
        self.nonzero_prop = nonzero_ct / total_ct
            
        if self.verbosity > 0:
            print(f'G nonzero entries: {nonzero_ct}'
                  f' of {total_ct} ({self.nonzero_prop * 100:.2f}%)')


    def calc_Laplacian(self, recalc_W: bool = False) -> None:
        """
        Computes the symmetric k-NN graph's Laplacian matrix.
        """
        if (self.W is None) or (recalc_W):
            self.calc_W(self.x)

        # calc diagonal entries of weighted degree matrix D
        # and create sparse matrix D
        # note W is sparse -> np.matrix object -> always 2d
        # but coo_array requires 1d vector inputs -> 'A1' attribute
        # of np.matrix is equiv. to 'np.squeeze'
        # https://stackoverflow.com/questions/59944558/why-does-squeeze-not-work-on-sparse-arrays
        self.diagD = np.ravel(np.sum(self.W, axis=1))
        diag_idx = np.arange(self.n)
        D = coo_array(
            (self.diagD, (diag_idx, diag_idx)), 
            shape=(self.n, self.n)
        )
        
        # for sanity check, before constants
        # self.L = (D - self.W)

        self.eta, self.c_eta, self.c_d = get_eta_objects(
            self.d_manifold, 
            self.eta_type
        )
        constant_L = (1. / (self.c_eta * self.n)) * \
            ((self.n * self.c_d) / self.k) ** (1 + 2 / self.d_manifold)
        self.L = constant_L * (D - self.W) # sparse csr_array
        # print(self.L)
        
        """
        # Calder and Trillos implementation [slightly refactored]
        # note that is builds in density (rho) adjustment into the constant for W,
        # to get eigenvalues 2, 6, 12... instead of needing to adjust Lf like we do
        # https://github.com/jwcalder/kNNSpectralRates/blob/main/sphere_rates.py
        
        import graphlearning as gl
        from graphlearning.weightmatrix import knnsearch, knn
        from scipy.special import gamma

        self.k = self.n ** (4 / (self.d_manifold + 4))
        m = self.d_manifold
        sigma = ms.n_dim_ball_vol(m) / (m + 2.)
        alpha = np.pi**(m/2.) / gamma(m/2. + 1.)
        alphaup = np.pi**((m+1.) / 2.) / gamma((m+1.)/2. + 1.)
        rho = 1. / ((m + 1.) * alphaup)
        constant_W = (2. * rho**(2./m)) / sigma
        constant_L = ((self.n * alpha/self.k)**(1. + 2./m)) / self.n
        # self.L = constant_L * (D - constant_W * self.W)

        # https://jwcalder.github.io/GraphLearning/weightmatrix.html
        # note: 'knnsearch' includes self-neighborship!
        knn_ind, knn_dist = knnsearch(self.x, self.k)
        W = knn(
            None,
            self.k,
            knn_data=(knn_ind, knn_dist),
            kernel='uniform'
        )
        L = constant_W * gl.graph(W).laplacian()
        self.L = constant_L * L
        """
        
          
        
        

            

class EpsilonGraph:
    """
    Class for an epsilon-graph G and its
    weighted adjacency (W), degree (D),
    and graph Laplacian (L) matrices.

    Note that for the choice of kernel function
    eta, a constant c_eta needs to be calculated
    in order to compute the graph Laplacian. This
    calculation is an integral (see manuscript).
    Example: for eta = exp(-y) for 1-d manifold, 
    use 1 / z, where z is 
    "integral -1 to 1 |y|^2 * e^(-|y|) dy"
    [in Wolfram Alpha input].
    """
    
    def __init__(
        self, 
        x: np.ndarray,
        n: int,
        d_manifold: int, 
        distance: str = 'euclidean',
        eps: Union[float, str] = 'auto',
        eps_constant: float = 1.0,
        eps_quantile: float = 0.5,
        eta_type: str = 'indicator',
        verbosity: int = 0
    ) -> None:
        """
        Args:
            x: set of points on the manifold.
            n: number of data points/vertices.
            eps: epsilon parameter (float value),
                or a string key for the mode of 
                its calculation, e.g. 'auto' or
                'quantile'.
            d_manifold: dimension of the manifold (integer > 0).
            distance: string key for distance metric
                between points in a dataset x.
            eps_constant: constant multiple for epsilon
                in 'auto' mode.
            eps_quantile: value for calculation of
                epsilon via the 'quantile' method. This
                ensures an (approximate) fixed percent of 
                the data is truncated.
            verbosity: controls print output during 
                class' method executions.
        """
        self.x = x
        self.n = n
        self.d_manifold = d_manifold
        self.distance = distance
        self.dists = None
        self.eps_constant = eps_constant
        self.eps_quantile = eps_quantile
        self.eta_type = eta_type
        self.verbosity = verbosity

        # set epsilon based on string key or float
        self.set_epsilon(eps)

        # set eta c_eta, and c_d
        self.eta, self.c_eta, self.c_d = get_eta_objects(
            self.d_manifold, 
            self.eta_type
        )

        self.G = None
        self.W = None
        self.diagD = None
        self.L = None

    
    def calc_distances(self, x: np.ndarray | spmatrix) -> np.ndarray:
        """
        Note that x can be dense or a scipy sparse matrix:
            https://scikit-learn.org/1.5/modules/generated
            /sklearn.metrics.pairwise.euclidean_distances.html
        """
        if self.distance == "euclidean":
            if self.x.ndim == 1:
                self.x = self.x.reshape(-1, 1)
            dists = euclidean_distances(self.x, self.x)
            if self.verbosity > 1:
                print('distances\n', dists)
        else:
            raise NotImplementedError(
                f'\'{distance}\' distance type not implemented!'
            )
            dists = None
        return dists

    
    def set_epsilon(self, eps) -> None:
        if eps == "quantile":
            self.dists = self.calc_distances(self.x)
            triu_idx = np.triu_indices_from(self.dists, k=1)
            self.eps = np.quantile(
                self.dists[triu_idx],
                self.eps_quantile
            )
        elif eps == "auto":
            self.eps = self.eps_constant * \
                (np.log(self.n) / self.n) \
                    ** (1. / (self.d_manifold + 4))
        else:
            self.eps = eps
            
        if self.verbosity > 0:
            print(f'eps = {self.eps:.4f}')

    
    def calc_W(self, recalc_dists = False) -> None:
        r"""
        G is constructed with 1s and 0s, whether
        $|x_i - x_j| \leq \epsilon$.

        W is constructed where G has 1s, by
        $W_{i,j} = \eta(\frac{|x_i - x_j|}{\epsilon})$,
        where $\eta$ is a kernel function with certain
        restrictions (cf. Example 2.6).
        """
        if (not isinstance(self.dists, np.ndarray)) \
        or recalc_dists:
            self.dists = self.calc_distances(self.x)
        
        # truncate/discard values where dists > epsilon
        row_idx, col_idx = np.where(self.dists <= self.eps)

        # G is a sparse graph with an edge/1 where above is true 
        self.G = coo_array(
            (np.ones(row_idx.shape[0]), (row_idx, col_idx)), 
            shape=(self.n, self.n)
        )

        # calc graph sparsity stat
        nonzero_ct = row_idx.shape[0]
        total_ct = self.n ** 2
        self.nonzero_prop = nonzero_ct / total_ct
        
        if self.verbosity > 0:
            print(f'G nonzero entries: {nonzero_ct}'
                  f' of {total_ct} ({self.nonzero_prop * 100:.2f}%)')

        # where G has 1, the weighted adjacency matrix W has 
        # some positive kernel value
        k_vals = self.eta(
            self.dists[row_idx, col_idx] / self.eps
        )
        
        # if self.verbosity > 1:
        #     print('kernel_vals\n', k_vals, '\n')

        # generate sparse W
        self.W = coo_array(
            (k_vals, (row_idx, col_idx)), 
            shape=(self.n, self.n)
        ).tocsr()
        
    
    def calc_Laplacian(self) -> None:
        """
        Computes the graph Laplacian matrix.
        """
        if (self.G is None) \
        or (self.W is None) \
        or (recalc_G_and_W):
            self.calc_W(self.x)

        # calc diagonal entries of weighted degree matrix D
        # and create sparse matrix D
        self.diagD = np.sum(self.W, axis=1)
        diag_idx = np.arange(self.n)
        D = coo_array(
            (self.diagD, (diag_idx, diag_idx)), 
            shape=(self.n, self.n)
        )

        # calc graph Laplacian (sparse type: csr_array)
        constant =  1. / (self.c_eta * self.n * \
                          self.eps ** (self.d_manifold + 2))
        self.L = constant * (D - self.W)

