"""
Functions for geometric scattering / wavelets.
"""
import data_utilities as du
import numpy as np
import torch
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    Iterable,
    Callable
)
import scipy
import scipy.sparse
from scipy.sparse import (
    identity
)


def scattering_moments(
    Wjf: np.ndarray,
    Q: int = 4
) -> np.ndarray:
    """
    Computes Q scattering moments (q-norms) of 
    wavelet-filtered functions on a manifold,
    as defined in the MFCN manuscript. Note that
    the MFCN paper theory only applied to second-
    order moments (q=2).

    Args:
        Wjf: array of j-orders of wavelet 
            filters applied to a (normalized
            evaluated) function Pnf on the manifold,
            of shape (num_wavelets, n).
        Q: max scattering moment order
            (min is 1).
    Returns:
        Array of scattering moments for each
        wavelet, of shape (num_wavelets, Q).
    """
    axis = 1 if Wjf.ndim > 1 else 0
    moments_arr = np.array([
        np.linalg.norm(x=np.abs(Wjf), ord=q, axis=axis) \
        for q in range(1, Q + 1)
    ])
    if moments_arr.ndim == 1:
        moments_arr = np.expand_dims(moments_arr, axis=1)
    elif moments_arr.ndim == 3:
        moments_arr = np.squeeze(moments_arr)

    if moments_arr.ndim != 2:
        raise Exception(
            f"Output has an unexpected ndim: {moments_arry.ndim}"
        )
    
    return moments_arr
    

"""
spectral filters
"""

def spectral_lowpass_filter(
    eigenvals: np.ndarray | torch.Tensor,
    c: Optional[float] = None,
    device: Optional[str] = None
) -> np.ndarray | torch.Tensor:
    r"""
    Computes a (non-wavelet) simple lowpass spectral 
    filter of the form $w(\lambda) = e^{-c\lambda}$.
    
    Args:
        eigenvals: eigenvalues array or tensor,
            ascending, with shape (k, ).
        c: optional constant to multiply
            lambda by (to adjust smoothing).
        device: optional device string code (e.g. 
            'cuda:0') for placing tensors [currently
            unused].
    Returns:
        Array or tensor of wavelet operators, of shape 
        (1, k).
    """
    if torch.is_tensor(eigenvals):
        exp, unsqueeze = torch.exp, torch.unsqueeze
    else:
        exp, unsqueeze = np.exp, np.expand_dims
    if c is not None:
        eigenvals *= c
    return unsqueeze(exp(-eigenvals), 0) 
    

def spectral_wavelets(
    eigenvals: np.ndarray | torch.Tensor, 
    J: int = 4, 
    include_low_pass: bool = True,
    spectral_c: Optional[float] = None,
    device: Optional[str] = None
) -> np.ndarray | torch.Tensor:
    r"""
    Computes a set of spectral wavelet operators 
    based on eigenvalues 1...kappa, for orders 
    $0 \leq j \leq J$ ($\leq J + 1$ if including 
    low-pass). These operators are defined:

    $$w_{j}(\lambda) = e^{-2^{j-1} c \lambda} - 
    e^{-2^{j} c \lambda$$

    for $1 \leq j \leq J$.

    Args:
        eigenvals: eigenvalues array or tensor, sorted
            ascending; shape (k, )
        J: max wavelet order.
        include_low_pass: whether to include the low-pass
            wavelet filter.
        spectral_c: optional constant c (not equal to 1)
            in the above operator definition (a scalar multiplier
            of the eigenvalues).
        device: optional device string code (e.g. 'cuda:0')
            for placing tensors.
    Returns:
        Array or tensor of wavelet operators, of shape (k, J+1),
        or shape (k, J+2) with low pass.
    """
    if torch.is_tensor(eigenvals):
        arange, exp, outer, stack = torch.arange, torch.exp, torch.outer, torch.stack
        # print(eigenvals.dim())
    else: # numpy
        arange, exp, outer, stack = np.arange, np.exp, np.outer, np.stack
        
    # scale eigenvalues if a 'spectral_c' is provided
    if spectral_c is not None:
        eigenvals *= spectral_c

    # initialize empty list to store wavelets
    wavelets = [None] * (J + 2) if include_low_pass else [None] * (J + 1)

    # pre-calc array of (exponentiated) powers of -2^{j} * lambda:
    # (powers of -2 [rows]) x (eigenvalues [cols]) -> shape (J+1, k)
    dilations = 2 ** arange(0, J + 1)
    if (torch.is_tensor(eigenvals)) and (device is not None):
        dilations = dilations.to(device)
    # print('eigenvals.shape:', eigenvals.shape) # (num_graphs, k)
    dil_evec_arr = exp(-outer(dilations, eigenvals))

    # first wavelet: 1 - t = 1 - exp(-lambda)
    wavelets[0] = 1. - exp(-eigenvals)

    # middle wavelets: t^{2^{j-1}} - t^{2^{j}}
    # -> need row differences: W_0 = 1st row - 2nd; W_1 = 2nd minus 3rd, etc.
    # -> get J + 1 rows in resulting array: shape = (J + 1, k)
    pLambda_arr = dil_evec_arr[:J] - dil_evec_arr[1:(J + 1)]
    for i in arange(0, J): # J
        # insert into middle of wavelets list
        wavelets[i + 1] = pLambda_arr[i]

    # low-pass wavelet: t^{2^{J}} (= final row of dil_evec_arr)
    if include_low_pass:
        # put at end of wavelets list
        wavelets[J + 1] = dil_evec_arr[-1]

    wavelets = stack(wavelets, -1)
    # print('wavelets.shape:', wavelets.shape) # (k, J+1 or J+2)
    return wavelets


def wavelet_spectral_convolutions(
    Pnf: np.ndarray,
    eigenvals: np.ndarray,
    wavelet_filters: np.ndarray,
    J: int,
    include_low_pass: bool,
    eigenvectors: np.ndarray,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates 0th-, 1st-, and 2nd-order spectral
    filtrations of a function on a manifold, using
    wavelet filters.

    Args:
        Pnf: normalized sampled function values on the 
            manifold.
        eigenvals: array of eigenvalues (of a graph 
            Laplacian built on the geometry of the
            manifold); shape (k, ).
        wavelet_filters: array of wavelet filters 
            (e.g. from the 'spectral_wavelets'
            function above); shape (J+1, k) or
            (J+2, k) if including lowpass filter.
        J: max wavelet filter order.
        include_low_pass: whether to include the
            'lowpass' ($e^{2^{J}}$) wavelet filter.
        eigenvectors: array of eigenvectors (of a graph 
            Laplacian built on the geometry of the
            manifold); shape (n, k).
        verbosity: integer controlling print output 
            volume as this method runs.
    Returns:
        3-tuple of the 0th-, 1st-, and 2nd-order 
        convolutions. Shapes:
            0th: (n, ) [this is just Pnf in the Fourier domain]
            1st: (J+1, n) or (J+2, n) if incl. lowpass
            2nd: (J(J+1)/2, n) or ((J+1)(J+2)/2, n) if incl. lowpass
    """
    if verbosity > 0:
        print('Calculating wavelet spectral convolutions...')
            
    # 0
    # zero-order wavelet spectral convolutions
        # note: we still use Fourier domain, even though 0th- 
        # order scattering moment is ||f||^q
        # thus, if k < n, kfourier_Pnf != Pnf
    kfourier_coeffs_Pnf = np.einsum(
        'n,nk->k',
        Pnf,
        eigenvectors
    )

    # don't sum across k; re-used in Wjf_1
    kfourier_series_Pnf = np.einsum(
        'k,nk->nk',
        kfourier_coeffs_Pnf,
        eigenvectors
    )
    Wjf_0 = np.sum(kfourier_series_Pnf, axis=1)
    if verbosity > 0:
        print(f'\tWjf_0.shape: {Wjf_0.shape}') # (n, ) or (n, k)?

    
    # 1
    # first-order wavelet spectral convolutions
        # (l = J+1) wavelet k-filters on Fourier k-series of f
    if verbosity > 1:
        print('wavelet_filters.shape:', wavelet_filters.shape)
        print('kfourier_series_Pnf.shape:', kfourier_series_Pnf.shape)
    Wjf_1 = np.einsum(
        # 'lk,nk->ln',
        'kl,nk->ln',
        wavelet_filters, 
        kfourier_series_Pnf
    )
    if verbosity > 0:
        print(f'\tWjf_1.shape: {Wjf_1.shape}') # (J+1, n)


    # 2
    # second-order spectral convolutions, using (J(J+1)/2) j'-wavelets 
        # for each jth-order wavelet:
        # apply j' wavelet k-filters to |W_j f| in the Fourier k-domain,
        # where 0 <= j < j' <= J, sum across ks, then take |W_j-prime f|
        # note: can skip j = J, since if j = J, j' !> j
    Wjprime_convs = []
    J_lowpass_adj = J + 1 if include_low_pass else J
    
    for j in range(J_lowpass_adj):
        # print(f'j = {j}')
        
        # get |Wjf| as Fourier k-series
        kfourier_coeffs_absWjf = np.einsum(
            'n,nk->k',
            np.abs(Wjf_1[j]),
            eigenvectors
        )
        kfourier_series_absWjf = np.einsum(
            'k,nk->nk',
            kfourier_coeffs_absWjf,
            eigenvectors
        )

        # loop through jprimes > j
        for jprime in range(j + 1, J_lowpass_adj + 1):
            # print(f'\tjprime = {jprime}')
            jprime_conv = np.einsum(
                'k,nk->n',
                wavelet_filters[:, jprime],
                kfourier_series_absWjf
            )
            Wjprime_convs.append(jprime_conv)
    
    Wjf_2 = np.array(Wjprime_convs)
    if verbosity > 0:
        print(f'\tWjf_2.shape: {Wjf_2.shape}') # (J(J+1)/2, n)

    return Wjf_0, Wjf_1, Wjf_2


def get_spectral_wavelets_scat_moms_dict(
    L: scipy.sparse.spmatrix | np.ndarray,
    eigenvals: np.ndarray,
    eigenvecs: np.ndarray,
    Pnf: np.ndarray,
    J: int = 4,
    Q: int = 4,
    include_low_pass: bool = True,
    verbosity: int = 0
) -> Dict[str, np.ndarray]:
    """
    Generates a dictionary of geometric scattering 
    moments, using spectral wavelet filters.
    
    Args:
        L: (sparse) array of the graph Laplacian.
        eigenvals: array of eigenvalues of L; shape
            (k, ).
        eigenvecs: array of eigenvectors of L;
            shape (n, k)
        Pnf: array of normalized-evaluated values
            of a function defined on the manifold.
        J: max wavelet filter order.
        Q: max scattering moment.
        include_low_pass: whether to include the
            'lowpass' ($e^{2^{J}}$) wavelet filter.
        verbosity: integer controlling print
            output as this function executes.
    Returns:
        Dictionary of scattering moment values (arrays),
        keyed by moment (int).
    """
    wavelet_filters = spectral_wavelets(
        eigenvals=eigenvals, 
        J=J,
        include_low_pass=include_low_pass
    )
    
    Wjf_0, Wjf_1, Wjf_2 = wavelet_spectral_convolutions(
        Pnf=Pnf,
        eigenvals=eigenvals,
        wavelet_filters=wavelet_filters,
        J=J,
        include_low_pass=include_low_pass,
        eigenvectors=eigenvecs,
        verbosity=verbosity
    )

    # create dict of spectral scattering moments 
    # note: diff. orders have diff. numbers of moments / array shapes
    spectral_sm_dict = {
        i: scattering_moments(Wjf, Q) \
        for i, Wjf in enumerate((Wjf_0, Wjf_1, Wjf_2))
    }
    return spectral_sm_dict



"""
lazy random walk (diffusion) wavelets

P is the LRW matrix: $P \approx e^{-L}$, 
and so $P^t \approx e^{-t L}$.
"""

def fast_lazy_rw_wavelet_filtrations(
    P: scipy.sparse.spmatrix | torch.Tensor,
    x: np.ndarray | torch.Tensor,
    J: int = 4,
    include_lowpass: bool = True,
    rebatched_dense_3d_size: Tuple[int] | torch.Size = None
) -> np.ndarray | torch.Tensor:
    r"""
    Faster (sparser) implementation of lazy random
    walk ('P') matrix 1st-order wavelet filtrations 
    on a signal vector x. Skips computing increasingly
    dense powers of P, by these steps:
    
    1. Compute $y_t = P^t x$ recursively via $y_t = P y_{t-1}$,
       (only using P, and not its powers, which grow denser).
    2. Subtract $y_{2^{j-1}} - y_{2^{j}}$.
    3. The result is $W_j x = (P^{2^{j-1}} - P^{2^j}) x$.
    (Thus, we never form the matrices P^4 etc, which get denser
    with as the power increases.)

    Args:
        P: sparse lazy random walk matrix, shape 
            (n_nodes, n_nodes) [single P] or (batch_size * 
            n_nodes, batch_size * n_nodes) [batched block-
            diagonal Ps].
        x: vector (e.g. of normalized sampled function values 
            on the manifold), shape (n_nodes, ) [single x], or
            (batch_size * n_channels, batch_size * n_nodes)
            [batched block-diagonal xs].
        J: max wavelet filter order.
        include_lowpass: whether to include the 'lowpass'
            wavelet filtration, $P^{2^J} x$.
        rebatched_dense_indices: array of COO (coordinates) of
            indices for reshaping outputs back into a dense 3-d
            batch tensor, if x has been sparse block-diagonalized.
    Returns:
        Array of filtered vectors, shape (J+2, n) [if including
        lowpass; else (J+1, n)].
    """
    powers_to_save = 2 ** np.arange(J + 1)
    # print('x.shape:', x.shape)
    sbd_tensors = False

    # single P and x (np array x / scipy.sparse P)
    if isinstance(x, np.ndarray):
        stack = np.stack
        concat = np.concatenate

        # first entry in ys is P^0 x = I x = x
        # need first entry to be 2d, (n, 1) vector to prevent broadcasting
        # in subtraction step below
        if x.ndim == 1:
            y_t = np.expand_dims(x.copy(), axis=1)
            ys = [np.expand_dims(x, axis=1)]
        elif x.ndim == 2:
            y_t = x.copy()
            ys = [x]
        else:
            raise Exception(f"'x' has incorrect ndim={x.ndim}")
        
        for j in range(1, max(powers_to_save) + 1):
            # sparse matrix @ dense vector mult -> dense vector
            # print('P.shape:', P.shape)
            # print('y_t.shape:', y_t.shape)
            y_t = P @ y_t
            if j in powers_to_save:
                ys.append(y_t)
        # if include_lowpass:
        #     lowpass_Wjx = np.expand_dims(ys[-1], axis=0)
    
    # torch tensors: single P, or batched Ps in 2d sparse block-diagonals
    elif torch.is_tensor(P) and torch.is_tensor(x):
        stack = torch.stack
        concat = torch.concatenate

        if x.dim() == 1:
            y_t = x.detach().clone().unsqueeze(dim=1)
            ys = [x.unsqueeze(dim=1)]
        elif x.dim() == 2:
            y_t = x.detach().clone()
            ys = [x]
        else:
            raise Exception(f"'x' has incorrect ndim={x.ndim}")
        
        # batched sparse block-diagonal Ps and xs
        if x.is_sparse and rebatched_dense_3d_size is not None:
            # if xs are sparse, they have been sparse
            # block-diagonalized into 2d; the order of 2 sparse 
            # matrices must be switched for multiplication;
            # and the sparse 2d product matrix must be reconstructed
            # into dense 3d from coo indices (since torch.reshape is 
            # not implemented for sparse tensors)
            sbd_tensors = True
            
            for j in range(1, max(powers_to_save) + 1):
                # multiply 2d sparse block-diag. tensors
                y_t = torch.sparse.mm(y_t, P)
                if j in powers_to_save:
                    ys.append(y_t) # list of tensors w/ 3d shape
            # if include_lowpass:
            #     lowpass_Wjx = ys[-1]

        # single P and x tensors (not batched in block-diag. matrices)
        else:
            y_t = y_t.unsqueeze(dim=1)
            for j in range(1, max(powers_to_save) + 1):
                # sparse matrix @ dense vector -> dense vector
                y_t = torch.sparse.mm(P, y_t)
                if j in powers_to_save:
                    ys.append(y_t.squeeze())
            # if include_lowpass:
            #     lowpass_Wjx = ys[-1].unsqueeze(dim=0)
             
    # print(ys)
    # compute final wavelet filtrations:
    # subtract adjacent saved [P^{2^j} x]s, store in list
    Wjxs = [ys[j - 1] - ys[j] for j in range(1, J + 2)]
    if include_lowpass:
        Wjxs.append(ys[-1])

    # shapes = [Wjx.shape for Wjx in Wjxs]
    # print(shapes)
        
    if not sbd_tensors:
        Wjxs = stack(Wjxs) # dim=0 / axis=0 [default]
        # if include_lowpass:
        #     Wjxs = concat((Wjxs, lowpass_Wjx), axis=0)

    # if dealing with batched block-diag. tensors, transform
    # resulting filtrations (Wjxs) into a dense (3+1)-d tensor,
    # stacking in new dim (for new, multiple filtrations for each x vector)
    # at index 2; want shape = (batch_size, n_channels, n_filters, n)
    if sbd_tensors:
        
        rebatched_dense_3d_indices = du.get_rebatched_3d_dense_indices(
            rebatched_dense_3d_size
        )
        
        Wjxs = torch.stack([
            torch.sparse_coo_tensor(
                indices=rebatched_dense_3d_indices,
                values=Wjx._values(), 
                size=rebatched_dense_3d_size, 
                requires_grad=False
            ).to_dense() \
            for Wjx in Wjxs
        ], dim=2)
        
    return Wjxs


def fast_second_order_lazy_rw_wavelet_filtrations(
    P: scipy.sparse.spmatrix,
    Wjxs: np.ndarray,
    J: int = 4,
    include_lowpass: bool = True
) -> np.ndarray:
    r"""
    Computes second-order wavelet filtrations,
    defined as:
    $W_{j'} | W_j x |$ for $j < j' \leq J$
    Implemented for scipy sparse P matrices and
    numpy array filtered xs (Wjxs).
    
    Args:
        P: sparse lazy random walk matrix, shape 
            (n_nodes, n_nodes).
        Wjxs: array of first-order filtrations, each of
        shape (J+2, n) if including lowpass.
        J: max wavelet filter order.
        include_lowpass: whether to include the 'lowpass'
            wavelet filtration, $P^{2^J} x.$
    Returns:
        Array of filtered vectors, shape ((J+1)(J+2)/2, n) [if 
        including lowpass; else (J(J+1)/2, n)].
    """
    Wjprimexs = []
    absWjxs_j_end = -1 if include_lowpass else J
    
    # take modulus of all first-order filtered xs
    absWjxs = np.abs(Wjxs)

    # for all 1st-order filtered xs
    for j, absWjx in enumerate(absWjxs[:absWjxs_j_end]):
        # print(f'j = {j}')
    
        # apply all Wj filters again to |Wjx|, but
        # only keep Wj' results where j' > j;
        # this is efficient because we have to do the entire
        # recursion up to P^J for each |Wjx| (each 'new x'), but we
        # only keep features where smoother wavelets (with higher
        # powers of P) are applied to Wjxs from less-smooth wavelets 
        # with lower powers of P (again, where j' > j) 
        Wjprimex = fast_lazy_rw_wavelet_filtrations(
            P, 
            absWjx, 
            J, 
            include_lowpass
        )[(j + 1):]
        # print(Wjprimex.shape)
        Wjprimexs.append(Wjprimex)

    if isinstance(Wjprimexs[0], np.ndarray):
        return np.concatenate(Wjprimexs, axis=0)
    elif torch.is_tensor(Wjprimexs[0]):
        return torch.cat(Wjprimexs, dim=0)


def get_P_wavelets_scat_moms_dict(
    P: scipy.sparse.spmatrix | np.ndarray,
    Pnf: np.ndarray,
    J: int = 4,
    Q: int = 4,
    include_lowpass: bool = True,
    verbosity: int = 0
) -> Dict[str, np.ndarray]:
    r"""
    Generates a dictionary of 0th, 1st-, and 2nd-
    order geometric scattering moments, using 'P' 
    (lazy random walk) wavelet filters.
    
    Args:
        P: (sparse) array of the lazy random walk
            'P' matrix.
        Pnf: array of normalized-evaluated values
            of a function defined on the manifold.
        J: max wavelet filter order.
        Q: max scattering moment.
        include_low_pass: whether to include the
            'lowpass' ($e^{2^{J}}$) wavelet filter.
        verbosity: integer controlling print
            output as this function executes.
    Returns:
        Dictionary of scattering moment values (arrays),
        keyed by moment (int).
    """
    # P_wavelets = lazy_rw_wavelets(
    #     P=P, 
    #     J=J,
    #     include_low_pass=include_low_pass
    # ) # shape (J+1, N_PTS_ON_MANIFOLD, N_PTS_ON_MANIFOLD)
    
    Wjf_1 = fast_lazy_rw_wavelet_filtrations(
        P=P,
        x=Pnf,
        J=J,
        include_lowpass=include_lowpass
    )

    Wjf_2 = fast_second_order_lazy_rw_wavelet_filtrations(
        P=P,
        Wjxs=Wjf_1,
        J=J,
        include_lowpass=include_lowpass
    )
    
    # create dict of P-wavelet scattering moments 
    # 0th-order are just SMs of Pnf itself
    P_wavelets_sm_dict = {
        i: scattering_moments(Wjf, Q) \
        for i, Wjf in enumerate((Pnf, Wjf_1, Wjf_2))
    }
    return P_wavelets_sm_dict

