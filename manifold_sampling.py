"""
Methods for manifold sampling and functions:
circle, sphere, torus (to varying degrees of
development).
"""
import pickle
import numpy as np
from numpy.random import RandomState
from typing import (
    Callable,
    Tuple,
    List,
    Optional
)
from scipy.special import sph_harm
# scipy sparse matrix type hinting
from scipy.sparse.base import spmatrix


"""
Evaluation/sampling of a function on manifold
"""

def rand_bandlimit_evec_signal(
    eigenvectors: np.ndarray,
    max_evec_i: int,
    random_state: np.random.RandomState,
    coef_sample_dist_key: str = 'uniform',
    coef_save_dir: Optional[str] = None,
    coef_save_picklename: str = 'coefs.pkl'
) -> np.ndarray:
    """
    Creates a bandlimited 1-channel signal
    from a random linear combination of
    kappa eigenvectors.

    Args:
        eigenvectors: array of shape (N, k).
        max_evec_i: max eigenvector index (starting
            from index 0) to combine to create signal,
            combining eigenvectors 1...max_evec_i
            inclusive. We exclude eigenvector 0
            since it is constant/trivial from the
            graph Laplacian.
        random_state: random number generator 
            for sampling coefficients.
        coef_sample_dist_key: 'uniform' or
            'normal' distribution, for sampling
            the linear combination coefficients
            from.
        coef_save_dir: if not None, the path to which
            a list of the randomly-sampled coefficient
            values will be pickled.
    Returns:
        Array of signal values, shape (N, ).
    """
    # sample kappa coefficients
    # 'unif' from [-1., 1.]
    if 'unif' in coef_sample_dist_key.lower():
        coefs = random_state.uniform(
            low=-1., 
            high=1., 
            size=max_evec_i
        )

    # 'norm' from N(0, 1)
    elif 'norm' in coef_sample_dist_key.lower():
        coefs = random_state.normal(
            loc=0., 
            scale=1., 
            size=max_evec_i
        )
    else:
        raise NotImplementedError(
            f"Coefficient sampling distribution not implemented."
        )
    # print(coefs)
    # create linear combination of kappa eigenvectors
    bandlimit_signal = np.einsum(
        'k,Nk->N',
        coefs,
        eigenvectors[:, 1:(max_evec_i + 1)]
    )

    # optional: save randomly-sampled coefs
    if coef_save_dir is not None:
        coef_save_path = coef_save_dir + f"/{coef_save_picklename}"
        with open(coef_save_path, "wb") as f:
            pickle.dump(coefs.tolist(), f, protocol=pickle.HIGHEST_PROTOCOL)  
    
    return bandlimit_signal


def norm_eval_op(
    fx: np.ndarray | spmatrix, 
    n: int
) -> np.ndarray | spmatrix:
    r"""
    Normalized evaluation operator, for
    pointwise evaluation of sampled
    function values from points on a 
    manifold.
    
    Necessary to make sure that the eigen-
    vectors of the graph Laplacian are
    approximately equal to the eigenfunctions
    on the manifold: the $\sqrt(n)$ denominator
    comes from the expected value of pointwise
    function evaluations.

    Args:
        fx: array of sampled values of a 
            function on a manifold, shape (n, ).
        n: sample size.
    Returns:
        Array of normalized function values.
    """
    return fx / np.sqrt(n)
    

def get_manifold_coords_as_fn_vals(
    coords: np.ndarray | spmatrix,
    axes: str | Tuple[int],
    norm_evaluate: bool = True
) -> np.ndarray | spmatrix:
    """
    Retrieves (a subset) the extrinsinc
    coordinates of points sampled from a 
    manifold, to use as function values,
    if geometry is to also be construed as
    signal.

    Args:
        coords: ambient space coordinates
            of points on a manifold. Shape
            (n, ambient_dim). If sparse,
            'csc_matrix' is best for column-wise
            slicing here.
        axes: which coordinate axes to include
            as function values.
        norm_evaluate: optional flag whether to
            apply the normalization-evaluation
            operator to the values.
    Returns:
        Array of (optionally normalize-evaluated)
        function values, of shape (n_points, n_axes). 
    """
    if axes == 'all':
        fx = coords
    else:
        fx = coords[:, axes]

    if norm_evaluate:
        norm_fx = norm_eval_op(
            fx=fx, 
            n=coords.shape[0]
        )
    return norm_fx


def gaussian_noise(
    d: int, 
    n: int,
    seed: int,
    var_constant: Optional[float] = 1.
) -> np.ndarray:
    r"""
    Produces a sample of multivariate Gaussian noise
    of size $n$ from:
    $\mathcal{N}(0, \frac{1}{\sqrt(d)}I_{d \times d})$,
    where $d$ is the dimensionality.

    Args:
        d: dimension of sample.
        n: number of samples.
        seed: seed integer to pass to numpy random
            number generator.
        var_constant: scalar adjustment to error
            variance. 0. < var_constant < 1. produce
            smaller variance; var_constant > 1. 
            produces larger variance.
    Returns:
        Array of noise samples of shape (n, d), which
        can be added to a 'signal' array.
    """
    if var_constant is None:
        var_constant = 1.
    rng = np.random.default_rng(seed)
    mean = np.zeros(d)
    cov = np.diag(np.repeat(var_constant / np.sqrt(d), d))
    noise_sample = rng.multivariate_normal(mean, cov, n)
    return noise_sample


def n_dim_ball_vol(d: int, r: float = 1.) -> float:
    """
    Computes the volume of a d-dimensional
    sphere/ball. Note that the volume vanishes
    as d gets large.

    Ref:
    https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    https://davidegerosa.com/nsphere/
    
    Args:
        d: dimension of the ball.
        r: radius of the ball.
    Returns:
        Volume of d-dimensional ball.
    """
    unit_ball_vols = {
        0: 1.,
        1: 2.,
        2: 3.141592653589793, # np.pi,
        3: 4.1887902047863905, # (4. / 3.) * np.pi,
        4: 4.934802200544679, # np.pi ** 2 / 2.,
        5: 5.263789013914324, # (8. / 15.) * np.pi ** 2,
        6: 5.167712780049969, # np.pi ** 3 / 6.,
        7: 4.724765970331401, # (16. / 105.) * np.pi ** 3,
        8: 4.058712126416768, # np.pi ** 4 / 24.,
        9: 3.2985089027387064, # (32. / 945.) * np.pi ** 4,
        10: 2.550164039877345, # np.pi ** 5 / 120.,
        11: 1.8841038793898999, # (64. / 10395.) * np.pi ** 5,
        12: 1.3352627688545893, # np.pi ** 6 / 720.,
        13: 0.910628754783283, # (128. / 135135.) * np.pi ** 6,
        14: 0.5992645293207919, # np.pi ** 7 / 5040.,
        15: 0.38144328082330436 # (256. / 2027025.) * np.pi ** 7
    }
    if d <= 15:
        vol = unit_ball_vols[d]
    else:
        from scipy.special import gamma
        vol = (np.pi ** (d / 2.) / gamma(d / 2 + 1))
    # if ball has non-unit radius, multiply by r^d
    if r != 1.:
        vol *= (r ** d)
    return vol
    

def convert_complex_to_R2(
    c_nums: np.ndarray
) -> np.ndarray:
    """
    Converts an 1-d array of complex numbers
    to a 2-d array of points on the real plane.

    Args:
        c_nums: 1-d array of complex numbers.

    Returns:
        2-d array of points on the real plane,
        of shape (n, 2).
    """
    return np.array([
        np.array([c.real, c.imag]) for c in c_nums
    ])
    


"""
Non-uniform density sampling
"""
def get_polar_nonunif_p(
    v: np.ndarray, 
    mode: str = 'oversample',
    alpha: float = 1.,
    squared: bool = False
) -> np.ndarray:
    r"""
    Generates a linearly nonuniform/biased probability
    density based on the poles (max/min) of a
    vector v. For example: v could be the (symmetric)
    z-coordinates of a sphere, for which the default 
    settings of this function would generate a density the 
    oversamples towards the z-axis poles up to a ratio of 2.

    Args:
        v: vector of values, of length n.
        mode: if 'oversample', sampling
            is denser towards the poles of v;
            if 'undersample,' sampling is
            less dense towards the poles.
        alpha: constant which varies the overall
            density ratio, alpha \in (0., 2.)
        squared: whether to square the relationship
            between probability and extremity along
            v: results in a slower probability drop-
            off/ramp-up from the poles.
    Returns:
        Vector of nonuniform probabilities, of
        length n.
    """
    y = np.abs(v) - np.abs(np.min(v))
    if squared:
        y = -(y ** 2)
        
    if mode == 'undersample':
        y = np.max(v) - alpha * y
    elif mode == 'oversample':
        y = np.max(v) + (alpha / 2) * y

    # normalize to sum to 1. (probability density).
    p = y / np.sum(y)
    return p


def get_linear_alpha_from_density_ratio(
    density_ratio: float
) -> float:
    """
    For the above function 'get_polar_nonunif_p()',
    with 'squared=False' (i.e. a linear nonuniformity),
    calculate the alpha value corresponding to the
    desired density ratio.

    Args:
        density_ratio: density ratio for nonuniform sampling.
    Returns:
        alpha value, to pass to 'get_polar_nonunif_p()'.
    """
    alpha = 2. * (density_ratio - 1.) / density_ratio
    return alpha


def get_nonunif_p_from_schedule(
    v: np.ndarray, 
    power: float = 0.25, 
    p_schedule: List = None
) -> np.ndarray:
    """
    Generates an array of probabilities for a
    vector v (abs. values; minmax-scaled onto 
    [0, 1]), according to a p-schedule.

    Args:
        v: vector of values.
        power: controls the exponential
            decay of sampling probability
            along the absolute values of v,
            when the formulaic method is used.
        p_schedule: list of 2-tuples, where first
            is lower cutoff of a subset of [0, 1] 
            for rescaled v-values, and second is
            the relative probability mass assigned
            to that subset.
    Returns:
        Vector of probabilities on [0, 1].
    """
    # rescale v to [0, 1]
    v = np.abs(v)
    v_min, v_max = np.min(v), np.max(v)
    v_range = v_max - v_min
    v_rescaled = (v - v_min) / (v_range)

    # calculate probabilities using schedule or method
    # if p_schedule is not None:
    v_rescaled_copy = v_rescaled.copy()
    for i, ps in enumerate(p_schedule):
        key0 = ps[0]
        if i < (len(p_schedule) - 1):
            key1 = p_schedule[i + 1][0]
            mask = (v_rescaled_copy >= key0) & (v_rescaled_copy < key1)
        else:
            mask = (v_rescaled_copy >= key0)
        v_rescaled[mask] = ps[1]
    p = v_rescaled / np.sum(v_rescaled)
    # else: 
    #     # formulaic method: less probability the greater
    #     # the absolute value of v relative to others
    #     v_transform = (1 - v_rescaled) ** power
    #     p = v_transform / np.sum(v_transform)
    
    return p


def get_nonunif_sample_idx(
    n: int,
    p: np.ndarray,
    random_state: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sample indexes for a vector v, 
    given a transform function that defines 
    the sampling density over the values in v.
    
    Args:
        n: size of sample.
        p: vector of probabilities.
        random_state: numpy RandomState object.
    Returns:
        Sampled indexes.
    """
    # draw i.i.d. manifold pt indexes according to 
    # non-uniform z_probs
    nonunif_sample_idx = random_state.choice(
        a=np.arange(len(p)), 
        size=n, 
        replace=True, 
        p=p
    )
    return nonunif_sample_idx


"""
Circle
"""

def unif_circle_thetas(
    n: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Uniformly samples theta (arc angle) values
    on the circle.
    
    Args:
        n: number of points to sample.
        random_state: numpy RandomState object.
    Returns: array of theta angle values,
        on [0, 2*pi] radians.
    """
    thetas = 2. * np.pi * random_state.uniform(
        low=0., 
        high=1., 
        size=n
    )
    return thetas


def apply_circ_harm(thetas: np.ndarray) -> np.ndarray:
    """
    Generates values for a simple linear combination of
    circular harmonic functions.

    Args:
        thetas: array of theta (circle angle) values.
    Returns:
        Array of function output values.
    """
    f_vals = np.cos(thetas) \
        + np.cos(2. * thetas) \
        + np.cos(3. * thetas)
    # f_vals = np.cos(thetas) / np.sqrt(np.pi) \
    #     + 2. * np.cos(2. * thetas) / np.sqrt(np.pi)
    return f_vals


"""
Sphere
"""

def convert_sph_coords_to_eucl(
    r: float, 
    thetas: np.ndarray, 
    psis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts spherical coordinates to Euclidean, 
    for a fixed radius r.

    Args:
        r: radius of sphere.
        thetas: array of azimuthal angle values in [0, 2*pi].
        psis: array of polar angle values in [0, pi].
    Returns:
        3-tuple of x, y, z coordinates.
    """
    a = r * np.sin(psis) # common to x, y
    x = a * np.cos(thetas)
    y = a * np.sin(thetas)
    z = r * np.cos(psis)
    return x, y, z


def unif_sphere_thetas_psis(
    n: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """
    Generates random uniform points lying on a spherical
    manifold, using inverse transform sampling.
        The manifold can be defined in spherical coordinates:
    r (radial distance, r >= 0, fixed/same for all pts
    on manifold); psi (polar angle, 0 to pi rads); theta 
    (azimuth, 0 to 2pi rads).

    Ref: 
    https://mathworld.wolfram.com/SpherePointPicking.html
    
    Args:
        n: number of points to sample.
        random_state: numpy RandomState object.
    Returns:
      Numpy arrays of thetas and psis.
    """
    # thetas: azimuthal coords on [0, 2*pi]
    sph_thetas = 2. * np.pi * random_state.uniform(
        low=0., 
        high=1., 
        size=n
    )

    # psis: polar coordinates on [0, pi]
    sph_psis = np.arccos(
        2. * random_state.uniform(
            low=0., 
            high=1., 
            size=n
        ) - 1.
    )
    return sph_thetas, sph_psis


def get_real_sph_harm(
    m: int, 
    deg: int, 
    phi: float, 
    theta: float
) -> float:
    """
    Returns the real part of a spherical harmonic.
    
    Adapted from:
        https://scipython.com/blog/visualizing-the-real-
        forms-of-the-spherical-harmonics/
        
    Args:
        m: harmonic order.
        deg: harmonic degree (often denoted 'l').
        theta: azimuthal coordinate in [0, 2*pi].
        phi: polar coordinate in [0, pi].
    Returns:
        The spherical harmonic value from the real
        functional form.
    """
    Y = sph_harm(m, deg, theta, phi)
    # cast complex function forms to real function forms
    if m != 0:
        Y_l = Y.imag if m < 0 else Y.real
        Y = np.sqrt(2.) * (-1.)**m * Y_l
    else: # where m == 0
        Y = Y.real
    
    # Linear combination of Y_l,m and Y_l,-m to create the real form.
    # if m < 0:
    #     Y = np.sqrt(2.) * (-1.)**m * Y.imag
    # elif m > 0:
    #     Y = np.sqrt(2.) * (-1.)**m * Y.real
    return Y


def gen_rand_lin_combo_sph_harm_params(
    n_terms: int,
    max_deg: int,
    random_state: np.random.RandomState
) -> dict:
    """
    Spherical harmonic functions are eigenfunctions,
    with eigenvalues l(l + 1) and multiplicity (2l + 1), 
    where 'l' is the degree of the harmonic. They are a
    special case of the Fourier series, for the sphere.

    For our use, keep `max_deg` low, so that the maximum
    number of eigenpairs (kappa) to reconstruct the Fourier 
    coefficients of x (values of a function on a manifold, 
    where this function is a random linear combination of
    spherical harmonics) is also low.

    We also sample only once from an order, which avoids the
    eigenvalue multiplicity in spherical harmonics.

    Args:
        n_terms: number of spherical harmonic functions
        max_deg: maximum degree spherical harmonic.
        random_state: np random number generator state.
    Returns: 
        List of dictionaries containing parameters for a 
        composite spherical harmonics function (a random 
        linear combination of harmonic functions), WITHOUT 
        repeated degrees.
    """
    # draw random function degrees as integers from 1...max_deg
    deg_range = np.arange(1, max_deg + 1)
    rand_degs = random_state.choice(
        a=deg_range,
        size=n_terms,
        replace=False
    )
    # draw random weights from a st. normal
    rand_weights = random_state.randn(n_terms)

    params = [None] * n_terms
    for i in range(n_terms):
        wt = rand_weights[i]
        deg = rand_degs[i]
        # order of sph. harmonic is an integer from -deg to +deg
        m = random_state.randint(low=-deg, high=deg, size=1)
        params[i] = {
            'weight': wt,
            'degree': deg,
            'order': m
        }
    return params


def apply_sph_harm_lc(
    sph_harm_params: List[dict],
    theta: np.ndarray, 
    psi: np.ndarray
) -> np.ndarray:
    """
    Calculates an array of function output values, where the function
    is the composite/linear combination of spherical harmonic
    functions defined in the 'sph_harm_params' argument.
        
    Args:
        sph_harm_params: parameter set for spherical harmonics, 
            generated by 'gen_rand_lin_combo_sph_harm_params'.
        theta: array of azimuthal coordinate values, of length n.
        psi: array of polar coordinate values, of length n.
    Returns:
        Array of function values of shape (n, ).
    """
    num_fns = len(sph_harm_params)
    wtd_Ys = [None] * num_fns
    for i in range(num_fns):
        params = sph_harm_params[i]
        wtd_Ys[i] = params['weight'] * get_real_sph_harm(
            m=params['order'], 
            deg=params['degree'],
            theta=theta,
            phi=psi
        )
    return np.sum(wtd_Ys, axis=0)


"""
Ellipsoid
"""
def unif_ellipsoid_xyz(
    n: int,
    abc: Tuple[float, float, float],
    random_state: np.random.RandomState,
    sphere_oversample_mult: float = 10.
) -> np.ndarray:
    """
    Generates approximately random uniform points lying on 
    the surface of a 2-ellipsoid in 3-d Cartesian space, via 
    rejection sampling.

    Notes: 
    - ellipsoid sampling is approximate: for efficiency, we generate
    a large pool of spherical points and sample according to
    a normalized probability distribution across all spherical
    points; true uniform ellipsoid sampling would use this method, 
    accepting or rejecting one spherical sample at a time.
    - if, e.g., b >> a and c >> a, this sampling method
    will be very inefficient.

    Ref: 
    https://math.stackexchange.com/a/982833
    
    Args:
        n: number of points to sample.
        abc: 3-tuple of a, b, and c ellipsoid
            parameters (constants for x, y, and z,
            respectively).
        random_state: numpy RandomState object.
        initial_sphere_oversample_mult: multiple of n
            to oversample points on the sphere. The 
            larger this multiple is, the better the
            approximation of uniform ellipsoid sampling.
    Returns:
      Numpy array of x, y, z Cartesian coordinates for 
      uniform points on a 2-ellipsoid surface.
    """

    # init empty arrays for ellipsoid pt coords
    ellip_x, ellip_y, ellip_z = [], [], []
    
    # initial sample size on the 2-sphere
    sph_n = sphere_oversample_mult * n

    while True:
        # generate uniform points on 2-sphere
        sph_thetas, sph_psis = unif_sphere_thetas_psis(
            int(sph_n), 
            random_state
        )
        sph_x, sph_y, sph_z = convert_sph_coords_to_eucl(
            r=1.,
            thetas=sph_thetas,
            psis=sph_psis
        )
    
        # reject points with probability mu_xyz / mu_max
        a, b, c = abc
        s_x, s_y, s_z = b * c, a * c, a * b
        mu_xyz = np.sqrt(
            (s_x * sph_x) ** 2 \
            + (s_y * sph_y) ** 2 \
            + (s_z * sph_z) ** 2
        )
        mu_max = max((s_x, s_y, s_z))
        p_reject = mu_xyz / mu_max
        # normalize probability
        p_reject = p_reject / np.sum(p_reject)
    
        # sample point indexes
        ellip_sample_idx = random_state.choice(
            a=np.arange(len(p_reject)), 
            size=n, 
            replace=False, 
            p=p_reject
        )
    
        # transform and add accepted points to coord arrays
        ellip_x.extend(a * sph_x[ellip_sample_idx])
        ellip_y.extend(b * sph_y[ellip_sample_idx])
        ellip_z.extend(c * sph_z[ellip_sample_idx])
    
        # check if enough ellipsoid points have been sampled
        # if so, return; if not, run again (with fewer sphere samples)
        if len(ellip_x) >= n:
            return ellip_x[:n], ellip_y[:n], ellip_z[:n]
        else:
            sph_n = sphere_oversample_mult * (n - len(ellip_x))
        

def ellipsoid_max_min_curv(
    abc: Tuple[float, float, float]
) -> Tuple[float, float]:
    r"""
    If $a \geq b \geq c > 0$, then:
    Max curvature = $(a / bc)^2$, and
    min curvature = $(c / ab)^2$.

    Ref:
    https://math.stackexchange.com/a/1484866

    Args:
        abc: 3-tuple of 'stretch' coefficients
        for x, y, and z coordinates in a
        2-ellipsoid.
    Returns:
        2-tuple of min and max curvature.
    """
    # max_abc_i, min_abc_i = np.argmax(abc), np.argmin(abc)
    # other_abc_i = 3 - max_abc_i - min_abc_i
    # max_abc, min_abc, other_abc = (
    #     abc[max_abc_i], 
    #     abc[min_abc_i],
    #     abc[other_abc_i]
    # )
    # max_curv = (max_abc / (min_abc * other_abc)) ** 2
    # min_curv = (min_abc / (max_abc * other_abc)) ** 2
    
    curvs = (
        (abc[0] / (abc[1] * abc[2])) ** 2,
        (abc[1] / (abc[0] * abc[2])) ** 2,
        (abc[2] / (abc[0] * abc[1])) ** 2,
    )
    min_curv = min(curvs)
    max_curv = max(curvs)
    if min_curv > max_curv:
        print(f'Error: min curvature > max! abc = {abc}')
    return min_curv, max_curv


def generate_ellipsoid_manifold_objs(
    args
) -> List[dict]:
    """
    Generates ellipsoid manifold objects:
    coordinate arrays, target values, abc values, 
    etc.
        
    Args:
        args: Args dataclass instance, containing
        arguments for dataset creation.
    Returns:
        A list of dictionaries of objects, one 
        per manifold.
    """
    manifolds_dictl = [None] * args.N_MANIFOLDS
    
    # init RandomStates
    abc_rs = np.random.RandomState(seed=args.ABC_SEED)
    sampling_rs = np.random.RandomState(seed=args.POINT_SAMPLING_SEED)
    
    # random axis a, b, c (ellipsoid stretch parameters) sets
    abcs = np.stack((
        abc_rs.uniform(low=args.MIN_ABC, high=args.MAX_ABC, size=(args.N_MANIFOLDS,)),
        abc_rs.uniform(low=args.MIN_ABC, high=args.MAX_ABC, size=(args.N_MANIFOLDS,)),
        abc_rs.uniform(low=args.MIN_ABC, high=args.MAX_ABC, size=(args.N_MANIFOLDS,))
    ), axis=-1)
    
    # create 2-ellipsoid manifolds from abc value sets
    for i, abc in enumerate(abcs):
        manifolds_dictl[i] = {}

        # sample unif pts on ellipsoid
        manifold = np.stack(
            unif_ellipsoid_xyz(
                n=args.N_PTS_ON_MANIFOLD, 
                abc=abc,
                random_state=sampling_rs
            ), 
            axis=-1
        )
        # print(f'manifold.shape: {manifold.shape}') # (args.N_MANIFOLDS, 3)
        
        min_curv, max_curv = ellipsoid_max_min_curv(abc)
        manifolds_dictl[i]['abc'] = abc
        manifolds_dictl[i]['manifold'] = manifold
        manifolds_dictl[i]['min_curvature'] = min_curv
        manifolds_dictl[i]['max_curvature'] = max_curv
        manifolds_dictl[i]['curvature_ratio'] = max_curv / min_curv

    return manifolds_dictl
    

"""
Torus
"""

def unif_torus_thetas_sample(
    n: int, 
    r: float, 
    R: float,
    random_state: np.random.RandomState,
    overdraw_ratio: float = 2.1
) -> np.ndarray:
    """
    Generates a list of uniformly sampled theta values for a
    2-d torus in 3-d space, using a rejection sampling algorithm.
    The torus manifold is parameterized by theta (angle around the
    tube circle) and eta (angle about the z axis).
        Thetas have the support [0, 2pi). Etas constrain theta values,
    but can be sampled independently and uniformly across their 
    support [0, 2pi). If they didn't, naively uniformly-sampled thetas 
    would be oversampled in regions of high curvature.
        Ref: https://arxiv.org/abs/1206.6913v1, esp. p. 5.
    
    Args:
        n: number of points to sample.
        r: radius of the torus tube.
        R: radius from center of hole to center of tube.
        random_state: numpy RandomState object.
        overdraw_ratio: multiple to overdraw, since
            rejection sampling leaves fewer accepted samples.
    Returns:
      Numpy array of valid theta samples.
    """
    final_thetas = []

    # rejection sampling needs random num. of iterations
    # since it's random how many samples will be in acceptance region
    while len(final_thetas) < n:
        draw_n = int((n - len(final_thetas)) * overdraw_ratio)
        print('draw_n:', draw_n)
        
        # (over)sample uniformly from 2-d Lebesgue measure box of 
        # (theta, eta) supports
        unif_thetas = random_state.uniform(
            low=0, 
            high=(2 * np.pi), 
            size=draw_n
        )
        unif_etas = random_state.uniform(
            low=0, 
            high=(1 / np.pi), 
            size=draw_n
        )
        
        # rejection sampling: only keep theta if eta < 1 + (r/R) * cos(theta)
        reject_f = (1.0 + (r / R) * np.cos(unif_thetas)) / (2.0 * np.pi)
        accept_idx = np.where(unif_etas < reject_f)
        accept_thetas = unif_thetas[accept_idx]
        print('len(accept_thetas):', len(accept_thetas))
        final_thetas.extend(accept_thetas)
        
    return np.array(final_thetas[:n])

