from pprint import pformat
from time import time
from typing import Union, Iterable, Callable, List, Dict  # noqa

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import slogdet
from scipy.stats import (norm as sp_norm, truncnorm as sp_truncnorm)
from scipy.special import digamma, loggamma
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
import tensorflow as tf


KernelType = Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]


def truncnorm(a=np.infty, b=np.infty, loc=0, scale=1):
    a, b = (a - loc) / scale, (b - loc) / scale
    return sp_truncnorm(a=a, b=b, loc=loc, scale=scale)


def logdet(A):
    s, ret_val = slogdet(A)
    if s < 0:
        raise ValueError(f"A must be positive definite")
    return ret_val


def _calc_a_sqrd_mu(a_mu, a_sigma):
    return a_mu.T @ a_mu + a_sigma


# noinspection PyPep8Naming
def _calc_G_stats(N, P, G_mu, G_sigma, Km):
    G_sqrd_μ = G_mu.T @ G_mu + N * G_sigma
    KmtimesG_μ = G_mu.T.reshape(1, N * P) @ Km
    return G_sqrd_μ, KmtimesG_μ


# noinspection PyPep8Naming
def _calc_b_e_stats(b_e_mu, b_e_sigma, P):
    e_μ = b_e_mu[:1, 1: P + 1]
    e_sqrd_μ = e_μ.T @ e_μ + b_e_sigma[1: P + 1, 1: P + 1]

    b_μ = b_e_mu[0, 0]
    b_sqrd_μ = b_μ**2 + b_e_sigma[0, 0]

    etimesb_μ = e_μ * b_μ + b_e_sigma[0, 1: P + 1]

    return b_sqrd_μ, e_sqrd_μ, etimesb_μ


def plot_distplot(data, name, alpha=0.3, ax1=None, ax2=None,
                  skip_n=None, cut_n=None, step_n=None, max_data=None,
                  min_data=None, plothist=True, plotdist=True,
                  **kwargs):
    if ax1 is None:
        ax1 = plt.figure(figsize=(8, 8)).gca()
    if ax2 is None and plotdist and plothist:
        ax2 = ax1.twinx()
    elif ax2 is None and (not plotdist or not plothist):
        ax2 = ax1
    if skip_n is not None:
        data = data[skip_n:]
    if cut_n is not None:
        data = data[:cut_n]
    if step_n is not None:
        data = data[::step_n]
    if max_data is not None:
        data = data[data <= max_data]
    if min_data is not None:
        data = data[data >= min_data]

    if plothist:
        sns.distplot(data, kde=False, ax=ax1, **kwargs)
        ax1.set_ylabel('Count')
    if plotdist:
        sns.kdeplot(data, ax=ax2, **kwargs)
        ax2.set_ylabel('Density')
    ax1.set_xlabel(f'{name} value')
    return ax1, ax2


def create_tf_variable(np_var, name, dtype):
    return tf.Variable(np_var, dtype=dtype, trainable=True, name=name)


def create_tf_constant(np_var, name):
    return tf.constant(np_var, dtype=tf.float64, name=name)


def create_elbo_graph(N, P):
    float_dtype = tf.float64
    placeholders = {}
    variables = {}
    factors = {}
    with tf.device('/device:CPU:0'):
        tf_N = create_tf_constant(N, name='tf_N')
        tf_P = create_tf_constant(P, name='tf_P')
        tf_LOG2PI = create_tf_constant(LOG2PI, name='tf_LOG2PI')

        tf_hyp_lambda_alpha = create_tf_variable(0, 'tf_hyp_lambda_alpha', float_dtype)
        variables['_hyp_lambda_alpha'] = tf_hyp_lambda_alpha
        tf_hyp_lambda_beta = create_tf_variable(0, 'tf_hyp_lambda_beta', float_dtype)
        variables['_hyp_lambda_beta'] = tf_hyp_lambda_beta

        tf_lambda_alpha = tf.placeholder(float_dtype, shape=N, name='tf_lambda_alpha')
        placeholders['lambda_alpha'] = tf_lambda_alpha
        tf_lambda_beta = tf.placeholder(float_dtype, shape=N, name='tf_lambda_beta')
        placeholders['lambda_beta'] = tf_lambda_beta

        tf_hyp_gamma_alpha = create_tf_variable(0, 'tf_hyp_gamma_alpha', float_dtype)
        variables['_hyp_gamma_alpha'] = tf_hyp_gamma_alpha
        tf_hyp_gamma_beta = create_tf_variable(0, 'tf_hyp_gamma_beta', float_dtype)
        variables['_hyp_gamma_beta'] = tf_hyp_gamma_beta

        tf_gamma_alpha = tf.placeholder(float_dtype, name='tf_gamma_alpha')
        placeholders['gamma_alpha'] = tf_gamma_alpha
        tf_gamma_beta = tf.placeholder(float_dtype, name='tf_gamma_beta')
        placeholders['gamma_beta'] = tf_gamma_beta

        tf_hyp_omega_alpha = create_tf_variable(0, 'tf_hyp_omega_alpha', float_dtype)
        variables['_hyp_omega_alpha'] = tf_hyp_omega_alpha
        tf_hyp_omega_beta = create_tf_variable(0, 'tf_hyp_omega_beta', float_dtype)
        variables['_hyp_omega_beta'] = tf_hyp_omega_beta

        tf_omega_alpha = tf.placeholder(float_dtype, shape=P, name='tf_omega_alpha')
        placeholders['omega_alpha'] = tf_omega_alpha
        tf_omega_beta = tf.placeholder(float_dtype, shape=P, name='tf_omega_beta')
        placeholders['omega_beta'] = tf_omega_beta

        tf_sigma_g = create_tf_variable(0, 'tf_sigma_g', float_dtype)
        variables['sigma_g'] = tf_sigma_g

        tf_a_mu = tf.placeholder(float_dtype, shape=(1, N), name='tf_a_mu')
        placeholders['a_mu'] = tf_a_mu
        tf_a_sigma = tf.placeholder(float_dtype, shape=(N, N), name='tf_a_sigma')
        placeholders['a_sigma'] = tf_a_sigma
        tf_a_sqrd_mu = tf.placeholder(float_dtype, shape=(N, N), name='tf_a_sqrd_mu')
        placeholders['a_sqrd_mu'] = tf_a_sqrd_mu

        tf_G_mu = tf.placeholder(float_dtype, shape=(N, P), name='tf_G_mu')
        placeholders['G_mu'] = tf_G_mu
        tf_G_sigma = tf.placeholder(float_dtype, shape=(P, P), name='tf_G_sigma')
        placeholders['G_sigma'] = tf_G_sigma
        tf_G_sqrd_mu = tf.placeholder(float_dtype, shape=(P, P), name='tf_G_sqrd_mu')
        placeholders['G_sqrd_mu'] = tf_G_sqrd_mu

        tf_b_e_mu = tf.placeholder(float_dtype, shape=(1, P + 1), name='tf_b_e_mu')
        placeholders['b_e_mu'] = tf_b_e_mu
        tf_b_e_sigma = tf.placeholder(float_dtype, shape=(P + 1, P + 1), name='tf_b_e_sigma')
        placeholders['b_e_sigma'] = tf_b_e_sigma
        tf_b_sqrd_mu = tf.placeholder(float_dtype, name='tf_b_sqrd_mu')
        placeholders['b_sqrd_mu'] = tf_b_sqrd_mu
        tf_e_sqrd_mu = tf.placeholder(float_dtype, shape=(P, P), name='tf_e_sqrd_mu')
        placeholders['e_sqrd_mu'] = tf_e_sqrd_mu
        tf_etimesb_mu = tf.placeholder(float_dtype, shape=(1, P), name='tf_etimesb_mu')
        placeholders['etimesb_mu'] = tf_etimesb_mu

        tf_f_mu = tf.placeholder(float_dtype, shape=(1, N), name='tf_f_mu')
        placeholders['f_mu'] = tf_f_mu
        tf_f_sigma = tf.placeholder(float_dtype, shape=(1, N), name='tf_f_sigma')
        placeholders['f_sigma'] = tf_f_sigma

        tf_KmKm = tf.placeholder(float_dtype, shape=(N, N), name='tf_KmKm')
        placeholders['KmKm'] = tf_KmKm

        tf_KmtimesG_mu = tf.placeholder(float_dtype, shape=(1, N), name='tf_KmtimesG_mu')
        placeholders['KmtimesG_mu'] = tf_KmtimesG_mu

        tf_lower_bound = tf.placeholder(float_dtype, shape=N, name='tf_lower_bound')
        placeholders['lower_bound'] = tf_lower_bound
        tf_upper_bound = tf.placeholder(float_dtype, shape=N, name='tf_upper_bound')
        placeholders['upper_bound'] = tf_upper_bound

        tf_output = tf_b_e_mu @ tf.transpose(tf.concat((tf.ones((N, 1), dtype=float_dtype), tf_G_mu), axis=1))
        tf_alpha_norm = tf_lower_bound - tf_output
        tf_beta_norm = tf_upper_bound - tf_output
        tf_normalization = (
            tf.contrib.distributions.Normal(0., 1.).cdf(tf.cast(tf_beta_norm, tf.float32)) -
            tf.contrib.distributions.Normal(0., 1.).cdf(tf.cast(tf_alpha_norm, tf.float32))
        )
        tf_normalization = tf.cast(
            tf.where(tf_normalization == 0, tf.ones((1, N), dtype=tf.float32), tf_normalization), float_dtype
        )
        factors['tf_normalization'] = tf_normalization

        tf_lb = tf.Variable(0, name='tf_lb', dtype=float_dtype, trainable=False)
        # log(p(λ))
        tf_log_p_λ = tf.reduce_sum(
            (tf_hyp_lambda_alpha - 1) * (tf.digamma(tf_lambda_alpha) + tf.log(tf_lambda_beta))
            - tf_lambda_alpha * tf_lambda_beta / tf_hyp_lambda_beta
            - tf.lgamma(tf_hyp_lambda_alpha)
            - tf_hyp_lambda_alpha * tf.log(tf_hyp_lambda_beta)
        )
        factors['tf_log_p_λ'] = tf_log_p_λ
        tf_lb += tf_log_p_λ

        # log(p(a | λ))
        tf_log_p_a_λ = (
            - 0.5 * tf.reduce_sum(tf_lambda_alpha * tf_lambda_beta * tf.matrix_diag_part(tf_a_sqrd_mu))
            - 0.5 * tf_N * tf_LOG2PI
            + 0.5 * tf.reduce_sum(tf.digamma(tf_lambda_alpha) + tf.log(tf_lambda_beta))
        )
        factors['tf_log_p_a_λ'] = tf_log_p_a_λ
        tf_lb += tf_log_p_a_λ

        # log(p(G | a, Km))
        tf_log_p_G_a_Km = (
            - 0.5 * tf_sigma_g**(-2) * tf.reduce_sum(tf.matrix_diag_part(tf_G_sqrd_mu))
            + tf_sigma_g**(-2) * tf_a_mu @ tf.transpose(tf_KmtimesG_mu)
            - 0.5 * tf_sigma_g**-2 * tf.reduce_sum(tf_KmKm * tf_a_sqrd_mu)
            - 0.5 * tf_N * tf_P * (tf_LOG2PI + 2 * tf.log(tf_sigma_g))
        )[0, 0]
        factors['tf_log_p_G_a_Km'] = tf_log_p_G_a_Km
        tf_lb += tf_log_p_G_a_Km

        # log(p(γ))
        tf_log_p_γ = (
            (tf_hyp_gamma_alpha - 1) * (tf.digamma(tf_gamma_alpha) + tf.log(tf_gamma_beta))
            - tf_gamma_alpha * tf_gamma_beta / tf_hyp_gamma_beta
            - tf.lgamma(tf_hyp_gamma_alpha)
            - tf_hyp_gamma_alpha * tf.log(tf_hyp_gamma_beta)
        )
        factors['tf_log_p_γ'] = tf_log_p_γ
        tf_lb += tf_log_p_γ

        # log(p(b | γ))
        tf_log_p_b_γ = (
            - 0.5 * tf_gamma_alpha * tf_gamma_beta * tf_b_sqrd_mu
            - 0.5 * (tf_LOG2PI - (tf.digamma(tf_gamma_alpha) + tf.log(tf_gamma_beta)))
        )
        factors['tf_log_p_b_γ'] = tf_log_p_b_γ
        tf_lb += tf_log_p_b_γ

        # log(p(ω))
        tf_log_p_ω = tf.reduce_sum(
            (tf_hyp_omega_alpha - 1) * (tf.digamma(tf_omega_alpha) + tf.log(tf_omega_beta))
            - tf_omega_alpha * tf_omega_beta / tf_hyp_omega_beta
            - tf.lgamma(tf_hyp_omega_alpha)
            - tf_hyp_omega_alpha * tf.log(tf_hyp_omega_beta)
        )
        factors['tf_log_p_ω'] = tf_log_p_ω
        tf_lb += tf_log_p_ω

        # log(p(e | ω))
        tf_log_p_e_ω = (
            - 0.5 * tf.reduce_sum(tf_omega_alpha * tf_omega_beta * tf.matrix_diag_part(tf_e_sqrd_mu))
            - 0.5 * tf_P * tf_LOG2PI
            + 0.5 * tf.reduce_sum(tf.digamma(tf_omega_alpha) + tf.log(tf_omega_beta))
        )
        factors['tf_log_p_e_ω'] = tf_log_p_e_ω
        tf_lb += tf_log_p_e_ω

        # log(p(f | b, e, G))
        tf_log_p_f_b_e_G = (
            - 0.5 * (tf_f_mu @ tf.transpose(tf_f_mu) + tf.reduce_sum(tf_f_sigma))
            + (tf.expand_dims(tf_b_e_mu[0, 1:], 0) @ tf.transpose(tf_G_mu)) @ tf.transpose(tf_f_mu)
            + tf.reduce_sum(tf_b_e_mu[0, 0] * tf_f_mu)
            - 0.5 * tf.reduce_sum(tf_e_sqrd_mu * tf_G_sqrd_mu)
            - tf.reduce_sum(tf_etimesb_mu @ tf.transpose(tf_G_mu))
            - 0.5 * tf_N * tf_b_sqrd_mu
            - 0.5 * tf_N * tf_LOG2PI
        )[0, 0]
        factors['tf_log_p_f_b_e_G'] = tf_log_p_f_b_e_G
        tf_lb += tf_log_p_f_b_e_G

        # log(q(λ))
        tf_log_q_λ = tf.reduce_sum(
            - tf_lambda_alpha
            - tf.log(tf_lambda_beta)
            - tf.lgamma(tf_lambda_alpha)
            - (1 - tf_lambda_alpha) * tf.digamma(tf_lambda_alpha)
        )
        factors['tf_log_q_λ'] = tf_log_q_λ
        tf_lb -= tf_log_q_λ

        # log(q(a))
        tf_log_q_a = (
            - 0.5 * tf_N * (tf_LOG2PI + 1)
            - 0.5 * tf.linalg.slogdet(tf_a_sigma)[1]
        )
        factors['tf_log_q_a'] = tf_log_q_a
        tf_lb -= tf_log_q_a

        # log(q(G))
        tf_log_q_G = (
            - 0.5 * tf_N * tf_P * (tf_LOG2PI + 1)
            - 0.5 * tf_N * tf.linalg.slogdet(tf_G_sigma)[1]
        )
        factors['tf_log_q_G'] = tf_log_q_G
        tf_lb -= tf_log_q_G

        # log(q(γ))
        tf_log_q_γ = (
            - tf_gamma_alpha
            - tf.log(tf_gamma_beta)
            - tf.lgamma(tf_gamma_alpha)
            - (1 - tf_gamma_alpha) * tf.digamma(tf_gamma_alpha)
        )
        factors['tf_log_q_γ'] = tf_log_q_γ
        tf_lb -= tf_log_q_γ

        # log(q(ω))
        tf_log_q_ω = tf.reduce_sum(
            - tf_omega_alpha
            - tf.log(tf_omega_beta)
            - tf.lgamma(tf_omega_alpha)
            - (1 - tf_omega_alpha) * tf.digamma(tf_omega_alpha)
        )
        factors['tf_log_q_ω'] = tf_log_q_ω
        tf_lb -= tf_log_q_ω

        # log(q(b, e))
        tf_log_q_b_e = (
            - 0.5 * (tf_P + 1) * (tf_LOG2PI + 1)
            - 0.5 * tf.linalg.slogdet(tf_b_e_sigma)[1]
        )
        factors['tf_log_q_b_e'] = tf_log_q_b_e
        tf_lb -= tf_log_q_b_e

        # log(q(f))
        tf_log_q_f = tf.reduce_sum(
            - 0.5 * (tf_LOG2PI + tf_f_sigma)
            - tf.log(tf_normalization)
        )
        factors['tf_log_q_f'] = tf_log_q_f
        tf_lb -= tf_log_q_f

    opt = tf.train.AdamOptimizer()
    opt_op = opt.minimize(-tf_lb, var_list=list(variables.values()))

    return opt_op, tf_lb, placeholders, variables, factors


LOG2PI = np.log(2 * np.pi)


# noinspection PyPep8Naming,PyMethodMayBeStatic,PyArgumentList
class BEMKL(BaseEstimator, ClassifierMixin):
    def __init__(self, kernels: Iterable[KernelType],
                 random_state: Union[None, int, np.random.RandomState]=None,
                 hyp_lambda_alpha: float=1, hyp_lambda_beta: float=1,
                 hyp_gamma_alpha: float=1, hyp_gamma_beta: float=1,
                 hyp_omega_alpha: float=1, hyp_omega_beta: float=1, max_iter: int=200,
                 margin: float=1, sigma_g: float=0.1,
                 e_null_thrsh: float=1e-6, a_null_thrsh: float=1e-6,
                 k_norm_type: str='kernel',
                 filter_kernels: bool=True, filter_sv: bool=True,
                 hyperopt_enabled: bool=False, hyperopt_max_iter: int=100, hyperopt_every: int=5,
                 hyperopt_tol: float=1, calculate_bounds: bool=True,
                 verbose: bool=False, init_vars: dict=None) -> None:
        """
        :param kernels: iterable of kernels used to build the kernel matrix. A kernel is a function k(A, B, *args)
                        that takes a np.ndarray A and a np.ndarray B as its only arguments, and returns a single float,
                        if A and B are 1-D arrays, a 1-D array, if A or B is a 1-D array and the other is a 2-D array,
                        or a 2-D array, if both A and B are 2-D arrays.
        :param random_state: default: None. If int, random_state is the seed used by the random number generator.
                             If RandomState instance, random_state is the random number generator. If None, the random
                             number generator is the RandomState instance used by np.random.
        :param hyp_lambda_alpha: alpha parameter for the sample weights' prior Gamma distribution.
        :param hyp_lambda_beta: beta parameter for the sample weights' prior Gamma distribution.
        :param hyp_gamma_alpha: alpha parameter for the bias' prior Gamma distribution.
        :param hyp_gamma_beta: beta parameter for the bias' prior Gamma distribution.
        :param hyp_omega_alpha: alpha parameter for the kernel weights' prior Gamma distribution.
        :param hyp_omega_beta: beta parameter for the kernel weights' prior Gamma distribution.
        :param max_iter: Maximum number of iterations of the BEMKL algorithm for a single run.
        :param margin: controls scaling ambiguity, and places a low-density region between two classes.
        :param sigma_g: standard deviation of intermediate representations.
        :param e_null_thrsh: members of the e_μ vector with an absolute value lower than this will be considered zero.
                             Defaults to 1e-6.
        :param a_null_thrsh: members of the a_μ vector with an absolute value lower than this will be considered zero.
                             Defaults to 1e-6.
        :param k_norm_type: one of `none`, `pca`, `frob` or `kernel`, default `kernel`. Type of normalization to apply
                            to the kernel matrices:
                            * `none`: no normalization.
                            * `pca`: PCA whitening.
                            * `frob`: divide each matrix by its Frobenius norm.
                            * `kernel`: set
                               K~_{i, j} = K_{i, j}/ sqrt(K_{i, i}, K_{j, j}).
        :param filter_kernels: whether to eliminate kernels with corresponding null e_μ factor.
        :param filter_sv: whether to eliminate training points with corresponding null a_μ factor.
        :param filter_sv: whether to eliminate training points with corresponding null a_μ factor.
        :param hyperopt_enabled: whether to try to minimize the ELBO wrt. the model's hyperparameters. If True,
                                     at each iteration in the posterior update loop, `hyperopt_max_iter` rounds
                                     of hyperparameter optimization are performed.
        :param hyperopt_max_iter: number of rounds of hyperparameter optimization to perform.
        :param hyperopt_every: if `hyperopt_enabled` is True, run hyper parameter optimization every
                                   `hyperopt_every` steps of the fit loop.
        :param init_vars: dict with initial values for `a_mu`, `a_sigma`, `G_mu`, `G_sigma`, `f_mu`, and `f_sigma`.
                          If None, these variables are randomly initialized.

        For gamma priors, you can experiment with three different (alpha, beta)
        values
        (1, 1) => default priors
        (1e-10, 1e+10) => good for obtaining sparsity
        (1e-10, 1e-10) => good for small sample size problems
        """
        self.total_time: float = None
        self.params: dict = {}
        self.set_params(kernels=kernels, random_state=random_state,
                        hyp_lambda_alpha=hyp_lambda_alpha, hyp_lambda_beta=hyp_lambda_beta,
                        hyp_gamma_alpha=hyp_gamma_alpha, hyp_gamma_beta=hyp_gamma_beta,
                        hyp_omega_alpha=hyp_omega_alpha, hyp_omega_beta=hyp_omega_beta,
                        max_iter=max_iter, margin=margin,
                        sigma_g=sigma_g, e_null_thrsh=e_null_thrsh,
                        a_null_thrsh=a_null_thrsh, k_norm_type=k_norm_type,
                        filter_kernels=filter_kernels, filter_sv=filter_sv,
                        hyperopt_enabled=hyperopt_enabled,
                        hyperopt_max_iter=hyperopt_max_iter,
                        hyperopt_every=hyperopt_every,
                        hyperopt_tol=hyperopt_tol,
                        calculate_bounds=calculate_bounds,
                        verbose=verbose, init_vars=init_vars)
        # Training variables
        self.X_train: np.ndarray = None
        self.Km_train: np.ndarray = None
        self.Km_train_flat: np.ndarray = None
        self.a_mu: np.ndarray = None
        self.a_sigma: np.ndarray = None
        self.b_e_mu: np.ndarray = None
        self.b_e_sigma: np.ndarray = None
        self.G_mu: np.ndarray = None
        self.G_sigma: np.ndarray = None
        self.f_mu: np.ndarray = None
        self.f_sigma: np.ndarray = None

        self.Km_norms: np.ndarray = None
        self.N: int = None
        self.P: int = None

        self.lambda_alpha: float = None
        self.lambda_beta: float = None
        self.gamma_alpha: float = None
        self.gamma_beta: float = None
        self.omega_alpha: float = None
        self.omega_beta: float = None
        self.KmKm: np.ndarray = None
        self.a_sqrd_mu: np.ndarray = None
        self.G_sqrd_mu: np.ndarray = None
        self.KmtimesG_mu: np.ndarray = None
        self.b_sqrd_mu: np.ndarray = None
        self.e_sqrd_mu: np.ndarray = None
        self.etimesb_mu: np.ndarray = None
        self.lower_bound: np.ndarray = None
        self.upper_bound: np.ndarray = None
        self.normalization: np.ndarray = None
        self.bounds: List = None
        self.opt_op: tf.Operation = None
        self.tf_lb: tf.Tensor = None
        self.tf_placeholders: Dict[str, tf.Tensor] = None
        self.tf_variables: Dict[str, tf.Tensor] = None
        self.tf_factors: Dict[str, tf.Tensor] = None

    def get_params(self, deep=True):
        return self.params

    # noinspection PyAttributeOutsideInit
    def set_params(self, **params):
        if 'kernels' in params:
            self._kernels = np.asarray(params['kernels'])
        if 'random_state' in params:
            self._random_state = params['random_state']
            if not isinstance(self.random_state, np.random.RandomState):
                self._random_state = np.random.RandomState(self.random_state)
        if 'hyp_lambda_alpha' in params:
            self._hyp_lambda_alpha = params['hyp_lambda_alpha']
        if 'hyp_lambda_beta' in params:
            self._hyp_lambda_beta = params['hyp_lambda_beta']
        if 'hyp_gamma_alpha' in params:
            self._hyp_gamma_alpha = params['hyp_gamma_alpha']
        if 'hyp_gamma_beta' in params:
            self._hyp_gamma_beta = params['hyp_gamma_beta']
        if 'hyp_omega_alpha' in params:
            self._hyp_omega_alpha = params['hyp_omega_alpha']
        if 'hyp_omega_beta' in params:
            self._hyp_omega_beta = params['hyp_omega_beta']
        if 'max_iter' in params:
            self._max_iter = params['max_iter']
        if 'margin' in params:
            self._margin = params['margin']
        if 'sigma_g' in params:
            self._sigma_g = params['sigma_g']
        if 'e_null_thrsh' in params:
            self._e_null_thrsh = params['e_null_thrsh']
        if 'a_null_thrsh' in params:
            self._a_null_thrsh = params['a_null_thrsh']
        if 'k_norm_type' in params:
            self._k_norm_type = params['k_norm_type']
        if 'filter_kernels' in params:
            self._filter_kernels = params['filter_kernels']
        if 'filter_sv' in params:
            self._filter_sv = params['filter_sv']
        if 'verbose' in params:
            self._verbose = params['verbose']
        if 'init_vars' in params:
            self._init_vars = params['init_vars']
        if 'hyperopt_enabled' in params:
            self.hyperopt_enabled = params['hyperopt_enabled']
        if 'hyperopt_max_iter' in params:
            self.hyperopt_max_iter = params['hyperopt_max_iter']
        if 'hyperopt_every' in params:
            self.hyperopt_every = params['hyperopt_every']
        if 'hyperopt_tol' in params:
            self.hyperopt_tol = params['hyperopt_tol']
        if 'calculate_bounds' in params:
            self.calculate_bounds = params['calculate_bounds']

        self.params.update(params)
        return self

    @property
    def kernels(self):
        return self._kernels

    @property
    def random_state(self):
        return self._random_state

    @property
    def hyp_lambda_alpha(self):
        return self._hyp_lambda_alpha

    @property
    def hyp_lambda_beta(self):
        return self._hyp_lambda_beta

    @property
    def hyp_gamma_alpha(self):
        return self._hyp_gamma_alpha

    @property
    def hyp_gamma_beta(self):
        return self._hyp_gamma_beta

    @property
    def hyp_omega_alpha(self):
        return self._hyp_omega_alpha

    @property
    def hyp_omega_beta(self):
        return self._hyp_omega_beta

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def margin(self):
        return self._margin

    @property
    def sigma_g(self):
        return self._sigma_g

    @property
    def e_null_thrsh(self):
        return self._e_null_thrsh

    @property
    def a_null_thrsh(self):
        return self._a_null_thrsh

    @property
    def k_norm_type(self):
        return self._k_norm_type

    @property
    def filter_kernels(self):
        return self._filter_kernels

    @property
    def filter_sv(self):
        return self._filter_sv

    @property
    def verbose(self):
        return self._verbose

    @property
    def init_vars(self):
        return self._init_vars

    def _create_kernel_matrix(self, X1, X2, Km_norms=None):
        N1, _ = X1.shape
        N2, _ = X2.shape
        P = len(self.kernels)
        Km = np.zeros((P, N1, N2))
        calc_norm = False
        if Km_norms is None:
            Km_norms = [None] * P
            if self.k_norm_type == 'frob':
                Km_norms = np.ones(P)
            calc_norm = True

        for i, k in enumerate(self.kernels):
            kmi = k(X1, X2)
            kmi_norm = Km_norms[i]
            if self.k_norm_type == 'pca':
                if calc_norm:
                    kmi_norm = PCA(whiten=True).fit(kmi)
                    Km_norms[i] = kmi_norm
                Km[i, :, :] = kmi_norm.transform(kmi)
            elif self.k_norm_type == 'frob':
                if calc_norm:
                    kmi_norm = np.linalg.norm(kmi, ord='fro')
                    Km_norms[i] = kmi_norm
                Km[i, :, :] = kmi / kmi_norm
            elif self.k_norm_type == 'kernel':
                k_norm = np.sqrt(np.diag(kmi))
                Km[i, :, :] = kmi / np.outer(k_norm, k_norm)
        return Km, Km_norms

    def _init_λ(self):
        N = self.N
        self.lambda_alpha = (self.hyp_lambda_alpha + 0.5) * np.ones(N)
        self.lambda_beta = self.hyp_lambda_beta * np.ones(N)

    def _init_a(self):
        N = self.N
        self.a_mu = self.random_state.randn(1, N)
        self.a_sigma = np.eye(N)

    def _init_G(self, y_train):
        P = self.P
        N = self.N
        y_train = np.atleast_2d(np.sign(y_train)).T
        self.G_mu = ((np.abs(self.random_state.randn(N, P)) + self.margin) *
                     y_train)
        self.G_sigma = np.eye(P)

    def _init_γ(self):
        self.gamma_alpha = self.hyp_gamma_alpha + 0.5
        self.gamma_beta = self.hyp_gamma_beta

    def _init_ω(self):
        P = self.P
        self.omega_alpha = (self.hyp_omega_alpha + 0.5) * np.ones(P)
        self.omega_beta = self.hyp_omega_beta * np.ones(P)

    def _init_b_e(self):
        P = self.P
        self.b_e_mu = np.ones((1, P + 1))
        self.b_e_mu[0, 0] = 0
        self.b_e_sigma = np.eye(P + 1)

    def _init_f(self, y_train):
        N = self.N
        self.f_mu = (abs(self.random_state.randn(N)) + self.margin)
        self.f_mu *= np.sign(y_train)
        self.f_sigma = np.ones(N)

    def _calc_KmKm(self, Km):
        P, N, _ = Km.shape
        KmKm = np.zeros((N, N))
        for i in range(P):
            KmKm += Km[i, :, :].T @ Km[i, :, :]
        return KmKm

    def _update_λ(self):
        a_sqrd_mu = self.a_sqrd_mu
        self.lambda_beta = 1 / (1 / self.hyp_lambda_beta + 0.5 * np.diag(a_sqrd_mu))

    def _update_a(self):
        N = self.N
        sigma_g = self.sigma_g
        lambda_alpha = self.lambda_alpha
        lambda_beta = self.lambda_beta
        KmKm = self.KmKm
        KmtimesG_mu = self.KmtimesG_mu

        self.a_sigma = np.linalg.solve(
            np.diag(lambda_alpha * lambda_beta) + KmKm / sigma_g**2,
            np.eye(N)
        )
        self.a_mu = KmtimesG_mu @ self.a_sigma / sigma_g**2
        self.a_sqrd_mu = _calc_a_sqrd_mu(self.a_mu, self.a_sigma)
        # a_sqrd_μ[np.abs(a_sqrd_μ) < self.a_null_thrsh] = 0

    def _update_G(self):
        N = self.N
        P = self.P
        sigma_g = self.sigma_g
        e_sqrd_mu = self.e_sqrd_mu
        Km = self.Km_train_flat
        a_mu = self.a_mu
        f_mu = self.f_mu
        b_e_mu = self.b_e_mu
        etimesb_mu = self.etimesb_mu

        self.G_sigma = np.linalg.solve(
            (np.eye(P) / sigma_g**2 + e_sqrd_mu),
            np.eye(P)
        )
        self.G_mu = ((a_mu @ Km.T).reshape(P, N).T / sigma_g**2 +
                     np.outer(f_mu, b_e_mu[0, 1: P + 1]) -
                     etimesb_mu) @ self.G_sigma
        self.G_sqrd_mu, self.KmtimesG_mu =\
            _calc_G_stats(N, P, self.G_mu, self.G_sigma, Km)

    def _update_γ(self):
        b_sqrd_mu = self.b_sqrd_mu
        self.gamma_beta = 1 / (1 / self.hyp_gamma_beta + 0.5 * b_sqrd_mu)

    def _update_ω(self):
        e_sqrd_mu = self.e_sqrd_mu
        self.omega_beta = 1 / (1 / self.hyp_omega_beta + 0.5 * np.diag(e_sqrd_mu))

    def _update_b_e(self):
        N = self.N
        P = self.P
        gamma_alpha = self.gamma_alpha
        gamma_beta = self.gamma_beta
        G_mu = self.G_mu
        omega_alpha = self.omega_alpha
        omega_beta = self.omega_beta
        G_sqrd_mu = self.G_sqrd_mu
        f_mu = self.f_mu

        self.b_e_sigma = np.linalg.solve(
            np.r_[
                '1,2',
                np.r_['1,2',
                      [gamma_alpha * gamma_beta + N],
                      G_mu.sum(axis=0)].T,
                np.r_['0,2',
                      G_mu.sum(axis=0),
                      np.diag(omega_alpha * omega_beta) + G_sqrd_mu]
            ],
            np.eye(P + 1)
        )
        self.b_e_mu = (
            (np.atleast_2d(f_mu) @ np.r_['1,2', np.ones((N, 1)), G_mu]) @
            self.b_e_sigma
        )
        # b_e_μ[0, 1:][np.abs(b_e_μ[0, 1:]) < self.e_null_thrsh] = 0
        self.b_sqrd_mu, self.e_sqrd_mu, self.etimesb_mu =\
            _calc_b_e_stats(self.b_e_mu, self.b_e_sigma, P)

    def _update_f(self):
        N = self.N
        G_mu = self.G_mu
        b_e_mu = self.b_e_mu
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        output = b_e_mu @ np.r_['1,2', np.ones((N, 1)), G_mu].T
        alpha_norm = lower_bound - output
        beta_norm = upper_bound - output
        normalization = sp_norm.cdf(beta_norm) - sp_norm.cdf(alpha_norm)
        normalization[normalization == 0] = 1
        self.normalization = normalization
        self.f_mu = (
            output +
            (sp_norm.pdf(alpha_norm) - sp_norm.pdf(beta_norm)) /
            normalization
        )
        self.f_sigma = (
            1 +
            (alpha_norm * sp_norm.pdf(alpha_norm) -
             beta_norm * sp_norm.pdf(beta_norm)) / normalization -
            (sp_norm.pdf(alpha_norm) -
             sp_norm.pdf(beta_norm))**2 / normalization**2
        )

    def optimize_elbo(self):
        if self.hyperopt_enabled and self.opt_op is None:
            self.opt_op, self.tf_lb, self.tf_placeholders, self.tf_variables, self.tf_factors =\
                create_elbo_graph(self.N, self.P)
        opt_op = self.opt_op
        placeholders = self.tf_placeholders
        variables = self.tf_variables
        factors = self.tf_factors
        tf_lb = self.tf_lb

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            feed_dict = {
                ph: getattr(self, ph_name)
                for ph_name, ph in placeholders.items()
            }
            for var_name, tf_var in variables.items():
                session.run(tf_var.assign(getattr(self, var_name)))
            for i in range(self.hyperopt_max_iter):
                session.run(opt_op, feed_dict=feed_dict)
                if self.verbose and ((i + 1) % self.verbose == 0):
                    optimal_hyperparams = {
                        var_name: tf_var.eval(feed_dict=feed_dict)
                        for var_name, tf_var in variables.items()
                    }
                    print(f"HyperParamIter: {i + 1}. Bound: {tf_lb.eval(feed_dict=feed_dict)}\n"
                          f"Hyperparams: {pformat(optimal_hyperparams)}")
            lb = tf_lb.eval(feed_dict=feed_dict)
            optimal_hyperparams = {
                var_name: tf_var.eval()
                for var_name, tf_var in variables.items()
            }
            factors = {
                var_name: tf.reduce_mean(tf_var).eval(feed_dict=feed_dict)
                for var_name, tf_var in factors.items()
            }
        return lb, optimal_hyperparams, factors

    def _calc_elbo(self):
        N = self.N
        P = self.P
        lambda_alpha = self.lambda_alpha
        lambda_beta = self.lambda_beta
        a_sqrd_mu = self.a_sqrd_mu
        sigma_g = self.sigma_g
        G_sqrd_mu = self.G_sqrd_mu
        a_mu = self.a_mu
        KmKm = self.KmKm
        KmtimesG_mu = self.KmtimesG_mu
        gamma_alpha = self.gamma_alpha
        gamma_beta = self.gamma_beta
        b_sqrd_mu = self.b_sqrd_mu
        omega_alpha = self.omega_alpha
        omega_beta = self.omega_beta
        e_sqrd_mu = self.e_sqrd_mu
        f_mu = self.f_mu
        f_sigma = self.f_sigma
        b_e_mu = self.b_e_mu
        G_mu = self.G_mu
        etimesb_mu = self.etimesb_mu
        a_sigma = self.a_sigma
        G_sigma = self.G_sigma
        b_e_sigma = self.b_e_sigma
        normalization = self.normalization

        lb = 0
        factors = {}
        # log(p(λ))
        log_p_λ = np.sum(
            (self.hyp_lambda_alpha - 1) * (digamma(lambda_alpha) + np.log(lambda_beta))
            - lambda_alpha * lambda_beta / self.hyp_lambda_beta
            - loggamma(self.hyp_lambda_alpha)
            - self.hyp_lambda_alpha * np.log(self.hyp_lambda_beta)
        ).real
        lb += log_p_λ
        factors['log_p_λ'] = log_p_λ

        # log(p(a | λ))
        log_p_a_λ = (
            - 0.5 * np.sum(lambda_alpha * lambda_beta * np.diag(a_sqrd_mu))
            - 0.5 * N * LOG2PI
            + 0.5 * np.sum(digamma(lambda_alpha) + np.log(lambda_beta))
        )
        lb += log_p_a_λ
        factors['log_p_a_λ'] = log_p_a_λ

        # log(p(G | a, Km))
        log_p_G_a_Km = (
            - 0.5 * sigma_g**(-2) * np.sum(np.diag(G_sqrd_mu))
            + sigma_g**(-2) * a_mu @ KmtimesG_mu.T
            - 0.5 * sigma_g**-2 * np.sum(KmKm * a_sqrd_mu)
            - 0.5 * N * P * (LOG2PI + 2 * np.log(sigma_g))
        )[0, 0]
        lb += log_p_G_a_Km
        factors['log_p_G_a_Km'] = log_p_G_a_Km

        # log(p(γ))
        log_p_γ = (
            (self.hyp_gamma_alpha - 1) * (digamma(gamma_alpha) + np.log(gamma_beta))
            - gamma_alpha * gamma_beta / self.hyp_gamma_beta
            - loggamma(self.hyp_gamma_alpha)
            - self.hyp_gamma_alpha * np.log(self.hyp_gamma_beta)
        ).real
        lb += log_p_γ
        factors['log_p_γ'] = log_p_γ

        # log(p(b | γ))
        log_p_b_γ = (
            - 0.5 * gamma_alpha * gamma_beta * b_sqrd_mu
            - 0.5 * (LOG2PI - (digamma(gamma_alpha) + np.log(gamma_beta)))
        )
        lb += log_p_b_γ
        factors['log_p_b_γ'] = log_p_b_γ

        # log(p(ω))
        log_p_ω = np.sum(
            (self.hyp_omega_alpha - 1) * (digamma(omega_alpha) + np.log(omega_beta))
            - omega_alpha * omega_beta / self.hyp_omega_beta
            - loggamma(self.hyp_omega_alpha)
            - self.hyp_omega_alpha * np.log(self.hyp_omega_beta)
        ).real
        lb += log_p_ω
        factors['log_p_ω'] = log_p_ω

        # log(p(e | ω))
        log_p_e_ω = (
            - 0.5 * np.sum(omega_alpha * omega_beta * np.diag(e_sqrd_mu))
            - 0.5 * P * LOG2PI
            + 0.5 * np.sum(digamma(omega_alpha) + np.log(omega_beta))
        )
        lb += log_p_e_ω
        factors['log_p_e_ω'] = log_p_e_ω

        # log(p(f | b, e, G))
        log_p_f_b_e_G = (
            - 0.5 * (f_mu @ f_mu.T + np.sum(f_sigma))
            + (b_e_mu[0, 1:] @ G_mu.T) @ f_mu.T
            + np.sum(b_e_mu[0, 0] * f_mu)
            - 0.5 * np.sum(e_sqrd_mu * G_sqrd_mu)
            - np.sum(etimesb_mu @ G_mu.T)
            - 0.5 * N * b_sqrd_mu
            - 0.5 * N * LOG2PI
        )[0, 0]
        lb += log_p_f_b_e_G
        factors['log_p_f_b_e_G'] = log_p_f_b_e_G

        # log(q(λ))
        log_q_λ = np.sum(
            - lambda_alpha
            - np.log(lambda_beta)
            - loggamma(lambda_alpha)
            - (1 - lambda_alpha) * digamma(lambda_alpha)
        ).real
        lb -= log_q_λ
        factors['log_q_λ'] = log_q_λ

        # log(q(a))
        log_q_a = (
            - 0.5 * N * (LOG2PI + 1)
            - 0.5 * logdet(a_sigma)
        )
        lb -= log_q_a
        factors['log_q_a'] = log_q_a

        # log(q(G))
        log_q_G = (
            - 0.5 * N * P * (LOG2PI + 1)
            - 0.5 * N * logdet(G_sigma)
        )
        lb -= log_q_G
        factors['log_q_G'] = log_q_G

        # log(q(γ))
        log_q_γ = (
            - gamma_alpha
            - np.log(gamma_beta)
            - loggamma(gamma_alpha)
            - (1 - gamma_alpha) * digamma(gamma_alpha)
        ).real
        lb -= log_q_γ
        factors['log_q_γ'] = log_q_γ

        # log(q(ω))
        log_q_ω = np.sum(
            - omega_alpha
            - np.log(omega_beta)
            - loggamma(omega_alpha)
            - (1 - omega_alpha) * digamma(omega_alpha)
        ).real
        lb -= log_q_ω
        factors['log_q_ω'] = log_q_ω

        # log(q(b, e))
        log_q_b_e = (
            - 0.5 * (P + 1) * (LOG2PI + 1)
            - 0.5 * logdet(b_e_sigma)
        )
        lb -= log_q_b_e
        factors['log_q_b_e'] = log_q_b_e

        # log(q(f))
        log_q_f = np.sum(
            - 0.5 * (LOG2PI + f_sigma)
            - np.log(normalization)
        )
        lb -= log_q_f
        factors['log_q_f'] = log_q_f

        return lb.real, {}, factors

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BEMKL":
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).flatten()
        self.X_train = X_train
        N, _ = X_train.shape
        assert len(y_train) == N
        P = len(self.kernels)
        Km, Km_norms = self._create_kernel_matrix(X_train, X_train)
        self.Km_norms = Km_norms
        self.Km_train = Km
        self.N = N
        self.P = P

        self._init_λ()
        self._init_γ()
        self._init_ω()
        self._init_b_e()
        if not self.init_vars:
            self._init_a()
            self._init_G(y_train)
            self._init_f(y_train)
            self._init_vars = {
                'a_mu': self.a_mu,
                'a_sigma': self.a_sigma,
                'G_mu': self.G_mu,
                'G_sigma': self.G_sigma,
                'f_mu': self.f_mu,
                'f_sigma': self.f_sigma,
            }
        else:
            init_vars = self._init_vars
            self.a_mu = init_vars['a_mu']
            self.a_sigma = init_vars['a_sigma']
            self.G_mu = init_vars['G_mu']
            self.G_sigma = init_vars['G_sigma']
            self.f_mu = init_vars['f_mu']
            self.f_sigma = init_vars['f_sigma']

        self.KmKm = self._calc_KmKm(Km)
        self.Km_train_flat = Km.reshape((P * N, N))

        self.lower_bound = -1e40 * np.ones(N)
        self.lower_bound[y_train > 0] = self.margin
        self.upper_bound = 1e40 * np.ones(N)
        self.upper_bound[y_train < 0] = -self.margin

        if self.calculate_bounds or self.hyperopt_enabled:
            self.bounds: List = [None] * self.max_iter

        self.a_sqrd_mu = _calc_a_sqrd_mu(self.a_mu, self.a_sigma)
        self.G_sqrd_mu, self.KmtimesG_mu =\
            _calc_G_stats(N, P, self.G_mu, self.G_sigma, self.Km_train_flat)
        self.b_sqrd_mu, self.e_sqrd_mu, self.etimesb_mu =\
            _calc_b_e_stats(self.b_e_mu, self.b_e_sigma, P)

        self.total_time = 0
        init_time = time()
        for i in range(self.max_iter):
            self._update_λ()
            self._update_a()
            self._update_G()
            self._update_γ()
            self._update_ω()
            self._update_b_e()
            self._update_f()

            if self.hyperopt_enabled and i > 0 and i % self.hyperopt_every == 0:
                prev_lb, *_ = self._calc_elbo()
                lb, optimal_hyperparams, factors = self.optimize_elbo()
                try:
                    assert lb >= prev_lb or prev_lb - lb < self.hyperopt_tol
                except AssertionError:
                    print(lb, prev_lb)
                    raise
                self.bounds[i] = lb, optimal_hyperparams, factors
                self.set_params(hyp_lambda_alpha=optimal_hyperparams['_hyp_lambda_alpha'],
                                hyp_lambda_beta=optimal_hyperparams['_hyp_lambda_beta'],
                                hyp_gamma_alpha=optimal_hyperparams['_hyp_gamma_alpha'],
                                hyp_gamma_beta=optimal_hyperparams['_hyp_gamma_beta'],
                                hyp_omega_alpha=optimal_hyperparams['_hyp_omega_alpha'],
                                hyp_omega_beta=optimal_hyperparams['_hyp_omega_beta'],
                                sigma_g=optimal_hyperparams['sigma_g'])
                self._init_λ()
                self._init_γ()
                self._init_ω()
            elif self.calculate_bounds:
                self.bounds[i] = self._calc_elbo()

            if self.verbose and ((i + 1) % self.verbose == 0):
                if self.bounds:
                    last_bound = self.bounds[i][0]
                    print(f"Iter: {i + 1}. Bound: {last_bound:.4f}")
                else:
                    print(f"Iter: {i + 1}")

        self.total_time = time() - init_time
        if self.verbose:
            print(f"Iterations total time: {self.total_time:.4f}")

        self.total_kernels = P
        self.total_sv = N

        self.nr_sv_used = len(np.argwhere(
            np.abs(self.a_mu[0]) > self.a_null_thrsh
        ))
        self.nr_kernels_used = len(np.argwhere(
            np.abs(self.b_e_mu[0, 1:]) > self.e_null_thrsh
        ))

        self.b_e_mu_orig = b_e_mu = self.b_e_mu.copy()
        self.b_e_sigma_orig = b_e_sigma = self.b_e_sigma.copy()
        self.kernels_orig = self.kernels.copy()

        if self.filter_kernels:
            e_mu = b_e_mu[0, 1:]
            b_mu = b_e_mu[0, 0]
            mask = np.abs(e_mu) > self.e_null_thrsh
            e_mu = e_mu[mask]
            self._kernels = self.kernels[mask]
            b_e_mu = np.r_[b_mu, e_mu].reshape(1, len(e_mu)+1)
            mask = np.r_[[True], mask]
            b_e_sigma = b_e_sigma[mask][:, mask]

        self.a_mu_orig = a_mu = self.a_mu.copy()
        self.a_sigma_orig = a_sigma = self.a_sigma.copy()
        if self.filter_sv:
            mask = np.abs(a_mu[0]) > self.a_null_thrsh
            a_mu = a_mu[0, mask]
            a_sigma = a_sigma[mask][:, mask]
            self.X_train = self.X_train[mask]

        return self

    def _binarize(self, y_pred_proba: np.ndarray):
        y_pred = np.argwhere(y_pred_proba >= 0.5)[:, 1]
        y_pred[y_pred == 0] = -1
        return y_pred.astype(int)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred_proba = self.predict_proba(X_test)
        return self._binarize(y_pred_proba)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.k_norm_type == 'kernel':
            Ntr, _ = self.X_train.shape
            X = np.r_[self.X_train, X_test]
            Km, _ = self._create_kernel_matrix(X, X)
            Km = Km[:, Ntr:, :Ntr]
        else:
            Km, _ = self._create_kernel_matrix(X_test, self.X_train,
                                               Km_norms=self.Km_norms)

        self.Km_test = Km

        margin = self.margin
        a_mu = self.a_mu
        a_sigma = self.a_sigma
        b_e_mu = self.b_e_mu
        b_e_sigma = self.b_e_sigma
        P, N, _ = Km.shape

        G_mu = np.zeros((N, P))
        G_sigma = np.zeros((N, P))
        for i in range(P):
            G_mu[:, i] = a_mu @ Km[i, :, :].T
            G_sigma[:, i] = (
                self.sigma_g**2 +
                np.diag(Km[i, :, :] @ a_sigma.T @ Km[i, :, :].T)
            )

        G_mu_ext = np.r_['1,2', np.ones((N, 1)), G_mu]
        f_mu = b_e_mu @ G_mu_ext.T
        f_sigma = 1 + np.diag(G_mu_ext @ b_e_sigma @ G_mu_ext.T)
        pos: np.ndarray = 1 - sp_norm.cdf((margin - f_mu) / f_sigma)
        neg: np.ndarray = sp_norm.cdf((-margin - f_mu) / f_sigma)
        y_pred_proba = pos / (pos + neg)
        return np.r_['0,2', 1-y_pred_proba, y_pred_proba].T

    def plot_e(self, **kwargs):
        b_e_mu = self.b_e_mu
        if hasattr(self, 'b_e_mu_orig'):
            b_e_mu = self.b_e_mu_orig
        return plot_distplot(b_e_mu[0, 1:], r'$e_\mu$', **kwargs)

    def plot_a(self, **kwargs):
        a_mu = self.b_e_mu
        if hasattr(self, 'a_mu_orig'):
            a_mu = self.a_mu_orig
        return plot_distplot(a_mu[0], r'$a_\mu$', **kwargs)

    def plot_bounds(self, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs['ax']
            del kwargs['ax']
        else:
            ax = plt.figure(figsize=(8, 8)).gca()
        ax.plot(self.bounds, **kwargs)
        ax.set_ylabel('ELBO')
        ax.set_xlabel('Iteration')
        return ax
