from typing import Union, Iterable, Callable, List

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import slogdet
from scipy.stats import (norm as sp_norm, truncnorm as sp_truncnorm)
from scipy.special import digamma, loggamma
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA


KernelType = Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]


LOG2PI = np.log(2 * np.pi)


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
    e_μ = np.atleast_2d(b_e_mu[0, 1: P + 1])
    b_μ = b_e_mu[0, 0]
    b_sqrd_μ = b_μ**2 + b_e_sigma[0, 0]
    e_sqrd_μ = e_μ.T @ e_μ + b_e_sigma[1: P + 1, 1: P + 1]
    etimesb_μ = e_μ * b_μ + b_e_sigma[0, 1: P + 1]
    return b_sqrd_μ, e_sqrd_μ, etimesb_μ


def _plot_distplot(data, name, alpha=0.3, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs['ax']
        del kwargs['ax']
    else:
        ax = plt.figure(figsize=(8, 8)).gca()

    if 'ax2' in kwargs:
        ax2 = kwargs['ax2']
        del kwargs['ax2']
    else:
        ax2 = ax.twinx()

    sns.distplot(data, kde=False, ax=ax, **kwargs)
    sns.kdeplot(data, ax=ax2)
    ax.set_ylabel('Count')
    ax.set_xlabel(f'{name} value')
    ax2.set_ylabel('Density')
    return ax, ax2


# noinspection PyPep8Naming,PyMethodMayBeStatic,PyArgumentList
class BEMKL(BaseEstimator, ClassifierMixin):
    def __init__(self, kernels: Iterable[KernelType],
                 random_state: Union[None, int, np.random.RandomState]=None,
                 alpha_lambda: float=1, beta_lambda: float=1,
                 alpha_gamma: float=1, beta_gamma: float=1,
                 alpha_omega: float=1, beta_omega: float=1, max_iter: int=200,
                 margin: float=1, sigma_g: float=0.1, e_null_thrsh: float=1e-6,
                 a_null_thrsh: float=1e-6, k_norm_type: str='kernel',
                 filter_kernels: bool=True, filter_sv: bool=True,
                 verbose: bool=False, init_vars: dict=None) -> None:
        """
        :param kernels: iterable of kernels used to build the kernel matrix.
                        A kernel is a function k(A, B, *args) that takes a
                        np.ndarray A and a np.ndarray B as its only arguments,
                        and returns a single float, if A and B are 1-D arrays,
                        a 1-D array, if A or B is a 1-D array and the other is
                        a 2-D array, or a 2-D array, if both A and B are 2-D
                        arrays.
        :param random_state: int, RandomState instance or None, optional,
                             default: None
                             If int, random_state is the seed used by the
                             random number generator; If RandomState instance,
                             random_state is the random number generator;
                             If None, the random number generator is the
                             RandomState instance used by np.random.
        :param alpha_lambda: alpha parameter for the sample weights' prior
                             Gamma distribution.
        :param beta_lambda: beta parameter for the sample weights' prior Gamma
                            distribution.
        :param alpha_gamma: alpha parameter for the bias' prior Gamma
                            distribution.
        :param beta_gamma: beta parameter for the bias' prior Gamma
                           distribution.
        :param alpha_omega: alpha parameter for the kernel weights' prior Gamma
                            distribution.
        :param beta_omega: beta parameter for the kernel weights' prior Gamma
                           distribution.
        :param max_iter: Maximum number of iterations of the BEMKL algorithm
                         for a single run.
        :param margin: controls scaling ambiguity, and places a low-density
                       region between two classes.
        :param sigma_g: standard deviation of intermediate representations.
        :param e_null_thrsh: members of the e_μ vector with an absolute value
                             lower than this will be considered zero.
                             Defaults to 1e-6.
        :param a_null_thrsh: members of the a_μ vector with an absolute value
                             lower than this will be considered zero.
                             Defaults to 1e-6.
        :param k_norm_type: one of `none`, `pca`, `frob` or `kernel`, default
                            `kernel`. Type of normalization to apply to the
                            kernel matrices:
                            * `none`: no normalization.
                            * `pca`: PCA whitening.
                            * `frob`: divide each matrix by its Frobenius norm.
                            * `kernel`: set
                               K~_{i, j} = K_{i, j}/ sqrt(K_{i, i}, K_{j, j}).
        :param filter_kernels: whether to eliminate kernels with corresponding
                               null e_μ factor.
        :param filter_sv: whether to eliminate training points with
                          corresponding null a_μ factor.
        :param verbose: whether to print progress messages.

        For gamma priors, you can experiment with three different (alpha, beta)
        values
        (1, 1) => default priors
        (1e-10, 1e+10) => good for obtaining sparsity
        (1e-10, 1e-10) => good for small sample size problems
        """
        self.params: dict = {}
        self.set_params(kernels=kernels, random_state=random_state,
                        alpha_lambda=alpha_lambda, beta_lambda=beta_lambda,
                        alpha_gamma=alpha_gamma, beta_gamma=beta_gamma,
                        alpha_omega=alpha_omega, beta_omega=beta_omega,
                        max_iter=max_iter, margin=margin,
                        sigma_g=sigma_g, e_null_thrsh=e_null_thrsh,
                        a_null_thrsh=a_null_thrsh, k_norm_type=k_norm_type,
                        filter_kernels=filter_kernels, filter_sv=filter_sv,
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
        if 'alpha_lambda' in params:
            self._λ_α = params['alpha_lambda']
        if 'beta_lambda' in params:
            self._λ_β = params['beta_lambda']
        if 'alpha_gamma' in params:
            self._γ_α = params['alpha_gamma']
        if 'beta_gamma' in params:
            self._γ_β = params['beta_gamma']
        if 'alpha_omega' in params:
            self._ω_α = params['alpha_omega']
        if 'beta_omega' in params:
            self._ω_β = params['beta_omega']
        if 'max_iter' in params:
            self._max_iter = params['max_iter']
        if 'margin' in params:
            self._margin = params['margin']
        if 'sigma_g' in params:
            self._σ_g = params['sigma_g']
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
        self.params.update(params)
        return self

    @property
    def kernels(self):
        return self._kernels

    @property
    def random_state(self):
        return self._random_state

    @property
    def λ_α(self):
        return self._λ_α

    @property
    def λ_β(self):
        return self._λ_β

    @property
    def γ_α(self):
        return self._γ_α

    @property
    def γ_β(self):
        return self._γ_β

    @property
    def ω_α(self):
        return self._ω_α

    @property
    def ω_β(self):
        return self._ω_β

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def margin(self):
        return self._margin

    @property
    def σ_g(self):
        return self._σ_g

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
        self.lambda_alpha = (self.λ_α + 0.5) * np.ones(N)
        self.lambda_beta = self.λ_β * np.ones(N)

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
        self.gamma_alpha = self.γ_α + 0.5
        self.gamma_beta = self.γ_β

    def _init_ω(self):
        P = self.P
        self.omega_alpha = (self.ω_α + 0.5) * np.ones(P)
        self.omega_beta = self.ω_β * np.ones(P)

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
        self.lambda_beta = 1 / (1 / self.λ_β + 0.5 * np.diag(a_sqrd_mu))

    def _update_a(self):
        N = self.N
        sigma_g = self.σ_g
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
        sigma_g = self.σ_g
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
        self.gamma_beta = 1 / (1 / self.γ_β + 0.5 * b_sqrd_mu)

    def _update_ω(self):
        e_sqrd_mu = self.e_sqrd_mu
        self.omega_beta = 1 / (1 / self.ω_β + 0.5 * np.diag(e_sqrd_mu))

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

    # noinspection PyUnresolvedReferences
    def _calc_elbo(self):
        N = self.N
        P = self.P
        lambda_alpha = self.lambda_alpha
        lambda_beta = self.lambda_beta
        a_sqrd_mu = self.a_sqrd_mu
        sigma_g = self._σ_g
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
            (self.λ_α - 1) * (digamma(lambda_alpha) + np.log(lambda_beta))
            - lambda_alpha * lambda_beta / self.λ_β
            - loggamma(self.λ_α)
            - self.λ_α * np.log(self.λ_β)
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
            (self.γ_α - 1) * (digamma(gamma_alpha) + np.log(gamma_beta))
            - gamma_alpha * gamma_beta / self.γ_β
            - loggamma(self.γ_α)
            - self.γ_α * np.log(self.γ_β)
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
            (self.ω_α - 1) * (digamma(omega_alpha) + np.log(omega_beta))
            - omega_alpha * omega_beta / self.ω_β
            - loggamma(self.ω_α)
            - self.ω_α * np.log(self.ω_β)
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

        return lb.real, factors

    def fit(self, X_train: np.ndarray, y_train: np.ndarray)\
            -> "BEMKL":
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

        self.bounds: List = [None] * self.max_iter

        self.a_sqrd_mu = _calc_a_sqrd_mu(self.a_mu, self.a_sigma)
        self.G_sqrd_mu, self.KmtimesG_mu =\
            _calc_G_stats(N, P, self.G_mu, self.G_sigma, Km)
        self.b_sqrd_mu, self.e_sqrd_mu, self.etimesb_mu =\
            _calc_b_e_stats(self.b_e_mu, self.b_e_sigma, P)
        for i in range(self.max_iter):
            self._update_λ()
            self._update_a()
            self._update_G()
            self._update_γ()
            self._update_ω()
            self._update_b_e()
            self._update_f()

            self.bounds[i] = self._calc_elbo()
            if self.verbose and (i % self.verbose == 0 or
                                 i == self.max_iter - 1):
                print(f"Iter: {i}. Bound: {self.bounds[i][0]}")

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

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.predict_proba(X_test)
        y_pred = np.argwhere(y_pred >= 0.5)[:, 1]
        y_pred[y_pred == 0] = -1
        return y_pred.astype(int)

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
                self.σ_g**2 +
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
        return _plot_distplot(self.e_mu_orig, r'$e_\mu$', **kwargs)

    def plot_a(self, **kwargs):
        return _plot_distplot(self.a_mu_orig, r'$a_\mu$', **kwargs)

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
