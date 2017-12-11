from typing import Union, Iterable, Callable, NewType

import numpy as np
from numpy.linalg import slogdet
from scipy.stats import (norm as sp_norm, truncnorm as sp_truncnorm)
from scipy.spatial.distance import cdist
from scipy.special import polygamma
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


KernelType = NewType('KernelType',
                     Callable[[np.ndarray, np.ndarray],
                              Union[float, np.ndarray]])


LOG2PI = np.log(2 * np.pi)


def truncnorm(a=np.infty, b=np.infty, loc=0, scale=1):
    a, b = (a - loc) / scale, (b - loc) / scale
    return sp_truncnorm(a=a, b=b, loc=loc, scale=scale)


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


# noinspection PyPep8Naming,PyMethodMayBeStatic,PyArgumentList
class BEMKL(BaseEstimator, ClassifierMixin):
    def __init__(self, kernels: Iterable[KernelType],
                 random_state: Union[None, int, np.random.RandomState]=None,
                 alpha_lambda: float=1, beta_lambda: float=1,
                 alpha_gamma: float=1, beta_gamma: float=1,
                 alpha_omega: float=1, beta_omega: float=1, max_iter: int=200,
                 margin: float=1, sigma_g: float=0.1, e_null_thrsh: float=1e-6,
                 a_null_thrsh: float=1e-6, k_norm_type: str='kernel',
                 verbose: bool=False):
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
        :param k_norm_type: one of `none`, `pca`, `norm` or `kernel`, default
                            `kernel`. Type of normalization to apply to the
                            kernel matrices:
                            * `none`: no normalization.
                            * `pca`: PCA whitening.
                            * `norm`: divide each matrix by its Frobenius norm.
                            *
        :param verbose: whether to print progress messages.

        For gamma priors, you can experiment with three different (alpha, beta)
        values
        (1, 1) => default priors
        (1e-10, 1e+10) => good for obtaining sparsity
        (1e-10, 1e-10) => good for small sample size problems
        """
        self.params = {}
        self.set_params(kernels=kernels, random_state=random_state,
                        alpha_lambda=alpha_lambda, beta_lambda=beta_lambda,
                        alpha_gamma=alpha_gamma, beta_gamma=beta_gamma,
                        alpha_omega=alpha_omega, beta_omega=beta_omega,
                        max_iter=max_iter, margin=margin,
                        sigma_g=sigma_g, e_null_thrsh=e_null_thrsh,
                        a_null_thrsh=a_null_thrsh, k_norm_type=k_norm_type,
                        verbose=verbose)
        self.X_train = None
        self.a_mu = None
        self.a_sigma = None
        self.b_e_mu = None
        self.b_e_sigma = None
        self.sigma_g = None
        self.Km_norms = None

    def get_params(self, deep=True):
        return self.params

    # noinspection PyAttributeOutsideInit
    def set_params(self, **params):
        if 'kernels' in params:
            self.kernels = params['kernels']
        if 'random_state' in params:
            self.random_state = params['random_state']
            if not isinstance(self.random_state, np.random.RandomState):
                self.random_state = np.random.RandomState(self.random_state)
        if 'alpha_lambda' in params:
            self.λ_α = params['alpha_lambda']
        if 'beta_lambda' in params:
            self.λ_β = params['beta_lambda']
        if 'alpha_gamma' in params:
            self.γ_α = params['alpha_gamma']
        if 'beta_gamma' in params:
            self.γ_β = params['beta_gamma']
        if 'alpha_omega' in params:
            self.ω_α = params['alpha_omega']
        if 'beta_omega' in params:
            self.ω_β = params['beta_omega']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
        if 'margin' in params:
            self.margin = params['margin']
        if 'sigma_g' in params:
            self.σ_g = params['sigma_g']
        if 'e_null_thrsh' in params:
            self.e_null_thrsh = params['e_null_thrsh']
        if 'a_null_thrsh' in params:
            self.a_null_thrsh = params['a_null_thrsh']
        if 'k_norm_type' in params:
            self.k_norm_type = params['k_norm_type']
        if 'verbose' in params:
            self.verbose = params['verbose']
        self.params.update(params)
        return self

    def _create_kernel_matrix(self, X1, X2, Km_norms=None):
        N1, _ = X1.shape
        N2, _ = X2.shape
        P = len(self.kernels)
        Km = np.zeros((P, N1, N2))
        calc_norm = False
        if Km_norms is None:
            Km_norms = [None] * P
            # Km_norms = np.ones(P)
            calc_norm = True

        for i, k in enumerate(self.kernels):
            kmi = k(X1, X2)
            kmi_norm = Km_norms[i]
            if calc_norm:
                kmi_norm = PCA(whiten=True).fit(kmi)
                Km_norms[i] = kmi_norm
            #     kmi_norm = np.linalg.norm(kmi, ord='fro')
            #     Km_norms[i] = kmi_norm
            # Km[i, :, :] = kmi / kmi_norm
            Km[i, :, :] = kmi_norm.transform(kmi)
        return Km, Km_norms

    def _init_λ(self, N):
        λ_α = (self.λ_α + 0.5) * np.ones(N)
        λ_β = self.λ_β * np.ones(N)
        return λ_α, λ_β

    def _init_a(self, N):
        a_μ = self.random_state.randn(1, N)
        a_Σ = np.eye(N)
        return a_μ, a_Σ

    def _init_G(self, y_train, P):
        N = len(y_train)
        y_train = np.atleast_2d(np.sign(y_train)).T
        G_μ = (np.abs(self.random_state.randn(N, P)) + self.margin) * y_train
        G_Σ = np.eye(P)
        return G_μ, G_Σ

    def _init_γ(self):
        γ_α = (self.γ_α + 0.5)
        γ_β = self.γ_β
        return γ_α, γ_β

    def _init_ω(self, P):
        ω_α = (self.ω_α + 0.5) * np.ones(P)
        ω_β = self.ω_β * np.ones(P)
        return ω_α, ω_β

    def _init_b_e(self, P):
        b_e_μ = np.ones((1, P + 1))
        b_e_μ[0, 0] = 0
        b_e_Σ = np.eye(P + 1)
        return b_e_μ, b_e_Σ

    def _init_f(self, y_train):
        N = len(y_train)
        f_μ = (abs(self.random_state.randn(N)) + self.margin)
        f_μ *= np.sign(y_train)
        f_σ = np.ones(N)
        return f_μ, f_σ

    def _calc_KmKm(self, Km):
        P, N, _ = Km.shape
        KmKm = np.zeros((N, N))
        for i in range(P):
            KmKm += Km[i, :, :].T @ Km[i, :, :]
        return KmKm

    def _update_λ(self, a_sqrd_mu):
        λ_β = 1 / (1 / self.λ_β + 0.5 * np.diag(a_sqrd_mu))
        return λ_β

    def _update_a(self, lambda_alpha, lambda_beta, KmKm, sigma_g, N,
                  KmtimesG_mu):
        a_Σ = np.linalg.solve(
            (np.diag(lambda_alpha * lambda_beta) + KmKm / sigma_g**2),
            np.eye(N)
        )
        a_μ = KmtimesG_mu @ a_Σ / sigma_g**2
        a_sqrd_μ = _calc_a_sqrd_mu(a_μ, a_Σ)
        a_sqrd_μ[np.abs(a_sqrd_μ) < self.a_null_thrsh] = 0
        return a_Σ, a_μ, a_sqrd_μ

    def _update_G(self, N, P, sigma_g, e_sqrd_mu, Km, a_mu, f_mu, b_e_mu,
                  etimesb_mu):
        G_Σ = np.linalg.solve(
            (np.eye(P) / sigma_g**2 + e_sqrd_mu),
            np.eye(P)
        )
        G_μ = ((a_mu @ Km.T).reshape(P, N).T / sigma_g**2 +
               np.outer(f_mu, b_e_mu[0, 1: P + 1]) -
               etimesb_mu) @ G_Σ
        G_sqrd_μ, KmtimesG_μ = _calc_G_stats(N, P, G_μ, G_Σ, Km)
        return G_Σ, G_μ, G_sqrd_μ, KmtimesG_μ

    def _update_γ(self, b_sqrd_mu):
        γ_β = 1 / (1 / self.γ_β + 0.5 * b_sqrd_mu)
        return γ_β

    def _update_ω(self, e_sqrd_mu):
        ω_β = 1 / (1 / self.ω_β + 0.5 * np.diag(e_sqrd_mu))
        return ω_β

    def _update_b_e(self, N, P, gamma_alpha, gamma_beta, G_mu, omega_alpha,
                    omega_beta, G_sqrd_mu, f_mu):
        b_e_Σ = np.linalg.solve(
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
        b_e_μ = ((np.atleast_2d(f_mu) @ np.r_['1,2', np.ones((N, 1)), G_mu]) @
                 b_e_Σ)
        b_e_μ[0, 1:][np.abs(b_e_μ[0, 1:]) < self.e_null_thrsh] = 0
        b_sqrd_μ, e_sqrd_μ, etimesb_μ =\
            _calc_b_e_stats(b_e_μ, b_e_Σ, P)
        return b_e_Σ, b_e_μ, b_sqrd_μ, e_sqrd_μ, etimesb_μ

    def _update_f(self, N, G_mu, b_e_mu, lower_bound, upper_bound):
        output = b_e_mu @ np.r_['1,2', np.ones((N, 1)), G_mu].T
        alpha_norm = lower_bound - output
        beta_norm = upper_bound - output
        normalization = sp_norm.cdf(beta_norm) - sp_norm.cdf(alpha_norm)
        normalization[normalization == 0] = 1
        f_μ = (
            output +
            (sp_norm.pdf(alpha_norm) - sp_norm.pdf(beta_norm)) /
            normalization
        )
        f_Σ = (
            1 +
            (alpha_norm * sp_norm.pdf(alpha_norm) -
             beta_norm * sp_norm.pdf(beta_norm)) / normalization -
            (sp_norm.pdf(alpha_norm) -
             sp_norm.pdf(beta_norm))**2 / normalization**2
        )
        return f_μ, f_Σ, normalization

    # noinspection PyUnresolvedReferences
    def _calc_appr_error(self, N, P, lambda_alpha, lambda_beta, a_sqrd_mu,
                         sigma_g, G_sqrd_mu, a_mu, KmKm, KmtimesG_mu,
                         gamma_alpha, gamma_beta, b_sqrd_mu, omega_alpha,
                         omega_beta, e_sqrd_mu, f_mu, f_sigma, b_e_mu, G_mu,
                         etimesb_mu, a_sigma, G_sigma, b_e_sigma,
                         normalization):
        lb = np.nan
        if self.verbose == 1:
            lb = 0

            # p(λ)
            lb = (lb +
                  sum((self.alpha_lambda - 1) * (polygamma([1], lambda_alpha) +
                      np.log(lambda_beta)) -
                      lambda_alpha * lambda_beta / self.beta_lambda -
                      np.log(np.gamma(self.alpha_lambda)) -
                      self.alpha_lambda * np.log(self.beta_lambda)))
            # p(a | λ)
            lb = (
                lb -
                0.5 * sum(lambda_alpha * lambda_beta *
                          np.diag(a_sqrd_mu)) -
                0.5 * (N * LOG2PI - sum(polygamma(lambda_alpha) +
                                        np.log(lambda_beta)))
            )
            # p(G | a, Km)
            lb = (
                lb -
                0.5 * sigma_g**(-2) * sum(np.diag(G_sqrd_mu)) +
                sigma_g**(-2).T @ a_mu @ KmtimesG_mu.T -
                0.5 * sigma_g**-2 * sum(sum(KmKm * a_sqrd_mu)) -
                0.5 * N * P * (LOG2PI + 2 * np.log(sigma_g)))
            # p(γ)
            lb = (
                lb +
                (self.alpha_gamma - 1) * (polygamma(gamma_alpha) +
                                          np.log(gamma_beta)) -
                gamma_alpha * gamma_beta / self.beta_gamma -
                np.log(np.gamma(self.alpha_gamma)) -
                self.alpha_gamma * np.log(self.beta_gamma)
            )
            # p(b | γ)
            lb = (
                lb -
                0.5 * gamma_alpha * gamma_beta * b_sqrd_mu -
                0.5 * (LOG2PI - (polygamma(gamma_alpha) + np.log(gamma_beta)))
            )
            # p(ω)
            lb = (
                lb +
                sum((self.alpha_omega - 1) * (polygamma(omega_alpha) +
                                              np.log(omega_beta)) -
                    omega_alpha * omega_beta / self.beta_omega -
                    np.log(np.gamma(self.alpha_omega)) -
                    self.alpha_omega * np.log(self.beta_omega))
            )
            # p(e | ω)
            lb = (
                lb -
                0.5 * sum(omega_alpha * omega_beta *
                          np.diag(e_sqrd_mu)) -
                0.5 * (P * LOG2PI - sum(polygamma(omega_alpha) +
                                        np.log(omega_beta)))
            )
            # p(f | b, e, G)
            lb = (lb -
                  0.5 * (f_mu @ f_mu.T + sum(f_sigma)) +
                  (b_e_mu[1: P] @ G_mu.T) @ f_mu.T +
                  sum(b_e_mu[1] * f_mu) -
                  0.5 * sum(sum(e_sqrd_mu * G_sqrd_mu)) -
                  sum(etimesb_mu @ G_mu.T) -
                  0.5 * N * b_sqrd_mu -
                  0.5 * N * LOG2PI)

            # q(λ)
            lb = (
                lb +
                sum(lambda_alpha +
                    np.log(lambda_beta) +
                    np.log(np.gamma(lambda_alpha)) +
                    (1 - lambda_alpha) * polygamma(lambda_alpha))
            )
            # q(a)
            lb = lb + 0.5 * (N * (LOG2PI + 1) + slogdet(a_sigma)[1])
            # q(G)
            lb = lb + 0.5 * N * (P * (LOG2PI + 1) + slogdet(G_sigma)[1])
            # q(γ)
            lb = (
                lb +
                gamma_alpha +
                np.log(gamma_beta) +
                np.log(np.gamma(gamma_alpha)) +
                (1 - gamma_alpha) * polygamma(gamma_alpha)
            )
            # q(ω)
            lb = (
                lb +
                sum(omega_alpha +
                    np.log(omega_beta) +
                    np.log(np.gamma(omega_alpha)) +
                    (1 - omega_alpha) * polygamma(omega_alpha))
            )
            # q(b, e)
            lb = lb + 0.5 * ((P + 1) * (LOG2PI + 1) + np.logdet(b_e_sigma))
            # q(f)
            lb = (lb +
                  0.5 * sum(LOG2PI + f_sigma) +
                  sum(np.log(normalization)))

        return lb

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BEMKL":
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).flatten()
        self.X_train = X_train
        N, _ = X_train.shape
        assert len(y_train) == N
        P = len(self.kernels)
        Km, Km_norms = self._create_kernel_matrix(X_train, X_train)
        self.Km_norms = Km_norms

        lambda_alpha, lambda_beta = self._init_λ(N)
        a_mu, a_sigma = self._init_a(N)
        G_mu, G_sigma = self._init_G(y_train, P)
        gamma_alpha, gamma_beta = self._init_γ()
        omega_alpha, omega_beta = self._init_ω(P)
        b_e_mu, b_e_sigma = self._init_b_e(P)
        f_mu, f_sigma = self._init_f(y_train)
        sigma_g = self.σ_g

        KmKm = self._calc_KmKm(Km)
        Km = Km.reshape((P * N, N))

        lower_bound = -1e40 * np.ones(N)
        lower_bound[y_train > 0] = self.margin
        upper_bound = 1e40 * np.ones(N)
        upper_bound[y_train < 0] = -self.margin

        bounds = np.zeros(self.max_iter)

        a_sqrd_mu = _calc_a_sqrd_mu(a_mu, a_sigma)
        G_sqrd_mu, KmtimesG_mu = _calc_G_stats(N, P, G_mu, G_sigma, Km)
        b_sqrd_mu, e_sqrd_mu, etimesb_mu =\
            _calc_b_e_stats(b_e_mu, b_e_sigma, P)

        for i in range(self.max_iter):
            lambda_beta = self._update_λ(a_sqrd_mu)

            a_sigma, a_mu, a_sqrd_mu = self._update_a(
                lambda_alpha, lambda_beta, KmKm, sigma_g, N, KmtimesG_mu
            )

            G_sigma, G_mu, G_sqrd_mu, KmtimesG_mu = self._update_G(
                N, P, sigma_g, e_sqrd_mu, Km, a_mu, f_mu, b_e_mu, etimesb_mu
            )

            gamma_beta = self._update_γ(b_sqrd_mu)

            omega_beta = self._update_ω(e_sqrd_mu)

            b_e_sigma, b_e_mu, b_sqrd_mu, e_sqrd_mu, etimesb_mu =\
                self._update_b_e(N, P, gamma_alpha, gamma_beta, G_mu,
                                 omega_alpha, omega_beta, G_sqrd_mu, f_mu)

            f_mu, f_sigma, normalization = self._update_f(
                N, G_mu, b_e_mu, lower_bound, upper_bound
            )
            bounds[i] = self._calc_appr_error(
                N, P, lambda_alpha, lambda_beta, a_sqrd_mu, sigma_g,
                G_sqrd_mu, a_mu, KmKm, KmtimesG_mu, gamma_alpha, gamma_beta,
                b_sqrd_mu, omega_alpha, omega_beta, e_sqrd_mu, f_mu, f_sigma,
                b_e_mu, G_mu, etimesb_mu, a_sigma, G_sigma, b_e_sigma,
                normalization
            )

        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.b_e_mu = b_e_mu
        self.b_e_sigma = b_e_sigma

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.predict_proba(X_test)
        y_pred = np.argwhere(y_pred >= 0.5)[:, 1]
        y_pred[y_pred == 0] = -1
        return y_pred.astype(int)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        Km, _ = self._create_kernel_matrix(X_test, self.X_train,
                                           Km_norms=self.Km_norms)
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
        sns.distplot(self.b_e_mu[0, 1:], **kwargs)


# noinspection PyPep8Naming
def poly_kernel(A, B, c, d):
    return (A @ B.T + c)**d


# noinspection PyPep8Naming
def gauss_kernel(A, B, sigma):
    return np.exp(- cdist(A, B, metric='sqeuclidean')/sigma)


# noinspection PyPep8Naming
def scoring(estimator, X_test, y_test):
    if 'iteration' not in dir(scoring):
        scoring.iteration = 0
    y_pred = estimator.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    bemkl_model = None
    if isinstance(estimator, BEMKL):
        bemkl_model = estimator
    if isinstance(estimator, Pipeline):
        bemkl_model = estimator.named_steps['bemkl']
    e_mu = bemkl_model.b_e_mu[0, 1:]
    a_mu = bemkl_model.a_mu
    X_train = bemkl_model.X_train
    if len(X_train) != len(X_test):
        # noinspection PyTypeChecker
        nr_kernels_used = len(np.argwhere(
            np.abs(e_mu) > bemkl_model.e_null_thrsh
        ))
        total_kernels = len(e_mu)
        nr_support_vect = len(np.argwhere(
            np.abs(a_mu) > bemkl_model.a_null_thrsh
        ))
        total_support_vect = len(a_mu)
        print(
            f"{scoring.iteration} - "
            f"Kernels: {nr_kernels_used}/{total_kernels} "
            f"({nr_kernels_used/total_kernels}). "
            f"SV: {nr_support_vect}/{total_support_vect} "
            f"({nr_support_vect/total_support_vect}). "
            f"Mean e: {e_mu.mean():0.4f}. "
            f"Median e: {np.median(e_mu):0.4f}. "
            f"Std e: {e_mu.std():0.4f}. "
         )
        scoring.iteration += 1
    return score
