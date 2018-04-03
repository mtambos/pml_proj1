#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from typing import Union, Iterable, Callable, NewType

from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import slogdet
from scipy.stats import (norm as sp_norm, truncnorm as sp_truncnorm)
from scipy.special import digamma, loggamma
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from numpy import newaxis as na

KernelType = NewType('KernelType',
                     Callable[[np.ndarray, np.ndarray],
                              Union[float, np.ndarray]])

# VARIOUS UTILS
# ==================
LOG2PI = np.log(2 * np.pi)

def truncnorm(a=np.infty, b=np.infty, loc=0, scale=1):
    a, b = (a - loc) / scale, (b - loc) / scale
    return sp_truncnorm(a=a, b=b, loc=loc, scale=scale) # standard gaussian truncated to the range [a, b]

def _calc_a_sqrd_mu(a_mu, a_sigma): 
    N, L      = a_mu.shape
    a_sqrd_mu = np.zeros((N, N, L))
    for o in range(L):
        a_sqrd_mu[:,:,o] = np.dot(a_mu[:, o][:,na], a_mu[:, o][na,:]) + a_sigma[:,:,o]
    return a_sqrd_mu

def _calc_G_stats(N, P, G_mu, G_sigma, Km):
    L         = G_mu.shape[2] # G_mu has shape (P, N, L)
    G_sqrd_mu = np.zeros((P, P, L))
    for o in range(L):
        G_sqrd_mu[:,:,o] = np.dot(G_mu[:,:, o], G_mu[:,:,o].T) + N*G_sigma
    KmtimesG_mu = np.dot(Km, np.swapaxes(G_mu, 0,1).reshape(N*P, L)) # Km has shape (N, N*P), result of shape (N, L)
    return G_sqrd_mu, KmtimesG_mu

def _calc_b_e_stats(b_e_mu, b_e_sigma, P, L): 
    e_μ = b_e_mu[L:L+P, 0][:, na] ; assert(e_μ.shape==(P,1))
    b_μ = b_e_mu[0:L, 0][:, na]   ; assert(b_μ.shape==(L,1))
    b_sqrd_μ  = np.dot(b_μ, b_μ.T) +  b_e_sigma[0:L, 0:L]                    
    e_sqrd_μ  = np.dot(e_μ, e_μ.T) +  b_e_sigma[L:L+P, L:L+P] 
    etimesb_μ = np.zeros((P,L))
    for o in range(L):
       etimesb_μ[:, o] = np.squeeze(e_μ) * b_μ[o,0] + b_e_sigma [L:L+P, o]                  
    return b_sqrd_μ, e_sqrd_μ, etimesb_μ

# CLASSIFIER
# ==================
class BEMKL_multilabel(BaseEstimator, ClassifierMixin):
    """
    Multiclass BEMKL classifier.

    The input data is provided as precomputed kernel matrices Km_train and Km_total
    (i.e. features X_train and X_test are not needed)
    """

    def __init__(self,Km_train,Km_total, 
             random_state: Union[None, int, np.random.RandomState]=None,
             alpha_lambda: float=1, beta_lambda: float=1,
             alpha_gamma: float=1, beta_gamma: float=1,
             alpha_omega: float=1, beta_omega: float=1, max_iter: int=200,
             margin: float=1, sigma_g: float=0.1, e_null_thrsh: float=1e-6,
             a_null_thrsh: float=1e-6, k_norm_type: str='kernel',
             filter_kernels: bool=True, filter_sv: bool=True,
             verbose: bool=False):
        self.params = {}
        self.set_params( Km_train = Km_train,Km_total=Km_total,random_state=random_state,
                        alpha_lambda=alpha_lambda, beta_lambda=beta_lambda,
                        alpha_gamma=alpha_gamma, beta_gamma=beta_gamma,
                        alpha_omega=alpha_omega, beta_omega=beta_omega,
                        max_iter=max_iter, margin=margin,
                        sigma_g=sigma_g, e_null_thrsh=e_null_thrsh,
                        a_null_thrsh=a_null_thrsh, k_norm_type=k_norm_type,
                        filter_kernels=filter_kernels, filter_sv=filter_sv,
                        verbose=verbose)
        self.a_mu      = None
        self.a_sigma   = None
        self.b_e_mu    = None
        self.b_e_sigma = None
        self.sigma_g   = None
        self.Km_norms  = None
        self.L         = None

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'Km_train' in params:
            self.Km_train = params['Km_train']
        if 'Km_total' in params:
            self.Km_total = params['Km_total']
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
        if 'filter_kernels' in params:
            self.filter_kernels = params['filter_kernels']
        if 'filter_sv' in params:
            self.filter_sv = params['filter_sv']
        if 'verbose' in params:
            self.verbose = params['verbose']
        self.params.update(params)
        return self

    # NORMALIZE KERNEL
    # ==================
    def _normalize_kernel_matrix(self, Km, Km_norms=None): # exact same function as for binary classification
        P, N1, N2 = Km.shape
        for i in range(P): # iterate over the P kernels
            kmi = Km[i,:,:] # kernel matrix for one kernel            
            self.k_norm_type == 'kernel' # default kernel normalization
            k_norm      = np.sqrt(np.diag(kmi)) # one value per sample, ie N values
            Km[i, :, :] = kmi / np.outer(k_norm, k_norm) # normalized kernel matrix
        return Km 

    def _calc_KmKm(self, Km): # Km has shape (N, N, P), exact same function as for binary classification
        _, N, P = Km.shape
        KmKm = np.zeros((N, N))
        for i in range(P):
            KmKm += np.dot(Km[:, :,i], Km[:, :, i].T)
        return KmKm # KmKm has shape (N, N)

    # INIT VARIABLES
    # ==================
    def _init_λ(self, N, L): # (N, L)-dimensional gamma
        λ_α = (self.λ_α + 0.5) * np.ones((N, L))
        λ_β = self.λ_β * np.ones((N, L))
        return λ_α, λ_β

    def _init_a(self, N, L, fixed_init=True):     
        if fixed_init:
            a_μ = np.ones((N, L))*0.5 # shape (N, L)
        else:
            a_μ = self.random_state.randn(N, L) # shape (N, L)
        a_Σ = np.tile(np.eye(N), (L,1,1)).T ; assert(a_Σ.shape==(N,N,L)) # shape (N, N, L)
        return a_μ, a_Σ

    def _init_G(self, y_train, P, L, fixed_init=True):  
        N            = y_train.shape[1]
        y_train_sign = np.sign(y_train).T ; assert(y_train_sign.shape==(N, L)) # converted to +1s and -1s values
        if fixed_init:
            G_μ = (np.abs(np.ones((P, N, L))*0.5) + self.margin) * y_train_sign[na,:,:] ; assert(G_μ.shape==(P, N, L))
        else:
            G_μ = (np.abs(self.random_state.randn(P, N, L)) + self.margin) * y_train_sign[na,:,:] ; assert(G_μ.shape==(P, N, L))
        G_Σ = np.eye(P) # shape (P, P)
        return G_μ, G_Σ

    def _init_γ(self, L):
        γ_α = (self.γ_α + 0.5) * np.ones((L, 1))
        γ_β = self.γ_β * np.ones((L, 1))
        return γ_α, γ_β

    def _init_ω(self, P):
        ω_α = (self.ω_α + 0.5) * np.ones((P,1))
        ω_β = self.ω_β * np.ones((P,1))
        return ω_α, ω_β

    def _init_b_e(self, P, L):
        b_e_μ = np.concatenate((np.zeros((L, 1)), np.ones((P, 1))), 0); assert(b_e_μ.shape==(L+P, 1))
        b_e_Σ = np.eye(P + L) # shape (P+L, P+L)
        return b_e_μ, b_e_Σ

    def _init_f(self, y_train, L, fixed_init=True): 
        N    = y_train.shape[1]
        if fixed_init:
            f_μ  = (abs(np.ones((L, N))*0.5) + self.margin)
        else:
            f_μ  = (abs(self.random_state.randn(L, N)) + self.margin)
        f_μ *= np.sign(y_train) # shape (L, N)
        f_σ  = np.ones((N, L))
        return f_μ, f_σ

    # UPDATE VARIABLES
    # ==================
    def _update_λ(self, a_sqrd_mu):
        _,N,L = a_sqrd_mu.shape
        λ_β = np.zeros((N, L))
        for o in range(L):
            λ_β[:, o] = 1. / ((1. / self.λ_β) + (0.5 * np.diag(a_sqrd_mu[:,:,o]))) # use initial λ_β only 
        return λ_β  
        
    def _update_a(self, lambda_alpha, lambda_beta, KmKm, sigma_g, N,KmtimesG_mu, L):
        a_Σ = np.zeros((N,N,L))
        a_μ = np.zeros((N, L))
        for o in range(L):
            a_Σ [:,:,o] = np.linalg.solve(np.diag(lambda_alpha[:,o]*lambda_beta[:,o]) + (KmKm/(sigma_g**2)), np.eye(N))
            a_μ [:, o]  = np.squeeze( np.dot(a_Σ[:,:,o], KmtimesG_mu [:,o][:,na]) ) / sigma_g**2
        a_sqrd_μ = _calc_a_sqrd_mu(a_μ, a_Σ)  
        return a_Σ, a_μ, a_sqrd_μ
  
    def _update_G(self, N, P, L, sigma_g, e_sqrd_mu, Km, a_mu, f_mu, b_e_mu,
                  etimesb_mu):
        G_Σ = np.linalg.solve( (np.eye(P) / sigma_g**2 + e_sqrd_mu), np.eye(P) ) ; assert(G_Σ.shape == (P,P))
        G_μ = np.zeros((P,N,L))
        for o in range(L):
            assert(np.matlib.repmat(etimesb_mu[:,o][:,na], 1,N).shape==(P,N))
            G_μ[:,:,o] = np.dot( G_Σ , (   (np.squeeze(np.dot(a_mu[:,o][:,na].T, Km))).reshape(N, P).T / sigma_g**2 +
                                            np.outer(b_e_mu[L:L+P, 0], f_mu[o,:]) -
                                            np.matlib.repmat(etimesb_mu[:,o][:,na], 1,N)    )   )
        G_sqrd_μ, KmtimesG_μ = _calc_G_stats(N, P, G_μ, G_Σ, Km)    
        return G_Σ, G_μ, G_sqrd_μ, KmtimesG_μ

    def _update_γ(self, b_sqrd_mu):
        γ_β = 1. / (1. / self.γ_β + 0.5 * np.diag(b_sqrd_mu)[:,na])
        return γ_β

    def _update_ω(self,e_sqrd_mu):
        ω_β = 1. / (1. / self.ω_β + 0.5 * np.diag(e_sqrd_mu)[:,na])
        return ω_β

    def _update_b_e(self, N, P, gamma_alpha, gamma_beta, G_mu, omega_alpha,
                    omega_beta, G_sqrd_mu, f_mu, L):
        a = np.diag(np.squeeze(gamma_alpha * gamma_beta)) + N*np.eye(L) ; assert(a.shape==(L,L))
        b = np.sum(G_mu, axis=1).T                          ; assert(b.shape==(L,P))
        c = np.sum(G_mu, axis=1)                            ; assert(c.shape==(P,L))
        d = np.diag(np.squeeze(omega_alpha * omega_beta))   ; assert(d.shape==(P,P))
        for o in range(L):
            d = d + G_sqrd_mu[:,:,o]
        b_e_sigma = np.vstack([np.hstack([a,b]), np.hstack([c,d])]) ; assert(b_e_sigma.shape==(L+P, L+P))
        b_e_sigma = np.linalg.solve(  b_e_sigma  , np.eye(P+L)  )   ; assert(b_e_sigma.shape==(L+P, L+P))
        b_e_μ = np.zeros((L+P, 1)) # but was already initialized
        b_e_μ [0:L, 0] =  np.sum(f_mu, axis=1)  
        for o in range(L):
            b_e_μ [L:L+P, 0] =  b_e_μ [L:L+P, 0] + np.squeeze((np.dot(G_mu[:,:,o],f_mu[o,:][na,:].T))) 
        b_e_μ = np.dot( b_e_sigma, b_e_μ)
        b_sqrd_μ, e_sqrd_μ, etimesb_μ = _calc_b_e_stats(b_e_μ, b_e_sigma, P, L)
        return b_e_sigma, b_e_μ, b_sqrd_μ, e_sqrd_μ, etimesb_μ
                 
    def _update_f(self, N, P, L, G_mu, b_e_mu, lower_bound, upper_bound):
        output = np.zeros((L, N))
        for o in range(L):
            a = np.vstack([b_e_mu[o,0][na,na], b_e_mu[L:L+P,0][:,na]])    ; assert(a.shape==(P+1, 1))
            b = np.vstack([np.ones((1,N)), G_mu[:,:,o]]).T               ; assert(b.shape==(N, P+1))
            output[o,:] = np.squeeze(np.dot(b, a))
        print("output :", output[0,0])
        alpha_norm = lower_bound - output
        beta_norm  = upper_bound - output
        normalization = sp_norm.cdf(beta_norm) - sp_norm.cdf(alpha_norm) ; assert(normalization.shape==(L,N))
        print("normalization:", normalization[0,0])
        normalization[normalization == 0] = 1 
        f_μ = (
            output +
            ((sp_norm.pdf(alpha_norm) - sp_norm.pdf(beta_norm)) / normalization)
        )
        f_Σ = (
            1 +
            (alpha_norm * sp_norm.pdf(alpha_norm) -
             beta_norm * sp_norm.pdf(beta_norm)) / normalization -
            (sp_norm.pdf(alpha_norm) -
             sp_norm.pdf(beta_norm))**2 / normalization**2
        )
        return f_μ, f_Σ, normalization       

    # TRAIN
    # ==================
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BEMKL_multilabel":
        # transform training input to numpy arrays, define dimensions
        K_train = np.asarray(self.Km_train) # assuming shape (P, N, N) 
        y_train = np.asarray(y_train) # assuming shape (L, N) 
        L, N         = y_train.shape # N =nb of training instances, L = nb of classes
        assert((K_train.shape[1] == N) and (K_train.shape[2] == N))
        P = K_train.shape[0] # P = number of kernels
        assert((K_train.shape[0] == P))
        # kernel normalization
        Km = self._normalize_kernel_matrix(K_train)           # Km shape is (P, N, N) 
        Km = Km.transpose(1,2,0) ; assert (Km.shape==(N,N,P)) # Km has shape (N, N, P)
        print("N, P, L: ", N, P, L)
        
        # init variables
        lambda_alpha, lambda_beta = self._init_λ(N, L)
        a_mu, a_sigma             = self._init_a(N, L)
        G_mu, G_sigma             = self._init_G(y_train, P, L)
        gamma_alpha, gamma_beta   = self._init_γ(L)
        omega_alpha, omega_beta   = self._init_ω(P)
        b_e_mu, b_e_sigma         = self._init_b_e(P, L)
        f_mu, f_sigma             = self._init_f(y_train, L)
        sigma_g                   = self.σ_g
        print('INIT--------------------------------------------------------')
        print('lambda_beta ',lambda_beta[0,0])
        print('a_mu',a_mu[0,0])
        print('a_sigma',a_sigma[0,0])
        print('G_mu',G_mu[0,0])
        print('G_sigma',G_sigma[0,0])
        print('gamma_beta ',gamma_beta[0,0])
        print('omega_beta ',omega_beta[0,0])
        print('b_e_mu',b_e_mu[0,0])
        print('b_e_sigma',b_e_sigma[0,0])
        print('f_mu',f_mu[0,0])
        print('f_sigma',f_sigma[0,0])
        print()
        print()
        # utils
        KmKm = self._calc_KmKm(Km)  # KmKm shape is (N, N)
        Km   = Km.reshape((N, N*P)) # Km shape is now (N, N*P)
        print('KmKm',KmKm[10,20])
        lower_bound = -1e40 * np.ones((L, N))
        lower_bound[y_train > 0] = self.margin
        upper_bound = 1e40 * np.ones((L, N))
        upper_bound[y_train < 0] = -self.margin 

        self.bounds = np.zeros(self.max_iter)

        a_sqrd_mu = _calc_a_sqrd_mu(a_mu, a_sigma)
        G_sqrd_mu, KmtimesG_mu = _calc_G_stats(N, P, G_mu, G_sigma, Km)
        b_sqrd_mu, e_sqrd_mu, etimesb_mu = _calc_b_e_stats(b_e_mu, b_e_sigma, P, L)
        print('a_sqrd_mu',a_sqrd_mu[0,0])
        print('G_sqrd_mu',G_sqrd_mu[0,0])
        print('KmtimesG_mu',KmtimesG_mu[0,0])
        print('b_sqrd_mu',b_sqrd_mu[0,0])
        print('e_sqrd_mu',e_sqrd_mu[0,0])
        print('etimesb_mu',etimesb_mu[0,0])

        for i in range(self.max_iter):
            print()
            print('iter{}-------------------------------'.format(i))
            lambda_beta = self._update_λ(a_sqrd_mu)
            
            a_sigma, a_mu, a_sqrd_mu = self._update_a(
                lambda_alpha, lambda_beta, KmKm, sigma_g, N, KmtimesG_mu, L
            )

            G_sigma, G_mu, G_sqrd_mu, KmtimesG_mu = self._update_G(
                N, P, L, sigma_g, e_sqrd_mu, Km, a_mu, f_mu, b_e_mu, etimesb_mu
            )

            gamma_beta = self._update_γ(b_sqrd_mu)

            omega_beta = self._update_ω(e_sqrd_mu)

            b_e_sigma, b_e_mu, b_sqrd_mu, e_sqrd_mu, etimesb_mu =\
                self._update_b_e(N, P, gamma_alpha, gamma_beta, G_mu,
                                 omega_alpha, omega_beta, G_sqrd_mu, f_mu, L)

            f_mu, f_sigma, normalization = self._update_f(
                N, P, L, G_mu, b_e_mu, lower_bound, upper_bound)

            print('lambda_beta ',lambda_beta[0,0])
            print('a_mu',a_mu[0,0])
            print('a_sigma',a_sigma[0,0])
            print('a_sqrd_mu',a_sqrd_mu[0,0])
            print('G_mu',G_mu[0,0])
            print('G_sigma',G_sigma[0,0])
            print('G_sqrd_mu',G_sqrd_mu[0,0])
            print('KmtimesG_mu',KmtimesG_mu[0,0])
            print('gamma_beta ',gamma_beta[0,0])
            print('omega_beta ',omega_beta[0,0])
            print('b_e_mu',b_e_mu[0,0])
            print('b_e_sigma',b_e_sigma[0,0])
            print('b_sqrd_mu',b_sqrd_mu[0,0])
            print('e_sqrd_mu',e_sqrd_mu[0,0])
            print('etimesb_mu',etimesb_mu[0,0])
            print('f_mu',f_mu[0,0])
            print('f_sigma',f_sigma[0,0])
        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.b_e_mu = b_e_mu
        self.b_e_sigma = b_e_sigma
        self.L = L
        return self

    # TEST
    # ==================
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        K_total = self.Km_total
        y_pred = self.predict_proba(K_total)
        y_result = -1 * np.ones(y_pred.shape)
        y_result [ np.argmax(y_pred, axis=0), np.arange(y_pred.shape[1]) ] = np.ones(y_pred.shape[1])
        y_result = y_result.astype(int)
        assert(np.allclose(y_result.sum(axis=0), np.ones(y_pred.shape[1])*(2-self.L)))
        return y_result

    def predict_proba(self,X_test: np.ndarray) -> np.ndarray:
        if self.k_norm_type == 'kernel':
            Ntr = self.Km_train.shape[1] # nb of training instances
            Km = self._normalize_kernel_matrix(self.Km_total)
            Km     = Km[:, Ntr:, :Ntr]
        margin = self.margin
        a_mu = self.a_mu
        a_sigma = self.a_sigma
        b_e_mu = self.b_e_mu
        b_e_sigma = self.b_e_sigma
        Km = Km.transpose(2,1,0) # shape(Ntrain, Ntest,P)
        _, N, P = Km.shape # nb of test instances, Note: N is Ntest
        L = self.L
        G_mu    = np.zeros((P, N, L))
        G_sigma = np.zeros((P, N, L))
        for o in range(L):
            for m in range(P):
                G_mu[m, :, o]     =  np.squeeze( np.dot( a_mu [:,o][:,na].T ,       Km [:,:,m]  )) 
                G_sigma [m,:, o]  =  self.σ_g**2  + np.diag(Km[:,:,m].T @ a_sigma [:,:, o] @ Km[:,:,m] )
        f_mu    = np.zeros((L,N))
        f_sigma = np.zeros((L,N))
        for o in range(L):
            a = np.vstack([b_e_mu[o,0][na,na], b_e_mu[L:L+P,0][:,na]])    ; assert(a.shape==(P+1, 1))
            b = np.vstack([np.ones((1,N)), G_mu[:,:,o]]).T                ; assert(b.shape==(N, P+1))
            f_mu[o,:] = np.squeeze(np.dot(b, a))
            a = b_e_sigma [o,o] [na, na]
            b = b_e_sigma [o, L:L+P] [na,:]
            c = b_e_sigma [L:L+P, o] [:,na]
            d = b_e_sigma [L:L+P, L:L+P]
            f_sigma[o,:] = 1. + np.diag( (np.vstack([np.ones((1,N)), G_mu[:,:,o]])).T     @  
                                          np.vstack([np.hstack([a,b]), np.hstack([c,d])]) @
                                          np.vstack([np.ones((1,N)), G_mu[:,:,o]]) )
        
        pos: np.ndarray = 1 - sp_norm.cdf((margin - f_mu) / f_sigma)
        neg: np.ndarray = sp_norm.cdf((-margin - f_mu) / f_sigma)
        y_pred_proba = pos / (pos + neg)
        return y_pred_proba

