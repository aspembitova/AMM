import math
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import scipy.stats
import numpy as np
import scipy.linalg as spla
import os
import numbers
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from scipy.linalg import sqrtm
from arch import arch_model
%matplotlib inline
from pathlib import Path
from os import listdir
from os.path import isfile, join
from datetime import datetime

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates
#plt.style.use("bmh") #"seaborn-deep" "seaborn-paper"
from functools import reduce



def multivaraiate_normal_sampler(mean, covariance, n_samples=1):
    
    L = spla.cholesky(covariance)
    Z = np.random.normal(size=(n_samples, covariance.shape[0]))
    
    return Z.dot(L) + mean


def multivariate_student_t_sampler(mean, cov, dof, n_samples=1):
    #multivariate student-t distribution has covariance matrix equals to (dof/(dof-2) * Sigma) and the same mean
    m = mean.shape[0]
    u = np.random.gamma(dof / 2., 2. / dof, size = (n_samples,1))
    Y = multivaraiate_normal_sampler(np.zeros((m, )), cov, n_samples)
    
    return Y / np.tile(np.sqrt(u), [1,m]) + mean


def simulate_Brownian_mult(So, mu, sigma, mQ, returns, n):
    #maxs = returns.max().values*1.3 # restriction on max/min values of returns
    #mins = returns.min().values*1.3
    white_noise = multivaraiate_normal_sampler(mu, mQ, n)
    d = returns.shape[1]
    ret = np.zeros((n, d))
    S = np.zeros([n, d])
    

    for t in range(1, n):
        ret[t] = mu + np.sqrt(sigma2)*white_noise[t][0]
        #ret[t] = ret[t].clip(max=maxs, min=mins)
    
        S[t, :] = S[t-1, :] * (1 + ret[t,:] / 100)
        St = S[t, :]
        St[np.where(St <0)] = 0 #to avoid negative prices
        S[t, :] = St
    return ret, S


def fit_GARCH(ret, dist):

    scaling_const = 1.0 #/ ret.std()

    am = arch_model(ret * scaling_const,
                    mean='ARX', lags=1, # mean = Constant, ARX, HARX + the number of lags
                    vol='Garch', p=1, o=0, q=1, # vol = Garch, EGARCH, HARCH + the number of lags
                    dist=dist) # dist = Normal, t, skewstudent, ged

    res = am.fit(update_freq=0, disp='off')
    alpha = pd.DataFrame(res.params).filter(regex='alpha', axis=0).values[0][0]
    beta = pd.DataFrame(res.params).filter(regex='beta', axis=0).values[0][0]

    # try parts of t-s to get stationary model
    cut_p = np.arange(0.05,0.6,0.05)
    i = 0
    while ((alpha + beta) > 0.9999) & (i < len(cut_p)):
        y = np.int(len(ret) * cut_p[i])
        ret_cut = ret[y:]
        am = arch_model(ret_cut * scaling_const,
                    mean='ARX', lags=1, # mean = Constant, ARX, HARX + the number of lags
                    vol='Garch', p=1, o=0, q=1, # vol = Garch, EGARCH, HARCH + the number of lags
                    dist=dist) # dist = Normal, t, skewstudent, ged
        i = i + 1
        res = am.fit(update_freq=0, disp='off')
        alpha = pd.DataFrame(res.params).filter(regex='alpha', axis=0).values[0][0]
        beta = pd.DataFrame(res.params).filter(regex='beta', axis=0).values[0][0]
    
    return am, res, scaling_const


def fit_mult_GARCH(returns, dist):
    
    eps = pd.DataFrame(columns=returns.columns)
    distribution_params = {}
    df_params = pd.DataFrame(columns=returns.columns)

    # Fit marginal distributions for each asset
    ret_dim = returns.shape[0]
    for col in returns.columns:
        print(col)
        am, res, scaling_const = fit_GARCH(returns[col], dist)
        cond_vol = res.conditional_volatility.to_numpy().reshape(len(res.resid),1)
        a = np.empty(len(returns[col]) - len(res.resid))
        a[:] = np.nan

        eps[col] = returns[col].values / np.concatenate([a, cond_vol.T[0]])
        
        alpha = pd.DataFrame(res.params).filter(regex='alpha', axis=0).values[0][0]
        beta = pd.DataFrame(res.params).filter(regex='beta', axis=0).values[0][0]
        nu = pd.DataFrame(res.params).filter(regex='nu', axis=0).values[0][0]
        omega = pd.DataFrame(res.params).filter(regex='omega', axis=0).values[0][0]
        a0 = res.params[0]
        a1 = res.params[1]
        df_params[col] = [alpha, beta, omega, a0, a1, scaling_const] 
        
        print(alpha + beta, nu)
        distribution_params[col] = (am, res, scaling_const)
        
    return distribution_params, df_params, eps


def simulate_Student_t_GARCH_mult(So, dof, n, mQ, returns, df_params):

    d = returns.shape[1]
    
    maxs = returns.max().values*1.3 # restriction on max/min values of returns
    mins = returns.min().values*1.3 # restriction on max/min values of returns

    ret = np.zeros((n, d))
    stdres = np.zeros((n, d))
    sigma2 = np.zeros((n, d))
    
    [alpha, beta, omega, a0, a1, scaling_const] = df_params.values
    
    mu = np.zeros((1, d)) # initial mu
    sigma2[0] = omega / (1 - alpha - beta) # initial sigma
    
    white_noise = np.sqrt((dof - 2) / dof) * multivariate_student_t_sampler(np.zeros((d, )), mQ, dof, n) 

    ret[0] = returns[-1:].values
    S = np.zeros([n, d])
    S[0, :] = So
    #loop over sims
    for t in range(1, n):
        
        mu = a0 + a1 * ret[t-1]
        sigma2[t] = omega + alpha * (ret[t-1] - mu)**2 + beta * sigma2[t-1]
        ret[t] = mu + np.sqrt(sigma2[t]) * white_noise[t]
        ret[t] = ret[t].clip(max=maxs, min=mins)
        stdres[t] = ret[t] / np.sqrt(sigma2[t])
            
        S[t, :] = S[t-1, :] * (1 + ret[t,:] / 100)
        St = S[t, :]
        St[np.where(St <0)] = 0 #to avoid negative prices
        S[t, :] = St
    
    return ret, S               


