"""Coefficient-only IRLS from statsmodels QuantReg (BSD-3-Clause).

Adapted from statsmodels.regression.quantile_regression, version 0.15.0.
Author of the upstream Python module: Vincent Arel-Bundock; original IRLS
algorithm: Shapour Mohammadi; translation: Christian Prinoth. The iteration
arithmetic, 1e-6 residual floor, cycle check, and stopping rule are preserved.
Asymptotic iid-pair covariance calculations are omitted because Figure 4 uses
whole-identity bootstrap/permutation inference. See STATSMODELS_LICENSE.txt.
"""
from __future__ import annotations
import numpy as np


def fit_coefficients(y,X,q=.5,max_iter=2000,p_tol=1e-6):
    xstar=X;beta=np.ones(X.shape[1]);diff=10.;cycle=False;n_iter=0;history=[]
    while n_iter<max_iter and diff>p_tol and not cycle:
        n_iter+=1;beta0=beta
        xtx=np.dot(xstar.T,X);xty=np.dot(xstar.T,y)
        beta=np.dot(np.linalg.pinv(xtx),xty)
        residual=y-np.dot(X,beta)
        small=np.abs(residual)<.000001
        residual[small]=((residual[small]>=0)*2-1)*.000001
        residual=np.where(residual<0,q*residual,(1-q)*residual)
        residual=np.abs(residual);xstar=X/residual[:,np.newaxis]
        diff=np.max(np.abs(beta-beta0));history.append(beta)
        if n_iter>=300 and n_iter%100==0:
            for ii in range(2,10):
                if np.all(beta==history[-ii]):cycle=True;break
    valid=not cycle and n_iter<max_iter and bool(np.isfinite(beta).all())
    return beta,valid
