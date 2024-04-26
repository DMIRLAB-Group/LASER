import sys

import numpy

sys.path.append('..')

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


def e_x_estimator(x, w):
    """estimate P(W_i=1|X_i=x)"""
    log_reg = LogisticRegression().fit(x, w)
    return log_reg


def naive_estimator(t, y):
    """estimate E[Y|T=1] - E[Y|T=0]"""
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    y1 = y[index_t1,]
    y0 = y[index_t0,]

    tau = np.mean(y1) - np.mean(y0)
    return tau


def ipw_estimator(x, t, y):
    """estimate ATE using ipw method"""
    propensity_socre_reg = e_x_estimator(x, t)
    propensity_socre = propensity_socre_reg.predict_proba(x)
    propensity_socre = propensity_socre[:, 1][:, None]  # prob of treatment=1

    ps1 = 1. / np.sum(t / propensity_socre)
    y1 = ps1 * np.sum(y * t / propensity_socre)
    ps0 = 1. / np.sum((1. - t) / (1. - propensity_socre))
    y0 = ps0 * np.sum(y * ((1. - t) / (1 - propensity_socre)))
    # print((1. - t).sum())
    # print(t.sum())

    tau = y1 - y0
    return tau


def s_learner_estimator(x, t, y, regression=LinearRegression()):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        s_learner: naive estimator using same regression function
    """
    x_t = np.concatenate((x, t), axis=1)
    regression.fit(X=x_t, y=y)
    x_t1 = np.concatenate((x, numpy.ones_like(t)), axis=1)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)), axis=1)
    y1 = regression.predict(X=x_t1)
    y0 = regression.predict(X=x_t0)

    tau = np.mean(y1 - y0)
    return tau


def t_learner_estimator(x, t, y, regression_1=LinearRegression(), regression_0=LinearRegression()):
    """ estimate E(Y|X,T=1)-E(Y|X,T=0)
        t_learner: naive estimator using different regression function
    """
    index_t1 = np.squeeze(t == 1)
    index_t0 = np.squeeze(t == 0)
    x_t1 = np.concatenate((x[index_t1,], t[index_t1,]), axis=1)
    x_t0 = np.concatenate((x[index_t0,], t[index_t0,]), axis=1)
    regression_1.fit(X=x_t1, y=y[index_t1,])
    regression_0.fit(X=x_t0, y=y[index_t0,])
    x_t1 = np.concatenate((x, numpy.ones_like(t)), axis=1)
    x_t0 = np.concatenate((x, numpy.zeros_like(t)), axis=1)
    y1 = regression_1.predict(X=x_t1)
    y0 = regression_0.predict(X=x_t0)

    tau = np.mean(y1 - y0)
    return tau


def x_learner_estimator():
    pass


def double_robust_estimator(x, t, y):
    pass


def tmle_estimator(x,t,y):
    pass