# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

""" Hmmmm """

from sklearn.base import clone


def residuals_gpr(gpr, X, y, train, test, weights=None, Xeval=None,
                  return_gpr=False, clone_gpr=True):
    """
    Fit a 1D Gaussian process regressor (GPR) on the training set and score the
    fit on the test set. If `Xeval` is not `None` evaluates the GPR on it,
    otherwise assumes that `Xeval = X`. Optionally returns the fitted GPR.


    Parameters
    ----------
    gpr : py:class:`sklearn.gaussian_process.GaussianProcessRegressor`
        The unfitted GPR instance.
    X : 1-dimensional array
        The input samples of shape (n_samples, ).
    y : 1-dimensional array
        The target values of shape (n_samples, ).
    train : 1-dimensional array
        The training samples' indices.
    test : 1-dimensional array
        The test samples' indices.
    weights : 1-dimensional array, optional
        The target weights. GPR does not consider weights while fitting, used
        only for scoring. By default `None`.
    Xeval : 2-dimensional array, optional.
        Supplementary array on which to evaluate the GPR if specified.
    return_gpr: bool, optional
        Whether to also return the fitted GPR. By default False.
    clone_gpr : bool, optional
        Whether to clone the GPR instance. By default True.


    Returns
    -------
    residuals : 1-dimensional array
        The GPR residuals defined as `ytrue - ypred`.
    score : float
        The :math:`R^2` regression score on the test set.
    yeval : 1-dimensional array
        Array of GPR predictions corresponding to `Xeval`. If `Xeval` was not
        specified is `None`.
    gpr : py:class:sklearn.gaussian_process.GaussianProcessRegressor, optional
        The fitted GPR instance.
    """
    if X.ndim > 1:
        raise TypeError("`X` must be a 1-dimensional array.")
    if Xeval is not None and Xeval.ndim > 1:
        raise TypeError("`Xeval` must be a 1-dimensional array.")
    X = X.reshape(-1, 1)
    Xeval = X if Xeval is None else Xeval.reshape(-1, 1)

    gpr = clone(gpr) if clone_gpr else gpr
    gpr.fit(X[train], y[train])
    residuals = y - gpr.predict(X)
    score = gpr.score(X[test], y[test],
                      weights[test] if weights is not None else None)
    yeval = gpr.predict(Xeval)

    out = (residuals, score, yeval)
    out = out + (gpr,) if return_gpr else out
    return out
