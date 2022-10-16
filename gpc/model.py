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

"""Regressor functions and partial correlation functions."""

import numpy
from scipy import stats
from sklearn.base import clone
from tqdm import tqdm

from .utils import (train_test_from_mask, kernel_weights)


CORRMEASURES = {"pearson": stats.pearsonr,
                "spearman": stats.spearmanr}


def run_reg(reg, x, y, test_mask=None, weights=None, clone_reg=True):
    """
    Fit a regressor and return the predicted value and score. If available
    performs a train-test split to fit and evaluate the regressor.

    Parameters
    ----------
    reg : regressor instance
        The unfitted regressor.
    x : 1-dimensional array
        The input samples of shape (n_samples, ).
    y : 1-dimensional array
        The target values of shape (n_samples, ).
    test_mask : 1-dimensional array, optional
        Boolean array of shape `(n_samples, )`, where `True` indicates that the
        sample belongs to the test set. If `None` does not perform a train-test
        split.
    weights : 1-dimensional array, optional
        The target weights. Not considered while fitting, used only for
        scoring. By default `None`.
    clone_reg : bool, optional
        Whether to clone the regressor. By default `True`.

    Returns
    -------
    ypred : 1-dimensional array
        The predicted values of shape (n_samples, ).
    score : float
        The :math:`R^2` score. Evaluated on the test set if available.
    """
    if x.ndim != 1:
        raise TypeError("`x` must be a 1-dimensional array.")
    x = x.reshape(-1, 1)
    # Test-train split
    if test_mask is None:
        train = test = numpy.arange(test_mask.size)
    else:
        train, test = train_test_from_mask(test_mask)

    reg = clone(reg) if clone_reg else reg
    reg.fit(x[train], y[train])

    ypred = reg.predict(x)
    score = reg.score(x[test], y[test],
                      weights[test] if weights is not None else None)
    return ypred, score


def run_reg_folds(reg, x, y, test_masks, weights=None, verbose=True):
    """
    Run the regressor over cross-validation (CV) folds defined by `test_masks`.

    Parameters
    ----------
    reg : regressor instance
        The unfitted regressor.
    x : 1-dimensional array
        The input samples of shape (n_samples, ).
    y : 1-dimensional array
        The target values of shape (n_samples, ).
    test_mask : 2-dimensional array, optional
        Boolean array of shape `(n_folds, n_samples)`, where `True` indicates
        that the sample belongs to the test set.
    weights : 1-dimensional array, optional
        The target weights of shape `(n_samples, )`. Not considered while
        fitting, used only for scoring. By default `None`.
    verbose : bool, optional
        Verbosity flag of the folding iterator. By default `True`.

    Returns
    -------
    ypred : 2-dimensional array
        Predicted values of shape `(n_samples, n_folds)`.
    score : float or 1-dimensional array
        1-dimensional array of each fold's score of shape (`n_folds`, ).
    """
    Nfolds, Nsamples = test_masks.shape

    ypred = numpy.full((Nsamples, Nfolds), numpy.nan)
    score = numpy.full(Nfolds, numpy.nan)


    iters = range(Nfolds)
    iters = tqdm(iters) if verbose else iters
    for i in iters:
        mask = test_masks[i, :]
        ypred[:, i], score[i] = run_reg(reg, x, y, mask, weights=weights)

    return ypred, score


def get_reg_residuals(reg, x, y, z, test_mask=None, weights=None, partial=False,
                      verbose=True):
    """
    Calculate residuals on `x` and `y` while predicting their values based
    solely on `z`. `x` can contain several features, however the fit is always
    simply from one feature to one prediction.

    Parameters
    ----------
    reg : regressor instance
        The unfitted regressor.
    x : 1- or 2-dimensional array
        Correlation features of shape `(n_samples, )` or
        `(n_samples, n_xfeatures)`.
    y : 1-dimensional array
        Correlation feature to be correlated with `x` of shape `(n_samples, )`.
    z : 1-dimensional array
        Control feature of shape `(n_samples, )`.
    test_mask : 2-dimensional array, optional
        Boolean array of shape `(n_folds, n_samples)`, where `True` indicates
        that the sample belongs to the test set. Alternatively `(n_samples, )`
        can also be accepted.
    weights : 1-dimensional array, optional
        The target weights of shape `(n_samples, )`. Not considered while
        fitting, used only for scoring. By default `None`.
    partial : bool, optional
        Whether to perform a `partial` partial correlation, in which case the
        residuals of `y` with respect to `z` are not calculated and taken to
        be `y`.
    verbose : bool, optional
        Verbosity flag of the folding iterator. By default `True`.

    Returns
    -------
    dxz : 3-dimensional array
        Residuals of `true` - `predicted` of `x` of shape
        (`n_samples`, `n_xfeatures`, `n_folds`).
    dyz : 3-dimensional array
        Residuals of `true` - `predicted` of `y` of shape
        (`n_samples`, `n_folds`).
    fullout : dict
        Additional output. Keys are `(xz, scorexz, yz, scoreyz)`. The array
        shapes follow the notation above and the scores are :math:`R^2`.
    """
    # Enforce that y and z are 1-dimensional
    if y.ndim != 1 or z.ndim != 1:
        raise TypeError("`y` and `z` must be a 1-dimensional arrays.")
    # If no CV folding of the mask reshape it to match the following code
    test_mask = test_mask.reshape(1, -1) if test_mask.ndim == 1 else test_mask
    # If x is 1dim reshape to fit the remainder of the code
    x = x.reshape(-1, 1) if x.ndim == 1 else x

    # Preallocate arrays
    Nsamp, Nxfeat = x.shape
    Nfolds = test_mask.shape[0]
    xz = numpy.full((Nsamp, Nxfeat, Nfolds), numpy.nan)
    scorexz = numpy.full((Nxfeat, Nfolds), numpy.nan)

    # Fit the GP for the `x` features
    for i in range(Nxfeat):
        xz[:, i, :], scorexz[i, :] = run_reg_folds(
            reg, z, x[:, i], test_mask, weights, verbose)

    if partial:
        yz, scoreyz = numpy.zeros_like(y), numpy.full_like(scorexz, numpy.nan)
    else:
        yz, scoreyz = run_reg_folds(reg, z, y, test_mask, weights, verbose)

    fullout = {"xz": xz,
               "scorexz": scorexz,
               "yz": yz,
               "scoreyz": scoreyz}

    dxz = numpy.full_like(xz, numpy.nan)
    dyz = numpy.full((Nsamp, Nfolds), numpy.nan)
    for i in range(Nfolds):
        dxz[..., i] = x - xz[..., i]
        dyz[..., i] = y - yz[..., i]

    return dxz, dyz, fullout


def _pick_correlation(corr):
    """
    Boilerplate function to return correlation statistic by name from a
    predefined dictionary and print a useful error message if needed.

    Paramteres
    ----------
    corr : string
        Correlation statistic. By default `spearman`, supported
        are `[spearman, pearson]`.

    Returns
    -------
    corr_func : py:func
        Correlation function that returns the correlation coefficient and
        p-value.
    """
    keys = CORRMEASURES.keys()
    if corr not in keys:
        raise ValueError("Supported correlators are: `{}`."
                         .format(list(CORRMEASURES.keys())))
    return CORRMEASURES[corr]


def partial_correlation(dxz, dyz, corr="spearman"):
    """
    Calculate the partial correlation between columns of `dxz` and `dyz`.
    Relies on having previously calculated their residuals with respect to a
    control variable `z`.

    Parameters
    ----------
    dxz : 2-dimensional array
        Array of residuals of shape `(n_samples, nxfeats)`.
    dyz : 1-dimensional array
        Array of residuals of shape `(n_samples, )`.
    corr : string, optional
        Correlation statistic. By default `spearman`, supported
        are `[spearman, pearson]`. By default `"spearman"`.

    Returns
    -------
    out : 2-dimensional array
        Array of shape `(n_xfeats, 2)`, where the 2nd index represents the
        correlation coefficient and its p-value.
    """
    corr = _pick_correlation(corr)

    Nxfeat = dxz.shape[1]
    out = numpy.full((Nxfeat, 2), numpy.nan)
    # Calculate the correlation of every feature in dxz with dyz
    for i in range(Nxfeat):
        out[i, :] = corr(dxz[:, i], dyz)

    return out
