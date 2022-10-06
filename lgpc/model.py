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

import numpy
from sklearn.base import clone
from tqdm import tqdm

from .utils import train_test_from_mask


def run_gpr(gpr, x, y, test_mask=None, weights=None, clone_gpr=True):
    """
    Fit a 1D Gaussian process regressor (GPR) and return the predicted value
    and score. If available performs a train-test split to fit and evaluate
    the GPR.

    Parameters
    ----------
    gpr : py:class:`sklearn.gaussian_process.GaussianProcessRegressor`
        The unfitted GPR instance.
    x : 1-dimensional array
        The input samples of shape (n_samples, ).
    y : 1-dimensional array
        The target values of shape (n_samples, ).
    test_mask : 1-dimensional array, optional
        Boolean array of shape `(n_samples, )`, where `True` indicates that the
        sample belongs to the test set. If `None` does not perform a train-test
        split.
    weights : 1-dimensional array, optional
        The target weights. GPR does not consider weights while fitting, used
        only for scoring. By default `None`.
    return_gpr: bool, optional
        Whether to also return the fitted GPR. By default False.
    clone_gpr : bool, optional
        Whether to clone the GPR instance. By default True.

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

    gpr = clone(gpr) if clone_gpr else gpr
    gpr.fit(x[train], y[train])

    ypred = gpr.predict(x)
    score = gpr.score(x[test], y[test],
                      weights[test] if weights is not None else None)
    return ypred, score


def run_gpr_folds(gpr, x, y, test_masks, weights=None, verbose=True, return_full=False):

    Nrepeat, Nsamples = test_masks.shape

    ypred = numpy.full((Nrepeat, Nsamples), numpy.nan)
    score = numpy.full(Nrepeat, numpy.nan)


    iters = range(Nrepeat)
    iters = tqdm(iters) if verbose else iters
    for i in iters:
        mask = test_masks[i, :]
        ypred[i, :], score[i] = run_gpr(gpr, x, y, mask, weights=weights)

    ypred = ypred if return_full else numpy.mean(ypred, axis=0)
    return ypred, score


def partial_correlation(x, y, z, gpr, test_mask=None, partial=False, verbose=True):
    """Simplify theif else statements. Allow for X having more dimensions."""

    from scipy.stats import spearmanr


    if test_mask.ndim > 1:
        xz, score_xz = run_gpr_folds(gpr, z, x, test_mask, verbose=verbose)
        if partial:
            yz, score_yz = run_gpr_folds(gpr, z, y, test_mask, verbose=verbose)
        else:
            yz, score_yz = y, numpy.full_like(score_xz, numpy.nan)
    else:
        xz, score_xz = run_gpr(gpr, z, x, test_mask)
        if partial:
            yz, score_yz = run_gpr(gpr, z, y, test_mask)
        else:
            yz, score_yz = y, numpy.nan

    dxz = x - xz
    dyz = y - yz

    print(score_xz, score_yz)
    print(spearmanr(x, y))
    print(spearmanr(dxz, dyz))
    return dxz, dyz

