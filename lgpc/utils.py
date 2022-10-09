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

"""Various support functions."""

import numpy


def train_test_from_mask(test_mask):
    """
    Return train and test indices from a test mask.

    Parameters
    ----------
    test_mask : 1-dimensional array
        Boolean array of shape `(n_samples, )`, where `True` indicates that the
        sample belongs to the test set.

    Returns
    -------
    train : 1-dimensional array
        Training set indices.
    test : 1-dimensional array
        Test set indices.
    """
    if test_mask.ndim != 1:
        raise TypeError("`test_mask` must be a 1-dimensional array.")
    _x = numpy.arange(test_mask.size)
    return _x[~test_mask], _x[test_mask]


def fold_average(arr, weights=None):
    """
    Calculates average over folds of an `n`-dimensional array, assuming that
    the folds are under the last index.

    Parameters
    ----------
    arr : n-dimensional array
        Array to be averaged over the last index.
    weights : NOT SUPPORTED
        DESCR.

    Returns
    -------
    arr : n-dimensional array
        Array averaged over the last index.
    """
    if weights is not None:
        raise NotImplementedError("Weighting schemes are not implemented yet.")
    return numpy.mean(arr, axis=-1)


def kernel_weights(p, loc, scale, kernel):
    """
    Probability density at `p` of a kernel defined by `loc` and `scale`.

    Parameters
    ----------
    p : 1-dimensional array
        Where to evaluate the kernel distribution.
    loc : float
        The location of the distribution.
    scale : float
        The scale of the distribution.
    kernel : str
        Kernel, allowed choices are `["gaussian", "tophat"]`.

    Returns
    -------
    weights : 1-dimensional array
        The kernel weights.
    """
    allowed = ["gaussian", "tophat"]

    if kernel == "gaussian":
        dist = stats.norm(loc, scale)
    elif kernel == "tophat":
        dist = stats.norm(loc, scale)
    else:
        raise ValueError("Allowed kernels are `{}`.".format(allowed))

    weights = dist.pdf(p)
    weights /= weights.sum()
    return weights
