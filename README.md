# Generalised Partial Correlation

Calculation of the [partial correlation](https://en.wikipedia.org/wiki/Partial_correlation) between $x$ and $y$ while controlling for the effect of a third variable $z$. The partial correlation is calculated as a correlation between $x$ and $y$ following a removal of their mean trend dependence on $z$.


The canonical partial correlation assumes that the mean is removed by a linear fit, whereas here any `scikit-learn` model can be used, in particular Gaussian process regressors.


## Installation

The package can be pip-installed with, e.g. `python -m pip install --user .` executed in a folder with `setup.py` of `gpc`.


# Usage

An example notebook is provided [here](https://github.com/Richard-Sti/gpc/blob/main/scripts/example.ipynb), however see below for a short example.



```python
...

model = LinearRegression()

dxz, dyz, fullout = gpc.get_reg_residuals(model, X, y, z, test_masks, verbose=False)
# Average the residuals over folds
dxz = gpc.fold_average(dxz)
dyz = gpc.fold_average(dyz)

pc = gpc.partial_correlation(dxz, dyz)
```

In the example above the model can be similarly replaced with a [Gaussian process regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).