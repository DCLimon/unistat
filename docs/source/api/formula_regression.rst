========================
Formula-Based Regression
========================

Module for regression statistics based on ``patsy``-/``R``-style formulae.

This module provides an abstract base class and concrete implementations for
performing regression analyses using `statsmodels <statsmodels-homepage_>`_.
It supports linear regression, logistic regression, and log-binomial regression,
with features like variance inflation factor (VIF) calculation, standardized
regressions, odds/risk ratios, and formatted output.

Assumes input data are pandas ``Series``/``DataFrames``. Handles boolean columns
specially in standardization.

.. automodule:: unistat.formula_regression
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FormulaSides, reg, std_reg, X, y, X_std