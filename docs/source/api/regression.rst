=================
Simple Regression
=================

Module for regression statistics.

This module provides an abstract base class and concrete implementations for
performing regression analyses using statsmodels. It supports linear regression,
logistic regression, and log-binomial regression, with features like variance
inflation factor (VIF) calculation, standardized regressions, odds/risk ratios,
and formatted output.

Assumes input data are pandas Series/DataFrames. Boolean columns are not subject
to standardization.

.. automodule:: unistat.regression
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: bool_cols, reg, std_reg, X_std