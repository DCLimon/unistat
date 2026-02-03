"""Module for regression statistics.

This module provides an abstract base class and concrete implementations for
performing regression analyses using statsmodels. It supports linear regression,
logistic regression, and log-binomial regression, with features like variance
inflation factor (VIF) calculation, standardized regressions, odds/risk ratios,
and formatted output.

Dependencies
------------
* abc: For abstract base classes.
* numpy: For numerical operations.
* pandas: For data manipulation.
* scipy: For statistical functions (z-score).
* statsmodels: For regression models and VIF.

Classes
-------
RegressionStats
    Abstract base class for regression statistics.
LogitStats
    Class for logistic regression statistics.
LinRegStats
    Class for linear regression statistics.
LogBinStats
    Experimental class for log-binomial regression statistics.

Notes
-----
Assumes input data are pandas Series/DataFrames. Handles boolean columns
specially in standardization.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .exceptions import warn_experimental


class RegressionStats(ABC):
    """Abstract base class for regression statistics.

    Provides common functionality for regression models, including data
    preparation, standardization, VIF calculation, and properties for
    regression results.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Independent variables (features).
    y : pd.Series
        Dependent variable (target).
    bool_col_names : list[str] or str or None, optional
        Names of boolean columns in X to exclude from standardization.

    Attributes
    ----------
    bool_cols : list[str] or None
        List of boolean column names.
    X : pd.DataFrame
        Feature DataFrame. `NaN` values are removed and all columns are
        converted to `float64`.
    y : pd.DataFrame
        Target DataFrame. `NaN` values are removed and all columns are converted
        to `float64`.
    reg : statsmodels regression result
        Fitted regression model.
    X_std : pd.DataFrame
        `X`, with all non-Boolean columns transformed to Z-scores.
    std_reg : statsmodels regression result
        Fitted standardized regression model.
    _df : pd.DataFrame
        Concatenated DataFrame of X and y.

    Notes
    -----
    Observations with any missing data in either `X` *or* `y` are dropped.
    """

    def __init__(self, X, y, bool_col_names: list | str | None = None):
        self._df = self._concat_xy(X, y)
        self.bool_cols = bool_col_names
        self.X = (
            self._df.drop(columns=[y.name])
            .apply(pd.to_numeric, errors='coerce')
            .astype('float64')
        )
        self.y = (
            self._df[[y.name]]
            .apply(pd.to_numeric, errors='coerce')
            .astype('float64')
        )
        self.reg = None
        self.std_reg = None

    def __str__(self):
        """
        String representation of the regression results.

        Returns
        -------
        str
            Formatted string with VIF (if applicable) and regression summaries.
        """
        if len(self.X.columns) >= 2:
            print_string = (f'{str(self.vif_matrix())}\n'
                            f'{self.reg.summary2().as_text()}\n')
        else:
            print_string = f'{self.reg.summary2().as_text()}\n'

        if not self._all_bool_cols:
            print_string += f'{self.std_reg.summary2().as_text()}\n'

        return print_string

    @property
    def _all_bool_cols(self):
        """Check if all columns in X are boolean.

        Returns
        -------
        bool
            True if all columns are boolean, False otherwise.
        """
        if self.bool_cols is None:
            return False
        else:
            return len(self.bool_cols) == self.X.shape[1]

    @property
    def bool_cols(self):
        """Get the list of boolean column names.

        Returns
        -------
        list[str] or None
            List of boolean columns.
        """
        return self._bool_cols

    @bool_cols.setter
    def bool_cols(self, bool_col_names):
        if isinstance(bool_col_names, str):
            self._bool_cols = [bool_col_names]
        elif isinstance(bool_col_names, list):
            self._bool_cols = bool_col_names
        elif bool_col_names is None:
            self._bool_cols = None
        else:
            raise ValueError('bool_col_names must be a str or list of str')

    @property
    def X_std(self):
        """Standardized version of X (z-scores), excluding boolean columns.

        Returns
        -------
        pd.DataFrame
            Standardized features.
        """
        # Create a standardized version of feature column (z-score)
        if self.bool_cols is not None:
            X_num = self.X[[col for col in self.X.columns
                            if col not in self.bool_cols]]
            X_num_std = X_num.apply(stats.zscore, axis='index', ddof=1,
                                    nan_policy='omit')
            X_num_std.columns = [f'{col}_std' for col in X_num_std.columns]
            X_std = pd.concat(
                [
                    X_num_std,
                    self.X[self.bool_cols]
                ],
                axis='columns'
            )
        else:
            X_std = self.X.apply(stats.zscore, axis='index', ddof=1,
                                 nan_policy='omit')
            X_std.columns = [f'{col}_std' for col in X_std.columns]
        return X_std

    @abstractmethod
    def _run_regression(self, standardize: bool = False):
        """Abstract method to run the regression model.

        Parameters
        ----------
        standardize : bool, optional
            Whether to use standardized features. Defaults to False.

        Returns
        -------
        statsmodels regression result
            Fitted model.
        """
        if not standardize:
            exog = self.X
        else:
            exog = self.X_std
        endog = self.y

        reg = sm.OLS(
            endog=endog,
            exog=sm.add_constant(exog),
            missing='drop'
        ).fit()

        return reg

    @property
    def reg(self):
        """Get the fitted regression model.

        Returns
        -------
        statsmodels regression result
            Fitted model.
        """
        return self._reg

    @reg.setter
    def reg(self, result):
        if result is None:
            self._reg = self._run_regression()
        else:
            self._reg = result

    @property
    def std_reg(self):
        """Get the fitted standardized regression model.

        Returns
        -------
        statsmodels regression result
            Fitted standardized model.
        """
        return self._std_reg

    @std_reg.setter
    def std_reg(self, result):
        if result is None:
            self._std_reg = self._run_regression(standardize=True)
        else:
            self._std_reg = result

    def vif_matrix(self):
        """Calculate variance inflation factors (VIF) for feature DataFrame.

        Returns
        -------
        pd.Series
            VIF values for each feature.

        Raises
        ------
        ValueError
            If fewer than 2 columns in X.
        """
        if len(self.X.columns) < 2:
            raise ValueError('VIF requires at least 2 columns')
        X = self.X.dropna().astype(float)

        vif_matrix = pd.Series(
            [variance_inflation_factor(X.values, i)
             for i in range(X.shape[1])],
            name='vif',
            index=X.columns,
        )

        return vif_matrix

    @staticmethod
    def _concat_xy(X, y):
        """Concatenate X and y, handle data types, drop NaNs.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Features.
        y : pd.Series
            Target.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame.
        """
        df = pd.concat([X, y], axis='columns').dropna(axis='index')
        for col in df.columns:
            if df[col].dtype == pd.BooleanDtype():
                df[col] = df[col].astype(int)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)
            elif df[col].dtype == pd.Int64Dtype():
                df[col] = df[col].astype(int)
        return df


class LogitStats(RegressionStats):
    """Class for logistic regression statistics.

    Extends RegressionStats for logistic regression using Logit model.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Predictor observations.
    y : pd.Series
        Binary outcome observations.
    bool_col_names : list[str] or str or None, optional
        String names of Boolean columns to exclude from standardization.
    """
    def __init__(self, X, y, bool_col_names: list | str | None = None):
        super().__init__(X, y, bool_col_names)

    def __str__(self):
        """String representation with VIF, summary, and odds ratios.

        Returns
        -------
        str
            Formatted results.
        """
        if len(self.X.columns) >= 2:
            print_string = (
                f'{str(self.vif_matrix())}\n'
                f'{self.reg.summary2().as_text()}\n'
                f'{self.logit_or()}\n'
            )
        else:
            print_string = (
                f'{self.reg.summary2().as_text()}\n'
                f'{self.logit_or()}\n'
            )

        if not self._all_bool_cols:
            print_string += (
                f'{self.std_reg.summary2().as_text()}\n'
                f'{self.logit_or(standardize=True)}\n'
            )

        return print_string

    def _run_regression(self, standardize: bool = False):
        """Fit logistic regression.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized features. Defaults to False.

        Returns
        -------
        statsmodels.discrete.discrete_model.LogitResults
            Fitted model.
        """
        if not standardize:
            exog = self.X
        else:
            if self._all_bool_cols:
                return None
            else:
                exog = self.X_std
        endog = self.y

        logit = sm.Logit(
            endog=endog,
            exog=sm.add_constant(exog),
            missing='drop'
        ).fit()

        return logit

    def logit_or(self, standardize: bool = False) -> pd.DataFrame:
        """Calculate odds ratios by predictor, with 95% confidence intervals.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized model. Defaults to False.

        Returns
        -------
        pd.DataFrame
            Odds ratios with 95% CI.

        Raises
        ------
        ValueError
            If all columns are boolean and standardize is True.
        """
        if standardize:
            if not self._all_bool_cols:
                model = self.std_reg
            else:
                raise ValueError('Standardized logit cannot be run when all '
                                 'columns are boolean.')
        else:
            model = self.reg

        output = pd.DataFrame(model.params, columns=['OR'])
        output[['95% CI lower', '95% CI upper']] = model.conf_int()
        return output.map(np.exp)

    def pretty_print_or(self,
                        standardize: bool = False,
                        label: bool = True) -> None:
        """Print formatted ORs & 95% CIs for easy copy-pasting.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized model. Defaults to False.
        label : bool, optional
            Include parameter labels. Defaults to True.

        Raises
        ------
        ValueError
            If all columns are boolean and standardize is True.
        """
        if standardize:
            if self._all_bool_cols:
                raise ValueError('Standardized logit cannot be run when all '
                                 'columns are boolean.')
            else:
                ratios = self.logit_or(standardize=True)
        else:
            ratios = self.logit_or()

        for idx in ratios.index:
            row = ratios.loc[idx, :]

            print_string = (
                f'OR {row['OR']:.2f} '
                f'({row['95% CI lower']:.2f} - {row['95% CI upper']:.2f})'
            )
            if label:
                print_string = f'{idx}: ' + print_string

            print(print_string)


class LinRegStats(RegressionStats):
    r"""Class for linear regression statistics.

    Extends RegressionStats for ordinary least squares (OLS) regression.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Predictor observations.
    y : pd.Series
        Numeric outcome observations.
    bool_col_names : list[str] or str or None, optional
        Boolean columns to exclude from standardization.

    Notes
    -----
    `unistat` does NOT standardize the values of `y` for linear regression.
    Typically, "standardized regression" refers to a transformation of
    :math:`y \sim X` such that
    :math:`\text{SD}\left( y \right) \sim \text{SD}\left( X \right)`; a
    coefficient :math:`\beta` is thus interpreted as a 1-S.D. increase in
    :math:`X` conferring a :math:`\beta` S.D. increase in :math:`y`. We find
    this to be difficult to interpret, with no benefit beyond adherence to
    convention.

    Instead, `unistat` opts for "X-standardized regression". That is, since only
    :math:`X` is Z-scored, :math:`y \sim X` is transformed such that
    :math:`y \sim \text{SD}\left( X \right)`. Here, a coefficient :math:`\beta`
    is interpreted as a 1-S.D. increase in :math:`X` conferring an absolute
    increase of :math:`\beta` units in :math:`y`. This is more easily
    interpretable, while still allowing comparison of the relative strengths of
    all :math:`X` predictors.
    """

    def __init__(self, X, y, bool_col_names: list | str | None = None):
        super().__init__(X, y, bool_col_names)

    def _run_regression(self, standardize: bool = False):
        """Run OLS linear regression.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized features. Defaults to False.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResults
            Fitted model.
        """
        if not standardize:
            exog = self.X
        else:
            if self._all_bool_cols:
                return None
            else:
                exog = self.X_std
        endog = self.y

        reg = sm.OLS(
            endog=endog,
            exog=sm.add_constant(exog),
            missing='drop'
        ).fit()

        return reg

    def pretty_print_coefs(self,
                           standardize: bool = False,
                           label: bool = True) -> None:
        """
        Print formatted regression coefficients for easy copy-pasting.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized model. Defaults to False.
        label : bool, optional
            Include parameter labels. Defaults to True.

        Raises
        ------
        ValueError
            If all columns are boolean and standardize is True.
        """
        # Use standardized regression results as necessary
        if standardize:
            if self._all_bool_cols:
                raise ValueError('Standardized LinReg cannot be run when all '
                                 'columns are boolean.')
            else:
                reg = self.std_reg
        else:
            reg = self.reg

        # Make df of coefs & 95% CIs
        coef_df = pd.DataFrame(
            data={
                'Coef': reg.params,
                '95% CI lower': reg.conf_int()[0],
                '95% CI upper': reg.conf_int()[1],
            }
        )

        for idx in coef_df.index:
            row = coef_df.loc[idx, :]

            print_string = (
                f'{row['Coef']:.3f} '
                f'({row['95% CI lower']:.3f} - {row['95% CI upper']:.3f})'
            )
            if label:
                print_string = f'{idx}: ' + print_string

            print(print_string)


class LogBinStats(RegressionStats):
    """Class for log-binomial regression statistics (experimental).

    Extends RegressionStats for generalized linear model with binomial family
    and log link.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Independent variables.
    y : pd.Series
        Dependent variable (binary).
    bool_col_names : list[str] or str or None, optional
        Boolean columns to exclude from standardization.

    Warnings
    --------
    This class is experimental; use with caution and verify results.
    """

    def __init__(self, X, y, bool_col_names: list | str | None = None):
        warn_experimental('''
            This feature is experimental and has not been fully tested nor 
            optimized; calculations may be incorrect, and/or errors may occur. 
            In critical applications, output should be verified for accuracy 
            and/or LogitStats preferred instead.
        ''')
        super().__init__(X, y, bool_col_names)

    def __str__(self):
        """String representation with VIF, summary, and risk ratios.

        Returns
        -------
        str
            Formatted results.
        """
        if len(self.X.columns) >= 2:
            print_string = (
                f'{str(self.vif_matrix())}\n'
                f'{self.reg.summary2().as_text()}\n'
                f'{self.logbin_rr()}\n'
            )
        else:
            print_string = (
                f'{self.reg.summary2().as_text()}\n'
                f'{self.logbin_rr()}\n'
            )

        if not self._all_bool_cols:
            print_string += (
                f'{self.std_reg.summary2().as_text()}\n'
                f'{self.logbin_rr(standardize=True)}\n'
            )

        return print_string

    def _run_regression(self, standardize: bool = False):
        """Run log-binomial regression.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized features. Defaults to False.

        Returns
        -------
        statsmodels.genmod.generalized_linear_model.GLMResults
            Fitted model.
        """
        if not standardize:
            exog = self.X
        else:
            if self._all_bool_cols:
                return None
            else:
                exog = self.X_std
        endog = self.y

        logbin = sm.GLM(
            endog=endog,
            exog=sm.add_constant(exog),
            family=sm.families.Binomial(link=sm.families.links.Log()),
            missing='drop'
        ).fit(
            start_params=(
                [np.log(0.5)]
                + [np.float64(0) for i in range(0, len(self.X.columns))]
            ),
            method='irls',
            maxiter=1000
        )

        return logbin

    def logbin_rr(self, standardize: bool = False) -> pd.DataFrame:
        """Calculate risk ratios by predictor with 95% confidence intervals.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized model. Defaults to False.

        Returns
        -------
        pd.DataFrame
            Risk ratios with 95% CI.

        Raises
        ------
        ValueError
            If all columns are boolean and standardize is True.
        """
        if standardize:
            if not self._all_bool_cols:
                model = self.std_reg
            else:
                raise ValueError('Standardized logit cannot be run when all '
                                 'columns are boolean.')
        else:
            model = self.reg

        output = pd.DataFrame(np.exp(model.params), columns=['RR'])
        output[['95% CI lower', '95% CI upper']] = model.conf_int()
        return output.map(np.exp)

    def pretty_print_rr(self,
                        standardize: bool = False,
                        label: bool = True) -> None:
        """Print formatted risk ratios w/ 95% CIs for easy copy-pasting.

        Parameters
        ----------
        standardize : bool, optional
            Use standardized model. Defaults to False.
        label : bool, optional
            Include parameter labels. Defaults to True.

        Raises
        ------
        ValueError
            If all columns are boolean and standardize is True.
        """
        if standardize:
            if self._all_bool_cols:
                raise ValueError('Standardized logit cannot be run when all '
                                 'columns are boolean.')
            else:
                ratios = self.logbin_rr(standardize=True)
        else:
            ratios = self.logbin_rr()

        for idx in ratios.index:
            row = ratios.loc[idx, :]

            print_string = (
                f'OR {row['RR']:.2f} '
                f'({row['95% CI lower']:.2f} - {row['95% CI upper']:.2f})'
            )
            if label:
                print_string = f'{idx}: ' + print_string

            print(print_string)
