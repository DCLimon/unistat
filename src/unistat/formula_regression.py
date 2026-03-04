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
import ast
import re
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# For functions giving separate results for Patsy formula LHS & RHS
FormulaSides = namedtuple('FormulaSides', ['lhs', 'rhs'])


class FormulaRegression(ABC):
    """Abstract base class for formula-based regression statistics.

    Provides common functionality for regression models, including data
    preparation, standardization, VIF calculation, and properties for
    regression results.

    Uses Wilkinson formulae (akin to R-style formulae), which are implemented
    in `statsmodels` via the `patsy` package.

    Parameters
    ----------
    formula : str
        `statsmodels`/`patsy`/`R`-style formula defining the regression model.
    data : pd.DataFrame
        Data for the model. `data` must contain (at a minimum) all variables
        referenced by `formula`.

    Attributes
    ----------
    X : pd.DataFrame
        Feature DataFrame used in model, after transformation by Patsy. Includes
        'Intercept' column.
    y : pd.Series
        Target Series used in model.
    reg : statsmodels regression result
        Fitted regression model.
    X_std : pd.DataFrame
        `X`, with all non-Boolean columns transformed to Z-scores.
    std_reg : statsmodels regression result
        Fitted standardized regression model.
    data : pd.DataFrame
        Input DataFrame after filtering for columns in formula, dropping NaNs,
        and converting all columns to `float`.
    """

    def __init__(self, formula: str, data: pd.DataFrame):
        self.formula = formula
        self._formula_cols = self.extract_formula_cols(self.formula)
        self.data = data[self._formula_cols.lhs + self._formula_cols.rhs]
        self.data = self._standardize_dtypes()
        self._coerce_occult_bools()

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

        if hasattr(self, 'std_reg'):
            print_string += f'{self.std_reg.summary2().as_text()}\n'

        return print_string

    @property
    def _X_num_names(self) -> list[str]:
        return (
            self.data[self._formula_cols.rhs]
            .select_dtypes(include='number')
            .columns.tolist()
        )

    def _standardize_dtypes(self):
        df = self.data.dropna()

        X_bool_cols: list[str] = (
            df[self._formula_cols.rhs]
            .select_dtypes(include=['bool', 'boolean'])
            .columns.tolist()
        )
        X_num_cols: list[str] = self._X_num_names

        # Ensure y is float
        for col in self._formula_cols.lhs:
            df[col] = df[col].astype(float)

        # Ensure all bools are boolean, nums are floats
        df[X_bool_cols] = df[X_bool_cols].astype(bool)
        df[X_num_cols] = df[X_num_cols].astype(float)

        return df

    def _coerce_occult_bools(self) -> None:
        for col in self._X_num_names:
            # If only values in column are 0 & 1:
            # - coerce column to bool
            # - edit formula to add C()
            if self.data[col].astype(float).isin([0, 1]).all():
                self.data[col] = self.data[col].astype(bool)
                self.formula = self.formula.replace(
                    col, f'C({col}, Treatment(False))'
                )

    @staticmethod
    def _extract_cols_from_termlist(termlist: list[str]) -> list[str]:
        """Extract unique column names from a patsy termlist.

        `termlist` can derive from the left- (LHS) or right-hand (RHS) side of
        a `patsy` formula, but not both.

        This function parses each term in the termlist using patsy, then
        processes each factor to identify underlying column names, handling
        special cases like C(), Q(), I(), and function calls (e.g., np.log).

        Parameters
        ----------
        termlist : list[str]
            `patsy`-derived list of terms from LHS/RHS of a `patsy` formula.

        Returns
        -------
        list[str]
            Sorted list of unique column names extracted from the formula.

        Examples
        --------
        >>> extract_cols_from_termlist([
        ... 'ca_grams_4h', 'C(ca_gluconate_bool)', 'ca_grams_4h',
        ... 'C(ca_gluconate_bool)', 'I(age ** 2)', 'np.log(iss)', 'C(moi_pen)',
        ... 'C(sex_female)', 'mortality_24h'
        ... ])
        [
            'age',
            'ca_gluconate_bool',
            'ca_grams_4h',
            'iss',
            'moi_pen',
            'mortality_24h',
            'sex_female',
        ]
        """
        cols = set()
        for term in termlist:
            for factor in term.factors:
                code = factor.code.strip()
                if code.startswith('C(') or code.startswith('Q('):
                    inner = code[code.find('(') + 1:code.rfind(')')].strip()
                    if ',' in inner:
                        inner = inner.split(',', 1)[0].strip()
                    if inner.startswith(("'", '"')) and inner.endswith(("'", '"')):
                        inner = inner[1:-1]
                    cols.add(inner)
                elif code.startswith('I('):
                    expr = code[2:-1].strip()
                    tree = ast.parse(expr)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            cols.add(node.id)
                elif '.' in code:  # e.g., np.log(col)
                    func, arg = code.rsplit('.', 1)
                    if arg.endswith(')'):
                        arg = arg[arg.find('(') + 1: arg.rfind(')')].strip()
                    cols.add(arg)
                else:
                    cols.add(code)
        return list(cols)

    @staticmethod
    def extract_formula_cols(formula: str) -> FormulaSides:
        """Extract unique column names from a `patsy`/`R` formula string.

        This function parses the formula using `patsy`, then processes each factor
        to identify underlying column names, handling special cases like C(), Q(),
        I(), and function calls (e.g., np.log).

        Parameters
        ----------
        formula : str
            The patsy-compatible formula string (e.g., 'y ~ x + C(cat) + np.log(z)').

        Returns
        -------
        list[str]
            Sorted list of unique column names extracted from the formula.

        Examples
        --------
        >>> extract_columns_from_formula(
        ...     'mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2) '
        ...     '+ np.log(iss) + C(moi_pen) + C(sex_female)'
        ... )
        [
            'age',
            'ca_gluconate_bool',
            'ca_grams_4h',
            'iss',
            'moi_pen',
            'mortality_24h',
            'sex_female'
        ]
        """
        # Use Patsy ModelDesc to structure the formula
        model_desc = patsy.ModelDesc.from_formula(formula)
        result = FormulaSides(
            lhs=FormulaRegression._extract_cols_from_termlist(
                model_desc.lhs_termlist
            ),
            rhs=FormulaRegression._extract_cols_from_termlist(
                model_desc.rhs_termlist
            )
        )

        return result

    @abstractmethod
    def _run_regression(self, standardize: bool = False):
        if not standardize:
            model = sm.OLS.from_formula(self.formula, self.data)
            self._reg_cache = model.fit()
            return self._reg_cache
        elif self._formula_std is None or self._data_std is None:
            return None
        else:
            model = sm.OLS.from_formula(self._formula_std, self._data_std)
            self._std_reg_cache = model.fit()
            return self._std_reg_cache

    @property
    def reg(self):
        # Run regression if never done prior (which defines self._reg_cache)
        if not hasattr(self, '_reg_cache'):
            self._run_regression(standardize=False)
        return self._reg_cache

    @property
    def std_reg(self):
        # Run regression if never done prior (which defines self._reg_cache)
        if not hasattr(self, '_std_reg_cache'):
            self._run_regression(standardize=True)
        return self._std_reg_cache

    @property
    def _X_num_std_names(self) -> list[str]:
        """
        Lazily computed list of standardized numeric predictor column names.

        Returns the original numeric RHS column names with '_std' appended.
        Empty list if no numeric predictors.

        Cached for efficiency; invalidate with ``del self._X_num_std_names_cache``.
        """
        if not hasattr(self, '_X_num_std_names_cache'):
            num_cols = self._X_num_names
            self._X_num_std_names_cache = (
                [col + '_std' for col in num_cols] if num_cols else []
            )
        return self._X_num_std_names_cache

    @property
    def _data_std(self) -> pd.DataFrame | None:
        """
        Lazily computed version of ``self.data`` with numeric RHS columns standardized.

        Numeric columns are centered and scaled (ddof=1) via ``patsy.standardize``,
        renamed with '_std' suffix, and appended after non-numeric columns.
        Non-numeric columns are unchanged.

        Returns ``None`` if no numeric predictors.

        Cached for efficiency; invalidate with ``del self._data_std_cache``.
        """
        if not hasattr(self, '_data_std_cache'):
            num_cols = self._X_num_names
            std_names = self._X_num_std_names

            if not num_cols:
                self._data_std_cache = None
            else:
                std_df = self.data.copy()

                # Standardize only numeric RHS columns
                standardized = patsy.standardize(
                    self.data[num_cols],
                    center=True,
                    rescale=True,
                    ddof=1
                )

                # Rename to *_std
                standardized = standardized.rename(
                    columns=dict(zip(num_cols, std_names))
                )

                # Drop original numerics and append standardized versions
                std_df = std_df.drop(columns=num_cols)
                std_df = pd.concat([std_df, standardized], axis=1)

                self._data_std_cache = std_df

        return self._data_std_cache

    @property
    def _formula_std(self) -> str | None:
        """
        Lazily computed formula where numeric RHS columns are replaced with their _std versions.
        Returns None if no numeric columns to standardize.
        """
        if not hasattr(self, '_formula_std_cache'):
            if not self._X_num_names:
                self._formula_std_cache = None
            else:
                f = self.formula
                for orig, std in zip(self._X_num_names, self._X_num_std_names):
                    f = re.sub(r'\b' + re.escape(orig) + r'\b', std, f)
                self._formula_std_cache = f
        return self._formula_std_cache

    @property
    def _dmatrices(self):
        dmatrices = patsy.highlevel.dmatrices(
            self.formula,
            self.data,
            return_type='dataframe'
        )

        # Make LHS a series if only single outcome column
        lhs = (
            dmatrices[0].iloc[:, 0]
            if len(dmatrices[0]) == 1
            else dmatrices[0]
        )

        return FormulaSides(
            lhs=lhs,
            rhs=dmatrices[1]
        )

    @property
    def X(self):
        return self._dmatrices.rhs

    @property
    def y(self):
        return self._dmatrices.lhs

    @property
    def _dmatrices_std(self):
        dmatrices = patsy.highlevel.dmatrices(
            self._formula_std,
            self._data_std,
            return_type='dataframe'
        )

        # Make LHS a series if only single outcome column
        lhs = (
            dmatrices[0].iloc[:, 0]
            if len(dmatrices[0]) == 1
            else dmatrices[0]
        )

        return FormulaSides(
            lhs=lhs,
            rhs=dmatrices[1]
        )

    @property
    def X_std(self):
        return self._dmatrices_std.rhs

    @property
    def _y_std(self):
        return self._dmatrices_std.lhs

    def vif_matrix(self, drop_ints: bool = False):
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
        if len(self.X) < 2:
            raise ValueError('VIF requires at least 2 modelled X columns')

        X = self.X
        if drop_ints:
            X = X.drop(columns=[col for col in X.columns if ':' in col])

        vif_matrix = pd.Series(
            [variance_inflation_factor(X.values, i)
             for i in range(X.shape[1])],
            name='vif',
            index=X.columns,
        )

        return vif_matrix

    
class FormulaLogit(FormulaRegression):
    def __init__(self, formula: str, data: pd.DataFrame):
        super().__init__(formula, data)

    def __str__(self):
        """String representation with VIF, summary, and odds ratios.

        Returns
        -------
        str
            Formatted results.
        """
        with pd.option_context('display.max_colwidth', None,
                               'max_colwidth', None,
                               'display.width', 256):
            if len(self.X.columns) >= 2:
                print_string = (
                    f'{str(self.vif_matrix())}\n'
                    f'{self.reg.summary2().as_text()}\n'
                    f'{self.logit_or().to_string()}\n'
                )
            else:
                print_string = (
                    f'{self.reg.summary2().as_text()}\n'
                    f'{self.logit_or().to_string()}\n'
                )

            if hasattr(self, 'std_reg'):
                print_string += (
                    f'{self.std_reg.summary2().as_text()}\n'
                    f'{self.logit_or(standardize=True).to_string()}\n'
                )

            return print_string

    def _run_regression(self, standardize: bool = False):
        if not standardize:
            model = sm.Logit.from_formula(self.formula, self.data)
            self._reg_cache = model.fit()
            return self._reg_cache
        elif self._formula_std is None or self._data_std is None:
            return None
        else:
            model = sm.Logit.from_formula(self._formula_std, self._data_std)
            self._std_reg_cache = model.fit()
            return self._std_reg_cache

    def logit_or(self, standardize: bool = False):
        if not standardize:
            model = self.reg
        elif self.std_reg is None:
            raise ValueError('Standardized logit cannot be run when all '
                             'predictors are categorical/boolean.')
        else:
            model = self.std_reg

        output = pd.DataFrame(model.params, columns=['OR'])
        output[['95% CI lower', '95% CI upper']] = model.conf_int()
        return output.map(np.exp)


class FormulaLinReg(FormulaRegression):
    def __init__(self, formula: str, data: pd.DataFrame):
        super().__init__(formula, data)

    def _run_regression(self, standardize: bool = False):
        if not standardize:
            model = sm.OLS.from_formula(self.formula, self.data)
            self._reg_cache = model.fit()
            return self._reg_cache
        elif self._formula_std is None or self._data_std is None:
            return None
        else:
            model = sm.OLS.from_formula(self._formula_std, self._data_std)
            self._std_reg_cache = model.fit()
            return self._std_reg_cache
