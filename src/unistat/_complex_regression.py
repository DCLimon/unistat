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
from typing import Optional
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.api as sm
from statsmodels import formula
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .exceptions import warn_experimental
from .regression import RegressionStats


class ComplexRegression:
    def __init__(self,
                 data: pd.DataFrame,
                 y: str,
                 X_num: list[str] | str = None,
                 X_bool: list[str] | str = None,
                 X_cat: list[str] | str = None,
                 X_int: list[list[str]] | list[str] = None):
        self._X_num_names, self._X_bool_names, self._X_cat_names = (
            self._format_X_args(X_num, X_bool, X_cat)
        )

        if isinstance(y, str) and len(y) > 0:
            self._y_name = y
        else:
            raise TypeError('y must be a non-empty string.')

        self._X_int_names = self._format_X_int(X_int)
        self._ensure_X_int_terms_duplicated()

        self._data = data[self._unique_col_names].dropna()
        self._df = self._standardize_dtypes(self._data)

    @staticmethod
    def _format_X_args(*args: list[str] | str | None):
        """Check & format X_num/X_bool/X_cat during instantiation."""

        def format_X_arg(X_arg):
            # Return None if None
            if X_arg is None:
                return None
            # str -> list[str] is X_arg is string
            elif isinstance(X_arg, str):
                if len(X_arg) != 0:
                    return [X_arg]
                else:
                    raise TypeError('X_num, X_bool, and X_cat cannot contain '
                                    'empty strings.')
            elif isinstance(X_arg, list):
                if all(isinstance(item, str) for item in X_arg):
                    if not any(len(string) == 0 for string in X_arg):
                        return X_arg
                    else:
                        raise TypeError('X_num, X_bool, and X_cat cannot '
                                        'contain empty strings.')
                else:
                    raise TypeError('X_num, X_bool, and X_cat must be '
                                    'list[str] or str.')
            else:
                raise TypeError('X_num, X_bool, and X_cat must be list[str] '
                                'or str.')

        if len(args) == 1:
            return format_X_arg(args[0])

        output = []
        for arg in args:
            output.append(format_X_arg(arg))
        return tuple(output)

    @staticmethod
    def _format_X_int(X_int: list[list[str]] | list[str] | None):
        """Check & format X_int during instantiation."""
        if X_int is None:
            return None
        # If X_int is list[Any]
        elif isinstance(X_int, list):
            # If X_int is list[str], self.X_int: list[list[str]]
            if all(isinstance(item, str) for item in X_int):
                # Ensure no empty strings
                if any(len(substring) == 0 for substring in X_int):
                    raise TypeError('X_int cannot contain empty strings.')
                else:
                    return [X_int]
            # If X_int is list[list[Any]]
            elif all(isinstance(item, list) for item in X_int):
                for sublist in X_int:
                    # Raise error if list[list[str]]
                    if not all(isinstance(subitem, str) for subitem in sublist):
                        raise TypeError('X_int must be '
                                        'list[list[str]] or list[str].')
                    # Ensure no empty strings
                    elif any(len(substring) == 0 for substring in sublist):
                        raise TypeError('X_int cannot contain empty strings.')
                    else:
                        return X_int
            else:
                raise TypeError('X_int must be list[list[str]] or list[str].')
        else:
            raise TypeError('X_int must be list[list[str]] or list[str].')

    def _ensure_X_int_terms_duplicated(self):
        """Ensure all X_int terms appear in X_num, X_bool, X_cat.

        Necessary to ensure correct typing.
        """
        if self._X_int_names is not None:
            int_cols = set(
                substring
                for sublist in self._X_int_names
                for substring in sublist
            )

            X_lists = [self._X_num_names, self._X_bool_names,
                             self._X_cat_names]
            X_lists = [X_list for X_list in X_lists if X_list is not None]
            other_X_cols = set(
                col
                for X_list in X_lists
                for col in X_list
            )

            if not int_cols.issubset(other_X_cols):
                raise ValueError('All terms listed in X_int must also appear in '
                                 'X_num, X_bool, or X_cat.')

    @property
    def _unique_col_names(self) -> list[str]:
        unique_col_names = [self._y_name]
        for X in [self._X_num_names, self._X_bool_names, self._X_cat_names]:
            if X is not None:
                unique_col_names.extend(X)

        return unique_col_names

    def _standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._X_num_names is not None:
            df[self._X_num_names] = df[self._X_num_names].astype(float)
        if self._X_bool_names is not None:
            df[self._X_bool_names] = df[self._X_bool_names].astype(bool)
        if self._X_cat_names is not None:
            df[self._X_cat_names] = df[self._X_cat_names].astype('category')

        return df

    def _names_to_terms(self):
        # Create _X_int_terms
        if self._X_int_names is None:
            self._X_int_terms = None
            flat_int_terms = []
        else:
            self._X_int_terms = []  # self._X_int_names with C() added
            # For each interaction
            for sublist in self._X_int_names:
                # Create blank list to add to self._X_int_terms
                sublist_xformed = []
                # For each interaction term
                for int_term in sublist:
                    # Transform with C() if int_term is Boolean
                    if self._X_bool_names and (int_term in self._X_bool_names):
                        sublist_xformed.append(f'C({int_term})')
                    # Transform with C() if int_term is Categorical
                    elif self._X_cat_names and (int_term in self._X_cat_names):
                        sublist_xformed.append(f'C({int_term})')
                    # No transform if int_term is numeric
                    elif self._X_num_names:
                        sublist_xformed.append(int_term)
                self._X_int_terms.append(sublist_xformed)
            # Create flattened _X_int_terms to check non-interactional terms
            flat_int_terms = [
                term for sublist in self._X_int_names for term in sublist
            ]

        # Transform nums
        if self._X_num_names is None:
            self._X_num_terms = None
        else:
            self._X_num_terms = []
            for num_term in self._X_num_names:
                if num_term not in flat_int_terms:
                    self._X_num_terms.append(num_term)
            if len(self._X_num_terms) == 0:
                self._X_num_terms = None

        # Transform bools
        if self._X_bool_names is None:
            self._X_bool_terms = None
        else:
            self._X_bool_terms = []
            for bool_term in self._X_bool_names:
                if bool_term not in flat_int_terms:
                    self._X_bool_terms.append(f'C({bool_term})')
            if len(self._X_bool_terms) == 0:
                self._X_bool_terms = None

        # Transform categoricals
        if self._X_cat_names is None:
            self._X_cat_terms = None
        else:
            self._X_cat_terms = []
            for cat_term in self._X_cat_names:
                if cat_term not in flat_int_terms:
                    self._X_cat_terms.append(f'C({cat_term})')
            if len(self._X_cat_terms) == 0:
                self._X_cat_terms = None

    @property
    def formula(self):
        self._names_to_terms()
        formula = ''

        # Add numeric terms
        if self._X_num_terms is not None:
            for num_term in self._X_num_terms:
                formula += f' + {num_term}'
        # Add boolean terms
        if self._X_bool_terms is not None:
            for bool_term in self._X_bool_terms:
                formula += f' + {bool_term}'
        # Add categorical terms
        if self._X_cat_terms is not None:
            for cat_term in self._X_cat_terms:
                formula += f' + {cat_term}'
        # Add interaction terms
        if self._X_int_terms is not None:
            for int_list in self._X_int_terms:
                # Add first interaction term
                int_formula_str = f'{int_list[0]}'
                # Add each successive interaction term
                for int_term in int_list[1:]:
                    int_formula_str += f'*{int_term}'
                # Add interaction string to formula
                formula += f' + {int_formula_str}'

        # Final processing
        if len(formula) == 0:
            formula = '1'
        else:
            formula = formula[3:]  # Remove initial ' + '
        formula = f'{self._y_name} ~ {formula}'

        return formula
