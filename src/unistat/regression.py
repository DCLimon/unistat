
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class RegressionStats(ABC):
    def __init__(self, X, y, bool_col_names: list | str | None = None):
        self._df = self._concat_xy(X, y)
        self.bool_cols = bool_col_names
        self.X = self._df.drop(columns=[y.name])
        self.y = self._df[[y.name]]
        self.reg = None
        self.std_reg = None

    def __str__(self):
        print_string = f'{self.reg.summary2()}\n'
        if not self._all_bool_cols:
            print_string += f'{self.std_reg.summary2()}\n'
        return print_string

    @property
    def _all_bool_cols(self):
        if self.bool_cols is None:
            return False
        else:
            return len(self.bool_cols) == self.X.shape[1]

    @property
    def bool_cols(self):
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
            raise ValueError("bool_col_names must be a str or list of str")

    @property
    def X_std(self):
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
        if not standardize:
            exog = self.X
        else:
            exog = self.X_std
        endog = self.y

        logit = sm.OLS(
            endog=endog,
            exog=sm.add_constant(exog),
            missing='drop'
        ).fit()

        return logit

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, result):
        if result is None:
            self._reg = self._run_regression()
        else:
            self._reg = result

    @property
    def std_reg(self):
        return self._std_reg

    @std_reg.setter
    def std_reg(self, result):
        if result is None:
            self._std_reg = self._run_regression(standardize=True)
        else:
            self._std_reg = result

    @staticmethod
    def _concat_xy(X, y):
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
    def __init__(self, X, y, bool_col_names: list | str | None = None):
        super().__init__(X, y, bool_col_names)

    def __str__(self):
        print_string = (
            f'{self.reg.summary2()}\n'
            f'{self.logit_or()}\n'
        )
        if not self._all_bool_cols:
            print_string += (
                f'{self.std_reg.summary2()}\n'
                f'{self.logit_or(standardize=True)}\n'
            )
        return print_string

    def _run_regression(self, standardize: bool = False):
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
    def __init__(self, X, y, bool_col_names: list | str | None = None):
        super().__init__(X, y, bool_col_names)

    def _run_regression(self, standardize: bool = False):
        if not standardize:
            exog = self.X
        else:
            if self._all_bool_cols:
                return None
            else:
                exog = self.X_std
        endog = self.y

        logit = sm.OLS(
            endog=endog,
            exog=sm.add_constant(exog),
            missing='drop'
        ).fit()

        return logit

    def pretty_print_coefs(self,
                           standardize: bool = False,
                           label: bool = True) -> None:
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