"""Module for statistical analysis of continuous data.

This module provides classes for analyzing relationships between two continuous
or categorical series, including correlation tests, two-sample comparisons, and
grouped statistical tests. It supports both parametric and nonparametric methods.

Dependencies
------------
* typing: for Literal
* collections: For namedtuple.
* numpy: For numerical operations.
* pandas: For data manipulation and series handling.
* scipy: For statistical tests (Pearson, Spearman, t-test, Mann-Whitney U).
* .exceptions:
  * For custom exceptions: SeriesNameCollisionError & SeriesNameCollisionWarning

Classes
-------
CorrStats
    Class for correlation analysis (Pearson or Spearman).
TwoSeriesStats
    Class for two-sample statistical comparisons.
TwoSampleStats
    Class for two-sample comparisons with a boolean grouping variable.
ControlTestStats
    Named tuple for storing confidence intervals of control and test groups.

Notes
-----
Assumes input series are appropriately typed (e.g., continuous for numerical
tests, boolean for grouping). Handles missing data by dropping NaNs.
"""
import warnings
from typing import Literal
from collections import namedtuple
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
from scipy import stats
from .exceptions import SeriesNameCollisionError, SeriesNameCollisionWarning


class CorrStats:
    """Class for computing correlation statistics between two continuous series.

    Supports Pearson (parametric) or Spearman (nonparametric) correlation.

    Parameters
    ----------
    x : pd.Series
        First continuous variable.
    y : pd.Series
        Second continuous variable.
    parametric : bool, optional
        If True, compute Pearson correlation; otherwise, Spearman. Defaults to True.

    Attributes
    ----------
    x : pd.Series
        First variable after cleaning.
    y : pd.Series
        Second variable after cleaning.
    parametric : bool
        Whether to use parametric (Pearson) or nonparametric (Spearman) method.
    _df : pd.DataFrame
        Concatenated DataFrame of x and y with NaNs dropped.
    """
    def __init__(self, x: pd.Series, y: pd.Series, parametric: bool = True):
        self._df = pd.concat([x, y], axis='columns').dropna(axis='index')
        self.x = self._df[x.name]
        self.y = self._df[y.name]
        self.parametric = parametric

    def __str__(self):
        """String representation of correlation results.

        Returns
        -------
        str
            Formatted string with correlation coefficient, p-value, and sample
            size.
        """
        if self.parametric:
            print_str = (
                f"{self.x.name} vs. {self.y.name} Pearson's r:\n"
                f"r = {self.stat:.4f}\n"
                f"p = {self.p:.4f}\n"
                f"n = {self.n}\n"
            )
        else:
            print_str = (
                f"{self.x.name} vs. {self.y.name} Spearman's r:\n"
                f"rho = {self.stat:.4f}\n"
                f"p = {self.p:.4f}\n"
                f"n = {self.n}\n"
            )
        return print_str

    @property
    def result(self):
        """Compute correlation test result.

        Returns
        -------
        scipy.stats._stats_py.PearsonRResult or scipy.stats._stats_py.SpearmanrResult
            Result object with statistic and p-value.
        """
        if self.parametric:
            return stats.pearsonr(self.x, self.y, alternative='two-sided')
        else:
            return stats.spearmanr(self.x, self.y, alternative='two-sided')

    @property
    def n(self):
        """Sample size after dropping NaNs.

        Returns
        -------
        int
            Number of observations.
        """
        return len(self._df.index)

    @property
    def stat(self):
        """Correlation coefficient (r for Pearson, rho for Spearman).

        Returns
        -------
        float
            Correlation statistic.
        """
        return self.result.statistic

    @property
    def p(self):
        """P-value of the correlation test.

        Returns
        -------
        float
            P-value.
        """
        return self.result.pvalue


class TwoSeriesStats:
    """Compare 2 continuous series using parametric or nonparametric tests.

    Supports asymptotic t-tests (parametric) or Mann-Whitney U-tests
    (nonparametric).

    Parameters
    ----------
    test : pd.Series
        Test group data.
    control : pd.Series
        Control group data.
    parametric : bool, optional
        If True, use t-test; otherwise, Mann-Whitney U. Defaults to True.
    alpha_level : float, optional
        Significance level for confidence intervals and tests. Defaults to 0.05.

    Attributes
    ----------
    test : pd.Series
        Cleaned test group data.
    control : pd.Series
        Cleaned control group data.
    parametric : bool
        Whether to use parametric or nonparametric methods.
    alpha : float
        Significance level.

    See Also
    --------
    TwoSampleStats :
        Same tests, starting from a Boolean group column (x) and a numeric
        outcome column (y).
    TwoSeriesPermutation :
        Permutation hypothesis tests for 2-sample statistics, including t-test &
        Mann-Whitney U-test.
    """

    def __init__(self,
                 test: pd.Series,
                 control: pd.Series,
                 parametric: bool = True,
                 alpha_level: float = .05):
        if test.name == control.name:
            SeriesNameCollisionWarning('`test` and `control` have '
                                       'the same column name.')
            test = test.rename(f'{test.name}_TEST')
            control = control.rename(f'{test.name}_CTRL')
        self.test = test.dropna().reset_index(drop=True)
        self.control = control.dropna().reset_index(drop=True)
        self.parametric = parametric
        self.alpha = alpha_level

    def __str__(self):
        """String representation of summary statistics and test results.

        Returns
        -------
        str
            Formatted summary and test results (t-test or Mann-Whitney U)."""
        if self.parametric:
            return (f'{self.parametric_summ_stats()}\n'
                    f'{self.t_test()}')
        else:
            return (f'{self.nonparametric_summ_stats()}\n'
                    f'{self.mwu_test()}')

    def conf_int(self,
                 pct_ci: float = None,
                 dist: Literal['t', 'normal', 'z'] = 't'):
        """Calculate CI of mean of test and control groups, based on SEM.

        `pct_ci` can be used to custom-define a CI width. By default, 1 - alpha
        is used, as set in the TwoSeriesStats object.

        Parameters
        ----------
        pct_ci : float, optional
            Confidence level (e.g., 0.95 for 95%). Defaults to 1 - alpha.
        dist: {'t', 'normal'}, optional
            Parent distribution to use when calculating SEM. Defaults to 't'.

        Returns
        -------
        ControlTestStats
            Named tuple with confidence intervals for control and test groups.

        See Also
        --------
        TwoSeriesBootstrap :
            Bootstrapped CIs. Useful for finding CI around a statistic other
            than the sample mean (may be useful for median, as in a skewed
            sample), or for cases of small N.

        Notes
        -----

        SEM is always calculated using sample SD (1 DoF). CI can be calculated
        using either the normal (Z) distribution or Student's t-distribution;
        the latter will give slightly wider CI, all else being equal.

        For small N, Gurland & Trepathi (1971) report that CI is too narrow by
        25% for N=2, and ~5% for N=6, and provide an equation for calculating CI
        underestimation. Sokal & Rohlf (1981) give a formula for a correction
        factor to calculate unbiased CI for N < 20, which is not implemented
        in this method.

        For N > 100, Student's t-distribution and Gaussian normal distribution
        are approximately equivalent. Nonetheless, t-distribution is the default
        form used here.

        Rather than considering correction factors, our recommendation for small
        N, or to define CI about a measure of central tendency other than the
        sample mean, is to use a bootstrapped CI, as implemented in the
        ``resampling`` module.
        """
        if pct_ci is None:
            pct_ci = 1 - self.alpha

        if dist == 't':
            test_ci = stats.t.interval(
                pct_ci,
                df=self.test.count() - 1,
                loc=self.test.mean(),
                scale=self.test.std(ddof=1) / np.sqrt(self.test.count())
            )
            control_ci = stats.t.interval(
                pct_ci,
                df=self.control.count() - 1,
                loc=self.control.mean(),
                scale=self.control.std(ddof=1) / np.sqrt(self.control.count())
            )

        elif dist in ('normal', 'z'):
            test_ci = stats.norm.interval(
                pct_ci,
                loc=self.test.mean(),
                scale=self.test.std(ddof=1) / np.sqrt(self.test.count())
            )
            control_ci = stats.norm.interval(
                pct_ci,
                loc=self.control.mean(),
                scale=self.control.std(ddof=1) / np.sqrt(self.control.count())
            )

        else:
            raise ValueError("`dist` can be: {'t', 'normal', 'z'}.")

        return ControlTestStats(control=control_ci, test=test_ci)

    def parametric_summ_stats(self, alpha_level: float = None):
        """Compute parametric summary statistics (mean, std, CI, etc.).

        Parameters
        ----------
        alpha_level : float, optional
            Significance level for CI. Defaults to self.alpha.

        Returns
        -------
        pd.DataFrame
            Summary statistics for test and control groups.
        """
        if alpha_level is None:
            alpha_level = self.alpha

        ci_res = self.conf_int(1 - alpha_level)

        summ_stats = pd.DataFrame(
            data={
                self.test.name: {
                    'n': self.test.count(),
                    'mean': self.test.mean(),
                    'std': self.test.std(),
                    'min': self.test.min(),
                    'max': self.test.max(),
                    f'{100 * (1 - alpha_level):.0f}%_CI_lower': ci_res.test[0],
                    f'{100 * (1 - alpha_level):.0f}%_CI_upper': ci_res.test[1],
                },

                self.control.name: {
                    'n': self.control.count(),
                    'mean': self.control.mean(),
                    'std': self.control.std(),
                    'min': self.control.min(),
                    'max': self.control.max(),
                    f'{100 * (1 - alpha_level):.0f}%_CI_lower': (
                        ci_res.control[0]
                    ),
                    f'{100 * (1 - alpha_level):.0f}%_CI_upper': (
                        ci_res.control[1]
                    ),
                },
            },
        )
        return summ_stats

    def t_test(self, equal_var: bool = False):
        """Perform 2-independent-samples t-test (Welch's or Student's).

        Defaults to Welch's t-test for heteroskedastic samples. Delacre et al.
        (2017) recommends routinely use of Welch's test for unequal variance
        over Student's method, rather than selective use of Welch's test. The
        loss in power from Welch's vs. Student's test in cases of equal variance
        is minimal, whereas the reduction in Type I error rate from Welch's in
        cases of unequal variance is substantial; a strong determination of
        homoskedasticity is often not simple.

        Parameters
        ----------
        equal_var : bool, optional
            If True, assume equal variances (Student's t-test); otherwise, use
            Welch's. Defaults to False.

        Returns
        -------
        pd.Series
            Test results including t-statistic, degrees of freedom, and p-value.
        """
        result = stats.ttest_ind(
            a=self.test,
            b=self.control,
            equal_var=equal_var,
        )

        output = pd.Series(
            {
                't-statistic': result.statistic,
                'welch_df': result.df,
                'student_df': (
                    (self.test.count() - 1) + (self.control.count() - 1)
                ),
                'p-value': result.pvalue,
            }
        )

        return output

    def nonparametric_summ_stats(self, alpha_level: float = None):
        """Compute nonparametric summary statistics (quantiles, IQR).

        Parameters
        ----------
        alpha_level : float, optional
            Significance level for hypothesis direction. Defaults to self.alpha.

        Returns
        -------
        pd.DataFrame
            Summary statistics with quantiles and IQR, including hypothesis
            direction.
        """
        if alpha_level is None:
            alpha_level = self.alpha

        q_test = self.test.quantile([0, 0.25, 0.5, 0.75, 1])
        q_control = self.control.quantile([0, 0.25, 0.5, 0.75, 1])

        summ_stats = pd.DataFrame(
            data={
                self.test.name: {
                    'n': self.test.count(),
                    'min': q_test.loc[0],
                    'q1': q_test.loc[0.25],
                    'median': q_test.loc[0.5],
                    'q3': q_test.loc[0.75],
                    'max': q_test.loc[1],
                    'iqr': q_test.loc[0.75] - q_test.loc[0.25],
                },

                'Ha': '=',

                self.control.name: {
                    'n': self.control.count(),
                    'min': q_control.loc[0],
                    'q1': q_control.loc[0.25],
                    'median': q_control.loc[0.5],
                    'q3': q_control.loc[0.75],
                    'max': q_control.loc[1],
                    'iqr': q_control.loc[0.75] - q_control.loc[0.25],
                },
            },
        )

        if self.mwu_test().at['p-value'] < alpha_level:
            comparator = stats.mannwhitneyu(x=self.test, y=self.control,
                                            alternative='greater')
            if comparator.pvalue < alpha_level:
                summ_stats['Ha'] = '>'
            elif comparator.pvalue > alpha_level:
                comparator = stats.mannwhitneyu(x=self.test, y=self.control,
                                                alternative='less')
                if comparator.pvalue < alpha_level:
                    summ_stats['Ha'] = '<'
                else:
                    summ_stats['Ha'] = '!='
            else:
                summ_stats['Ha'] = '!='

        return summ_stats

    def mwu_test(self):
        """Perform Mann-Whitney U test.

        Returns
        -------
        pd.Series
            Test results including U-statistic and p-value.
        """
        result = stats.mannwhitneyu(x=self.test, y=self.control)

        desc_stats = pd.Series(
            {
                'U-statistic': result.statistic,
                'p-value': result.pvalue,
            }
        )
        return desc_stats


class TwoSampleStats(TwoSeriesStats):
    """Class for comparing a continuous outcome across two groups defined by a boolean variable.

    Extends TwoSeriesStats to split a continuous series by a boolean grouping variable.

    Parameters
    ----------
    bool_x : pd.Series
        Boolean variable defining groups.
    num_y : pd.Series
        Continuous outcome variable.
    parametric : bool, optional
        If True, use t-test; otherwise, Mann-Whitney U. Defaults to True.
    alpha_level : float, optional
        Significance level for tests and CIs. Defaults to 0.05.
    x_test_lvl : bool, optional
        Boolean level defining the test group. Defaults to True.

    Attributes
    ----------
    x : pd.Series
        Boolean grouping variable.
    y : pd.Series
        Continuous outcome variable.
    _test_x : bool
        Boolean level for test group.
    _df : pd.DataFrame
        Concatenated DataFrame of bool_x and num_y with NaNs dropped.

    Raises
    ------
    SeriesNameCollisionError
        If bool_x and num_y have the same name.
    ValueError
        If test or control group is empty after splitting.

    See Also
    --------
    TwoSeriesStats :
        Same tests, starting from 2 separate numeric outcome columns.
    TwoSamplePermutation :
        Permutation hypothesis tests for 2-sample statistics, including t-test &
        Mann-Whitney U-test.
    """
    def __init__(self,
                 bool_x: pd.Series, num_y: pd.Series,
                 parametric: bool = True,
                 alpha_level: float = .05,
                 x_test_lvl: bool = True):
        # Ensure names exist and are not the same
        bool_x = bool_x.rename(bool_x.name or 'x')
        num_y = num_y.rename(num_y.name or 'y')
        if bool_x.name == num_y.name:
            raise SeriesNameCollisionError('Both bool_x and num_y '
                                           'cannot have same names.')

        # Concat into a df in order to .dropna, then define x & y series
        self._df = (
            pd.concat(
                [
                    bool_x.astype('boolean'),
                    num_y,
                ],
                axis='columns'
            )
            .dropna(axis='index')
        )
        self.x = self._df[bool_x.name]
        self.y = self._df[num_y.name]

        # Build mask to slice test/control base on x_test_lvl
        self._test_x = bool(x_test_lvl)  # Ensure is type bool
        mask = (self.x == self._test_x).astype(bool)  # ensure safe masking

        # self.test is value of y where x == x_test_lvl
        test = (
            self.y[mask]
            .rename(f'{self.x.name}_{str(self._test_x).upper()}')
        )

        control = (
            self.y[~mask]
            .rename(f'{self.x.name}_{str((not self._test_x)).upper()}')
        )

        if test.empty or control.empty:
            raise ValueError('After cleaning/splitting, at least one group is '
                             'empty. Check x_test_lvl and values in bool_x.')

        super().__init__(test=test,
                         control=control,
                         parametric=parametric,
                         alpha_level=alpha_level)

    def __str__(self):
        """String representation of grouped summary statistics and test results.

        Returns
        -------
        str
            Formatted outcome name, summary, and test results.
        """
        if self.parametric:
            return (f'Outcome: {self.y.name}\n'
                    f'{self.parametric_summ_stats()}\n'
                    f'{self.t_test()}')
        else:
            return (f'Outcome: {self.y.name}\n'
                    f'{self.nonparametric_summ_stats()}\n'
                    f'{self.mwu_test()}')


ControlTestStats = namedtuple('ControlTestStats',
                              ['control', 'test'])


class MultiSeries1WayBGStats:
    def __init__(self,
                 *data: pd.Series | pd.DataFrame,
                 parametric: bool = True,
                 alpha_level: float = .05,
                 **named_data: pd.Series | pd.DataFrame):
        def parse_input_data(*input_data,
                             **input_named_data) -> list[pd.Series]:
            output_list = []


            def parse_input_df(input_df: pd.DataFrame) -> None:
                # Ensure at least 2 columns, then assign columns to series_list
                if input_df.shape[1] >= 2:
                    for col in input_df.columns:
                        output_list.append(input_df[col])
                else:
                    raise ValueError(
                        'If passing a single DataFrame as a positional `data` '
                        'arg or `data=df`kwarg, the DataFrame must have at '
                        'least 2 columns.'
                    )


            # Ensure exactly 1 paradigm used
            if input_data and input_named_data:
                raise ValueError('Cannot pass both `data` & `named_data`.')

            # If passing `data` as a kwarg: `data=pd.DataFrame()`
            elif 'data' in input_named_data:
                df = input_named_data.pop('data')

                # Multiple named_data
                if len(input_named_data) > 0:
                    ValueError(
                        'When using `data=df` argument to pass a DataFrame, '
                        '`*data` positional args must be omitted, and '
                        'no extra `**named_data` kwargs are permitted.'
                    )
                elif isinstance(df, pd.DataFrame):
                    parse_input_df(df)
                elif isinstance(df, pd.Series):
                    raise ValueError('Pass Series data via `*data` positional '
                                     'args or via `**named_data` kwargs.')
                else:
                    raise ValueError('`data=` must be pd.DataFrame or pd.Series.')

            # If passing a single DataFrame to `data` as arg
            elif (
                # pd.DataFrame arg
                (len(input_data) == 1)
                and isinstance(input_data[0], pd.DataFrame)
            ):
                parse_input_df(input_data[0])

            # If using `data` args
            elif input_data:
                # Ensure at least 2 args
                if len(input_data) >= 2:
                    for enum, series in enumerate(input_data):
                        # If `data` includes a DataFrame, ensure it only has 1 col
                        if isinstance(series, pd.DataFrame):
                            if series.shape[1] == 1:
                                series = series.iloc[:, 0]
                            else:
                                raise ValueError(
                                    'When passing multiple args to `data`, '
                                    'args must be pd.Series or single-column '
                                    'pd.DataFrame.'
                                )

                        # Assign name/number to each series
                        if hasattr(series, 'name'):
                            series.name += f'__{enum}'
                        else:
                            series.name = enum
                        output_list.append(series)

                else:
                    raise ValueError('When passing individual Series args to '
                                     '`data`, at least 2 args must be passed.')

            # If using `named_data` kwargs
            elif input_named_data:
                # Ensure at least 2 kwargs
                if len(input_named_data) >= 2:
                    for name, series in input_named_data.items():
                        # If `data` includes a DataFrame, ensure only has 1 col
                        if isinstance(series, pd.DataFrame):
                            if series.shape[1] == 1:
                                series = series.iloc[:, 0]
                            else:
                                raise ValueError(
                                    'When passing multiple kwargs to '
                                    '`named_data`, args must be a Series or '
                                    'single-column DataFrame.'
                                )

                        series.name = name
                        output_list.append(series)
                else:
                    raise ValueError('When using `named_data`, at least 2 '
                                     'kwargs must be passed.')

            # If `data` & `named_data` left blank
            else:
                raise ValueError('Must pass either `data` or `named_data`.')

            return output_list

        def check_duplicate_names(series_list: list[pd.Series]) -> list[str]:
            series_names: list = [series.name for series in series_list]

            # Set will be shorter than list if & only if duplicate column names
            if len(set(series_names)) < len(series_names):

                duplicates = series_names.copy()
                for name in set(series_names):
                    duplicates.remove(name)
                duplicates = str(set(duplicates))[1:-1]  # Remove {} wrapper

                raise SeriesNameCollisionError(
                    '`data` cannot have duplicate column names.\n'
                    f'Duplicates: {duplicates}'
                )
            else:
                return series_names

        series_list = parse_input_data(*data, **named_data)
        check_duplicate_names(series_list)

        series_list = [series.dropna().reset_index(drop=True)
                       for series in series_list]

        self.data = pd.concat(
            objs=series_list,
            axis='columns',
        )
        self.parametric = parametric
        self.alpha = alpha_level

    def __str__(self):
        """String representation of summary statistics and test results.

        Returns
        -------
        str
            Formatted summary and test results (t-test or Mann-Whitney U)."""
        if self.parametric:
            return (f'{self.parametric_summ_stats()}\n'
                    f'{self.anova()}')
        else:
            return (f'{self.nonparametric_summ_stats()}\n'
                    f'{self.kruskal_wallis()}')

    def conf_int(self,
                 pct_ci: float = None,
                 dist: Literal['t', 'normal', 'z'] = 't') -> pd.DataFrame:
        """Calculate CI of mean of all columns in `data`, based on SEM.

        `pct_ci` can be used to custom-define a CI width. By default, 1 - alpha
        is used, as set in the TwoSeriesStats object.

        Parameters
        ----------
        pct_ci : float, optional
            Confidence level (e.g., 0.95 for 95%). Defaults to 1 - alpha.
        dist: {'t', 'normal'}, optional
            Parent distribution to use when calculating SEM. Defaults to 't'.

        Returns
        -------
        pd.DataFrame
            Column names match `data.columns`; `index=['ci_lower', 'ci_upper']`.

        Notes
        -----

        SEM is always calculated using sample SD (1 DoF). CI can be calculated
        using either the normal (Z) distribution or Student's t-distribution;
        the latter will give slightly wider CI, all else being equal.

        For small N, Gurland & Trepathi (1971) report that CI is too narrow by
        25% for N=2, and ~5% for N=6, and provide an equation for calculating CI
        underestimation. Sokal & Rohlf (1981) give a formula for a correction
        factor to calculate unbiased CI for N < 20, which is not implemented
        in this method.

        For N > 100, Student's t-distribution and Gaussian normal distribution
        are approximately equivalent. Nonetheless, t-distribution is the default
        form used here.

        Rather than considering correction factors, our recommendation for small
        N, or to define CI about a measure of central tendency other than the
        sample mean, is to use a bootstrapped CI, as implemented in the
        ``resampling`` module. However, a bootstrapping class has not yet
        been implemented for multiclass (3+ level) cases.
        """
        if pct_ci is None:
            pct_ci = 1 - self.alpha

        res_dict = {}
        if dist == 't':
            for col in self.data.columns:
                res_dict[col] = stats.t.interval(
                    pct_ci,
                    df=self.data[col].count() - 1,
                    loc=self.data[col].mean(),
                    scale=(
                        self.data[col].std(ddof=1)
                        / np.sqrt(self.data[col].count())
                    )
                )

        elif dist in ('normal', 'z'):
            for col in self.data.columns:
                res_dict[col] = stats.norm.interval(
                    pct_ci,
                    loc=self.data[col].mean(),
                    scale=(
                        self.data[col].std(ddof=1)
                        / np.sqrt(self.data[col].count())
                    )
                )

        else:
            raise ValueError("`dist` can be: {'t', 'normal', 'z'}.")

        return pd.DataFrame(data=res_dict, index=['ci_lower', 'ci_upper'])

    def parametric_summ_stats(self, alpha_level: float = None) -> pd.DataFrame:
        """Compute parametric summary statistics (mean, std, CI, etc.).

        Parameters
        ----------
        alpha_level : float, optional
            Significance level for CI. Defaults to self.alpha.

        Returns
        -------
        pd.DataFrame
            Summary statistics for each column in `data`.
        """
        if alpha_level is None:
            alpha_level = self.alpha

        ci_res: pd.DataFrame = self.conf_int(1 - alpha_level)

        summ_stats = pd.DataFrame(
            data={
                col: {
                    'n': self.data[col].count(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    f'{100 * (1 - alpha_level):.0f}%_CI_lower': (
                        ci_res.at['ci_lower', col]
                    ),
                    f'{100 * (1 - alpha_level):.0f}%_CI_upper': (
                        ci_res.at['ci_upper', col]
                    ),
                } for col in self.data.columns.tolist()
            },
        )
        return summ_stats

    def anova(self, equal_var: bool = False) -> pd.Series:
        """Perform 1-way between-samples analysis of variance (ANOVA).

        Defaults to Welch's ANOVA for heteroskedastic samples. Delacre et al.
        (2019) recommends routinely use of Welch's F-test for unequal variance
        over Student's (Fisher's) method, rather than selective use of Welch's
        test. The loss in power from Welch's vs. Student's test in cases of
        equal variance is minimal, whereas the reduction in Type I error rate
        from Welch's incases of unequal variance is substantial; a strong
        determination of homoskedasticity is often not simple.

        Parameters
        ----------
        equal_var : bool, optional
            If True, assume equal variances (Fisher's ANOVA); otherwise, use
            Welch's. Defaults to False.

        Returns
        -------
        pd.Series
            Test results including F-statistic, degrees of freedom, and p-value.
        """
        result = stats.f_oneway(
            *[self.data[col].astype(float)
              for col in self.data.columns],
            equal_var=equal_var,
            nan_policy='omit',
            axis=0
        )

        output = pd.Series(
            {
                'F-statistic': result.statistic,
                'dof_B': self.data.shape[1] - 1,
                'dof_E': self.data.shape[0] - self.data.shape[1],
                'dof_T': self.data.shape[0] - 1,
                'p-value': result.pvalue,
            }
        )

        return output

    def nonparametric_summ_stats(self,
                                 alpha_level: float = None) -> pd.DataFrame:
        """Compute nonparametric summary statistics (quantiles, IQR).

        Parameters
        ----------
        alpha_level : float, optional
            Significance level for hypothesis direction. Defaults to self.alpha.

        Returns
        -------
        pd.DataFrame
            Summary statistics with quantiles and IQR.
        """
        if alpha_level is None:
            alpha_level = self.alpha

        quantiles = [0, 0.25, 0.5, 0.75, 1]

        summ_stats = pd.DataFrame(
            data={
                col: {
                    'n': self.data.count(),
                    'min': self.data[col].quantile(quantiles).loc[0],
                    'q1': self.data[col].quantile(quantiles).loc[0.25],
                    'median': self.data[col].quantile(quantiles).loc[0.5],
                    'q3': self.data[col].quantile(quantiles).loc[0.75],
                    'max': self.data[col].quantile(quantiles).loc[1],
                    'iqr': (self.data[col].quantile(quantiles).loc[0.75]
                            - self.data[col].quantile(quantiles).loc[0.25]),
                } for col in self.data.columns.tolist()
            },
        )

        return summ_stats

    def kruskal_wallis(self) -> pd.Series:
        """Perform 1-way between-samples analysis of variance (ANOVA).

        Defaults to Welch's ANOVA for heteroskedastic samples. Delacre et al.
        (2019) recommends routinely use of Welch's F-test for unequal variance
        over Student's (Fisher's) method, rather than selective use of Welch's
        test. The loss in power from Welch's vs. Student's test in cases of
        equal variance is minimal, whereas the reduction in Type I error rate
        from Welch's incases of unequal variance is substantial; a strong
        determination of homoskedasticity is often not simple.

        Returns
        -------
        pd.Series
            Test results including F-statistic, degrees of freedom, and p-value.
        """
        result = stats.kruskal(
            *[self.data[col].astype(float)
              for col in self.data.columns],
            nan_policy='omit',
            axis=0,
        )

        output = pd.Series(
            {
                'H-statistic': result.statistic,
                'dof': self.data.shape[1] - 1,
                'p-value': result.pvalue,
            }
        )

        return output


class MultiSample1WayBGStats(MultiSeries1WayBGStats):
    def __init__(self,
                 cat_x: pd.Series,
                 num_y: pd.Series,
                 cat_order: list[str] = None,
                 parametric: bool = True,
                 alpha_level: float = .05):
        # Combine into a df
        df = (
            pd.concat([cat_x, num_y], axis='columns')
            .dropna(how='any', axis='rows')
        )

        if len(cat_x.unique()) < 2:
            raise ValueError('`cat_x` must contain at least 2 categories.')

        # If not passing categorical ordering
        if cat_order is None:
            # Want to preserve native categorical ordering if cat_x was already
            # an ordered categorical
            if not is_categorical_dtype(df[cat_x.name].dtype):
                df[cat_x.name] = df[cat_x.name].astype('category')
            categories = df[cat_x.name].cat.categories.tolist()

        # If passing categorical ordering
        else:
            # Ensure all used categories appear in cat_order
            missing = (
                set(df[cat_x.name].unique()) - set(cat_order)
                - {float('nan'), None, pd.NA}
            )
            if missing:
                raise ValueError(f'Categories in cat_x not found in cat_order: '
                                 f'{missing}')
            df[cat_x.name] = (
                df[cat_x.name]
                .astype(CategoricalDtype(categories=cat_order, ordered=True))
            )
            categories = cat_order

        # Pivot to wide: columns = categories, values = num_y
        wide_df = df.pivot(columns=cat_x.name, values=num_y.name)

        # Warn/omit empty groups
        observed = wide_df.columns
        missing = set(categories) - set(observed)
        if missing:
            warnings.warn(f'Categories with no data after dropna '
                          f'(omitted): {missing}')

        super().__init__(data=wide_df,
                         parametric=parametric,
                         alpha_level=alpha_level)
