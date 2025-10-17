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
from typing import Literal
from collections import namedtuple
import numpy as np
import pandas as pd
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
