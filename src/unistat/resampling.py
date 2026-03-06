from typing import Optional, Literal
from collections.abc import Callable
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from ._types import VectorLike
from .exceptions import (
    ExperimentalWarning, SeriesNameCollisionError, SeriesNameCollisionWarning
)

################################################################################
# Bootstraps
################################################################################

type _BootstrapTestTypes = Literal['means', 'medians']


@dataclass
class BootResult:
    r"""Dataclass to store bootstrap resampling results.

    Attributes
    ----------
    obs_stat : float or int
       Observed test statistic (difference of means or medians).
    boot_mean : float or int
       Mean of the bootstrap distribution.
    boot_sem : float or int
       Standard error of the mean from the bootstrap distribution.
    boot_ci_lo : float or int
       Lower bound of the bias-corrected accelerated (BCa) confidence interval.
    boot_ci_hi : float or int
       Upper bound of the BCa confidence interval.
    p : float
       P-value from bootstrap hypothesis test (experimental).
    boot_ci_pct : float
       Confidence level percentage for the CI.
    boot_dist : np.ndarray
       Bootstrap distribution array.
    """

    obs_stat: float | int
    boot_mean: float | int
    boot_sem: float | int
    boot_ci_lo: float | int
    boot_ci_hi: float | int
    p: float
    boot_ci_pct: float
    boot_dist: np.ndarray

    def __str__(self):
        r"""String representation of bootstrap results.

        Returns
        -------
        str
           Formatted string with observed statistic, bootstrap mean, SEM, CI,
           and p-value.
        """
        return f'''
        Observed Test Stat: {self.obs_stat:.4f}
        Mean of Bootstrap Dist.: {self.boot_mean:.4f}
        SEM of Bootstrap Dist.: {self.boot_sem:.4f}
        BCa {self.boot_ci_pct:.0%} CI: {self.boot_ci_lo:.4f} to {self.boot_ci_hi:.4f}
        {f'P = {self.p:.5f}' if self.p is not None else 'No hypothesis testing performed.'}
        '''


class TwoSeriesBootstrap:
    r"""Class for bootstrapping CI & hypothesis tests on two continuous series.

    Supports difference of means or medians. Provides bootstrap distribution,
    CIs, and optional experimental p-value from a bootstrapped hypothesis test.

    Currently, by default, no hypothesis tests are performed nor p-values given.

    Parameters
    ----------
    test : pd.Series
       Test group data.
    control : pd.Series
       Control group data.
    test_type : {'means', 'medians'}, default 'means'
       Type of test statistic.
    alpha_level : float, default 0.05
       Significance level for CI.
    n_resamples : int, default 10_000
       Number of bootstrap resamples.
    rng : int, default 519
       Random seed for reproducibility.
    boot_p_value : bool, default False
       If True, compute experimental bootstrap P-value.

    Attributes
    ----------
    test : pd.Series
       Cleaned test data.
    control : pd.Series
       Cleaned control data.
    alpha : float
       Significance level.
    n_resamples : int
       Number of resamples.
    rng : int
       Random seed.
    boot_result : BootResult or None
       Stored bootstrap results.

    Raises
    ------
    ValueError
       If test_type is invalid.

    Notes
    -----
    Regarding the current WIP bootstrap hypothesis test, P-values are not
    currently displayed by default, as controlled by the ``boot_p_value``
    Boolean parameter.

    Under Frequentist statistical philosophy, a P-value represents the
    probability of getting results as-or-more-extreme than those observed, if
    the null hypothesis is true (this is even the assumption for classical
    calculated asymptotic hypothesis tests). In the context of bootstrap
    statistics, this would mean that a bootstrap distribution should represent
    the distribution of test statistics (mean, median, t-stat, U-stat) that
    would be observed *under the null hypothesis (H0)*, and the p-value would be
    the fraction of test statistic values in that bootstrapped distribution
    which are at least as extreme as what was actually observed.

    In contrast, when creating a bootstrap distribution to estimate CI or SEM
    (the primary use of bootstrapping), one creates a bootstrap distribution
    *under the assumption that the* **alternative hypothesis (Ha)** *is true*.
    Looking at how these bootstrap classes display results, this is apparent:
    we display bootstrapped means/CIs for the test statistic (difference in
    group means/medians) between test and control groups. Another way of
    phrasing a statistically significant difference in means (at the
    :math:`\alpha = .05` significance level), for example, would be to say that
    the 95% CI of difference in the means of the test and control samples does
    NOT include zero. Since the null hypothesis (at least
    in cases relevant to typical biostats) is that the means or medians are
    *equal* in the test and control groups (or that the difference in the means/
    medians is zero), the fact that the bootstrapped difference in means is not
    zero here shows that these bootstrap distributions for CI are NOT bootstraps
    under the null hypothesis. Indeed, when first making this module, that
    mistake was made, and it was found that even when there was an obvious
    difference between test vs. control groups, bootstrapped "p-values" taken
    from the same distribution as bootstrapped CIs returns a p-value near 0.50;
    to put it another way, this mistaken approach found that the observed
    difference in means was typically almost exactly the same as the average
    bootstrapped difference in means -- exactly what would be expected in a
    distribution representing the alternative hypothesis.

    While checking for overlapping CIs does indicate statistical significance in
    a Boolean fashion, it cannot give p-values. Instead, to calculate p-values
    via a bootstrapped methodology, in addition to the bootstrapped distribution
    under the Ha (used to estimate CIs), one must also use some method to
    bootstrap the null distribution in order to calculate p-value. Boos &
    Stefanski (2013) :cite:`boos_11_6` suggest 2 methods, as summarized in this
    `lecture by Alex Kaizer <https://www.alexkaizer.com/bios_6618/files/bios6618/W16/bootstrap_pvalue.pdf>`_:

    1. Combine all test and control observations (e.g. Hgb value for both blunt
    & penetrating trauma patients), then sample with replacement for the N in
    both test and control groups (number of blunt/penetrating patients) to
    create bootstrapped test and control samples.

       * On average, this approach will create equal means in each group
       * However, the intermingling of observations in each group means that if
         observed groups do not have equal variance, they will end up with
         equal variances during resampling

          * This is the same issue encountered in Welch vs. Student t-tests
          * This risks inflating Type I error rate if variances are unequal

    2. Create separate *observed* distributions for test & control, where each
    value represents how much the original value differed from its group mean.
    Each groups' bootstrap sample is sampled with replacement from its own
    group's mean-shifted sample.

       * Both test and control groups now have identical group means of 0
         (since the mean variation from the mean in a sample is 0, duh)
       * This also preserves unequal variances
       * This seems like a superior strategy, and will ultimately be
         implemented for this module
       * This method does require performing completely separate bootstrap
         samples for both test & control groups, if and only if the goal is
         to perform a hypothesis test and derive a p-value

    Due to coding challenges, a hybrid approach has been taken in the interim
    here, which will later be changed.

    * We take the bootstrap-under-Ha distribution (as used for CIs), and then
      shift it to be centered about the mean of the bootstrap distribution.
    * From this, we find how many bootstrap samples had a mean at least as
      extreme as what was observed. This is the p-value.
    * Since the bootstrap samples still allow admixture of test & control
      observations, this method is accurate, but still destroys any unequal
      variances between test & control groups

       * For this reason, this approach is not robust for deployment

    In contrast to bootstrapping, permutation tests (sampling without
    replacement, and then randomly assigning to test or control) generates
    a distribution under the null hypothesis. For this reason, permutation
    methods more easily generate a p-value, but are not (easily) fit to
    generating a CI for the test statistic, in the way bootstrapping is. While
    the acceptability of using both methods (i.e. bootstrap a CI and then
    permute a p-value), it doesn't immediately seem like an objectionable choice
    if both values are truly needed. This would probably be preferred even once
    bootstrapped hypothesis tests are fully implemented, since even that would
    require an entire separate resampling run. The only time when bootstrapped
    hypothesis tests are truly necessary is when the permutation test assumption
    (that test & control groups are exchangeable; that is, they have the same
    distributions if H0 is true) is violated. The proposed bootstrapped
    hypothesis testing method (#2 above) does not make this assumption.
    """

    _use_parametric_summ_stats = {'means': True, 'medians': False}

    def __init__(self,
                 test: pd.Series,
                 control: pd.Series,
                 test_type: _BootstrapTestTypes = 'means',
                 alpha_level: float = .05,
                 n_resamples: int = 10_000,
                 rng: Optional[int] = 519,
                 boot_p_value: bool = False):
        # Rename test & control if column names are same
        if test.name == control.name:
            warnings.warn(
                SeriesNameCollisionWarning(
                    '`test` and `control` have the same column name.'
                ),
                stacklevel=2
            )
            test = test.rename(f'{test.name}_TEST')
            control = control.rename(f'{test.name}_CTRL')

        self.test = test.dropna().reset_index(drop=True)
        self.control = control.dropna().reset_index(drop=True)

        if test_type in self._use_parametric_summ_stats.keys():
            self._test_type = test_type
            self._test_means: bool = self._use_parametric_summ_stats[test_type]
        else:
            raise ValueError(
                f"`test_type` must be one of "
                f"{self._use_parametric_summ_stats.keys()}"
            )

        self.alpha = alpha_level
        self._alt_hypothesis = '='
        self.n_resamples = n_resamples
        self.rng = rng
        self.boot_result: Optional[BootResult] = None

        # Can remove this attribute & argument once a robust bootstrap p-value
        # method has been implemented. For now, here to reduce confusion unless
        # p-value is truly desired
        self._calculate_boot_p = boot_p_value
        if self._calculate_boot_p:
            # Warn about current bootstrap p-value method
            warnings.warn(
                ExperimentalWarning(
                    feature='Bootstrapped p-value',
                    message='''
                            Currently, the bootstrapped hypothesis test p-value for mean &
                            median uses the assumption of approximate translation
                            equivariance in order to create a bootstrapped null distribution
                            of the test stat. This method pools variance (like Student's
                            t-test) in test & control. When samples have unequal variance,
                            Type I error can somewhat inflate (i.e. p-value is overly
                            optimistic), especially in small or skewed null distributions.

                            Of note, this does NOT affect mean/median permutation tests,
                            and, if bootstrapped Welch's t or Mann-Whitney U-tests were
                            implemented, would not be affected either under current methods.

                            The current bootstrap p-value method is valid but suboptimal for
                            hypothesis testing, though all other bootstrapped stats
                            including CI follow best practices.

                            In summary, use bootstrapping methods to calculate estimated
                            distribution stats (pop. mean, SEM, CIs), but prefer permutation
                            tests for hypothesis tests (has more distribution assumptions, 
                            so mean/SEM/CIs are difficult and not worth calculating. I have
                            not checked for if it's frowned upon to both bootstrap (for
                            distribution stats) & permute (for p-value); I see no obvious
                            issue, but no clear benefit beyond more numbers to throw in a
                            manuscript.

                            Further explanation & references for will be available in
                            TwoSeriesBootstrap.bootstrap docstring via:
                                `instance_name.bootstrap.__doc__`
                                - or -
                                `help(TwoSeriesBootstrap.bootstrap)`
                            '''
                ),
                stacklevel=2
            )

    @property
    def results(self):
        # Ensure bootstrapping has been run before printing
        if self.boot_result is None:
            self.bootstrap()

        return (
            f'{str(self.summ_stats)}\n'
            f'''
            Bootstrapped Hypothesis Test
                * Test Stat: Difference of {self._test_type.capitalize()}
                * Test Group: {self.test.name}
                * Control Group: {self.control.name}
            {str(self.boot_result)}
            '''
        )

    @property
    def summ_stats(self):
        if self._test_means:
            return self._parametric_summ_stats()
        else:
            return self._nonparametric_summ_stats()

    def _parametric_summ_stats(self):
        summ_stats = pd.DataFrame(
            data={
                self.test.name: {
                    'n': self.test.count(),
                    'mean': self.test.mean(),
                    'std': self.test.std(),
                    'min': self.test.min(),
                    'max': self.test.max(),
                },

                'Ha': self._alt_hypothesis,

                self.control.name: {
                    'n': self.control.count(),
                    'mean': self.control.mean(),
                    'std': self.control.std(),
                    'min': self.control.min(),
                    'max': self.control.max(),
                },
            },
        )
        return summ_stats

    def _nonparametric_summ_stats(self):
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

                'Ha': self._alt_hypothesis,

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

        return summ_stats

    def bootstrap(self, n_resamples: int = None):
        r"""Perform bootstrap resampling.

        Parameters
        ----------
        n_resamples : int, optional
           Number of resamples. Defaults to self.n_resamples.

        Returns
        -------
        BootResult
           Bootstrap results.

        Notes
        -----
        Uses BCa method for confidence intervals. P-value is experimental.
        Updates hypothesis direction in summary stats if p < alpha.
        """
        # Default to using self.n_resamples
        n_resamples = n_resamples or self.n_resamples

        # Use correct test statistic function
        if self._test_means:
            test_stat_func = self.mean_diff
        else:
            test_stat_func = self.median_diff

        boot_result = stats.bootstrap(
            data=(self.test, self.control),
            statistic=test_stat_func,
            n_resamples=n_resamples,
            vectorized=True,
            axis=0,
            paired=False,
            confidence_level=1 - self.alpha,
            alternative='two-sided',
            method='BCa',
            rng=self.rng
        )

        # .item() ensures test stat returned 1 and only 1 value
        obs_stat = test_stat_func(self.test, self.control)

        # Store returned bootstrap distribution
        boot_dist = boot_result.bootstrap_distribution

        # Mean & SEM of bootstrap distribution
        boot_mean = boot_dist.mean()
        boot_sem = boot_result.standard_error
        # confidence_level CI of the bootstrap distribution
        boot_ci_lo, boot_ci_hi = boot_result.confidence_interval

        if self._calculate_boot_p:
            p_val = (
                # Count of bootstrap stats >= positive value of observed stat
                ((boot_dist - boot_dist.mean()) >= np.abs(obs_stat)).sum()
                # Count of bootstrap stats <= negative value of observed stat
                + ((boot_dist - boot_dist.mean()) <= -np.abs(obs_stat)).sum()
                # +1 to numerator & denominator makes p-value bit more conservative
                + 1
            ) / (n_resamples + 1)

        # Add results to stored attribute
        self.boot_result = BootResult(
            obs_stat=obs_stat,
            boot_mean=boot_mean,
            boot_sem=boot_sem,
            boot_ci_lo=boot_ci_lo,
            boot_ci_hi=boot_ci_hi,
            p=p_val if self._calculate_boot_p else None,
            boot_ci_pct=1 - self.alpha,
            boot_dist=boot_dist,
        )

        # If p-value is significant, indicate direction in summ_stats
        if self._calculate_boot_p and (self.boot_result.p is not None):
            if self.boot_result.p < self.alpha:
                if self.boot_result.obs_stat > 0:
                    self._alt_hypothesis = '>'
                elif self.boot_result.obs_stat < 0:
                    self._alt_hypothesis = '<'
                else:
                    warnings.warn(
                        'Significant P value but no between-groups difference in '
                        'test statistic observed. Verify data/results.'
                    )
                    self._alt_hypothesis = '='
            else:
                self._alt_hypothesis = '='

        return self.boot_result

    @staticmethod
    def mean_diff(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ) -> float | np.float64:
        r"""Compute difference of means.

        Parameters
        ----------
        sample_a : VectorLike
           First sample.
        sample_b : VectorLike
           Second sample.
        axis : {0, 1, 'rows', 'columns'}, optional
           Axis to compute along. Defaults to 0.

        Returns
        -------
        float or np.float64
           Difference of means.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Calculate absolute difference in means
        return a.mean(axis=axis) - b.mean(axis=axis)

    @staticmethod
    def median_diff(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ) -> float | np.float64:
        r"""Compute difference of medians.

        Parameters
        ----------
        sample_a : VectorLike
           First sample.
        sample_b : VectorLike
           Second sample.
        axis : {0, 1, 'rows', 'columns'}, optional
           Axis to compute along. Defaults to 0.

        Returns
        -------
        float or np.float64
           Difference of medians.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Calculate absolute difference in means
        return np.median(a, axis=axis) -  np.median(b, axis=axis)


class TwoSampleBootstrap(TwoSeriesBootstrap):
    r"""Class for bootstrapping with a boolean grouping variable.

    Extends TwoSeriesBootstrap to split a continuous series by a boolean group.

    Parameters
    ----------
    bool_x : pd.Series
       Boolean grouping variable.
    num_y : pd.Series
       Continuous outcome variable.
    test_type : {'means', 'medians'}, optional
       Test statistic type. Defaults to 'means'.
    alpha_level : float, optional
       Significance level. Defaults to 0.05.
    n_resamples : int, optional
       Number of resamples. Defaults to 10,000.
    rng : int, optional
       Random seed. Defaults to 519.
    x_test_lvl : bool, optional
       Boolean level for test group. Defaults to True.
    boot_p_value : bool, optional
       Compute experimental p-value. Defaults to False.

    Attributes
    ----------
    x : pd.Series
       Grouping variable.
    y : pd.Series
       Outcome variable.

    Raises
    ------
    ValueError
       If group names collide or groups are empty after splitting.

    Notes
    -----
    Regarding the current WIP bootstrap hypothesis test, p-values are not
    currently displayed by default, as controlled by the ``boot_p_value``
    Boolean parameter.

    Under Frequentist statistical philosophy, a p-value represents the
    probability of getting results as-or-more-extreme than those observed, if
    the null hypothesis is true (this is even the assumption for classical
    calculated asymptotic hypothesis tests). In the context of bootstrap
    statistics, this would mean that a bootstrap distribution should represent
    the distribution of test statistics (mean, median, t-stat, U-stat) that
    would be observed *under the null hypothesis (H0)*, and the p-value would be
    the fraction of test statistic values in that bootstrapped distribution
    which are at least as extreme as what was actually observed.

    In contrast, when creating a bootstrap distribution to estimate CI or SEM
    (the primary use of bootstrapping), one creates a bootstrap distribution
    *under the assumption that the* **alternative hypothesis (Ha)** *is true*.
    Looking at how these bootstrap classes display results, this is apparent:
    we display bootstrapped means/CIs for the test statistic (difference in
    group means/medians) between test and control groups. Another way of
    phrasing a statistically significant difference in means (at the
    :math:`\alpha = .05` significance level), for example, would be to say that
    the 95% CI of difference in the means of the test and control samples does
    NOT include zero. Since the null hypothesis (at least
    in cases relevant to typical biostats) is that the means or medians are
    *equal* in the test and control groups (or that the difference in the means/
    medians is zero), the fact that the bootstrapped difference in means is not
    zero here shows that these bootstrap distributions for CI are NOT bootstraps
    under the null hypothesis. Indeed, when first making this module, that
    mistake was made, and it was found that even when there was an obvious
    difference between test vs. control groups, bootstrapped "p-values" taken
    from the same distribution as bootstrapped CIs returns a p-value near 0.50;
    to put it another way, this mistaken approach found that the observed
    difference in means was typically almost exactly the same as the average
    bootstrapped difference in means -- exactly what would be expected in a
    distribution representing the alternative hypothesis.

    While checking for overlapping CIs does indicate statistical significance in
    a Boolean fashion, it cannot give p-values. Instead, to calculate p-values
    via a bootstrapped methodology, in addition to the bootstrapped distribution
    under the Ha (used to estimate CIs), one must also use some method to
    bootstrap the null distribution in order to calculate p-value. Boos &
    Stefanski (2013) :cite:`boos_11_6` suggest 2 methods, as summarized in this
    `lecture by Alex Kaizer <https://www.alexkaizer.com/bios_6618/files/bios6618/W16/bootstrap_pvalue.pdf>`_:

    1. Combine all test and control observations (e.g. Hgb value for both blunt
       & penetrating trauma patients), then sample with replacement for the N in
       both test and control groups (number of blunt/penetrating patients) to
       create bootstrapped test and control samples.

       * On average, this approach will create equal means in each group.
       * However, the intermingling of observations in each group means that if
         observed groups do not have equal variance, they will end up with
         equal variances during resampling.

          * This is the same issue encountered in Welch vs. Student t-tests.
          * This risks inflating Type I error rate if variances are unequal.

    2. Create separate *observed* distributions for test & control, where each
       value represents how much the original value differed from its group mean.
       Each groups' bootstrap sample is sampled with replacement from its own
       group's mean-shifted sample.

       * Both test and control groups now have identical group means of 0
         (since the mean variation from the mean in a sample is 0).
       * This also preserves unequal variances.
       * This seems like a superior strategy, and will ultimately be
         implemented for this module.
       * This method does require performing completely separate bootstrap
         samples for both test & control groups, if and only if the goal is
         to perform a hypothesis test and derive a p-value.

    Due to coding challenges, a hybrid approach has been taken in the interim
    here, which will later be changed.

    * We take the bootstrap-under-Ha distribution (as used for CIs), and then
      shift it to be centered about the mean of the bootstrap distribution.
    * From this, we find how many bootstrap samples had a mean at least as
      extreme as what was observed. This is the p-value.
    * Since the bootstrap samples still allow admixture of test & control
      observations, this method is accurate, but still destroys any unequal
      variances between test & control groups.

       * For this reason, this approach is not robust for deployment.

    In contrast to bootstrapping, permutation tests (sampling without
    replacement, and then randomly assigning to test or control) generates
    a distribution under the null hypothesis. For this reason, permutation
    methods more easily generate a p-value, but are not (easily) fit to
    generating a CI for the test statistic, in the way bootstrapping is. While
    the acceptability of using both methods (i.e. bootstrap a CI and then
    permute a p-value), it doesn't immediately seem like an objectionable choice
    if both values are truly needed. This would probably be preferred even once
    bootstrapped hypothesis tests are fully implemented, since even that would
    require an entire separate resampling run. The only time when bootstrapped
    hypothesis tests are truly necessary is when the permutation test assumption
    (that test & control groups are exchangeable; that is, they have the same
    distributions if H0 is true) is violated. The proposed bootstrapped
    hypothesis testing method (#2 above) does not make this assumption.
    """

    def __init__(self,
                 bool_x: pd.Series, num_y: pd.Series,
                 test_type: _BootstrapTestTypes = 'means',
                 alpha_level: float = .05,
                 n_resamples: int = 10_000,
                 rng: Optional[int] = 519,
                 x_test_lvl: bool = True,
                 boot_p_value: bool = False):
        # Ensure names exist and are not the same
        bool_x = bool_x.rename(bool_x.name or 'x')
        num_y = num_y.rename(num_y.name or 'y')
        if bool_x.name == num_y.name:
            raise ValueError('Both bool_x and num_y cannot have same names.')

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
                         test_type=test_type,
                         alpha_level=alpha_level,
                         n_resamples=n_resamples,
                         rng=rng,
                         boot_p_value=boot_p_value)

    def __str__(self):
        r"""String representation of bootstrap results with outcome name.

        Returns
        -------
        str
           Formatted outcome, summary stats, and bootstrap results.
        """
        # Ensure bootstrapping has been run before printing
        if self.boot_result is None:
            self.bootstrap()

        return (
            f'Outcome: {self.y.name}\n\n'
            f'{str(self.summ_stats)}\n'
            f'''
            Bootstrapped Hypothesis Test
                * Test Stat: Difference of {self._test_type.capitalize()}
                * Test Group: {self.test.name}
                * Control Group: {self.control.name}
            {str(self.boot_result)}
            '''
        )


################################################################################
# Permutations
################################################################################

type _PermutationTestTypes = Literal['means', 'medians', 't_welch', 'mwu']


@dataclass
class PermResult:
    r"""Dataclass to store permutation test results.

    Attributes
    ----------
    obs_stat : float or int
        Observed test statistic.
    perm_dist : np.ndarray
        Permutation null distribution.
    p : float
        P-value from permutation test.
    """

    obs_stat: float | int
    p: float
    perm_dist: np.ndarray

    def __str__(self):
        r"""String representation of permutation results.

        Returns
        -------
        str
            Formatted observed statistic, mean/SEM of permutation distribution,
            and p-value.
        """
        return f'''
        Observed Test Stat: {self.obs_stat:.4f}
        P = {self.p:.5f}
        '''


class TwoSeriesPermutation:
    r"""Class for permutation hypothesis tests on two continuous series.

    Supports difference of means/medians, Welch's t, or Mann-Whitney U.

    Parameters
    ----------
    test : pd.Series
        Test group data.
    control : pd.Series
        Control group data.
    test_type : {'means', 'medians', 't_welch', 'mwu'}, optional
        Test statistic type. Defaults to 'means'.
    alpha_level : float, optional
        Significance level. Defaults to 0.05.
    n_resamples : int, optional
        Number of permutations. Defaults to 10,000.
    rng : int, default 519
        Random seed.

    Attributes
    ----------
    test : pd.Series
        Cleaned test data.
    control : pd.Series
        Cleaned control data.
    test_type : str
        Selected test type.
    alpha : float
        Significance level.
    n_resamples : int
        Number of resamples.
    rng : int
        Random seed.
    perm_result : PermResult or None
        Stored permutation results.

    Raises
    ------
    ValueError
        If test_type is invalid.
    """

    _test_display_names = {
        'means': 'Difference of means',
        'medians': 'Difference of medians',
        't_welch': 'Welch t-statistic',
        'mwu': 'M-W U-statistic',
    }
    _tests_using_parametric_summ_stats = ['means', 't_welch']
    _tests_using_nonparametric_summ_stats = ['medians', 'mwu']

    def __init__(self,
                 test: pd.Series,
                 control: pd.Series,
                 test_type: _PermutationTestTypes = 'means',
                 alpha_level: float = .05,
                 n_resamples: int = 10_000,
                 rng: Optional[int] = 519):
        # Rename test & control if column names are same
        if test.name == control.name:
            warnings.warn(
                SeriesNameCollisionWarning(
                    '`test` and `control` have the same column name.'
                ),
                stacklevel=2
            )
            test = test.rename(f'{test.name}_TEST')
            control = control.rename(f'{test.name}_CTRL')

        self.test = test.dropna().reset_index(drop=True)
        self.control = control.dropna().reset_index(drop=True)

        if test_type in self._test_display_names.keys():
            self.test_type = test_type
        else:
            raise ValueError(
                f"`test_type` must be one of {self._test_display_names.keys()}."
            )

        self.alpha = alpha_level
        self._alt_hypothesis = '='
        self.n_resamples = n_resamples
        self.rng = rng
        self.perm_result: Optional[PermResult] = None

    @property
    def summ_stats(self):
        r"""Get summary statistics based on test type.

        Returns
        -------
        pd.DataFrame
            Parametric or nonparametric summary.
        """
        if self.test_type in self._tests_using_parametric_summ_stats:
            return self._parametric_summ_stats()

        # Explicitly want to default to nonparametric summary stats
        else:
            return self._nonparametric_summ_stats()

    @property
    def test_stat_func(self) -> Callable:
        r"""Get the test statistic function based on test_type.

        Returns
        -------
        Callable
            Function to compute test statistic.
        """
        func_map = {
            'means': self.mean_diff,
            'medians': self.median_diff,
            't_welch': self.t_welch,
            'mwu': self.mwu,
        }

        return func_map[self.test_type]

    def _parametric_summ_stats(self):
        r"""Compute parametric summary statistics.

        Returns
        -------
        pd.DataFrame
            Summary with n, mean, std, min, max, Ha indicator.
        """
        summ_stats = pd.DataFrame(
            data={
                self.test.name: {
                    'n': self.test.count(),
                    'mean': self.test.mean(),
                    'std': self.test.std(),
                    'min': self.test.min(),
                    'max': self.test.max(),
                },

                'Ha': self._alt_hypothesis,

                self.control.name: {
                    'n': self.control.count(),
                    'mean': self.control.mean(),
                    'std': self.control.std(),
                    'min': self.control.min(),
                    'max': self.control.max(),
                },
            },
        )
        return summ_stats

    def _nonparametric_summ_stats(self):
        r"""Compute nonparametric summary statistics.

        Returns
        -------
        pd.DataFrame
            Summary with n, min, q1, median, q3, max, iqr, Ha indicator.
        """
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

                'Ha': self._alt_hypothesis,

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

        return summ_stats

    def permute(self, n_resamples: int = None):
        r"""Perform permutation test.

        Parameters
        ----------
        n_resamples : int, optional
            Number of permutations. Defaults to self.n_resamples.

        Returns
        -------
        PermResult
            Permutation results.

        Notes
        -----
        Uses 'independent' permutation type, and does not support paired
        samples. Updates Ha indicator in summ_stats if p < alpha.
        """
        # Default to using self.n_resamples
        n_resamples = n_resamples or self.n_resamples

        perm_result = stats.permutation_test(
            data=(self.test, self.control),
            statistic=self.test_stat_func,
            permutation_type='independent',
            n_resamples=n_resamples,
            vectorized=True,
            axis=0,
            alternative='two-sided',
            rng=self.rng
        )

        # Save output to attribute
        self.perm_result = PermResult(
            obs_stat=perm_result.statistic.item(),
            perm_dist=perm_result.null_distribution,
            p=perm_result.pvalue.item()
        )

        # If p-value is significant, indicate direction in summ_stats
        if self.perm_result.p < self.alpha:
            if self.perm_result.obs_stat > 0:
                self._alt_hypothesis = '>'
            elif self.perm_result.obs_stat < 0:
                self._alt_hypothesis = '<'
            else:
                warnings.warn(
                    'Significant P value but no between-groups difference in '
                    'test statistic observed. Verify data/results.'
                )
                self._alt_hypothesis = '='
        else:
            self._alt_hypothesis = '='

        return self.perm_result

    @staticmethod
    def mean_diff(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ):
        r"""Compute difference of means for permutation.

        Parameters
        ----------
        sample_a : VectorLike
            First permuted sample.
        sample_b : VectorLike
            Second permuted sample.
        axis : {0, 1, 'rows', 'columns'}, optional
            Axis to compute. Defaults to 0.

        Returns
        -------
        float or np.float64
            Difference of means.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Calculate absolute difference in means
        return a.mean(axis=axis) - b.mean(axis=axis)

    @staticmethod
    def median_diff(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ):
        r"""Compute difference of medians for permutation.

        Parameters
        ----------
        sample_a : VectorLike
            First permuted sample.
        sample_b : VectorLike
            Second permuted sample.
        axis : {0, 1, 'rows', 'columns'}, optional
            Axis to compute. Defaults to 0.

        Returns
        -------
        float or np.float64
            Difference of medians.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Calculate absolute difference in means
        return np.median(a, axis=axis) -  np.median(b, axis=axis)

    @staticmethod
    def t_welch(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ):
        r"""Calculate Welch's 2-independent t-stat for a single permuted sample.

        Takes in observations that have been randomly permuted into test &
        control samples, called sample_a & sample_b, respectively (to
        distinguish from actual observed test/control), and returns the
        t-statistic.

        Parameters
        ----------
        sample_a, sample_b : pd.Series or array_like
            A permuted set of observations for test or control sample,
            respectively.
        axis : {0, 1, 'rows', 'columns'}, default=0
            Axis of sample_a & sample_b on which to perform t-test; passed to
            `stats.ttest_ind()`.

        Returns
        -------
        float or NumPy float
            Welch's t-statistic for the permuted sample.

        Notes
        -----
        Computes an asymptotic Welch's t-test (`equal_var=False`; correction for
        heteroskedasticity), which is repeated for each permuted sample. Since
        unequal-varianced samples violates the exchangeability assumption for
        permutation testing, the permuted p-value is not exact for finite N,
        thus Type I error is not guaranteed. However, Janssen & Pauls (2005)
        Monte Carlo simulation study found that in samples with unequal
        variance, permuted Welch t-tests still control Type I error better than
        both standard asymptotic Welch's t-test, and better than permuted
        Student's t-test.

        Of note, passed `PermutationMethod` instance to `method` would use
        `stats.permutation_test` to compute identical results, since
        `self.permute()` just runs `stats.permutation_test` using
        `self.t_welch()` as the test statistic. In later releases, permutation
        t- & U-tests will be implemented using native SciPy method, but for now,
        all permutation stats methods are being kept housed in this class, and
        implementing the native SciPy method would be complex and temporary.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Run Welch's t-test & extract t-statistic
        test = stats.ttest_ind(
            a=a, b=b,
            axis=axis,
            equal_var=False,
            alternative='two-sided',
        )

        return test.statistic

    @staticmethod
    def mwu(
        sample_a: VectorLike,
        sample_b: VectorLike,
        axis: Literal[0, 1, 'rows', 'columns'] = 0
    ):
        r"""Calculate Mann-Whitney U test for a single permuted sample.

        Takes in observations that have been randomly permuted into test &
        control samples, called sample_a & sample_b, respectively (to
        distinguish from actual observed test/control), and returns the Mann-
        Whitney U-statistic.

        Parameters
        ----------
        sample_a : VectorLike
            permuted observations assigned to test sample
        sample_b : VectorLike
            permuted observations assigned to control sample
        axis : {0, 1, 'rows', 'columns'}, default=0
            axis to compute MWU test; passed to stats.mannwhitneyu

        Returns
        -------
        float or NumPy float
            U-statistic for the permuted sample.

        Notes
        -----
        Computes a standard non-resampling M-W U-test, which is repeated for
        each permuted sample. Per SciPy docs, `method='auto'` computes an
        exact test if 8 or fewer n_obs in either sample_a or sample_b.

        Of note, passed `PermutationMethod` instance to `method` would use
        `stats.permutation_test` to compute identical results, since
        `self.permute()` just runs `stats.permutation_test` using `self.mwu()`
        as the test statistic. In later releases, permutation t- & U-tests will
        be implemented using native SciPy method, but for now, all permutation
        stats methods are being kept housed in this class, and implementing
        the native SciPy method here would be complex and temporary.
        """
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert to NDArray)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Run Mann-Whitney U-test
        test = stats.mannwhitneyu(
            x=a, y=b,
            use_continuity=True,
            alternative='two-sided',
            axis=axis,
            method='auto',
        )

        return test.statistic

    def __str__(self):
        r"""String representation of permutation results.

        Runs permute if not done.

        Returns
        -------
        str
            Formatted summary and permutation results.
        """
        # Ensure permutation has been run before printing
        if self.perm_result is None:
            self.permute()

        return (
            f'{str(self.summ_stats)}\n'
            f'''
            Permutation Hypothesis Test
                * Test Stat: {self._test_display_names[self.test_type]}
                * Test Group: {self.test.name}
                * Control Group: {self.control.name}
            {str(self.perm_result)}
            '''
        )


class TwoSamplePermutation(TwoSeriesPermutation):
    r"""Class for permutation tests with a boolean grouping variable.

    Extends TwoSeriesPermutation to split continuous series by boolean group.

    Parameters
    ----------
    bool_x : pd.Series
        Boolean grouping.
    num_y : pd.Series
        Continuous outcome.
    test_type : {'means', 'medians', 't_welch', 'mwu'}, default 'means'
        Test type.
    alpha_level : float, default 0.05
        Significance level.
    n_resamples : int, default 10_000
        Permutations..
    rng : int, default 519
        Random seed.
    x_test_lvl : bool, default True
        Test level.

    Attributes
    ----------
    x : pd.Series
        Grouping.
    y : pd.Series
        Outcome.

    Raises
    ------
    SeriesNameCollisionError
        If names collide.
    ValueError
        If groups empty.
    """

    def __init__(self,
                 bool_x: pd.Series, num_y: pd.Series,
                 test_type: _PermutationTestTypes = 'means',
                 alpha_level: float = .05,
                 n_resamples: int = 10_000,
                 rng: Optional[int] = 519,
                 x_test_lvl: bool = True):
        # Ensure names exist and are not the same
        bool_x = bool_x.rename(bool_x.name or 'x')
        num_y = num_y.rename(num_y.name or 'y')
        if bool_x.name == num_y.name:
            raise SeriesNameCollisionError('Both bool_x and num_y cannot '
                                           'have same names.')

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
                         test_type=test_type,
                         alpha_level=alpha_level,
                         n_resamples=n_resamples,
                         rng=rng)

    def __str__(self):
        r"""String representation with outcome name.

        Returns
        -------
        str
            Formatted outcome, summary, permutation results.
        """
        # Ensure bootstrapping has been run before printing
        if self.perm_result is None:
            self.permute()

        return (
            f'Outcome: {self.y.name}\n\n'
            f'{str(self.summ_stats)}\n'
            f'''
            Permutation Hypothesis Test
                * Test Stat: {self._test_display_names[self.test_type]}
                * Test Group: {self.test.name}
                * Control Group: {self.control.name}
            {str(self.perm_result)}
            '''
        )
