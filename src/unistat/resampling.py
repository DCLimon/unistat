
# Standard
from typing import Optional, Literal
from collections.abc import Callable
from dataclasses import dataclass
import warnings
# 3rd party
import numpy as np
import pandas as pd
from scipy import stats
# Local
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
    obs_stat: float | int
    boot_mean: float | int
    boot_sem: float | int
    boot_ci_lo: float | int
    boot_ci_hi: float | int
    p: float
    boot_ci_pct: float
    boot_dist: np.ndarray

    def __str__(self):
        return f'''
        Observed Test Stat: {self.obs_stat:.4f}
        Mean of Bootstrap Dist.: {self.boot_mean:.4f}
        SEM of Bootstrap Dist.: {self.boot_sem:.4f}
        BCa {self.boot_ci_pct:.0%} CI: {self.boot_ci_lo:.4f} to {self.boot_ci_hi:.4f}
        P = {self.p if self.p is not None else float('nan'):.5f}
        '''


class TwoSeriesBootstrap:
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
            SeriesNameCollisionWarning('`test` and `control` have the '
                                       'same column name.')
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
        axis = {'rows': 0, 'columns': 1}.get(axis, axis)

        # Convert inputs once (this fixes the 'unused variable' issue)
        a = np.asarray(sample_a)
        b = np.asarray(sample_b)

        # Calculate absolute difference in means
        return np.median(a, axis=axis) -  np.median(b, axis=axis)


class TwoSampleBootstrap(TwoSeriesBootstrap):
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
    obs_stat: float | int
    p: float
    perm_dist: np.ndarray

    def __str__(self):
        return f'''
        Observed Test Stat: {self.obs_stat:.4f}
        P = {self.p:.5f}
        '''


class TwoSeriesPermutation:
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
            SeriesNameCollisionWarning('`test` and `control` have '
                                       'the same column name.')
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
        if self.test_type in self._tests_using_parametric_summ_stats:
            return self._parametric_summ_stats()

        # Explicitly want to default to nonparametric summary stats
        else:
            return self._nonparametric_summ_stats()

    @property
    def test_stat_func(self) -> Callable:
        func_map = {
            'means': self.mean_diff,
            'medians': self.median_diff,
            't_welch': self.t_welch,
            'mwu': self.mwu,
        }

        return func_map[self.test_type]

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

    def permute(self, n_resamples: int = None):
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
        """Calculate Welch's 2-independent t-stat for a single permuted sample.

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
        """Calculate Mann-Whitney U test for a single permuted sample.

        Takes in observations that have been randomly permuted into test &
        control samples, called sample_a & sample_b, respectively (to
        distinguish from actual observed test/control), and returns the Mann-
        Whitney U-statistic.

        Parameters
        ----------
            sample_a: permuted observations assigned to test sample
            sample_b: permuted observations assigned to control sample
            axis: axis to compute MWU test; passed to stats.mannwhitneyu

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
