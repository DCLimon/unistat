import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy import stats


class CorrStats:
    def __init__(self, x: pd.Series, y: pd.Series, parametric: bool = True):
        self._df = pd.concat([x, y], axis='columns').dropna(axis='index')
        self.x = self._df[x.name]
        self.y = self._df[y.name]
        self.parametric = parametric

    def __str__(self):
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
        if self.parametric:
            return stats.pearsonr(self.x, self.y, alternative='two-sided')
        else:
            return stats.spearmanr(self.x, self.y, alternative='two-sided')

    @property
    def n(self):
        return len(self._df.index)

    @property
    def stat(self):
        return self.result.statistic

    @property
    def p(self):
        return self.result.pvalue


class TwoSeriesStats:
    def __init__(self,
                 test: pd.Series,
                 control: pd.Series,
                 parametric: bool = True,
                 alpha_level: float = .05):
        if test.name == control.name:
            warnings.warn('`test` and `control` have the same column name.')
            test = test.rename(f'{test.name}_TEST')
            control = control.rename(f'{test.name}_CTRL')
        self.test = test.dropna().reset_index(drop=True)
        self.control = control.dropna().reset_index(drop=True)
        self.control = control.dropna()
        self.parametric = parametric
        self.alpha = alpha_level

    def __str__(self):
        if self.parametric:
            return (f'{self.parametric_summ_stats()}\n'
                    f'{self.t_test()}')
        else:
            return (f'{self.nonparametric_summ_stats()}\n'
                    f'{self.mwu_test()}')

    def conf_int(self, pct_ci: float = None):
        if pct_ci is None:
            pct_ci = 1 - self.alpha

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
        return ControlTestStats(control=control_ci, test=test_ci)

    def parametric_summ_stats(self, alpha_level: float = None):
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
        result = stats.ttest_ind(
            a=self.test,
            b=self.control,
            equal_var=equal_var,
        )

        output = pd.Series(
            {
                't-statistic': result.statistic,
                'scipy_df': result.df,
                'calc_df': (self.test.count() - 1) + (self.control.count() - 1),
                'p-value': result.pvalue,
            }
        )

        return output

    def nonparametric_summ_stats(self, alpha_level: float = None):
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
        result = stats.mannwhitneyu(x=self.test, y=self.control)

        desc_stats = pd.Series(
            {
                'U-statistic': result.statistic,
                'p-value': result.pvalue,
            }
        )
        return desc_stats


class TwoSampleStats(TwoSeriesStats):
    def __init__(self,
                 bool_x: pd.Series, num_y: pd.Series,
                 parametric: bool = True,
                 alpha_level: float = .05,
                 x_test_lvl: bool = True):
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
                         parametric=parametric,
                         alpha_level=alpha_level)

    def __str__(self):
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
