r"""Classes to run statistics based on contingency tables for categorical data.

MulticlassContingencyStats runs summary stats and :math:`\chi^2` test stats for
a contingency table with any number of IV & DV levels. BooleanContingencyStats
inherits from MulticlassContingencyStats, and is a special case for a 2x2
contingency table, also implementing Fisher's exact test.
"""
import warnings
from typing import Optional, Literal
import pandas as pd
from scipy import stats


class MulticlassContingencyStats:
    r"""Compute contingency table stats with arbitrary number of IV & DV levels.

    Take 2 pandas Series representing row and column variables, compute a
    contingency table, and provides methods for statistical tests like
    summary & chi-squared stats.

    Parameters
    ----------
    table_rows : pd.Series
        Series representing the row variable (typically predictor).
    table_cols : pd.Series
        Series representing the column variable (typically outcome).
    row_title : str, optional
        Title for the row index. Defaults to the name of `table_rows`.
    row_names : list[str], optional
        Custom names for the row levels.
    col_title : str, optional
        Title for the column index. Defaults to the name of `table_cols`.
    col_names : list[str], optional
        Custom names for the column levels.

    Attributes
    ----------
    idx_series : pd.Series
        The row variable series.
    col_series : pd.Series
        The column variable series.
    row_title : str
        Title for rows.
    row_names : list[str]
        Names for row levels.
    col_title : str
        Title for columns.
    col_names : list[str]
        Names for column levels.

    Returns
    -------
    MulticlassContingencyStats object

    See Also
    --------
    BooleanContingencyStats

    Notes
    -----
    This class assumes categorical data in the input series. For better
    structure, consider passing intervention and outcome series explicitly in
    future versions.
    """

    def __init__(self,
                 table_rows: pd.Series,
                 table_cols: pd.Series,
                 row_title: Optional[str] = None,
                 row_names: Optional[list[str]] = None,
                 col_title: Optional[str] = None,
                 col_names: Optional[list[str]] = None):
        # Thought: this might be better structured by taking in parameters for
        # `intervention: pd.Series` & `outcome: pd.Series`, instead of rows/cols
        # - Currently, it takes in tables & rows, and "hopefully" you input the
        #   correct `axis` value in the `.table()` method.
        # - Instead, have `__init__` take in an intervention & outcome, and
        #   have `.table()` output intervention as rows, and outcome as cols.
        #   - If desired, you could implement a `transpose: bool = False` param
        #     in `.table()` if user really wants axes swapped.
        self._df = (pd.concat([table_rows, table_cols], axis='columns')
                    .dropna(axis='index'))
        self.idx_series = self._df[table_rows.name]
        self.col_series = self._df[table_cols.name]
        self.row_title = row_title
        self.row_names = row_names
        self.col_title = col_title
        self.col_names = col_names

    def table(self, as_pct: bool = False,
              axis: Literal[0, 1, 'rows', 'columns'] = 'rows') -> pd.DataFrame:
        r"""Compute the contingency table.

        Parameters
        ----------
        as_pct : bool, optional
            If True, return percentages instead of counts. Defaults to False.
        axis : str or int, optional
            Axis for percentage calculation, required if `as_pct` is True:

            * 'rows' or 0: percentages across rows.
            * 'columns' or 1: percentages across columns.

        Returns
        -------
        pd.DataFrame
            Contingency table with margins (totals).

        Raises
        ------
        ValueError
            If `as_pct` is True and `axis` is None.
        """
        table = pd.crosstab(
            index=self.idx_series,
            columns=self.col_series,
            rownames=[self.row_title],
            colnames=[self.col_title],
            margins=True,
            margins_name='Totals'
        )

        # Could probably clean this up by making them properties
        if self.row_names is not None:
            table.index = pd.Index(self.row_names + ['col_totals'])
        else:
            table.index = pd.Index([idx for idx in table.index[:-1]]
                                    + ['col_totals'])

        if self.row_title is not None:
            table.index.name = self.row_title
        else:
            table.index.name = self.idx_series.name

        if self.col_names is not None:
            table.columns = pd.Index(self.col_names + ['row_totals'])
        else:
            table.columns = pd.Index([col for col in table.columns[:-1]]
                                     + ['row_totals'])

        if self.col_title is not None:
            table.columns.name = self.col_title
        else:
            table.columns.name = self.col_series.name

        if as_pct:
            if (axis == 'rows') or (axis == 0):
                table = table.div(
                    table.loc[:, 'row_totals'], axis='rows'
                ).mul(100)
            elif (axis == 'columns') or (axis == 1):
                table = table.div(
                    table.loc['col_totals', :], axis='columns'
                ).mul(100)
            elif axis is None:
                raise ValueError("If as_pct=True, axis must be 0/'rows' or "
                                 "1/'columns'.")

        return table

    def matrix(self):
        r"""Get the contingency matrix, without marginal totals.

        Returns
        -------
        pd.DataFrame
            Contingency table excluding row and column totals.
        """
        return self.table().drop(index='col_totals', columns='row_totals')

    def chi2(self, correction: bool = False):
        r"""Perform Chi-squared test of independence.

        Parameters
        ----------
        correction : bool, optional
            Apply Yates' continuity correction. Defaults to False.

        Returns
        -------
        scipy.stats._result_classes.Chi2ContingencyResult
            Result object containing statistic, p-value, dof, and expected
            frequencies.

        Warns
        -----
        UserWarning
            If any expected frequencies are less than 5.

        Notes
        -----
        Yates' continuity correction applies a correction factor for the fact
        that contingency table observations are discrete (i.e. the table of
        counts will always be integers), yet the chi-squared distribution is
        continuous; this is especially relevant in cases with small samples.

        Yates' correction works by subtracting 0.5 from the absolute difference
        between observed vs. expected frequencies in all cells. While
        statistically valid and unbiased, it has been argued that the effect of
        Yates' correction (increased p-value) is excessively conservative. For
        this reason, the ``correction`` parameter defaults to False. In cases
        of 2x2 contingency tables, with small N, Fisher's exact test should
        generally be preferred over chi-squared with Yates' correction. In non-
        2x2 cases, it may still be acceptable to go without it, especially since
        any significance in a non-2x2 chi-squared test is likely to be followed
        by pairwise 2x2 testing, which can utilize Fisher's test.
        """
        test = stats.contingency.chi2_contingency(self.matrix(),
                                                  correction=correction)
        exp_freq = (
            (test.expected_freq < 5).sum()
            / (test.expected_freq < 5).size
        )
        if exp_freq > 0:
            warnings.warn(f"Expected frequency < 5 in {exp_freq:.1%} of cells.")

        return test

    def print_results(self):
        r"""Print contingency tables and Chi-squared results."""
        # def line_length():
        #     left_title_col = max(
        #         len(self.row_title), len(self.col_title),
        #         (
        #             max(self.row_names, key=len) if self.row_names is not None
        #             else 5
        #         ),
        #         (
        #             max(self.co, key=len) if self.co is not None
        #             else 5
        #         ),
        #
        #     )

        chi2 = self.chi2()

        print(f'{self.row_title} vs. {self.col_title}\n', '='*40)
        print(f'Table (# Obs):\n'
              f'{self.table()}')
        print(f'-'*40,
              f'\nTable (% of Totals):\n'
              f'{self.table(as_pct=True)}')
        print('-'*40,
              f'\nChi^2 ToI:\n'
              f'X^2({chi2.dof}) = {chi2.statistic:.3f}, '
              f'p = {chi2.pvalue:.4f}')
        print('-'*40, '\n')


class BooleanContingencyStats(MulticlassContingencyStats):
    r"""Perform contingency statistics on boolean (2x2) tables.

    Extends MulticlassContingencyStats with methods specific to 2x2 tables,
    such as odds ratio and Fisher's exact test.

    Parameters
    ----------
    table_rows : pd.Series
        Series representing the row variable (typically predictor), with dtype
        collapsible to Boolean (True/False, 1/0, 1.0/0.0).
    table_cols : pd.Series
        Series representing the column variable (typically outcome), with dtype
        collapsible to Boolean (True/False, 1/0, 1.0/0.0).
    row_title : str, optional
        Title for the row index.
    row_names : list[str], optional
        Custom names for the row levels.
    col_title : str, optional
        Title for the column index.
    col_names : list[str], optional
        Custom names for the column levels.

    Returns
    -------
    BooleanContingencyStats object

    Warns
    -----
    UserWarning
        If any cell-wise expected frequencies < 5

    See Also
    --------
    MulticlassContingencyStats

    Notes
    -----
    Assumes the contingency table is 2x2.

    p-values are displayed for both the chi-square test of independence (ToI),
    and for an exact test like Fisher's. Deciding which test to report can
    follow Cochran's rule-of-thumb criteria [1]_ [2]_, which includes (but is
    not limited to) the following as indication for use of an exact test
    (Fisher's exact test in the original 1952 article) over chi-squared:

    * Any cell-wise expected frequency < 5
        * Actual rule is <20% must have expected frequency < 5, which means no
          cells can have low expected frequency in a 2x2 table.
    * N < 20
    * Cochran (1952) [1]_ recommends using Yates' correction if N > 40 but
      any expected frequency < 500; ``unistat`` does not implement this
      by default.

    By default, ```unistat`` *never* implements Yates' correction factor.
    Hasselblad & Lokhnygina (2007) [3]_ found that in **all** cases, Yates-
    corrected chi-squared is inferior to Fisher's exact test. Furthermore,
    they found that even Fisher's exact test is too conservative, and that,
    depending on sample size, Fisher's mid-p test or Barnard's exact test offer
    better power while maintaining target Type I error control.

    Alternative exact test(s) will be implemented in later releases; expect
    that at a minimum, this will include Boschloo's exact test.

    Lydersen et al. (2009) [4]_ compared multiple different exact tests, and
    noted the following:

    * Standard Fisher's exact test is near-uniformly too conservative, though it
      always maintains Type I error rate
    * Fisher's mid-p generally improves power, but occasionally violates Type I
      error rate.
    * Barnard's exact test is an excellent performer, but is computationally
      intensive to a prohibitive degree (exponential time complexity).
    * Boschloo's exact test (aka Fisher-Boschloo test) is an extension of
      Fisher's exact, and was considered the gold standard by Lydersen et al.;
      it is universally more powerful than traditional Fisher's exact & mid-p,
      and in trials did not violate target Type I error rate.
        * Further improved using the Berger-Boos correction, particularly for
          unbalanced designs (e.g. if survival occurs much more often than
          mortality) [4]_ [5]_
        * Standard Berger-Boos correction factor is :math:`\gamma = 0.001` [4]_
            * Not implemented by SciPy, though included in R ``Exact`` package

    References
    ----------
    ..  [1] Cochran, William G. The :math:`\text{\chi^2}` Test of Goodness of
        Fit. Ann. Math. Statist. 23 (3) 315 - 345, September 1952.
        doi: 10.1214/aoms/1177729380.

    ..  [2] Cochran, William G. The Combination of Estimates from Different
        Experiments" Biometrics 10, no. 1 (1954): 101–29. doi: 10.2307/3001666.

    .. [3] Hasselblad V, Lokhnygina Y. Tests for 2 x 2 tables in clinical
        trials. Journal of Modern Applied Statistical Methods. 2007;6:456–468.
        doi: 10.56801/10.56801/v6.i.318.

    .. [4] Lydersen S, Fagerland MW, Laake P. Recommended tests for association
        in 2 x 2 tables. Stat Med. 2009 Mar 30;28(7):1159-75.
        doi: 10.1002/sim.3531.

    ..  [5] Kang SH, Ahn CW. Tests for the homogeneity of two binomial
        proportions in extremely unbalanced 2 x 2 contingency tables. Stat Med.
        2008 Jun 30;27(14):2524-35. doi: 10.1002/sim.3055.
    """

    def __init__(self,
                 table_rows: pd.Series,
                 table_cols: pd.Series,
                 row_title: Optional[str] = None,
                 row_names: Optional[list[str]] = None,
                 col_title: Optional[str] = None,
                 col_names: Optional[list[str]] = None):
        super().__init__(table_rows, table_cols,
                         row_title, row_names,
                         col_title, col_names)

    def odds_ratio(self, kind: str = 'sample'):
        r"""Compute the odds ratio.

        Parameters
        ----------
        kind : str, optional
            Type of odds ratio: 'sample' (default), 'conditional', or 'unconditional'.

        Returns
        -------
        scipy.stats._result_classes.OddsRatioResult
            Result object with statistic and confidence interval.
        """
        odds_ratio = stats.contingency.odds_ratio(self.matrix(), kind=kind)
        return odds_ratio

    def fisher_exact(self, alternative='two-sided'):
        r"""Perform Fisher's exact test.

        Parameters
        ----------
        alternative : str, optional
            Alternative hypothesis: 'two-sided' (default), 'less', or 'greater'.

        Returns
        -------
        float
            p-value of the test.
        """
        p_val = (
            stats.fisher_exact(self.matrix(), alternative=alternative).pvalue
        )
        return p_val

    def print_results(self):
        r"""Print contingency tables, odds ratio, Chi-squared, and Fisher's exact results.

        Overrides the parent method to include 2x2-specific statistics.
        """
        chi2 = self.chi2()

        print(f'{self.row_title} vs. {self.col_title}\n', '='*40)
        print(f'Table (# Obs):\n'
              f'{self.table()}')
        print(f'-'*40,
              f'\nTable (% of Totals):\n'
              f'{self.table(as_pct=True)}')
        print('-'*40,
              f'\nOdds Ratio:\n'
              f'OR = {self.odds_ratio().statistic:.4f}, '
              f'95% CI {self.odds_ratio().confidence_interval().low:.4f} to '
              f'{self.odds_ratio().confidence_interval().high:.4f}')
        print('-'*40,
              f'\nChi^2 ToI:\n'
              f'X^2({chi2.dof}) = {chi2.statistic:.3f}, '
              f'p = {chi2.pvalue:.4f}')
        print('-'*40,
              f'\nFisher Exact Test:\n'
              f'p = {self.fisher_exact():.4f}')
