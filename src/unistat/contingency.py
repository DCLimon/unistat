"""Classes for statistics based on contingency tables for categorical data.

:class:`MulticlassContingencyStats` runs summary stats and :math:`\chi^2` test
stats for a contingency table with any number of IV & DV levels.

:class:`BooleanContingencyStats` inherits from
:class:`MulticlassContingencyStats`, and is a special case for a 2x2 contingency
table, also implementing Fisher's & Boschloo's exact tests.
"""
from abc import ABC, abstractmethod
import warnings
from typing import Optional, Literal
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from ._types import PCorrectionMethod
from .exceptions import ExpectedFrequencyWarning, warn_expected_frequency


class _ContingencyStats(ABC):
    r"""Compute contingency table stats with arbitrary number of IV & DV levels.

    Take 2 ``pandas`` Series representing row and column variables, compute a
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
    _ContingencyStats object

    See Also
    --------
    MulticlassContingencyStats
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
        self._df = (
            pd.concat(
                [
                    table_rows.astype('category'),
                    table_cols.astype('category'),
                ],
                axis='columns')
            .dropna(axis='index', how='any')
        )

        self.idx_series = self._df[table_rows.name]
        self.col_series = self._df[table_cols.name]

        self.row_title = row_title or self.idx_series.name
        self.col_title = col_title or self.col_series.name

        # Done to ensure correct ordering
        crosstab = self._crosstab
        self.row_names = row_names or crosstab.index.tolist()[:-1]
        self.col_names = col_names or crosstab.columns.tolist()[:-1]

        self._exp_freq_lt5()

    @property
    def _crosstab(self) -> pd.DataFrame:
        return pd.crosstab(
            index=self.idx_series,
            columns=self.col_series,
            rownames=[self.row_title],
            colnames=[self.col_title],
            margins=True,
            margins_name='Totals'
        )

    def get_table(
        self,
        as_pct: bool = False,
        axis: Literal[0, 1, 'rows', 'columns'] = 'rows'
    ) -> pd.DataFrame:
        r"""Compute the contingency table.

        Parameters
        ----------
        as_pct : bool, optional
            If True, return percentages instead of counts. Defaults to False.
        axis : str or int, optional
            Axis for percentage calculation, required if ``as_pct`` is True:

            * 'rows' or 0: percentages across rows.
            * 'columns' or 1: percentages across columns.

        Returns
        -------
        pd.DataFrame
            Contingency table with margins (totals).

        Raises
        ------
        ValueError
            If ``as_pct`` is True and ``axis`` is None.
        """
        table = self._crosstab.copy()
        table.index = pd.Index(
            data=self.row_names+['col_totals'],
            name=self.row_title
        )
        table.columns = pd.Index(
            data=self.col_names+['row_totals'],
            name=self.col_title
        )

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

    @property
    def matrix(self) -> pd.DataFrame:
        r"""Get the contingency matrix, without marginal totals.

        Returns
        -------
        pd.DataFrame
            Contingency table excluding row and column totals.
        """
        return (self.get_table()
                .drop(index='col_totals')
                .drop(columns='row_totals'))

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
        ExpectedFrequencyWarning
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
        test = stats.contingency.chi2_contingency(self.matrix,
                                                  correction=correction)
        self._exp_freq = test.expected_freq
        return test

    @property
    def exp_freq(self) -> pd.DataFrame:
        r"""Cell-wise expected frequencies."""
        if not(hasattr(self, '_exp_freq')):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ExpectedFrequencyWarning)
                self.chi2()
        return pd.DataFrame(
            self._exp_freq,
            index=self.matrix.index,
            columns=self.matrix.columns
        )

    @abstractmethod
    def print_results(self):
        r"""Print contingency tables and Chi-squared results."""
        pass

    def _exp_freq_lt5(self) -> None:
        """Check if any expected frequency is < 5 and issue warning."""
        if not hasattr(self, '_exp_freq'):
            _ = self.chi2()  # compute once, silently

        ratio = (self._exp_freq < 5).sum().sum() / self._exp_freq.size
        if ratio > 0:
            warn_expected_frequency(
                f'Expected frequency < 5 in {ratio:.1%} of cells.'
            )


class MulticlassContingencyStats(_ContingencyStats):
    r"""Compute contingency table stats for 3+ IV and/or DV levels.

    Take 2 pandas Series representing row and column variables, compute a
    contingency table, and provides methods for summary stats, chi-squared
    tests of independence, and post hoc testing.

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
    matrix : pd.DataFrame
        Crosstabulated frequency counts, with levels of ``idx_series`` and
        `col_series` as the index and columns, respectively. ``matrix`` does not
        include marginal row/column totals.
    exp_freq : pd.DataFrame
        Crosstabulated expected frequency counts. Format mirrors ``matrix``.

    Methods
    -------
    get_table(as_pct=False, axis='rows')
        Crosstabulated frequency counts with marginal row/column totals. Format
        otherwise mirrors ``matrix``.

    See Also
    --------
    BooleanContingencyStats

    Notes
    -----
    This class assumes categorical data in the input series. For better
    structure, consider passing intervention and outcome series explicitly in
    future versions.
    """
    _ResidualResult = namedtuple('ResidualResult',
                                 ['residuals', 'p_values', 'p_corr'])

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

    def __str__(self):
        with (
            pd.option_context(
                'display.max_columns', None,
                'display.width', 256,
                'display.max_colwidth', None,
                'display.expand_frame_repr', False
            )
        ):
            lines = []
            width = 60  # minimum

            title = f'{self.row_title} vs. {self.col_title}'
            width = max(width, len(title) + 4)

            lines.append('=' * width)
            lines.append(title.center(width))
            lines.append('=' * width)
            lines.append('')

            # Counts table
            table_str = str(self.get_table())
            table_lines = table_str.splitlines()
            table_w = (
                max(
                    len(line) for line in table_lines
                )
                if table_lines else 40
            )
            width = max(width, table_w + 4)

            lines.append('Contingency Table (counts):')
            lines.append('-' * width)
            lines.extend(table_lines)
            lines.append('')

            # Percentages table
            pct_str = str(self.get_table(as_pct=True))
            pct_lines = pct_str.splitlines()
            pct_w = max(len(line) for line in pct_lines) if pct_lines else 40
            width = max(width, pct_w + 4)

            lines.append('Table (% of totals):')
            lines.append('-' * width)
            lines.extend(pct_lines)
            lines.append('')

            # Omnibus chi-square
            chi2 = self.chi2()
            lines.append('Chi-square Test of Independence:')
            lines.append('-' * width)
            lines.append(f'χ²({chi2.dof}) = {chi2.statistic:.4f}, '
                         f'p = {chi2.pvalue:.5f}')
            lines.append('')

            # Post-hoc (only if significant)
            if chi2.pvalue <= 0.05:
                lines.append('Post-hoc results:')
                lines.append('-' * width)

                posthoc = self.post_hoc()
                if isinstance(posthoc, pd.DataFrame):  # pairwise case
                    posthoc_str = posthoc.to_string(
                        float_format='{:.4f}'.format,
                        justify='right',
                        col_space=12
                    )
                    posthoc_lines = posthoc_str.splitlines()
                    ph_w = (
                        max(
                            len(line) for line in posthoc_lines
                        )
                        if posthoc_lines else 40
                    )
                    width = max(width, ph_w + 4)
                    lines.extend(posthoc_lines)
                else:  # residuals case (namedtuple)
                    for name, df in zip(
                        ['Adjusted residuals',
                         'Crude p-values',
                         'Corrected p-values'],
                        posthoc
                    ):
                        lines.append(f'  {name}:')
                        df_str = df.to_string(float_format='{:.6f}'.format)
                        lines.extend(df_str.splitlines())
                        lines.append('')
                lines.append('')

            lines.append('=' * width)

            return '\n'.join(lines)

    def pairwise_post_hoc(self,
                          alpha: float = 0.05,
                          p_corr_method: PCorrectionMethod = 'holm'):
        r"""Perform pairwise chi-square or Boschloo exact post hoc tests.

        Appropriate if either rows or columns are binary.

        By default, Holm-Bonferroni correction is used, but all
        correction methods supported by
        `statsmodels.stats.multitest.multipletests()
        <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_
        are supported.

        Parameters
        ----------
        alpha : float, default .05
            Significance level to be used for p-value correction.
        p_corr_method : str, default 'holm'
            *p*-Value correction method.

        Returns
        -------
        pd.DataFrame
            Pairwise results condensed to a single DataFrame. Only returned
            when a pairwise test is performed.

        Raises
        ------
        NotImplementedError
            If both rows & columns have 3+ levels, which is technically
            possible, but currently, only the use of adjusted standardized
            residuals is supported for such cases.

        Examples
        --------
        >>> four_by_two.matrix
        Outcome    dv0  dv1
        Predictor
        iv0        102   63
        iv1         20    8
        iv2          3    7
        iv3          1   10
        >>> four_by_two.pairwise_post_hoc()
                       test  test_stat   p-value    p_corr
        iv0          X^2(1)   2.572025  0.108768  0.272463
        iv1          X^2(1)   2.095681  0.147716  0.272463
        iv2  Boschloo exact   0.090821  0.095690  0.272463
        iv3  Boschloo exact   0.000983  0.000722  0.003931

        The binary axis is automatically detected, and tested against each level
        of the axis with 3+ levels:

        >>> two_by_four.matrix
        Predictor  iv0  iv1  iv2  iv3
        Outcome
        dv0        102   20    3    1
        dv1         63    8    7   10
        >>> two_by_four.pairwise_post_hoc()
                       test  test_stat   p-value    p_corr
        iv0          X^2(1)   2.572025  0.108768  0.272463
        iv1          X^2(1)   2.095681  0.147716  0.272463
        iv2  Boschloo exact   0.090821  0.095690  0.272463
        iv3  Boschloo exact   0.000983  0.000722  0.003931
        """
        pairwise_results: list[pd.DataFrame] = []

        if (len(self.matrix.index) > 2) and (len(self.matrix.columns) > 2):
            raise NotImplementedError(
                'Pairwise post hoc testing when both IV & DV have 3+ levels is '
                'not currently supported.'
            )
        elif len(self.matrix.columns) == 2:
            new_index = self.matrix.index.tolist()
            old_index = self._crosstab.index.tolist()[:-1]

            for new_idx, old_idx in zip(new_index, old_index):
                pairwise_test = BooleanContingencyStats(
                    table_rows=(self.idx_series == old_idx),
                    row_title=new_idx,
                    row_names=['False', 'True'],
                    table_cols=self.col_series,
                    col_title=self.col_title,
                    col_names=self.col_names,
                )
                if (pairwise_test.exp_freq < 5).any(axis=None):
                    pairwise_test_type = 'Boschloo exact'
                    pairwise_test_stat = (pairwise_test
                                          .boschloo_exact().statistic)
                    pairwise_test_p = pairwise_test.boschloo_exact().pvalue
                else:
                    chi_sq_result = pairwise_test.chi2()
                    pairwise_test_type = f'X^2({chi_sq_result.dof:.0f})'
                    pairwise_test_stat = chi_sq_result.statistic
                    pairwise_test_p = chi_sq_result.pvalue

                pairwise_result = pd.DataFrame(
                    data={
                        'test': pairwise_test_type,
                        'test_stat': pairwise_test_stat,
                        'p-value': pairwise_test_p,
                    },
                    index=[new_idx],
                )
                pairwise_results.append(pairwise_result)

        elif len(self.matrix.index) == 2:
            new_cols = self.matrix.columns.tolist()
            old_cols = self._crosstab.columns.tolist()[:-1]

            for new_col, old_col in zip(new_cols, old_cols):
                pairwise_test = BooleanContingencyStats(
                    table_rows=self.idx_series,
                    row_title=self.row_title,
                    row_names=self.row_names,
                    table_cols=(self.col_series == old_col),
                    col_title=new_col,
                    col_names=['False', 'True'],
                )
                if (pairwise_test.exp_freq < 5).any(axis=None):
                    pairwise_test_type = 'Boschloo exact'
                    pairwise_test_stat = (pairwise_test
                                          .boschloo_exact().statistic)
                    pairwise_test_p = pairwise_test.boschloo_exact().pvalue
                else:
                    chi_sq_result = pairwise_test.chi2()
                    pairwise_test_type = f'X^2({chi_sq_result.dof:.0f})'
                    pairwise_test_stat = chi_sq_result.statistic
                    pairwise_test_p = chi_sq_result.pvalue

                pairwise_result = pd.DataFrame(
                    data={
                        'test': pairwise_test_type,
                        'test_stat': pairwise_test_stat,
                        'p-value': pairwise_test_p,
                    },
                    index=[new_col],
                )
                pairwise_results.append(pairwise_result)

        pairwise_df = pd.concat(pairwise_results, axis='rows')
        pairwise_df['p_corr'] = multipletests(
            pvals=pairwise_df['p-value'],
            alpha=alpha,
            method=p_corr_method,
        )[1]

        return pairwise_df

    def residuals_post_hoc(self,
                           alpha: float = 0.05,
                           p_corr_method: PCorrectionMethod = 'holm'):
        r"""Perform post hoc tests using adjusted standardized residuals.

        Preferred post hoc method if both rows & columns have 3+ levels (see
        **Notes**).

        Uses adjusted standardized residuals (a/k/a adjusted Pearson residuals;
        cell-wise Z-scores), which are converted to *p*-values and corrected.

        By default, Holm-Bonferroni correction is used, but all
        correction methods supported by
        `statsmodels.stats.multitest.multipletests()
        <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_
        are supported.

        Parameters
        ----------
        alpha : float, default .05
            Significance level to be used for p-value correction.
        p_corr_method : str, default 'holm'
            *p*-Value correction method.

        Returns
        -------
        residuals : pd.DataFrame
            Adjusted standardized residuals in same format as `self.matrix`.
            Shows direction of effect in each cell.
        p_values : pd.DataFrame
            Uncorrected *p*-values in same format as `self.matrix`.
        p_corr : pd.DataFrame
            Corrected *p*-values in same format as `self.matrix`.

        Raises
        ------
        NotImplementedError
            If both rows & columns have 3+ levels, which is technically
            possible, but currently, use of adjusted standardized residuals is
            mandatory for such cases.

        See Also
        --------
        pairwise_post_hoc : Pairwise tests if either variable is binary.

        Notes
        -----
        We prefer post hoc testing via adjusted standardized residuals only in
        cases where both rows & columns have 3+ levels.

        If either rows or columns are binary, *p*-values are identical for both
        levels of the binary factor (see **Examples**). If comparing
        :math:`k \times 2` rows & columns (where :math:`k \geq 3`), there would
        only be :math:`k` different *p*-values (duplicated twice). This method
        detects these cases, and only corrects for :math:`k` *p*-values, to
        avoid inflating type-II error rate by naively corrected :math:`2k` *p*-
        values.

        However, this approach is effectively equivalent to running :math:`k`
        pairwise :math:`\chi^2` tests without Yates correction. Use of the
        `pairwise_post_hoc()` method is preferred, since this will automatically
        use Boschloo exact tests when expected frequencies are <5.

        Examples
        --------
        >>> four_by_four.matrix
        Outcome    dv0  dv1  dv2  dv3
        Predictor
        iv0         70    0    0    0
        iv1         53    4    3    0
        iv2         28   17    4    2
        iv3         14    7    3    9

        Adjusted standardized residuals (a/k/a adjusted Pearson residuals,
        equivalent to Z-scores) indicate whether each cell was less (negative)
        or more (positive) frequent than expected by chance. *p*-Values indicate
        whether this difference is statistically significant.

        >>> four_by_four.residuals_post_hoc().residuals
        Outcome          dv0             dv1              dv2          dv3
        Predictor
        iv0         5.558156       -3.957284        -2.258185    -2.374231
        iv1         2.440597       -1.737651         0.141516    -2.125546
        iv2        -4.323559        4.913430         1.229105    -0.451581
        iv3        -5.155365        1.505522         1.307525     6.260734
        >>> four_by_four.residuals_post_hoc().p_values
        Outcome             dv0           dv1       dv2           dv3
        Predictor
        iv0        2.726397e-08  7.580683e-05  0.023934  1.758554e-02
        iv1        1.466299e-02  8.227228e-02  0.887462  3.354109e-02
        iv2        1.535320e-05  8.949665e-07  0.219032  6.515711e-01
        iv3        2.531376e-07  1.321900e-01  0.191034  3.831699e-10
        >>> four_by_four.residuals_post_hoc().p_corr
        Outcome             dv0       dv1       dv2           dv3
        Predictor
        iv0        4.089596e-07  0.000834  0.191473  1.582699e-01
        iv1        1.466299e-01  0.493634  1.000000  2.347877e-01
        iv2        1.842384e-04  0.000012  0.764137  1.000000e+00
        iv3        3.543927e-06  0.660950  0.764137  6.130718e-09

        When one axis is binary, compare the results `pairwise_hoc_hoc()` to
        `residuals_post_hoc()`:

        >>> four_by_two.matrix
        Outcome    dv0  dv1
        Predictor
        iv0        102   63
        iv1         20    8
        iv2          3    7
        iv3          1   10

        `pairwise_hoc_hoc()` uses either :math:`\chi^2` or Fisher exact tests,
        depending on expected frequencies for each comparison.

        >>> four_by_two.pairwise_post_hoc()
                       test  test_stat   p-value    p_corr
        iv0          X^2(1)   2.572025  0.108768  0.272463
        iv1          X^2(1)   2.095681  0.147716  0.272463
        iv2  Boschloo exact   0.090821  0.095690  0.272463
        iv3  Boschloo exact   0.000983  0.000722  0.003931

        `residuals_post_hoc()` gives similar results to pairwise :math:`\chi^2`
        tests (cf uncorrected *p*-values), but cannot use Boschloo (or Fisher)
        exact tests when expected frequencies are low (``iv2`` & ``iv3``):

        >>> four_by_two.residuals_post_hoc().p_values
        Outcome         dv0       dv1
        Predictor
        iv0        0.108768  0.108768
        iv1        0.147716  0.147716
        iv2        0.057318  0.057318
        iv3        0.000570  0.000570
        >>> four_by_two.residuals_post_hoc().p_corr
        Outcome         dv0       dv1
        Predictor
        iv0        0.217537  0.217537
        iv1        0.217537  0.217537
        iv2        0.171955  0.171955
        iv3        0.002279  0.002279
        """
        observed = self.matrix
        expected = self.exp_freq
        # residuals = observed - expected
        # std_residuals = (observed - expected)/np.sqrt(expected)
        grand_sum = observed.sum().sum()

        adj_std_residuals = (
            (observed - expected)
            / np.sqrt(
                expected
                .mul(
                    1 - observed.sum(axis='columns').div(grand_sum),
                    axis='rows'
                )
                .mul(
                    1 - observed.sum(axis='rows').div(grand_sum),
                    axis='columns'
                )
            )
        )

        # Z-score -> p-value
        p_vals = adj_std_residuals.abs().map(stats.norm.sf).mul(2)
        rows, cols = p_vals.shape

        # No correction needed if both axes are binary
        if rows == 2 and cols == 2:
            return self._ResidualResult(adj_std_residuals, p_vals, p_vals)

        if rows > 2 and cols > 2:
            p_to_correct = p_vals.to_numpy().ravel()
        elif cols == 2:
            p_to_correct = p_vals.iloc[:, -1].to_numpy()
        elif rows == 2:
            p_to_correct = p_vals.iloc[-1, :].to_numpy()
        else:
            raise ValueError('Unsupported shape: must be >2×>2, k×2, or 2×k')

        _, p_corr_vec, _, _ = multipletests(
            p_to_correct,
            alpha=alpha,
            method=p_corr_method
        )

        if rows > 2 and cols > 2:
            p_corr_array = p_corr_vec.reshape((rows, cols))
        elif cols == 2:
            p_corr_array = np.column_stack([p_corr_vec, p_corr_vec])
        else:  # rows == 2
            p_corr_array = np.vstack([p_corr_vec, p_corr_vec])

        p_corr = pd.DataFrame(
            p_corr_array,
            index=p_vals.index,
            columns=p_vals.columns
        )

        return self._ResidualResult(adj_std_residuals, p_vals, p_corr)

    def post_hoc(self,
                 alpha: float = 0.05,
                 p_corr_method: PCorrectionMethod = 'holm'):
        r"""Perform appropriate post hoc test, based on IV/DV levels.

        Convenience method to choose type of post hoc test.

        If either rows or columns are binary, results of pairwise
        :math:`\chi^2` or Fisher exact tests (as appropriate for expected
        frequencies) are returned with uncorrected & corrected *p*-values.

        If rows & columns both have 3+ levels, adjusted standardized residuals
        (a/k/a adjusted Pearson residuals; cell-wise Z-scores) are converted to
        *p*-values and corrected.

        By default, Holm-Bonferroni correction is used, but all
        correction methods supported by
        `statsmodels.stats.multitest.multipletests()
        <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_
        are supported.

        Parameters
        ----------
        alpha : float, default .05
            Significance level to be used for p-value correction.
        p_corr_method : str, default 'holm'
            *p*-Value correction method.

        Returns
        -------
        pd.DataFrame
            Pairwise results condensed to a single DataFrame. Only returned
            when a pairwise test is performed.
        residuals : pd.DataFrame
            Adjusted standardized residuals in same format as `self.matrix`.
            Shows direction of effect in each cell.
        p_values : pd.DataFrame
            Uncorrected *p*-values in same format as `self.matrix`.
        p_corr : pd.DataFrame
            Corrected *p*-values in same format as `self.matrix`.

        See Also
        --------
        pairwise_post_hoc : When either rows or columns are binary.
        residuals_post_hoc : Preferred when rows & columns have 3+ levels.
        """
        if len(self.matrix.index) == 2 or len(self.matrix.columns) == 2:
            return self.pairwise_post_hoc(alpha, p_corr_method)
        else:
            return self.residuals_post_hoc(alpha, p_corr_method)

    def print_results(self):
        r"""Print contingency tables and Chi-squared results."""
        warnings.warn(
            'As of unistat v0.3.0, `.print_results()` is deprecated.\n'
            'The preferred syntax for console output is now via `print()`. \n'
            'E.g., `my_contingency.print_results()` -> '
            '`print(my_contingency)`.',
            FutureWarning
        )

        chi2 = self.chi2()

        print(f'{self.row_title} vs. {self.col_title}\n', '='*40)
        print(f'Table (# Obs):\n'
              f'{self.get_table()}')
        print(f'-'*40,
              f'\nTable (% of Totals):\n'
              f'{self.get_table(as_pct=True)}')
        print('-'*40,
              f'\nChi^2 ToI:\n'
              f'X^2({chi2.dof}) = {chi2.statistic:.3f}, '
              f'p = {chi2.pvalue:.4f}')
        print('-'*40, '\n')

        if chi2.pvalue <= 0.05:
            if (len(self.matrix.index) > 2) and (len(self.matrix.columns) > 2):
                for table in self.residuals_post_hoc():
                    print(table.to_string())
            else:
                print(self.pairwise_post_hoc().to_string())


class BooleanContingencyStats(_ContingencyStats):
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
    ExpectedFrequencyWarning
        If any cell-wise expected frequencies < 5

    See Also
    --------
    MulticlassContingencyStats

    Notes
    -----
    Assumes the contingency table is 2x2.

    p-values are displayed for both the chi-square test of independence (ToI),
    and for an exact test like Fisher's. Deciding which test to report can
    follow Cochran's rule-of-thumb criteria :cite:`cochran_1952`
    :cite:`cochran_1954`, which includes (but is not limited to) the following
    as indication for use of an exact test (Fisher's exact test in the original
    1952 article) over chi-squared:

    * Any cell-wise expected frequency < 5
        * Actual rule is <20% must have expected frequency < 5, which means no
          cells can have low expected frequency in a 2x2 table.
    * N < 20
    * Cochran (1952) :cite:`cochran_1952` recommends using Yates' correction if
      N > 40 but any expected frequency < 500; ``unistat`` does not implement
      this by default.

    By default, ``unistat`` *never* implements Yates' correction factor.
    Hasselblad & Lokhnygina (2007) :cite:`hasselblad_2007` found that in **all**
    cases, Yates-corrected chi-squared is inferior to Fisher's exact test.
    Furthermore, they found that even Fisher's exact test is too conservative,
    and that, depending on sample size, Fisher's mid-p test or Barnard's exact
    test offer better power while maintaining target Type I error control.

    Alternative exact test(s) will be implemented in later releases; expect
    that at a minimum, this will include Boschloo's exact test.

    Lydersen et al. (2009) :cite:`lydersen_2009` compared multiple different
    exact tests, and noted the following:

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
          mortality) :cite:`lydersen_2009` :cite:`kang_2008`
        * Standard Berger-Boos correction factor is :math:`\gamma = 0.001`
          :cite:`lydersen_2009`

            * Not implemented by SciPy, though included in R ``Exact`` package
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

    def __str__(self):
        with (
            pd.option_context(
                'display.max_columns', None,
                'display.width', 256,
                'display.max_colwidth', None,
                'display.expand_frame_repr', False
            )
        ):
            lines = []
            width = 60  # minimum width; expand PRN

            title = f'{self.row_title} vs. {self.col_title}'
            width = max(width, len(title) + 4)

            lines.append('=' * width)
            lines.append(title.center(width))
            lines.append('=' * width)
            lines.append('')

            # Contingency table (counts)
            table_str = str(self.get_table())
            table_lines = table_str.splitlines()
            table_width = (
                max(
                    len(line) for line in table_lines
                )
                if table_lines else 40
            )
            width = max(width, table_width + 4)

            lines.append('Contingency Table (counts):')
            lines.append('-' * width)
            lines.extend(table_lines)
            lines.append('')

            # Percentages table
            pct_str = str(self.get_table(as_pct=True))
            pct_lines = pct_str.splitlines()
            pct_width = (
                max(
                    len(line) for line in pct_lines
                )
                if pct_lines else 40
            )
            width = max(width, pct_width + 4)

            lines.append('Table (% of totals):')
            lines.append('-' * width)
            lines.extend(pct_lines)
            lines.append('')

            # Odds Ratio
            or_res = self.odds_ratio()
            ci_low = or_res.confidence_interval().low
            ci_high = or_res.confidence_interval().high

            lines.append('Odds Ratio:')
            lines.append('-' * width)
            lines.append(f'OR = {or_res.statistic:.4f}, '
                         f'95% CI = {ci_low:.4f} to {ci_high:.4f}')
            lines.append('')

            # Chi-square
            chi2 = self.chi2()
            lines.append('Chi-square Test of Independence:')
            lines.append('-' * width)
            lines.append(f'χ²({chi2.dof}) = {chi2.statistic:.4f}, '
                         f'p = {chi2.pvalue:.5f}')
            lines.append('')

            # Boschloo exact
            bosch = self.boschloo_exact()
            lines.append('Boschloo Exact Test:')
            lines.append('-' * width)
            lines.append(f'p = {bosch.pvalue:.5f}')
            lines.append('')

            lines.append('=' * width)

            return '\n'.join(lines)

    def odds_ratio(self, kind: str = 'sample'):
        r"""Compute the odds ratio.

        Parameters
        ----------
        kind : str, optional
            Type of odds ratio: 'sample' (default), 'conditional', or
            'unconditional'.

        Returns
        -------
        scipy.stats._result_classes.OddsRatioResult
            Result object with statistic and confidence interval.
        """
        odds_ratio = stats.contingency.odds_ratio(self.matrix, kind=kind)
        return odds_ratio

    def fisher_exact(
            self,
            alternative: Literal['two-sided', 'less', 'greater']='two-sided'):
        r"""Perform Fisher's exact test.

        Parameters
        ----------
        alternative : str, default 'two-sided'
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.

        Returns
        -------
        float
            p-value of the test.
        """
        p_val = (
            stats.fisher_exact(self.matrix, alternative=alternative).pvalue
        )
        return p_val

    def boschloo_exact(
            self,
            alternative: Literal['two-sided', 'less', 'greater']='two-sided',
            n_sampling_points: int = 32):
        r"""Perform Boschloo's exact test.

        Parameters
        ----------
        alternative : str, default: 'two-sided'
            Alternative hypothesis: 'two-sided', 'less', or 'greater'
        n_sampling_points : int, default 32
            Number of sampling points used in the construction of the sampling
            method. See `scipy.stats.boschloo_exact() <bosch-exact>`_
            documentation for further detail.

        .. _bosch-exact: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html

        Returns
        -------
        statistic : float
            Test statistic for Boschloo's exact test, which is the lesser of the
            p-values given by 2 one-sided Fisher exact test.
        pvalue : float
            Boschloo's exact p-value.

        Notes
        -----
        Lydersen et al. (2009) :cite:`lydersen_2009` compared multiple different exact
        tests, and found Boschloo's exact test to be universally more powerful
        than both traditional and mid-p Fisher's exact tests. Boschloo's exact
        test can be further improved using the Berger-Boos correction,
        particularly for unbalanced designs (e.g. if survival occurs much more
        often than mortality) :cite:`lydersen_2009` :cite:`kang_2008`

        * Standard Berger-Boos correction factor is :math:`\gamma = 0.001`
          :cite:`lydersen_2009`

            * Not implemented by `SciPy <scipy-homepage_>`_, though included in
              R ``Exact`` package
            * May be implemented here in future update
        """
        return stats.boschloo_exact(self.matrix,
                                    alternative=alternative,
                                    n=n_sampling_points)

    def print_results(self):
        r"""Print tables, odds ratio, Chi-squared, and Boschloo exact results.

        Overrides the parent method to include 2x2-specific statistics.
        """
        warnings.warn(
            'As of unistat v0.3.0, `.print_results()` is deprecated.\n'
            'The preferred syntax for console output is now via `print()`. \n'
            'E.g., `my_contingency.print_results()` -> '
            '`print(my_contingency)`.',
            FutureWarning
        )

        chi2 = self.chi2()

        print(f'{self.row_title} vs. {self.col_title}\n', '='*40)
        print(f'Table (# Obs):\n'
              f'{self.get_table()}')
        print(f'-'*40,
              f'\nTable (% of Totals):\n'
              f'{self.get_table(as_pct=True)}')
        print('-'*40,
              f'\nOdds Ratio:\n'
              f'OR = {self.odds_ratio().statistic:.4f}, '
              f'95% CI {self.odds_ratio().confidence_interval().low:.4f} to '
              f'{self.odds_ratio().confidence_interval().high:.4f}')
        print('-'*40,
              f'\nChi^2 ToI:\n'
              f'X^2({chi2.dof}) = {chi2.statistic:.4f}, '
              f'p = {chi2.pvalue:.5f}')
        print('-'*40,
              f'\nBoschloo Exact Test:\n'
              f'p = {self.boschloo_exact().pvalue:.5f}')
