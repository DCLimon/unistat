
import warnings
from typing import Optional
import pandas as pd
from scipy import stats


class MulticlassContingencyStats:
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
              axis: str | int = 'rows') -> pd.DataFrame:
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
        return self.table().drop(index='col_totals', columns='row_totals')

    def chi2(self, correction: bool = False):
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

        print(f'{self.row_title} vs. {self.col_title}\n', '='*40)
        print(f'Table (# Obs):\n'
              f'{self.table()}')
        print(f'-'*40,
              f'\nTable (% of Totals):\n'
              f'{self.table(as_pct=True)}')
        print('-'*40,
              f'\nChi^2 ToI:\n'
              f'X^2({self.chi2().dof}) = {self.chi2().statistic:.3f}, '
              f'p = {self.chi2().pvalue:.4f}')
        print('-'*40, '\n')


class BooleanContingencyStats(MulticlassContingencyStats):
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
        odds_ratio = stats.contingency.odds_ratio(self.matrix(), kind=kind)
        return odds_ratio

    def fisher_exact(self, alternative='two-sided'):
        p_val = (
            stats.fisher_exact(self.matrix(), alternative=alternative).pvalue
        )
        return p_val

    def print_results(self):
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
              f'X^2({self.chi2().dof}) = {self.chi2().statistic:.3f}, '
              f'p = {self.chi2().pvalue:.4f}')
        print('-'*40,
              f'\nFisher Exact Test:\n'
              f'p = {self.fisher_exact():.4f}')