
################################################################################
# Setup
################################################################################

# Import standard packages
# Import 3rd party packages
import pandas as pd
# Import local modules
from tests.test_project.dataprep import get_outcome_df
from unistat import (
    CorrStats,
    TwoSeriesStats, TwoSampleStats,
    MultiSeries1WayBGStats, MultiSample1WayBGStats,
)


# Data Formatting ==============================================================

outcome_df = get_outcome_df()


################################################################################
# CorrStats Tests
################################################################################

def pearson_int_vs_float() -> CorrStats:
    return CorrStats(
        x=outcome_df['elemental_ca_grams_per_unit_4h'],
        y=outcome_df['los'],
        parametric=True
    )


def spearman_int_vs_float() -> CorrStats:
    return CorrStats(
        x=outcome_df['elemental_ca_grams_per_unit_4h'],
        y=outcome_df['los'],
        parametric=False
    )


def print_corr_tests() -> None:
    print(pearson_int_vs_float())
    print(spearman_int_vs_float())


################################################################################
# 2-Sample Tests
################################################################################

# TwoSeriesStats ===============================================================

def t_int_vs_int() -> TwoSeriesStats:
    mask = (outcome_df['elemental_ca_grams_per_unit_4h']
            > outcome_df['elemental_ca_grams_per_unit_4h'].median())
    test = outcome_df.loc[mask, 'los']
    control = outcome_df.loc[~mask, 'los']

    return TwoSeriesStats(
        test=test,
        control=control,
        parametric=True
    )


def mwu_int_vs_int() -> TwoSeriesStats:
    mask = (outcome_df['elemental_ca_grams_per_unit_4h']
            > outcome_df['elemental_ca_grams_per_unit_4h'].median())
    test = outcome_df.loc[mask, 'los']
    control = outcome_df.loc[~mask, 'los']

    return TwoSeriesStats(
        test=test,
        control=control,
        parametric=False
    )


# TwoSampleStats ===============================================================

def t_int_vs_bool() -> TwoSampleStats:
    mask = (outcome_df['elemental_ca_grams_per_unit_4h']
            > outcome_df['elemental_ca_grams_per_unit_4h'].median())

    return TwoSampleStats(
        bool_x=mask,
        num_y=outcome_df['los'],
        parametric=True
    )


def mwu_int_vs_bool() -> TwoSampleStats:
    mask = (outcome_df['elemental_ca_grams_per_unit_4h']
            > outcome_df['elemental_ca_grams_per_unit_4h'].median())

    return TwoSampleStats(
        bool_x=mask,
        num_y=outcome_df['los'],
        parametric=False
    )


# Results ======================================================================

def print_2sample_tests() -> None:
    print(t_int_vs_int())
    print(mwu_int_vs_int())
    print(t_int_vs_bool())
    print(mwu_int_vs_bool())


################################################################################
# Multilevel Tests
################################################################################

# MultiSeries1WayBGStats =======================================================

# +-------+
# | ANOVA |
# +-------+

def anova_ints_from_dataframe_arg() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        df,
        parametric=True
    )


def anova_ints_from_dataframe_kwarg() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        data=df,
        parametric=True
    )


def anova_ints_from_args() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        *tuple(df[col] for col in df.columns),
        parametric=True
    )


def anova_ints_from_kwargs() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        **{f'test{enum}': df[col] for enum, col in enumerate(df.columns)},
        parametric=True,
    )


# +----------------+
# | Kruskal-Wallis |
# +----------------+

def kw_ints_from_dataframe_arg() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        df,
        parametric=False
    )


def kw_ints_from_dataframe_kwarg() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        data=df,
        parametric=False
    )


def kw_ints_from_args() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        *tuple(df[col] for col in df.columns),
        parametric=False
    )


def kw_ints_from_kwargs() -> MultiSeries1WayBGStats:
    levels = ['< 23.25 mg/U', '23.25 - 31 mg/U',
              '31 - 46.5 mg/U', '>= 46.5 mg/U']
    seriess: list = []

    for level in levels:
        series = outcome_df.loc[
            outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'] == level,
            'los'
        ]
        series.name = series.name + f'__{level}'
        seriess.append(series)

    df = pd.concat(objs=seriess, axis='columns')

    return MultiSeries1WayBGStats(
        **{f'test{enum}': df[col] for enum, col in enumerate(df.columns)},
        parametric=False,
    )


# MultiSample1WayBGStats =======================================================

# +-------+
# | ANOVA |
# +-------+

def anova_ints_vs_categories() -> MultiSample1WayBGStats:
    return MultiSample1WayBGStats(
        cat_x=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        num_y=outcome_df['los'],
        cat_order=[
            '< 23.25 mg/U', '23.25 - 31 mg/U', '31 - 46.5 mg/U', '>= 46.5 mg/U',
        ],
        parametric=True
    )


def anova_ints_vs_ordered_categories() -> MultiSample1WayBGStats:
    return MultiSample1WayBGStats(
        cat_x=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        num_y=outcome_df['los'],
        parametric=True
    )


# +----------------+
# | Kruskal-Wallis |
# +----------------+

def kw_ints_vs_categories() -> MultiSample1WayBGStats:
    return MultiSample1WayBGStats(
        cat_x=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        num_y=outcome_df['los'],
        parametric=False
    )


def kw_ints_vs_ordered_categories() -> MultiSample1WayBGStats:
    return MultiSample1WayBGStats(
        cat_x=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        num_y=outcome_df['los'],
        cat_order=[
            '< 23.25 mg/U', '23.25 - 31 mg/U', '31 - 46.5 mg/U', '>= 46.5 mg/U',
        ],
        parametric=False
    )


def print_multilevel_stats() -> None:
    # MultiSeries1WayBGStats
    print(anova_ints_from_dataframe_arg())
    print(anova_ints_from_dataframe_kwarg())
    print(anova_ints_from_args())
    print(anova_ints_from_kwargs())
    print(kw_ints_from_dataframe_arg())
    print(kw_ints_from_dataframe_kwarg())
    print(kw_ints_from_args())
    print(kw_ints_from_kwargs())

    # MultiSample1WayBGStats
    print(anova_ints_vs_categories())
    print(anova_ints_vs_ordered_categories())
    print(kw_ints_vs_categories())
    print(kw_ints_vs_ordered_categories())


################################################################################
# Main
################################################################################

def main():
    print_corr_tests()
    print_2sample_tests()
    print_multilevel_stats()


if __name__ == '__main__':
    main()
