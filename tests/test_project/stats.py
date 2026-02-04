
################################################################################
# Setup
################################################################################

# Imports
import pandas as pd
from tests.test_project.dataprep import prep_data
from unistat import (
    MulticlassContingencyStats,
    LogitStats,
)


# Module Options ===============================================================

pd.set_option('future.no_silent_downcasting', True)
pd.set_option('mode.copy_on_write', 'warn')


# Data Formatting ==============================================================

outcome_df = prep_data()


################################################################################
# Run Stats
################################################################################

# All-form Ca dosing ===========================================================

# region Numeric predictors
def ratio_mort_logit():
    return LogitStats(
        X=outcome_df['ca_grams_per_unit_4h'].astype(float),
        y=outcome_df['mortality_24h']
    )
# endregion

# region Categorical predictors
def ratiobin_mort_contingency():
    return MulticlassContingencyStats(
        table_rows=outcome_df['mutex_ca_grams_per_unit_4h_binned'],
        table_cols=outcome_df['mortality_24h'],
        row_title='Ca dose (gluconate or CaCl2)',
        col_title='24h mortality'
    )
# endregion

# region Multivariable comparisons
def ratio_mort_mvlogit():
    return LogitStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',
            'age',
            'iss',
            'moi_pen',
            'sex_female']],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )
# endregion

def print_ratio_stats():
    from tests.test_project.posthoc import chisq_and_posthoc_corrected
    # Numeric
    print(ratio_mort_logit())

    # Categorical
    ratiobin_mort_contingency().print_results()
    print(chisq_and_posthoc_corrected(
        ratiobin_mort_contingency().matrix(),
        correction_method='holm'
    ))

    # Multivariable
    print(ratio_mort_mvlogit())


# Elemental Ca dosing ==========================================================

# region Numeric predictors
def elemental_mort_logit():
    return LogitStats(
        X=outcome_df['elemental_ca_grams_per_unit_4h'].astype(float),
        y=outcome_df['mortality_24h']
    )
# endregion

# region Categorical predictors
def elementalbin_mort_contingency():
    return MulticlassContingencyStats(
        table_rows=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        table_cols=outcome_df['mortality_24h'],
        row_title='Elemental Ca dose',
        col_title='24h mortality'
    )
# endregion

# region Multivariable comparisons
def elemental_mort_mvlogit():
    return LogitStats(
        X=outcome_df[[
            'elemental_ca_grams_per_unit_4h',
            'age',
            'iss',
            'moi_pen',
            'sex_female']],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )
# endregion

def print_elemental_stats():
    # Numeric
    print(elemental_mort_logit())

    # Categorical
    elementalbin_mort_contingency().print_results()

    # Multivariable
    print(elemental_mort_mvlogit())


# CaCl2 dosing =================================================================

# region Numeric predictors
def cacl2ratio_mort_logit():
    df = outcome_df[~outcome_df['ca_gluconate_bool']]

    return LogitStats(
        X=df['ca_grams_per_unit_4h'].astype(float),
        y=df['mortality_24h']
    )
# endregion

# region Categorical predictors
def cacl2ratiobin_mort_contingency():
    df = outcome_df[~outcome_df['ca_gluconate_bool']]

    return MulticlassContingencyStats(
        table_rows=df['mutex_ca_grams_per_unit_4h_binned'],
        table_cols=df['mortality_24h'],
        row_title='CaCl2 dose',
        col_title='24h mortality'
    )
# endregion

# region Multivariable comparisons
def cacl2ratio_mort_mvlogit():
    df = outcome_df[~outcome_df['ca_gluconate_bool']]

    return LogitStats(
        X=df[[
            'ca_grams_per_unit_4h',
            'age',
            'iss',
            'moi_pen',
            'sex_female']],
        y=df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )
# endregion

def print_cacl2_stats():
    # Numeric
    print(cacl2ratio_mort_logit())

    # Categorical
    cacl2ratiobin_mort_contingency().print_results()

    # Multivariable
    print(cacl2ratio_mort_mvlogit())


# Ca-gluconate dosing ==========================================================

# region Numeric predictors
def gluconateratio_mort_logit():
    df = outcome_df[outcome_df['ca_gluconate_bool']]

    return LogitStats(
        X=df['ca_grams_per_unit_4h'].astype(float),
        y=df['mortality_24h']
    )
# endregion

# region Categorical predictors
def gluconateratiobin_mort_contingency():
    df = outcome_df[outcome_df['ca_gluconate_bool']]

    return MulticlassContingencyStats(
        table_rows=df['mutex_ca_grams_per_unit_4h_binned'],
        table_cols=df['mortality_24h'],
        row_title='Ca-gluconate dose',
        col_title='24h mortality'
    )
# endregion

# region Multivariable comparisons
def gluconateratio_mort_mvlogit():
    df = outcome_df[outcome_df['ca_gluconate_bool']]

    return LogitStats(
        X=df[[
            'ca_grams_per_unit_4h',
            'age',
            'iss',
            'moi_pen',
            'sex_female']],
        y=df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )
# endregion

def print_gluconate_stats():
    # Numeric
    print(gluconateratio_mort_logit())

    # Categorical
    gluconateratiobin_mort_contingency().print_results()

    # Multivariable
    print(gluconateratio_mort_mvlogit())


################################################################################
# Main
################################################################################

def main():
    print_ratio_stats()
    print_elemental_stats()
    print_cacl2_stats()
    print_gluconate_stats()


if __name__ == '__main__':
    main()
