
################################################################################
# Setup
################################################################################

# Import standard packages
import pathlib
# Import 3rd party packages
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
# Import local modules
from tests.test_project.dataprep import prep_data
from unistat.formula_regression import FormulaLogit, FormulaLinReg


# Data Formatting ==============================================================

outcome_df = prep_data()


################################################################################
# Models
################################################################################

# All-form Ca dosing ===========================================================

def ratio_mort_mvlogit_linear():
    from tests.test_project.stats import ratio_mort_mvlogit
    return ratio_mort_mvlogit().reg


def ratio_mort_mvlogit_quadratic():
    df = outcome_df[['mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
                     'moi_pen', 'sex_female']].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('mortality_24h ~ I(ca_grams_per_unit_4h ** 2)'
                 '+ age + iss + moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def ratio_mort_mvlogit_logarithmic():
    df = outcome_df[['mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
                     'moi_pen', 'sex_female']].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('mortality_24h ~ np.log(ca_grams_per_unit_4h)'
                 '+ age + iss + moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


# Regressions adjusted for dose form ===========================================

def ratioform_mort_mvlogit():
    df = outcome_df[[
        'ca_grams_4h', 'ca_gluconate_bool', 'age', 'iss', 'moi_pen',
        'sex_female', 'mortality_24h'
    ]].dropna().astype(float)

    model = sm.Logit.from_formula(
        formula=(
            'mortality_24h ~ ca_grams_4h*ca_gluconate_bool + age + iss'
            '+ moi_pen + sex_female'
        ),
        data=df,
    ).fit()

    return model


################################################################################
# Tests of FormulaRegression
################################################################################

def sm_test_reg(formula: str, data: pd.DataFrame):
    model = sm.Logit.from_formula(formula=formula, data=data).fit()
    return model


formula = (
    "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2) "
    "+ iss + C(moi_pen, Treatment(False)) + C(sex_female)"
    "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
)
logit = FormulaLogit(
    formula=formula,
    data=outcome_df
)
print(logit)

linreg = FormulaLinReg(
    formula='los ~' + formula.split('~')[1],
    data=outcome_df
)
print(linreg)


################################################################################
# Main
################################################################################

def main():
    pass


if __name__ == '__main__':
    main()