
################################################################################
# Setup
################################################################################

# Import standard packages
# Import 3rd party packages
import statsmodels.api as sm
# Import local modules
from tests.test_project.dataprep import prep_data
from unistat import FormulaLinReg, FormulaLogit


# Data Formatting ==============================================================

outcome_df = prep_data()


################################################################################
# Hard-coded statsmodels Regression
################################################################################

# Logit ========================================================================

def statsmodels_mvlogit_linear():
    from tests.test_project.stats import ratio_mort_mvlogit
    return ratio_mort_mvlogit().reg


def statsmodels_mvlogit_quadratic():
    df = outcome_df[[
        'mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('mortality_24h ~ I(ca_grams_per_unit_4h ** 2) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvlogit_logarithmic():
    df = outcome_df[[
        'mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('mortality_24h ~ np.log(ca_grams_per_unit_4h) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvlogit_linear_interaction():
    df = outcome_df[[
        'ca_grams_4h', 'ca_gluconate_bool', 'age', 'iss', 'moi_pen',
        'sex_female', 'mortality_24h'
    ]].dropna().astype(float)

    model = sm.Logit.from_formula(
        formula=('mortality_24h ~ ca_grams_4h*ca_gluconate_bool + age + iss'
                 '+ moi_pen + sex_female'
        ),
        data=df,
    ).fit()

    return model


# Probit =======================================================================

def statsmodels_mvprobit_linear():
    df = outcome_df[[
        'mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.Probit.from_formula(
        formula=('mortality_24h ~ ca_grams_per_unit_4h + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvprobit_quadratic():
    df = outcome_df[[
        'mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.Probit.from_formula(
        formula=('mortality_24h ~ I(ca_grams_per_unit_4h ** 2) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvprobit_logarithmic():
    df = outcome_df[[
        'mortality_24h', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.Probit.from_formula(
        formula=('mortality_24h ~ np.log(ca_grams_per_unit_4h) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvprobit_linear_interaction():
    df = outcome_df[[
        'ca_grams_4h', 'ca_gluconate_bool', 'age', 'iss', 'moi_pen',
        'sex_female', 'mortality_24h'
    ]].dropna().astype(float)

    model = sm.Probit.from_formula(
        formula=('mortality_24h ~ ca_grams_4h*ca_gluconate_bool + age + iss'
                 '+ moi_pen + sex_female'
        ),
        data=df,
    ).fit()

    return model


# LinReg =======================================================================

def statsmodels_mvlinreg_linear():
    df = outcome_df[[
        'los', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('los ~ ca_grams_per_unit_4h + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvlinreg_quadratic():
    df = outcome_df[[
        'los', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('los ~ I(ca_grams_per_unit_4h ** 2) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvlinreg_logarithmic():
    df = outcome_df[[
        'los', 'ca_grams_per_unit_4h', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('los ~ np.log(ca_grams_per_unit_4h) + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


def statsmodels_mvlinreg_interaction():
    df = outcome_df[[
        'los', 'ca_grams_4h', 'ca_gluconate_bool', 'age', 'iss',
        'moi_pen', 'sex_female',
    ]].dropna().astype(float)

    model = sm.OLS.from_formula(
        formula=('los ~ ca_grams_4h*ca_gluconate_bool + age + iss'
                 '+ moi_pen + sex_female'),
        data=df,
    ).fit()

    return model


# Print Results ================================================================

def print_statsmodels_tests() -> None:
    sm_regressions = (
        statsmodels_mvlogit_linear(),
        statsmodels_mvlogit_quadratic(),
        statsmodels_mvlogit_logarithmic(),
        statsmodels_mvlogit_linear_interaction(),
        statsmodels_mvprobit_linear(),
        statsmodels_mvprobit_quadratic(),
        statsmodels_mvprobit_logarithmic(),
        statsmodels_mvprobit_linear_interaction(),
        statsmodels_mvlinreg_linear(),
        statsmodels_mvlinreg_quadratic(),
        statsmodels_mvlinreg_logarithmic(),
        statsmodels_mvlinreg_interaction(),
    )

    for reg in sm_regressions:
        print(reg.summary2())


################################################################################
# Tests of FormulaRegression
################################################################################

# FormulaLogit =================================================================

def nums_mvlogit() -> FormulaLogit:
    formula = ("mortality_24h ~ ca_grams_per_unit_4h + I(age**2)"
               "+ np.sqrt(iss)")

    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def cats_mvlogit() -> FormulaLogit:
    # Test if implicit boolean detection works
    df = outcome_df.copy()
    df['ca_gluconate_bool'] = df['ca_gluconate_bool'].astype(float)

    formula = (
        "mortality_24h ~ ca_gluconate_bool + C(sex_female)"
        "+ C(moi_pen, Treatment(False))"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
    )

    return FormulaLogit(
        formula=formula,
        data=df,
    )


def interaction_mvlogit() -> FormulaLogit:
    formula = (
        "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2) "
        "+ np.sqrt(iss) + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
    )
    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def cluster_random_effects_mvlogit() -> FormulaLogit:
    r"""Random intercepts for a single clustering factor, e.g. study site."""
    formula = (
        "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2)"
        "+ iss + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
        "+ (1 | C(fake_cluster_factor))"
    )
    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def crossed_random_effects_mvlogit() -> FormulaLogit:
    r"""Random intercepts for 2 crossed random effects.

    Random intercepts for each factor are completely independent and additive.
    Used for 2 REs where any combination of RE factors is possible.

    E.g. Each Pt (subject ID) may appear in multiple hospitals (site-level
    factor), and hospitals (site-level factor) can treat same Pt (subject ID)
    multiple times.

    E.g. 15 private practice surgeons share call across 5 different
    hospitals. If you wanted to analyze effects on operative time while
    accounting for differences between hospitals and between surgeons,
    categories for hospital and surgeon assigned to each patient would constitute
    a 2 completely crossed random effects, because a patient may have any of
    5 different hospital factors, and any of 15 different surgeon factors.
    Coded as ``+ (1 | hospital_id) + (1 | surgeon_id)``
    """
    formula = (
        "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2)"
        "+ iss + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
        "+ (1 | fake_cluster_factor) + (1 | fake_id_factor)"
    )
    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def nested_random_effects_mvlogit_0() -> FormulaLogit:
    r"""Random intercepts for 2 nested random effects.

    Used for 2 REs where variation of 1 RE factor (often individual-level)
    occurs within 1 and only 1 level of the other RE factor (often a site-level
    factor).

    E.g., if you wanted to account for both variation between hospitals (site-
    level factor) and between Pts *within each hospital* (individual-
    level), but each Pt is seen at *one and only one* hospital.

    E.g. 5 different hospitals each employ 6 surgeons who only operate at their
    respective hospitals. If you wanted to analyze effects on operative time
    while accounting for differences between hospitals and between surgeons,
    the RE for surgeon is completely nested within the RE for hospital, because
    each surgeon works at one and only one hospital. As a result, you cannot
    compare inter-surgeon variation independent of hospital-level effects; you
    can only compare how a given surgeon differs from the other 5 surgeons who
    operate at that hospital.
    This could be coded:
    * ``+ (1 | hospital_id/surgeon_id)``
    * or equivalently: ``+ (1 | hospital_id) + (1 | hospital_id:surgeon_id)``

    Random intercepts are still additive, but calculated as follows:
    * Random intercept for each ``hospital_id`` is calculated
    * Random intercept for how each ``surgeon_id`` differs from their respective
      ``hospital_id`` is calculated
    * The ``hospital_id`` intercept and ``surgeon_id`` offset are added to
      obtain a final intercept for each ``hospital_id:surgeon_id`` combination
    """
    formula = (
        "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2)"
        "+ iss + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
        "+ (1 | fake_cluster_factor/fake_id_factor)"
    )

    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def nested_random_effects_mvlogit_1() -> FormulaLogit:
    r"""Random intercepts for 2 nested random effects.

    Used for 2 REs where variation of 1 RE factor (often individual-level)
    occurs within 1 and only 1 level of the other RE factor (often a site-level
    factor).

    E.g., if you wanted to account for both variation between hospitals (site-
    level factor) and between Pts *within each hospital* (individual-
    level), but each Pt is seen at *one and only one* hospital.

    E.g. 5 different hospitals each employ 6 surgeons who only operate at their
    respective hospitals. If you wanted to analyze effects on operative time
    while accounting for differences between hospitals and between surgeons,
    the RE for surgeon is completely nested within the RE for hospital, because
    each surgeon works at one and only one hospital. As a result, you cannot
    compare inter-surgeon variation independent of hospital-level effects; you
    can only compare how a given surgeon differs from the other 5 surgeons who
    operate at that hospital.
    This could be coded:
    * ``+ (1 | hospital_id/surgeon_id)``
    * or equivalently: ``+ (1 | hospital_id) + (1 | hospital_id:surgeon_id)``

    Random intercepts are still additive, but calculated as follows:
    * Random intercept for each ``hospital_id`` is calculated
    * Random intercept for how each ``surgeon_id`` differs from their respective
      ``hospital_id`` is calculated
    * The ``hospital_id`` intercept and ``surgeon_id`` offset are added to
      obtain a final intercept for each ``hospital_id:surgeon_id`` combination
    """
    formula = (
        "mortality_24h ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2)"
        "+ iss + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
        "+ (1 | fake_cluster_factor) + (1 | fake_cluster_factor:fake_id_factor)"
    )

    return FormulaLogit(
        formula=formula,
        data=outcome_df
    )


def print_formula_logit_tests() -> None:
    logits = (
        nums_mvlogit(),
        cats_mvlogit(),
        interaction_mvlogit(),
        # cluster_random_effects_mvlogit(),
        # crossed_random_effects_mvlogit(),
        # nested_random_effects_mvlogit_0(),
        # nested_random_effects_mvlogit_1(),
    )

    for logit in logits:
        print(logit)


# FormulaLinReg ================================================================

def nums_mvlinreg() -> FormulaLinReg:
    formula = "los ~ ca_grams_per_unit_4h + I(age**2) + np.sqrt(iss)"

    return FormulaLinReg(
        formula=formula,
        data=outcome_df
    )


def cats_mvlinreg() -> FormulaLinReg:
    # Test if implicit boolean detection works
    df = outcome_df.copy()
    df['ca_gluconate_bool'] = df['ca_gluconate_bool'].astype(float)

    formula = (
        "los ~ ca_gluconate_bool + C(sex_female)"
        "+ C(moi_pen, Treatment(False))"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
    )

    return FormulaLinReg(
        formula=formula,
        data=df,
    )


def interaction_mvlinreg() -> FormulaLinReg:
    formula = (
        "los ~ ca_grams_4h*C(ca_gluconate_bool) + I(age**2) "
        "+ np.sqrt(iss) + C(moi_pen, Treatment(False)) + C(sex_female)"
        "+ C(mutex_ca_grams_per_unit_4h_binned, Treatment('< 0.25 g/U'))"
    )
    return FormulaLinReg(
        formula=formula,
        data=outcome_df
    )


def print_formula_lingreg_tests() -> None:
    linregs = (
        nums_mvlinreg(),
        cats_mvlinreg(),
        interaction_mvlinreg(),
    )

    for linreg in linregs:
        print(linreg)


################################################################################
# Main
################################################################################

def main():
    # print_statsmodels_tests()
    print_formula_logit_tests()
    print_formula_lingreg_tests()


if __name__ == '__main__':
    main()