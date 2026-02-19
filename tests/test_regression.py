
################################################################################
# Setup
################################################################################

# Import standard packages
# Import 3rd party packages
# Import local modules
from tests.test_project.dataprep import get_outcome_df
from unistat import LogitStats, LinRegStats, LogBinStats


# Data Formatting ==============================================================

outcome_df = get_outcome_df()
outcome_df['sex_female'] = outcome_df['sex_female'].astype(float)


################################################################################
# LinReg
################################################################################

# Mixed Types ==================================================================

def linreg_1float_2int_2bool() -> LinRegStats:
    return LinRegStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['los'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# All Numeric Case =============================================================

def linreg_1float_2_int() -> LinRegStats:
    return LinRegStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
        ]],
        y=outcome_df['los'],
    )


# All Boolean Case =============================================================

def linreg_2bool() -> LinRegStats:
    return LinRegStats(
        X=outcome_df[[
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['los'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# Print Results ================================================================

def print_linreg_tests() -> None:
    print(linreg_1float_2int_2bool())
    print(linreg_1float_2_int())
    print(linreg_2bool())


################################################################################
# Logit
################################################################################

# Mixed Types ==================================================================

def logit_1float_2int_2bool() -> LogitStats:
    return LogitStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# All Numeric Case =============================================================

def logit_1float_2_int() -> LogitStats:
    return LogitStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
        ]],
        y=outcome_df['mortality_24h'],
    )


# All Boolean Case =============================================================

def logit_2bool() -> LogitStats:
    return LogitStats(
        X=outcome_df[[
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# Print Results ================================================================

def print_logit_tests() -> None:
    print(logit_1float_2int_2bool())
    print(logit_1float_2_int())
    print(logit_2bool())


################################################################################
# LogBin
################################################################################

# Mixed Types ==================================================================

def logbin_1float_2int_2bool() -> LogBinStats:
    return LogBinStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# All Numeric Case =============================================================

def logbin_1float_2_int() -> LogBinStats:
    return LogBinStats(
        X=outcome_df[[
            'ca_grams_per_unit_4h',  # pd.Float64
            'age',  # pd.Int64
            'iss',  # pd.Int64
        ]],
        y=outcome_df['mortality_24h'],
    )


# All Boolean Case =============================================================

def logbin_2bool() -> LogBinStats:
    return LogBinStats(
        X=outcome_df[[
            'moi_pen',  # pd.Boolean
            'sex_female',  # boolean-like float
        ]],
        y=outcome_df['mortality_24h'],
        bool_col_names=['moi_pen', 'sex_female']
    )


# Print Results ================================================================

def print_logbin_tests() -> None:
    print(logbin_1float_2int_2bool())
    print(logbin_1float_2_int())
    print(logbin_2bool())


################################################################################
# Main
################################################################################

def main():
    print_linreg_tests()
    print_logit_tests()
    print_logbin_tests()


if __name__ == '__main__':
    main()
