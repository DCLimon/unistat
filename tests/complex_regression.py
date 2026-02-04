
################################################################################
# Setup
################################################################################

# Import standard packages
import pathlib
# Import 3rd party packages
import numpy as np
import pandas as pd
import polars as pl
# Import local modules
from tests.test_project.dataprep import prep_data
from unistat._complex_regression import ComplexRegression


# Data Formatting ==============================================================

outcome_df = prep_data()


################################################################################
# Functionality
################################################################################

test_regression = ComplexRegression(
    data=outcome_df,
    y='mortality_24h',
    X_num=['ca_grams_4h', 'age', 'iss'],
    X_bool=['moi_pen', 'sex_female', 'ca_gluconate_bool'],
    X_int=['ca_grams_4h', 'ca_gluconate_bool']
)


################################################################################
# Main
################################################################################

def main():
    pass


if __name__ == '__main__':
    main()
