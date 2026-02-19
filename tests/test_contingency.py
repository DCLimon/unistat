################################################################################
# Setup
################################################################################

# Import standard packages
import pandas as pd
# Import local modules
from tests.test_project.dataprep import get_outcome_df
from unistat import MulticlassContingencyStats, BooleanContingencyStats


# Data Formatting ==============================================================

outcome_df = get_outcome_df()


################################################################################
# Tests
################################################################################

# 2x2 Table ====================================================================

def two_by_two() -> BooleanContingencyStats:
    return BooleanContingencyStats(
        table_rows=outcome_df['ca_gluconate_bool'],  # Has missing values
        row_title='Ca Form',
        row_names=['CaCl2', 'Ca-gluconate'],
        table_cols=outcome_df['mortality_30d'],
        col_title='30d Mortality',
        col_names=['Survivor', 'Mortality'],
    )


def print_boolean_tests() -> None:
    print(two_by_two())


# Many x 2 =====================================================================

def four_by_two() -> MulticlassContingencyStats:
    return MulticlassContingencyStats(
        table_rows=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        row_title='Elemental Ca Dose',
        row_names=['≥46.5 mg/U', '31.0–46.5 mg/U',
                   '23.25–31.0 mg/U', '≤23.25 mg/U'],
        table_cols=outcome_df['ca_gluconate_bool'],
        col_title='Ca Form',
        col_names=['CaCl2', 'Ca-gluconate'],
    )


def two_by_four() -> MulticlassContingencyStats:
    return MulticlassContingencyStats(
        table_cols=outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        col_title='Elemental Ca Dose',
        col_names=['≥46.5 mg/U', '31.0–46.5 mg/U',
                   '23.25–31.0 mg/U', '≤23.25 mg/U'],
        table_rows=outcome_df['ca_gluconate_bool'],
        row_title='Ca Form',
        row_names=['CaCl2', 'Ca-gluconate'],
    )


def four_by_four() -> MulticlassContingencyStats:
    levels = ['<2U', '2-4U', '4-8U', '>=8U']
    seriess: list = []

    df = pd.cut(
        outcome_df['wb_total_4h'],
        bins=[0, 2., 4., 8.0, 1000],
        labels=levels,
        include_lowest=True
    )
    df = pd.concat(objs=[outcome_df['mutex_elemental_ca_mg_per_unit_4h_binned'],
                         df],
                   axis='columns')

    return MulticlassContingencyStats(
        table_cols=df['mutex_elemental_ca_mg_per_unit_4h_binned'],
        col_title='Elemental Ca Dose',
        col_names=['≥46.5 mg/U', '31.0–46.5 mg/U',
                   '23.25–31.0 mg/U', '≤23.25 mg/U'],
        table_rows=df['wb_total_4h'],
        row_title='WB xfsn range',
        # row_names=['CaCl2', 'Ca-gluconate'],
    )


def print_multiclass_tests() -> None:
    print(four_by_two())
    for table in four_by_two().residuals_post_hoc():
        print(table.to_string(), sep='\n')

    print(two_by_four())
    for table in two_by_four().residuals_post_hoc():
        print(table.to_string(), sep='\n')

    print(four_by_four())


################################################################################
# Main
################################################################################

def main():
    print_boolean_tests()
    print_multiclass_tests()


if __name__ == '__main__':
    main()
