
################################################################################
# Setup
################################################################################

import pathlib
import polars as pl
import pandas as pd


# Module Options ===============================================================

pd.set_option('future.no_silent_downcasting', True)
pd.set_option('mode.copy_on_write', 'warn')

working_parquet_path = r'tests\data\test_data_wb_calcium.parquet'


################################################################################
# Prep data
################################################################################

class IRLData:
    CA_GLUCONATE_ELEMENTAL_FRACTION = 0.093  # Per Rx label; actual is 0.093124
    CACL2_ELEMENTAL_FRACTION = 0.270  # Per Rx label

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path or (
            pathlib.Path(working_parquet_path)
        )
        self.df = None

    def import_parquet(self) -> pl.DataFrame:
        self.df = pl.read_parquet(
            source=self.path,
        )

        return self.df

    def clean_df(self) -> pl.DataFrame:
        self.df = self.df.rename({
            'Age': 'age',
            'Sex': 'sex_female',
            'MOI': 'moi_pen',
            'CPR': 'cpr',
            'Ca_prehosp': 'ca_prehosp_bool',
            'Doseprehospca': 'ca_prehosp_dose_g',
            'ISS': 'iss',
            'CA_tru_yesno': 'ca_tru_bool',
            'CA_TRU_DOSE': 'ca_tru_dose_g',
            # 'ArrivalCA': 'ica_arrival',
            'CAHR4': 'ica_4h',
            'HypoCA_at4': 'hypoca_4h_bool',
            'totalCAdose_by4': 'ca_grams_4h',
            'LTOWB_by4': 'wb_total_4h',
            'LTOWB_by24': 'wb_total_24h',
            'mortality_by30': 'mortality_30d',
            'Mortality_byhr24': 'mortality_24h',
            'Lessthanoneper3': 'lt_1g_per_3u',
            'Lessthanoneper2': 'lt_1g_per_2u',
            'Lessthanoneper4': 'lt_1g_per_4u',
            'Adequateoneper3': 'geq_1g_per_3u',
            'Adequateoneper2': 'geq_1g_per_2u',
            'Adequateoneper4': 'geq_1g_per_4u',
            'atleastoneper2': 'gt_1g_per_2u',
            'atleastoneper3': 'gt_1g_per_3u',
            'atleastoneper4': 'gt_1g_per_4u',
            'Atleastoneprbcorltowbper2': 'gt_2g_per_1u',
            'calciumtype': 'ca_gluconate_bool',
            'ionizedca_arrival': 'ica_arrival',
            'BleedingunplannedOR': 'unplanned_op_bleeding',
            'Bleedingother': 'other_bleeding',
            'DVTPE': 'vte_bool',
            'wenttoOR': 'went_to_or_bool',
            # 'los': 'los',
            'mortalityat4': 'mortality_4h',
            'mortality30min': 'mortality_30min',
            'gotsomecalcium': 'any_ca_bool',
            # 'elementalcalcium': 'elementalcalcium',
            'filter_$': 'filter_col',
        })

        drop_cols = ['ArrivalCA', 'filter_col', 'atl1per2n', 'atl1per3n',
                     'atl1per4n', 'countinousva', 'elementalcalcium']
        self.df = self.df.drop(drop_cols)

        # Make Ca type into a normal bool, then cast columns as booleans
        self.df = self.df.with_columns(pl.col('ca_gluconate_bool') == pl.lit(1))

        bool_cols = [
            'sex_female', 'moi_pen', 'cpr', 'ca_prehosp_bool', 'ca_tru_bool',
            'hypoca_4h_bool', 'mortality_30d', 'mortality_24h',
            'lt_1g_per_3u', 'lt_1g_per_2u', 'lt_1g_per_4u', 'geq_1g_per_3u',
            'geq_1g_per_2u', 'geq_1g_per_4u', 'gt_1g_per_2u', 'gt_1g_per_3u',
            'gt_1g_per_4u', 'gt_2g_per_1u', 'ca_gluconate_bool',
            'unplanned_op_bleeding', 'other_bleeding',
            'vte_bool',  'went_to_or_bool', 'mortality_4h', 'mortality_30min',
            'any_ca_bool',
        ]
        self.df = self.df.with_columns(
            [pl.col(col).cast(pl.Boolean) for col in bool_cols]
        )

        return self.df

    def synthesize_cols(self) -> pl.DataFrame:
        self.df = self.df.with_columns(
            # Ca (gluconate or chloride) grams per unit ========================
            (pl.col('ca_grams_4h') / pl.col('wb_total_4h'))
            .cast(pl.Float64)
            .alias('ca_grams_per_unit_4h'),

            # Calculate elemental Ca dose ======================================
            # For elemental Ca dose from Ca-gluconate, multiply the Ca-gluc dose
            # in grams by the w/w mass frac (9.3124%). Divide by 1000 for mg.
            pl.when(pl.col('ca_gluconate_bool'))
            .then(
                pl.col('ca_grams_4h')
                * pl.lit(self.CA_GLUCONATE_ELEMENTAL_FRACTION)
            )
            # For elemental Ca dose from CaCl2, multiply the CaCl2 dose in grams
            # by the w/w mass frac (27%). Divide by 1000 for mg.
            .when(~pl.col('ca_gluconate_bool'))
            .then(pl.col('ca_grams_4h') * pl.lit(self.CACL2_ELEMENTAL_FRACTION))
            .otherwise(None)
            .alias('elemental_ca_grams_4h'),
        ).with_columns(
            # Elemental Ca dose per unit WB ====================================
            (pl.col('elemental_ca_grams_4h') / pl.col('wb_total_4h'))
            .alias('elemental_ca_grams_per_unit_4h'),
        ).with_columns(
            # Ca grams per unit cut mutually exclusive bins ====================
            pl.when(pl.col('ca_grams_per_unit_4h').ge(0.5)).then(pl.lit(True))
            .when(pl.col('ca_grams_per_unit_4h').lt(0.5)).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_ca_geq0.5_grams_per_unit_4h'),

            pl.when(
                pl.col('ca_grams_per_unit_4h').lt(0.5)
                & pl.col('ca_grams_per_unit_4h').ge(1/3)
            ).then(pl.lit(True))
            .when(pl.col('ca_grams_per_unit_4h').lt(1/3)).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_ca_geq0.33_grams_per_unit_4h'),

            pl.when(
                pl.col('ca_grams_per_unit_4h').lt(1/3)
                & pl.col('ca_grams_per_unit_4h').ge(0.25)
            ).then(pl.lit(True))
            .when(pl.col('ca_grams_per_unit_4h').lt(0.25)).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_ca_geq0.25_grams_per_unit_4h'),

            pl.when(pl.col('ca_grams_per_unit_4h').lt(0.25)).then(pl.lit(True))
            .when(pl.col('ca_grams_per_unit_4h').ge(0.25)).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_ca_lt0.25_grams_per_unit_4h')
        )

        self.df = self.df.with_columns(
            # Elemental Ca grams per unit cut mutually exclusive bins ==========
            # Elemental Ca dose based on elemental Ca from the above grams of
            # Ca-gluconate
            pl.when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .ge(self.gluc_to_elemental(0.5))
            ).then(pl.lit(True))
            .when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .lt(self.gluc_to_elemental(0.5))
            ).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_elemental_ca_geq46.5mg_per_unit_4h'),

            pl.when(
                (
                    pl.col('elemental_ca_grams_per_unit_4h')
                    .lt(self.gluc_to_elemental(0.5))
                ) & (
                    pl.col('elemental_ca_grams_per_unit_4h')
                    .ge(self.gluc_to_elemental(1/3))
                )
            ).then(pl.lit(True))
            .when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .lt(self.gluc_to_elemental(1/3))
            ).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_elemental_ca_geq31mg_per_unit_4h'),

            pl.when(
                (
                    pl.col('elemental_ca_grams_per_unit_4h')
                    .lt(self.gluc_to_elemental(1/3))
                ) & (
                    pl.col('elemental_ca_grams_per_unit_4h')
                    .ge(self.gluc_to_elemental(0.25))
                )
            ).then(pl.lit(True))
            .when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .lt(self.gluc_to_elemental(0.25))
            ).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_elemental_ca_geq23.25mg_per_unit_4h'),

            pl.when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .lt(self.gluc_to_elemental(0.25))
            ).then(pl.lit(True))
            .when(
                pl.col('elemental_ca_grams_per_unit_4h')
                .ge(self.gluc_to_elemental(0.25))
            ).then(pl.lit(False))
            .otherwise(None)
            .cast(pl.Boolean)
            .alias('mutex_elemental_ca_lt23.25mg_per_unit_4h'),
        )

        self.df = self.df.with_columns(
            # Ca grams per unit categorical column =============================
            pl.when(pl.col('mutex_ca_geq0.5_grams_per_unit_4h'))
            .then(pl.lit('>= 0.5 g/U'))
            .when(pl.col('mutex_ca_geq0.33_grams_per_unit_4h'))
            .then(pl.lit('0.33 - 0.5 g/U'))
            .when(pl.col('mutex_ca_geq0.25_grams_per_unit_4h'))
            .then(pl.lit('0.25 - 0.33 g/U'))
            .when(pl.col('mutex_ca_lt0.25_grams_per_unit_4h'))
            .then(pl.lit('< 0.25 g/U'))
            .otherwise(None)
            .cast(pl.Enum([
                '>= 0.5 g/U', '0.33 - 0.5 g/U', '0.25 - 0.33 g/U', '< 0.25 g/U',
            ]))
            .alias('mutex_ca_grams_per_unit_4h_binned'),

            # Ca grams per unit categorical column =============================
            pl.when(pl.col('mutex_elemental_ca_geq46.5mg_per_unit_4h'))
            .then(pl.lit('>= 46.5 mg/U'))
            .when(pl.col('mutex_elemental_ca_geq31mg_per_unit_4h'))
            .then(pl.lit('31 - 46.5 mg/U'))
            .when(pl.col('mutex_elemental_ca_geq23.25mg_per_unit_4h'))
            .then(pl.lit('23.25 - 31 mg/U'))
            .when(pl.col('mutex_elemental_ca_lt23.25mg_per_unit_4h'))
            .then(pl.lit('< 23.25 mg/U'))
            .otherwise(None)
            .cast(pl.Enum([
                '>= 46.5 mg/U', '31 - 46.5 mg/U',
                '23.25 - 31 mg/U', '< 23.25 mg/U',
            ]))
            .alias('mutex_elemental_ca_mg_per_unit_4h_binned'),
        )

        return self.df

    def make_df(self) -> pl.DataFrame:
        self.import_parquet()
        self.clean_df()
        self.synthesize_cols()
        return self.df

    def polars(self) -> pl.DataFrame:
        return self.make_df()

    def pandas(self) -> pd.DataFrame:
        return (
            self.make_df()
            .to_pandas(use_pyarrow_extension_array=False)
            .convert_dtypes()
        )

    @staticmethod
    def gluc_to_elemental(gluc_dose: float | int) -> float:
        return float(gluc_dose * IRLData.CA_GLUCONATE_ELEMENTAL_FRACTION)


def prep_data():
    df = IRLData().pandas()

    # Remove CPR in prog
    df = df[~(df['cpr'].fillna(True))]

    # Filter out Pts receiving no Ca, no WB, or died <30min
    df = df[df['ca_grams_4h'] > 0]
    df = df[df['wb_total_4h'] > 0]
    # df = df[~df['mortality_30min']]

    return df


################################################################################
# Main
################################################################################

def main():
    global pl_df, pd_df
    data = IRLData()
    pl_df = data.polars()
    pd_df = data.pandas()


if __name__ == '__main__':
    main()
