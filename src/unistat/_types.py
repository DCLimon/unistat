
from typing import Literal
import numpy.typing as npt
import pandas as pd


# Pandas/NumPy compatibility
type VectorLike = pd.Series | npt.ArrayLike

# p-value correction method
type PCorrectionMethod = Literal[
    'bonferroni',
    'sidak',
    '`holm-sidak',
    'holm',
    '`simes-hochberg',
    'hommel',
    'fdr_bh',
    'fdr_by',
    'fdr_tsbh',
    'fdr_tsbky',
]


__all__ = ['VectorLike', 'PCorrectionMethod']
