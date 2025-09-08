
import numpy.typing as npt
import pandas as pd


# Pandas/NumPy compatibility
type VectorLike = pd.Series | npt.ArrayLike


__all__ = ['VectorLike']
