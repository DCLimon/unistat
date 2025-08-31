from .contingency import MulticlassContingencyStats, BooleanContingencyStats
from .regression import LogitStats, LinRegStats
from .continuous import CorrStats, TwoSampleStats, TwoSeriesStats

# Expose all core classes at top-level
MulticlassContingencyStats = MulticlassContingencyStats
BooleanContingencyStats = BooleanContingencyStats
LogitStats = LogitStats
LinRegStats = LinRegStats
CorrStats = CorrStats
TwoSampleStats = TwoSampleStats
TwoSeriesStats = TwoSeriesStats

__all__ = [
    'MulticlassContingencyStats',
    'BooleanContingencyStats',
    'LogitStats',
    'LinRegStats',
    'CorrStats',
    'TwoSampleStats',
    'TwoSeriesStats',
]