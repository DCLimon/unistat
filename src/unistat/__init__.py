from .contingency import MulticlassContingencyStats, BooleanContingencyStats
from .regression import LogitStats, LinRegStats
from .continuous import CorrStats, TwoSampleStats

# Expose all core classes at top-level
MulticlassContingencyStats = MulticlassContingencyStats
BooleanContingencyStats = BooleanContingencyStats
LogitStats = LogitStats
LinRegStats = LinRegStats
CorrStats = CorrStats
TwoSampleStats = TwoSampleStats

__all__ = [
    "MulticlassContingencyStats",
    "BooleanContingencyStats",
    "LogitStats",
    "LinRegStats",
    "CorrStats",
    "TwoSampleStats",
]