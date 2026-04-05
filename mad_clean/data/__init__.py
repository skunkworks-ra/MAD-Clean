"""mad_clean.data — dataset loaders and observation simulation."""

from .crumb    import CRUMB, CRUMB_MB, CRUMB_FRDEEP, CRUMB_AT17, CRUMB_MBHyb, CRUMB_CoMBo, CRUMB_NoMB, CRUMB_4Class
from .simulate import SimulateObservations

__all__ = [
    "CRUMB", "CRUMB_MB", "CRUMB_FRDEEP", "CRUMB_AT17",
    "CRUMB_MBHyb", "CRUMB_CoMBo", "CRUMB_NoMB", "CRUMB_4Class",
    "SimulateObservations",
]
