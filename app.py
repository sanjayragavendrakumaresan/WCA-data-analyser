"""Application composition root."""

from .menu_base import WCAMenuBase
from .statistics_menu import StatisticsMenuMixin
from .predictive_menu import PredictiveMenuMixin


class WCAMenu(WCAMenuBase, StatisticsMenuMixin, PredictiveMenuMixin):
    """Combined menu application assembled from focused mixins."""
    pass
