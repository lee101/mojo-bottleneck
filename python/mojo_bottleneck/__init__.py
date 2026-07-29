"""Bottleneck-compatible nan reductions and moving-window functions."""

from .api import (
    allnan,
    anynan,
    move_max,
    move_mean,
    move_min,
    move_std,
    move_sum,
    move_var,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanstd,
    nansum,
    nanvar,
)

__version__ = "0.1.0"

__all__ = [
    "nansum",
    "nanmean",
    "nanmedian",
    "nanvar",
    "nanstd",
    "nanmin",
    "nanmax",
    "nanargmin",
    "nanargmax",
    "anynan",
    "allnan",
    "move_sum",
    "move_mean",
    "move_var",
    "move_std",
    "move_min",
    "move_max",
]
