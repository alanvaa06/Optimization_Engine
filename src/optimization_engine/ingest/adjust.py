"""Putting a session range on the same scale as its adjusted close.

Several providers publish an adjusted close alongside an *unadjusted* open,
high and low — FMP does, Tiingo's raw fields do, and so does almost every
spreadsheet exported from a terminal, where an ``adj_close`` column sits at
the end of an otherwise raw OHLCV row.

Left alone that panel is not merely inconsistent, it is wrong in a way that
compounds. After a few percent of accumulated dividends the adjusted close
sits *outside* its own day's high and low, so any range-based volatility
estimator reads a bar that never existed, and the panel's own validation
rejects it — correctly, because a low above a close is not a thing that
happens.

The fix is the standard reconstruction: carry each day's own
``adjusted / raw`` ratio across the rest of that day's prices. It absorbs
whatever splits and dividends the provider has applied up to that date, needs
no corporate-actions history of its own, and leaves the panel internally
coherent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine.ingest import fields as F

#: The fields that move with the close and must share its scale. Volume is not
#: among them: it counts shares, and a split adjustment on it runs the other
#: way.
_SCALED_FIELDS: tuple[str, ...] = (F.OPEN, F.HIGH, F.LOW, F.VWAP)


def rescale_to_adjusted(
    series_by_field: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Scale the session range onto the adjusted close's basis.

    Args:
        series_by_field: Homogenized field name to that field's series, all
            sharing one index. Must contain both
            :data:`~optimization_engine.ingest.fields.CLOSE` (adjusted) and
            :data:`~optimization_engine.ingest.fields.CLOSE_RAW` (as printed)
            for anything to happen.

    Returns:
        A new mapping with open, high, low and VWAP multiplied by each date's
        ``CLOSE / CLOSE_RAW``. Where the raw close is missing or non-positive
        the ratio is unknowable, and that bar's range is dropped rather than
        left on the other scale: an unscaled bar beside an adjusted close is
        not a smaller error, it is a low above its own close, and it takes
        the whole panel down with it. A gap is something the panel is built
        to carry; a contradiction is not.
    """
    adjusted = series_by_field.get(F.CLOSE)
    raw = series_by_field.get(F.CLOSE_RAW)
    if adjusted is None or raw is None:
        return series_by_field

    usable = raw.where(raw > 0.0)
    ratio = pd.to_numeric(adjusted / usable, errors="coerce")
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    out = dict(series_by_field)
    for field in _SCALED_FIELDS:
        if field in out:
            out[field] = out[field] * ratio
    return out


def rescale_frames_to_adjusted(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """:func:`rescale_to_adjusted`, for whole per-field frames.

    Applied column-wise, so a panel where one identifier pays dividends and
    another does not is scaled correctly for both.

    Args:
        frames: ``field name -> frame``, which must carry both the adjusted
            and the raw close.

    Returns:
        The same mapping with every price field put on the adjusted scale. A
        cell with no usable ratio has its range dropped rather than left on
        the wrong scale.
    """
    adjusted = frames.get(F.CLOSE)
    raw = frames.get(F.CLOSE_RAW)
    if adjusted is None or raw is None:
        return frames

    aligned = raw.reindex(index=adjusted.index, columns=adjusted.columns)
    usable = aligned.where(aligned > 0.0)
    ratio = (adjusted / usable).replace([np.inf, -np.inf], np.nan)

    out = dict(frames)
    for field in _SCALED_FIELDS:
        if field in out:
            out[field] = out[field].reindex(
                index=adjusted.index, columns=adjusted.columns
            ) * ratio
    return out


__all__ = ["rescale_frames_to_adjusted", "rescale_to_adjusted"]
