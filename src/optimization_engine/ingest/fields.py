"""The homogenized vocabulary every provider is translated into.

Each vendor names the same number differently — ``Adj Close``, ``adjClose``,
``close_split_adjusted``, ``adjusted_close`` — and each has its own opinion
about what "close" already includes. If those names reach the optimizer, the
choice of provider leaks into every downstream module, and swapping Yahoo for
FMP becomes a refactor instead of a config change.

So the adapters translate once, at the edge, into the names below. Downstream
code only ever sees these. The prefix marks the block a field belongs to
(``m_`` for market data), which leaves room for fundamentals or macro blocks
later without a naming collision.

Two conventions worth stating explicitly, because getting them wrong is silent:

* **Adjustment.** :data:`CLOSE` is the *total-return* price — adjusted for both
  splits and dividends — because that is what a return series must be built
  from. :data:`CLOSE_RAW` is the unadjusted print, kept for reporting and for
  turning weights into share counts.
* **Volume.** :data:`VOLUME` is optional on purpose. An index has no volume,
  and the panel says so (:data:`InstrumentKind.INDEX`) rather than filling
  zeros that a liquidity model would happily read as "cannot trade this".
"""

from __future__ import annotations

from enum import Enum

#: Total-return price: split- and dividend-adjusted. The default series for
#: returns, covariance and every backtest.
CLOSE = "m_close"
#: Unadjusted closing print, as it appeared on the tape.
CLOSE_RAW = "m_close_raw"
OPEN = "m_open"
HIGH = "m_high"
LOW = "m_low"
#: Shares (or contracts) traded. ``None`` for instruments that have no volume.
VOLUME = "m_volume"
#: Volume-weighted average price, when the provider publishes it.
VWAP = "m_vwap"

#: The full market-data vocabulary, in the order a panel presents it.
MARKET_FIELDS: tuple[str, ...] = (
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    CLOSE_RAW,
    VOLUME,
    VWAP,
)

#: The minimum a provider must return to be useful at all. Everything the
#: engine does — returns, covariance, optimization, backtest — needs only this.
PRICE_ONLY: tuple[str, ...] = (CLOSE,)

#: Open/high/low/close without volume. What an index universe can actually
#: supply, and enough for range-based volatility estimators.
OHLC: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE)

#: Everything a liquid single-stock provider can give.
OHLCV: tuple[str, ...] = (OPEN, HIGH, LOW, CLOSE, VOLUME)

#: Fields whose values are prices, and so must be positive and mutually
#: ordered (low <= open/close <= high).
PRICE_FIELDS: frozenset[str] = frozenset({OPEN, HIGH, LOW, CLOSE, CLOSE_RAW, VWAP})

#: Fields that may legitimately be absent for a whole instrument rather than
#: just for a date. Their absence is a fact about the instrument, not a defect.
OPTIONAL_FIELDS: frozenset[str] = frozenset({VOLUME, VWAP, CLOSE_RAW})


class InstrumentKind(str, Enum):
    """What sort of thing an identifier refers to.

    This is not decoration. It decides whether missing volume is a data-quality
    problem or the expected state of the world, whether a price can be traded
    at all, and how the app labels the series. Providers set it from their own
    metadata when they publish one, and fall back to
    :attr:`UNKNOWN` rather than guessing.
    """

    EQUITY = "equity"
    ETF = "etf"
    #: A computed level — S&P 500, IPC, MSCI World. Has no volume and cannot
    #: be traded directly; the backtest treats it as infinitely liquid.
    INDEX = "index"
    FUND = "fund"
    FX = "fx"
    #: A yield or policy rate published as a level, not a tradeable price.
    RATE = "rate"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"

    @property
    def has_volume(self) -> bool:
        """Whether an instrument of this kind is expected to report volume.

        Used to decide whether a missing volume column is worth a warning.
        """
        return self in {
            InstrumentKind.EQUITY,
            InstrumentKind.ETF,
            InstrumentKind.CRYPTO,
            InstrumentKind.COMMODITY,
        }

    @property
    def is_tradeable(self) -> bool:
        """Whether the instrument can be held directly at its quoted price.

        Indices and rates cannot: a backtest on them is a study of the
        strategy, not a claim about an implementable portfolio.
        """
        return self not in {InstrumentKind.INDEX, InstrumentKind.RATE}


def normalize_fields(fields: object) -> tuple[str, ...]:
    """Validate and de-duplicate a requested field selection.

    Args:
        fields: An iterable of field names from this module's vocabulary.

    Returns:
        The requested fields in :data:`MARKET_FIELDS` order, without
        duplicates, always including :data:`CLOSE` — nothing downstream can
        work without it, and silently returning a panel with no price is a
        worse surprise than adding the column back.

    Raises:
        ValueError: If any name is not part of the vocabulary.
    """
    if isinstance(fields, str):
        requested = [fields]
    else:
        try:
            requested = [str(f) for f in fields]  # type: ignore[union-attr]
        except TypeError as exc:
            raise ValueError(f"fields must be iterable; got {type(fields).__name__}") from exc

    unknown = sorted(set(requested) - set(MARKET_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown field(s): {', '.join(unknown)}. "
            f"Known fields: {', '.join(MARKET_FIELDS)}."
        )
    selected = set(requested) | {CLOSE}
    return tuple(f for f in MARKET_FIELDS if f in selected)


__all__ = [
    "CLOSE",
    "CLOSE_RAW",
    "HIGH",
    "InstrumentKind",
    "LOW",
    "MARKET_FIELDS",
    "OHLC",
    "OHLCV",
    "OPEN",
    "OPTIONAL_FIELDS",
    "PRICE_FIELDS",
    "PRICE_ONLY",
    "VOLUME",
    "VWAP",
    "normalize_fields",
]
