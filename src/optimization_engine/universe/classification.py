"""Labels that know when they became true.

A sector map is not a constant. A name reclassified from Industrials to
Technology in 2022 was Industrials in 2019, and a backtest that groups it as
Technology throughout has quietly told its 2019 self about a decision taken
three years later. The effect is not academic: sector caps, sector-neutral
tilts and group risk decompositions all read this mapping, so a stale label
moves weights.

:class:`Classification` keeps the label *and* the date it took effect, and
refuses to answer a question about a dated label without a date to answer it
as of. A label with genuinely no history — an ISO currency code, a legal
domicile that has not moved — is declared static with
:meth:`Classification.static` and answers without one.

Before the first record there is no label, and the answer is ``None`` rather
than a guess: a name whose classification history starts in 2015 was not
"Unclassified" in 2014, it was simply not covered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from optimization_engine.universe.signal import UniverseError

#: Column names :meth:`Classification.from_history` reads by default.
ASSET_COLUMN = "asset"
LABEL_COLUMN = "label"
EFFECTIVE_FROM_COLUMN = "effective_from"


@dataclass(frozen=True)
class LabelRecord:
    """One label, and the date it took effect.

    Attributes:
        label: The classification value — a sector, a country, a currency.
        effective_from: The first date the label applies. ``None`` marks a
            static record: a label with no history, which answers without a
            date.
    """

    label: str
    effective_from: pd.Timestamp | None = None


class Classification:
    """Point-in-time labels for a universe.

    Build one with :meth:`static` for labels that have never moved, or with
    :meth:`from_history` for labels that have. The two can be mixed in one
    object: whether a date is required is decided per asset, by whether that
    asset's own records are dated.

    Args:
        records: ``asset -> records``, each a :class:`LabelRecord`.
        name: What this classification is called — ``"GICS sector"``,
            ``"Domicile"``. Used as the default layer name by
            :meth:`~optimization_engine.constraints.ConstraintLayer.from_classification`.

    Raises:
        UniverseError: If an asset carries both dated and undated records, so
            it is neither answerable without a date nor properly historical.
    """

    __slots__ = ("_records", "_name")

    def __init__(
        self,
        records: Mapping[str, Sequence[LabelRecord]],
        name: str = "Classification",
    ) -> None:
        cleaned: dict[str, tuple[LabelRecord, ...]] = {}
        for asset, entries in dict(records).items():
            ordered = sorted(
                entries,
                key=lambda r: (
                    r.effective_from is not None,
                    r.effective_from or pd.Timestamp.min,
                ),
            )
            dated = [r for r in ordered if r.effective_from is not None]
            if dated and len(dated) != len(ordered):
                raise UniverseError(
                    f"{asset!r} carries both dated and undated labels. A label "
                    "either has a history or it does not; pick one."
                )
            cleaned[str(asset)] = tuple(ordered)
        self._records = cleaned
        self._name = str(name)

    # -- construction -------------------------------------------------------

    @classmethod
    def static(
        cls, mapping: Mapping[str, Any], name: str = "Classification"
    ) -> Classification:
        """Labels with no history, answerable without a date.

        Args:
            mapping: ``asset -> label``. Empty and missing labels are dropped,
                so an unlabelled name is genuinely uncovered rather than
                labelled ``""``.
            name: What this classification is called.

        Returns:
            The :class:`Classification`. :meth:`label` on it never requires
            ``as_of``, and ignores one if given.
        """
        records = {
            str(asset): (LabelRecord(str(label)),)
            for asset, label in dict(mapping).items()
            if label is not None and str(label).strip() != ""
        }
        return cls(records, name=name)

    @classmethod
    def from_history(
        cls,
        frame: pd.DataFrame,
        *,
        asset_column: str = ASSET_COLUMN,
        label_column: str = LABEL_COLUMN,
        effective_from_column: str = EFFECTIVE_FROM_COLUMN,
        name: str = "Classification",
    ) -> Classification:
        """Dated labels, from a long frame of ``(asset, label, effective_from)``.

        Args:
            frame: One row per label change. Extra columns are ignored.
            asset_column: Column holding the asset identifier.
            label_column: Column holding the label.
            effective_from_column: Column holding the date the label took
                effect. Parsed with :func:`pandas.to_datetime`.
            name: What this classification is called.

        Returns:
            The :class:`Classification`. Every asset in it is dated, so
            :meth:`label` requires an ``as_of``.

        Raises:
            UniverseError: If a required column is missing, or any
                ``effective_from`` is empty or unparseable — a label change
                with no date is a label change that cannot be placed in time.
        """
        missing = [
            column
            for column in (asset_column, label_column, effective_from_column)
            if column not in frame.columns
        ]
        if missing:
            raise UniverseError(
                f"A classification history needs the column(s) {missing}; "
                f"got {list(frame.columns)}."
            )
        try:
            stamps = pd.to_datetime(frame[effective_from_column])
        except (TypeError, ValueError) as exc:
            raise UniverseError(
                f"{effective_from_column!r} could not be read as dates: {exc}"
            ) from exc
        if stamps.isna().any():
            bad = frame.loc[stamps.isna(), asset_column].astype(str).tolist()
            raise UniverseError(
                f"{effective_from_column!r} is empty for {sorted(set(bad))}. A "
                "label change with no date cannot be placed in time."
            )
        records: dict[str, list[LabelRecord]] = {}
        for asset, label, stamp in zip(
            frame[asset_column], frame[label_column], stamps
        ):
            records.setdefault(str(asset), []).append(
                LabelRecord(str(label), pd.Timestamp(stamp))
            )
        return cls(records, name=name)

    # -- shape --------------------------------------------------------------

    @property
    def name(self) -> str:
        """What this classification is called."""
        return self._name

    @property
    def assets(self) -> list[str]:
        """Every asset with at least one record, sorted."""
        return sorted(self._records)

    @property
    def is_dated(self) -> bool:
        """Whether any asset's label has a history.

        Returns:
            ``True`` when at least one record carries an ``effective_from``,
            which is what makes ``as_of`` mandatory for that asset.
        """
        return any(
            record.effective_from is not None
            for entries in self._records.values()
            for record in entries
        )

    def __repr__(self) -> str:
        """``Classification(<name>, assets=…, dated=…)``."""
        return (
            f"Classification({self._name!r}, assets={len(self._records)}, "
            f"dated={self.is_dated})"
        )

    # -- reading ------------------------------------------------------------

    def label(self, asset: str, as_of: Any = None) -> str | None:
        """The label an asset carried on a date.

        Args:
            asset: The asset identifier.
            as_of: The date to read as of. Required when the asset's records
                are dated; ignored when they are static.

        Returns:
            The label in force, or ``None`` when the asset is not covered at
            all, or when ``as_of`` precedes its first record — before that
            date the classification had nothing to say, which is not the same
            as a label of "unknown".

        Raises:
            UniverseError: If the asset's records are dated and ``as_of`` is
                ``None``. The whole point of this class is that the answer
                moves; returning the latest label would be exactly the
                look-ahead it exists to prevent.
        """
        entries = self._records.get(str(asset))
        if not entries:
            return None
        if entries[0].effective_from is None:
            return entries[-1].label
        if as_of is None:
            raise UniverseError(
                f"{asset!r} has a dated classification, so label() needs an "
                "as_of date. Reading the latest label at every point in "
                "history is the look-ahead this class exists to prevent."
            )
        stamp = pd.Timestamp(as_of)
        found: str | None = None
        for record in entries:
            if record.effective_from is not None and record.effective_from <= stamp:
                found = record.label
        return found

    def assignments(self, as_of: Any = None) -> dict[str, str]:
        """``asset -> label`` as of a date, ready for a constraint layer.

        Args:
            as_of: The date to read as of. Required when any asset is dated.

        Returns:
            One entry per asset that had a label on that date, in sorted
            asset order. Assets whose history had not started are left out
            rather than mapped to a placeholder bucket.

        Raises:
            UniverseError: If a dated asset is read without ``as_of``.
        """
        out: dict[str, str] = {}
        for asset in self.assets:
            found = self.label(asset, as_of)
            if found is not None:
                out[asset] = found
        return out

    def labels(self, as_of: Any = None) -> list[str]:
        """Every distinct label in force on a date, sorted.

        Args:
            as_of: The date to read as of. Required when any asset is dated.

        Returns:
            The label values, sorted. These are the buckets a layer built from
            this classification will carry.

        Raises:
            UniverseError: If a dated asset is read without ``as_of``.
        """
        return sorted(set(self.assignments(as_of).values()))

    def group_matrix(
        self, as_of: Any = None, assets: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """The membership matrix in force on a date.

        This is the form the constraint layers and the risk decomposition
        consume: one row per asset, one column per label, ``True`` where the
        asset carried that label on that date.

        Args:
            as_of: The date to read as of. Required when any asset is dated.
            assets: Restrict and order the rows. ``None`` uses every asset the
                classification knows, sorted. An asset with no label on that
                date gets an all-``False`` row, which is the honest reading:
                it belongs to no bucket, so no bucket limit binds it.

        Returns:
            A ``bool`` frame indexed by asset with one column per label,
            columns sorted.

        Raises:
            UniverseError: If a dated asset is read without ``as_of``.
        """
        mapping = self.assignments(as_of)
        rows = [str(a) for a in assets] if assets is not None else self.assets
        columns = sorted(set(mapping.values()))
        matrix = pd.DataFrame(False, index=pd.Index(rows, name="asset"), columns=columns)
        for asset, group in mapping.items():
            if asset in matrix.index:
                matrix.at[asset, group] = True
        return matrix

    def describe(self) -> str:
        """One line: the name, how many assets, and whether it is point-in-time.

        Returns:
            Something like ``"GICS sector: 42 assets, dated (as_of
            required)"``.
        """
        kind = "dated (as_of required)" if self.is_dated else "static"
        return f"{self._name}: {len(self._records)} assets, {kind}"


__all__ = [
    "ASSET_COLUMN",
    "EFFECTIVE_FROM_COLUMN",
    "LABEL_COLUMN",
    "Classification",
    "LabelRecord",
]
