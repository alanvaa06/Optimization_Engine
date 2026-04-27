"""Excel exporter for optimization results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def write_excel_report(path: str | Path, sheets: Mapping[str, pd.DataFrame | pd.Series]) -> Path:
    """Write a multi-sheet Excel workbook from a mapping of name → frame."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            sheet = name[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet, index=True)
    return p
