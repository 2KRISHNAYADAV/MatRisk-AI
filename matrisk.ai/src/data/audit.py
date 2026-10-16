"""
src.data.audit
---------------
Shared data-loading utilities for the MatRisk AI notebooks.

This module provides the small, stable API that every notebook in this
project imports: file locations, CSV/XLSX loaders, the README reader, a
chemical-formula parser, and the isotropic-elasticity identity used
throughout (nu implied by K, G).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# File locations (relative to data/raw)
# ---------------------------------------------------------------------
CSV_FILES = {
    "DS1": "DS1_material_properties_5500.csv",
    "DS2": "DS2_commodity_prices_10yr.csv",
    "DS3": "DS3_infrastructure_bridges_5000.csv",
    "DS4": "DS4_crossdomain_features_daily.csv",
    "DS5": "DS5_element_prices_monthly.csv",
    "DS6": "DS6_historical_failures_2000.csv",
}
XLSX_FILE = "MatRisk_AI_Sample_Datasets.xlsx"
README_FILE = "ReadME_for_Dataset.docx"

# date columns to parse per dataset (empty list = no date column)
DATE_COLS = {
    "DS1": [],
    "DS2": ["date"],
    "DS3": ["last_inspection_date"],
    "DS4": ["date"],
    "DS5": ["date"],
    "DS6": [],
}

# DS3 materials with non-zero Paris-law fatigue constants (data-driven,
# not asserted): metals undergo fatigue crack growth, concrete/FRP/timber
# do not, and the data encodes this as paris_m == 0 for the latter.
METAL_MATERIALS_DS3 = {
    "Aluminium Alloy", "Cast Iron", "Composite Steel-Concrete",
    "Stainless Steel 316L", "Steel (A36)", "Steel (A514 HY)",
    "Steel (A588 Weathering)",
}


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_csvs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all six DS CSVs with their date columns parsed."""
    raw_dir = Path(raw_dir)
    out = {}
    for key, fname in CSV_FILES.items():
        df = pd.read_csv(raw_dir / fname)
        for c in DATE_COLS[key]:
            df[c] = pd.to_datetime(df[c])
        out[key] = df
    return out


def _is_blank_row(row) -> bool:
    return all(v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" for v in row)


def load_xlsx_blocks(path: Path) -> dict[str, pd.DataFrame]:
    """
    Parse every sheet of the workbook into one or more tabular "blocks".

    The workbook mixes title rows, free-text notes, blank-row separators
    and (for DS1 and DS3) more than one table per sheet. This walks each
    sheet, treats a run of non-blank rows as a candidate block, and keeps
    the ones that look like a table (a header row of distinct, mostly
    string labels followed by at least one data row). Blocks are named
    "<sheet>::block1", "<sheet>::block2", ... in sheet order, except a
    block whose header contains "VALIDATION" which is named
    "<sheet>::VALIDATION" for readability.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    blocks: dict[str, pd.DataFrame] = {}

    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]

        # split into runs of non-blank rows
        runs, cur = [], []
        for r in rows:
            if _is_blank_row(r):
                if cur:
                    runs.append(cur)
                    cur = []
            else:
                cur.append(r)
        if cur:
            runs.append(cur)

        n = 0
        for run in runs:
            # find the header row: first row in the run where >=2 cells
            # are non-null strings and the row is not a single free-text
            # title/note (title/note rows have exactly one non-null cell)
            header_idx = None
            for i, r in enumerate(run):
                non_null = [v for v in r if v is not None and str(v).strip() != ""]
                str_like = [v for v in non_null if isinstance(v, str)]
                if len(non_null) >= 2 and len(str_like) >= 2:
                    header_idx = i
                    break
            if header_idx is None or header_idx >= len(run) - 1:
                continue  # title-only run or header with no data rows

            header = [str(h).strip() if h is not None else f"col{i}" for i, h in enumerate(run[header_idx])]
            # de-duplicate header labels
            seen: dict[str, int] = {}
            dedup = []
            for h in header:
                seen[h] = seen.get(h, 0) + 1
                dedup.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
            data_rows = run[header_idx + 1:]
            width = len(dedup)
            data_rows = [list(r[:width]) + [None] * (width - len(r)) for r in data_rows]
            df = pd.DataFrame(data_rows, columns=dedup)
            # drop fully-empty trailing columns
            df = df.dropna(axis=1, how="all")

            n += 1
            name = f"{sheet}::block{n}"
            if any("VALIDATION" in str(c).upper() for c in header if c):
                name = f"{sheet}::VALIDATION"
            blocks[name] = df.reset_index(drop=True)

    return blocks


def read_docx_text(path: Path) -> str:
    """Extract plain text (paragraphs) from the README .docx."""
    import docx

    d = docx.Document(str(path))
    lines = [p.text for p in d.paragraphs if p.text.strip() != ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Chemistry / physics helpers shared across notebooks
# ---------------------------------------------------------------------
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_elements(formula: str) -> dict[str, float]:
    """
    Parse a reduced chemical formula like 'Fe2O3' or 'Ti6Al4V' into
    {element_symbol: count}. Counts default to 1 when no number follows
    the symbol. set(parse_elements(f)) gives the set of elements present
    (dict iteration yields its keys).
    """
    out: dict[str, float] = {}
    for sym, num in _FORMULA_TOKEN.findall(str(formula)):
        if not sym:
            continue
        count = float(num) if num not in ("", None) else 1.0
        out[sym] = out.get(sym, 0.0) + count
    return out


def _poisson_from_KG(K, G):
    """Isotropic elasticity identity: nu implied by bulk modulus K and shear modulus G."""
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    return (3 * K - 2 * G) / (2 * (3 * K + G))


def young_from_KG(K, G):
    """Young's modulus E implied by (K, G): E = 9KG / (3K + G)."""
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    return (9 * K * G) / (3 * K + G)
