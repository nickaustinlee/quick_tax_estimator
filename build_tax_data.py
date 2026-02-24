"""
build_tax_data.py
-----------------
Reads all tax CSVs and writes a unified tax_data.yaml used by stock_tax_estimator.py.

Run:
    poetry run python build_tax_data.py [--output tax_data.yaml]

Re-run whenever the source CSVs change.
"""

import argparse
import csv
import os
import re
import sys
from typing import Optional

import yaml

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CSV filename map
# ---------------------------------------------------------------------------
CSV_FILES: dict = {
    "federal_brackets": {
        2025: "2025_federal_tax_brackets.csv",
        2026: "2026_federal_tax_brackets.csv",
    },
    "federal_std_deduction": {
        2025: "2025_federal_standard_deduction.csv",
        2026: "2026_federal_standard_deduction.csv",
    },
    "amt_exemptions": {
        2025: "2025_amt_exemptions.csv",
        2026: "2026_amt_exemptions.csv",
    },
    "amt_phaseout": {
        2025: "2025_amt_exemption_phaseout_thresholds.csv",
        2026: "2026_amt_exemption_phaseout_thresholds.csv",
    },
    "state_brackets": {
        2025: "2025_state_income_tax_brackets.csv",
        2026: "2026_state_income_tax_brackets.csv",
    },
    "ltcg_brackets": {
        2025: "2025_ltcg_brackets_federal.csv",
        2026: "2026_ltcg_brackets_federal.csv",
    },
}

SUPPORTED_YEARS = [2025, 2026]

# ---------------------------------------------------------------------------
# Hardcoded constants not present in any CSV
# ---------------------------------------------------------------------------

# AMT 26% / 28% ordinary income rate breakpoint.
# Source: Tax Foundation
#   2025: "the 28% AMT rate applies to excess AMTI of $239,100 for all taxpayers"
#   2026: "the 28% AMT rate applies to excess AMTI of $244,500 for all taxpayers"
# Applies to all filing statuses except married-filing-separately (ignored by this tool).
AMT_RATE_BREAKPOINTS: dict = {
    2025: 239100,
    2026: 244500,
}

# AMT exemption phaseout rate (cents reduced per dollar of AMTI over threshold).
# Source: Tax Foundation
#   2025: 25 cents per dollar (TCJA rate)
#   2026: 50 cents per dollar (OBBBA reverts to pre-TCJA phaseout rate — more aggressive)
AMT_PHASEOUT_RATES: dict = {
    2025: 0.25,
    2026: 0.50,
}

# NIIT thresholds — not inflation-adjusted.
NIIT_THRESHOLDS: dict = {
    "single": 200000,
    "married": 250000,
    "head_of_household": 200000,
}

# ---------------------------------------------------------------------------
# State name normalisation tables
# ---------------------------------------------------------------------------

# 2025 CSV uses abbreviated state names (period-terminated or bare for multi-word states).
STATE_ABBREV_TO_CODE: dict[str, str] = {
    "ala.": "AL", "alaska": "AK", "ariz.": "AZ", "ark.": "AR",
    "calif.": "CA", "colo.": "CO", "conn.": "CT", "del.": "DE",
    "fla.": "FL", "ga.": "GA", "hawaii": "HI", "idaho": "ID",
    "ill.": "IL", "ind.": "IN", "iowa": "IA", "kans.": "KS",
    "ky.": "KY", "la.": "LA", "maine": "ME", "md.": "MD",
    "mass.": "MA", "mich.": "MI", "minn.": "MN", "miss.": "MS",
    "mo.": "MO", "mont.": "MT", "nebr.": "NE", "nev.": "NV",
    "n.h.": "NH", "n.j.": "NJ", "n.m.": "NM", "n.y.": "NY",
    "n.c.": "NC", "n.d.": "ND", "ohio": "OH", "okla.": "OK",
    "ore.": "OR", "pa.": "PA", "r.i.": "RI", "s.c.": "SC",
    "s.d.": "SD", "tenn.": "TN", "tex.": "TX", "utah": "UT",
    "vt.": "VT", "va.": "VA", "wash.": "WA", "w.va.": "WV",
    "wis.": "WI", "wyo.": "WY", "d.c.": "DC",
}

# 2026 CSV uses full state names.
STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "washington dc": "DC",
}

# Full display names keyed by postal code (used in YAML for readability).
STATE_DISPLAY_NAMES: dict[str, str] = {
    v: k.title() for k, v in STATE_NAME_TO_CODE.items()
}
STATE_DISPLAY_NAMES["DC"] = "Washington DC"

# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

_FOOTNOTE_RE = re.compile(r"^\s*\([^)]+\)\s*$")


def parse_dollar(s: str) -> float:
    """Parse a dollar string like '$15,000' or '15000' to float. Returns 0 for n.a./empty."""
    s = s.strip().rstrip()
    if not s or s.lower() in ("n.a.", "n/a", "none", "na"):
        return 0.0
    s = s.lstrip("$").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_percent(s: str) -> float:
    """Parse '10%' or '1.00%' to 0.10. Returns 0 for n.a./empty/none."""
    s = s.strip()
    if not s or s.lower() in ("n.a.", "none", "na"):
        return 0.0
    s = s.rstrip("%")
    try:
        return round(float(s) / 100.0, 6)
    except ValueError:
        return 0.0


def parse_bracket_range(cell: str) -> tuple[float, Optional[float]]:
    """
    Parse a federal bracket range cell.
    '$0 to $11,925'  → (0.0, 11925.0)
    '$626,350 or more' → (626350.0, None)
    """
    cell = cell.strip().strip('"')
    if " to " in cell:
        lo, hi = cell.split(" to ", 1)
        return parse_dollar(lo), parse_dollar(hi)
    if " or more" in cell.lower():
        lo = cell.lower().replace(" or more", "")
        return parse_dollar(lo), None
    # Fallback
    return parse_dollar(cell), None


def parse_deduction_or_exemption(std_cell: str, exemp_cell: str) -> float:
    """
    Return the best available deduction as a dollar amount.
    - Values containing 'credit' are ignored (credits are not modelled as deductions).
    - 'n.a.', blank, 'none' → 0.
    - Returns max(standard_deduction, personal_exemption).
    """
    def _extract(cell: str) -> float:
        if "credit" in cell.lower():
            return 0.0
        return parse_dollar(cell)

    std = _extract(std_cell)
    exemp = _extract(exemp_cell)
    return max(std, exemp)


def build_brackets(rate_start_pairs: list[tuple[float, float]]) -> list[list]:
    """
    Convert a list of (rate, bracket_start) pairs to [[start, end, rate], ...].
    Sorted by bracket_start; last bracket has end = None (infinity).
    """
    if not rate_start_pairs:
        return []
    sorted_pairs = sorted(rate_start_pairs, key=lambda x: x[1])
    result = []
    for i, (rate, start) in enumerate(sorted_pairs):
        end = sorted_pairs[i + 1][1] if i + 1 < len(sorted_pairs) else None
        result.append([start, end, rate])
    return result


def _csv_path(category: str, year: int) -> str:
    if year not in CSV_FILES.get(category, {}):
        raise ValueError(f"No CSV mapped for {category}/{year}")
    filename = CSV_FILES[category][year]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing CSV: {filename}\n"
            f"Expected at: {path}"
        )
    return path


# ---------------------------------------------------------------------------
# Federal CSV parsers
# ---------------------------------------------------------------------------

_FILING_STATUS_COLS: dict[str, str] = {
    "single":            "For Single Filers",
    "married":           "For Married Individuals Filing Joint Returns",
    "head_of_household": "For Heads of Households",
}


def parse_federal_brackets(year: int, filing_status: str) -> list[list]:
    path = _csv_path("federal_brackets", year)
    col_name = _FILING_STATUS_COLS[filing_status]

    rows: list[tuple[float, Optional[float], float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        # Normalize non-breaking spaces and other whitespace variants in header cells
        header = [h.replace("\xa0", " ").strip() for h in raw_header]
        col_idx = header.index(col_name)
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            rate = parse_percent(row[0])
            start, end = parse_bracket_range(row[col_idx])
            rows.append((rate, start, end))

    # Chain brackets so the start of row[i+1] equals the end of row[i].
    # The CSVs sometimes have a $1 gap (e.g. 2026: "$0 to $12,400" then "$12,401 to ...").
    result = []
    for i, (rate, start, end) in enumerate(rows):
        actual_start = result[i - 1][1] if i > 0 else start
        result.append([actual_start, end, rate])
    return result


def parse_federal_std_deduction(year: int, filing_status: str) -> float:
    path = _csv_path("federal_std_deduction", year)
    fs_map = {
        "single":            "Single",
        "married":           "Married Filing Jointly",
        "head_of_household": "Head of Household",
    }
    target = fs_map[filing_status]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Filing Status"].strip() == target:
                return parse_dollar(row["Deduction Amount"])
    raise ValueError(f"Filing status '{filing_status}' not found in {path}")


def parse_amt_data(year: int, filing_status: str) -> dict:
    """Returns {exemption, phaseout_threshold}."""
    key = "Married Filing Jointly" if filing_status == "married" else "Unmarried Individuals"

    exemption_path = _csv_path("amt_exemptions", year)
    exemption = 0.0
    with open(exemption_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Filing Status"].strip() == key:
                exemption = parse_dollar(row["Exemption Amount"])

    phaseout_path = _csv_path("amt_phaseout", year)
    phaseout_threshold = 0.0
    with open(phaseout_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Filing Status"].strip() == key:
                phaseout_threshold = parse_dollar(row["Threshold"])

    return {"exemption": exemption, "phaseout_threshold": phaseout_threshold}


# ---------------------------------------------------------------------------
# State CSV parsers
# ---------------------------------------------------------------------------

def _is_footnote_row(cell: str) -> bool:
    """True if col[0] looks like a footnote annotation, e.g. '(a, b, c)'."""
    return bool(_FOOTNOTE_RE.match(cell.strip()))


def _strip_footnote_inline(name: str) -> str:
    """Strip trailing footnote annotations from a state name.
    'California (a, h, j)' → 'California'
    """
    return re.sub(r"\s*\([^)]+\)", "", name).strip()


def _state_entry() -> dict:
    return {
        "no_income_tax": False,
        "capital_gains_only": False,
        "capital_gains_deduction": {"single": 0.0, "married": 0.0},
        "capital_gains_brackets": [],
        "single_raw": [],   # [(rate, start), ...] accumulated during parse
        "mfj_raw": [],
        "std_ded_single": 0.0,
        "std_ded_mfj": 0.0,
    }


def _finalise_state(entry: dict) -> dict:
    """Convert raw (rate, start) pairs to [[start, end, rate]] brackets."""
    result = dict(entry)
    result["single_brackets"] = build_brackets(entry["single_raw"])
    result["mfj_brackets"]    = build_brackets(entry["mfj_raw"])
    del result["single_raw"]
    del result["mfj_raw"]
    return result


def _try_add_bracket_row(entry: dict, row: list[str]) -> None:
    """
    Given a CSV row, attempt to extract single and MFJ bracket data.
    col layout (both years): [0]state, [1]single_rate, [2]">", [3]single_start,
                              [4]mfj_rate, [5]">", [6]mfj_start, ...
    """
    try:
        rate_s_str = row[1].strip() if len(row) > 1 else ""
        brk_s_str  = row[3].strip() if len(row) > 3 else ""
        rate_m_str = row[4].strip() if len(row) > 4 else ""
        brk_m_str  = row[6].strip() if len(row) > 6 else ""
    except IndexError:
        return

    if rate_s_str and "%" in rate_s_str:
        rate_s = parse_percent(rate_s_str)
        brk_s  = parse_dollar(brk_s_str)
        entry["single_raw"].append((rate_s, brk_s))

    if rate_m_str and "%" in rate_m_str:
        rate_m = parse_percent(rate_m_str)
        brk_m  = parse_dollar(brk_m_str)
        entry["mfj_raw"].append((rate_m, brk_m))


def parse_state_brackets_2025(path: str) -> dict[str, dict]:
    """
    Parse the 2025 state CSV (abbreviated names, 2-row header, footnote continuation rows).
    Returns dict keyed by 2-letter state postal code.
    """
    states: dict[str, dict] = {}
    current_code: Optional[str] = None
    current_entry: Optional[dict] = None

    def _save_current():
        nonlocal current_code, current_entry
        if current_code and current_entry:
            states[current_code] = _finalise_state(current_entry)
        current_code = None
        current_entry = None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip group header row
        next(reader)  # skip column names row

        for row in reader:
            # Pad short rows
            while len(row) < 12:
                row.append("")

            col0 = row[0].strip()

            # Skip completely blank rows (state separator lines in the CSV)
            if not any(c.strip() for c in row):
                continue

            # --- Determine if this is an anchor (new state) or continuation row ---
            is_footnote = _is_footnote_row(col0)
            is_anchor   = bool(col0) and not is_footnote

            if is_anchor:
                _save_current()

                # Strip any inline footnote from the name and normalise
                raw_name = _strip_footnote_inline(col0).lower().rstrip(".")
                # The abbrev table keys are like "calif." so we try with and without
                code = STATE_ABBREV_TO_CODE.get(raw_name + ".") or \
                       STATE_ABBREV_TO_CODE.get(raw_name)
                if code is None:
                    print(f"  [warn] 2025: unknown state name '{col0}' — skipping",
                          file=sys.stderr)
                    continue

                current_code  = code
                current_entry = _state_entry()
                current_entry["name"] = STATE_DISPLAY_NAMES.get(code, code)

                # Check for no-income-tax states
                rate_cell = row[1].strip().lower()
                if rate_cell == "none":
                    current_entry["no_income_tax"] = True
                    continue

                # Washington capital-gains-only special case
                if "capital gains income only" in rate_cell:
                    current_entry["capital_gains_only"] = True
                    cg_ded = parse_dollar(row[7])
                    current_entry["capital_gains_deduction"]["single"]  = cg_ded
                    current_entry["capital_gains_deduction"]["married"] = parse_dollar(row[8]) or cg_ded
                    # Rate is in the string, e.g. "7.0% on capital gains income only"
                    m = re.search(r"([\d.]+)%", row[1])
                    if m:
                        rate = float(m.group(1)) / 100
                        current_entry["capital_gains_brackets"] = [[0, None, rate]]
                    # Extract deductions/exemptions for the record
                    current_entry["std_ded_single"] = parse_deduction_or_exemption(row[7], row[9])
                    current_entry["std_ded_mfj"]    = parse_deduction_or_exemption(row[8], row[10])
                    continue

                # Normal state: extract first bracket + deduction
                _try_add_bracket_row(current_entry, row)
                current_entry["std_ded_single"] = parse_deduction_or_exemption(row[7], row[9])
                current_entry["std_ded_mfj"]    = parse_deduction_or_exemption(row[8], row[10])

            else:
                # Continuation row (empty col0 or footnote annotation)
                if current_entry and not current_entry.get("no_income_tax") \
                        and not current_entry.get("capital_gains_only"):
                    _try_add_bracket_row(current_entry, row)

    _save_current()
    return states


def parse_state_brackets_2026(path: str) -> dict[str, dict]:
    """
    Parse the 2026 state CSV (full names, '- StateName' continuation rows).
    Returns dict keyed by 2-letter state postal code.
    """
    states: dict[str, dict] = {}
    current_code: Optional[str] = None
    current_entry: Optional[dict] = None
    wa_first_continuation = False   # flag to grab WA deduction from first continuation row

    def _save_current():
        nonlocal current_code, current_entry
        if current_code and current_entry:
            states[current_code] = _finalise_state(current_entry)
        current_code = None
        current_entry = None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            while len(row) < 12:
                row.append("")

            col0 = row[0].strip()
            if not col0:
                continue

            is_continuation = col0.startswith("- ")
            is_anchor       = not is_continuation

            if is_anchor:
                _save_current()
                wa_first_continuation = False

                raw_name = _strip_footnote_inline(col0).lower()
                code = STATE_NAME_TO_CODE.get(raw_name)
                if code is None:
                    print(f"  [warn] 2026: unknown state name '{col0}' — skipping",
                          file=sys.stderr)
                    continue

                current_code  = code
                current_entry = _state_entry()
                current_entry["name"] = STATE_DISPLAY_NAMES.get(code, code)

                rate_cell = row[1].strip().lower()
                if rate_cell == "none":
                    current_entry["no_income_tax"] = True
                    continue

                if "capital gains income only" in rate_cell:
                    current_entry["capital_gains_only"] = True
                    wa_first_continuation = True
                    continue

                # Normal anchor: first bracket + deductions
                _try_add_bracket_row(current_entry, row)
                current_entry["std_ded_single"] = parse_deduction_or_exemption(row[7], row[9])
                current_entry["std_ded_mfj"]    = parse_deduction_or_exemption(row[8], row[10])

            else:
                # Continuation row
                if current_entry is None:
                    continue

                if current_entry.get("capital_gains_only") and wa_first_continuation:
                    # First continuation row for WA: grab rate + deduction
                    wa_first_continuation = False
                    rate_s_str = row[1].strip()
                    rate_m_str = row[4].strip()
                    cg_ded_s = parse_dollar(row[7])
                    cg_ded_m = parse_dollar(row[8]) or cg_ded_s
                    current_entry["capital_gains_deduction"]["single"]  = cg_ded_s
                    current_entry["capital_gains_deduction"]["married"] = cg_ded_m
                    # Build capital gains brackets from continuation rows
                    rate = parse_percent(rate_s_str or rate_m_str)
                    brk_start = parse_dollar(row[3])
                    current_entry["capital_gains_brackets"].append([brk_start, None, rate])
                    continue

                if current_entry.get("capital_gains_only"):
                    # Additional WA capital gains bracket (e.g. 9% above $1M in 2026)
                    rate_s_str = row[1].strip()
                    rate = parse_percent(rate_s_str)
                    brk_start = parse_dollar(row[3])
                    if rate > 0:
                        # Fix the previous bracket's end and add this one
                        prev = current_entry["capital_gains_brackets"]
                        if prev:
                            prev[-1][1] = brk_start
                        current_entry["capital_gains_brackets"].append([brk_start, None, rate])
                    continue

                if not current_entry.get("no_income_tax"):
                    _try_add_bracket_row(current_entry, row)

    _save_current()
    return states


# ---------------------------------------------------------------------------
# LTCG CSV parser
# ---------------------------------------------------------------------------

# The LTCG CSV stores the lower threshold for each rate, not a range.
# Format:  "", "For Unmarried Individuals...", "For Married...", "For Heads..."
#          "0%",  "$0",       "$0",      "$0"
#          "15%", "$48,350",  "$96,700", "$64,750"
#          "20%", "$533,400", "$600,050","$566,700"

_LTCG_COL_MAP: dict[str, str] = {
    "single":            "For Unmarried Individuals, Taxable Income Over",
    "married":           "For Married Individuals Filing Joint Returns, Taxable Income Over",
    "head_of_household": "For Heads of Households, Taxable Income Over",
}


def parse_ltcg_brackets(year: int, filing_status: str) -> list[list]:
    path = _csv_path("ltcg_brackets", year)
    target_col = _LTCG_COL_MAP[filing_status]

    rows: list[tuple[float, float]] = []  # (rate, lower_bound)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.replace("\xa0", " ").strip() for h in next(reader)]
        col_idx = header.index(target_col)
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            rate = parse_percent(row[0])
            lower_bound = parse_dollar(row[col_idx])
            rows.append((rate, lower_bound))

    # Sort by lower bound and build [[start, end, rate], ...] with end=None for last.
    rows.sort(key=lambda x: x[1])
    result = []
    for i, (rate, start) in enumerate(rows):
        end = rows[i + 1][1] if i + 1 < len(rows) else None
        result.append([start, end, rate])
    return result


# ---------------------------------------------------------------------------
# Assemble full data structure
# ---------------------------------------------------------------------------

def build_year_data(year: int) -> dict:
    print(f"  Building federal data for {year}...")
    federal: dict = {}

    filing_statuses = ["single", "married", "head_of_household"]

    federal["brackets"] = {
        fs: parse_federal_brackets(year, fs) for fs in filing_statuses
    }
    federal["standard_deduction"] = {
        fs: parse_federal_std_deduction(year, fs) for fs in filing_statuses
    }
    federal["ltcg_brackets"] = {
        fs: parse_ltcg_brackets(year, fs) for fs in filing_statuses
    }
    federal["amt"] = {
        fs: parse_amt_data(year, fs) for fs in filing_statuses
    }
    federal["amt_rate_breakpoint"] = AMT_RATE_BREAKPOINTS[year]
    federal["amt_phaseout_rate"] = AMT_PHASEOUT_RATES[year]
    federal["niit_threshold"] = NIIT_THRESHOLDS

    print(f"  Building state data for {year}...")
    if year == 2025:
        raw_states = parse_state_brackets_2025(_csv_path("state_brackets", year))
    else:
        raw_states = parse_state_brackets_2026(_csv_path("state_brackets", year))

    # Convert to the final YAML-friendly structure
    states: dict = {}
    for code, sd in raw_states.items():
        states[code] = {
            "name":                   sd.get("name", code),
            "no_income_tax":          sd["no_income_tax"],
            "capital_gains_only":     sd["capital_gains_only"],
            "capital_gains_deduction": sd["capital_gains_deduction"],
            "capital_gains_brackets": sd["capital_gains_brackets"],
            "brackets": {
                "single":  sd.get("single_brackets", []),
                "married": sd.get("mfj_brackets", []),
            },
            "standard_deduction": {
                "single":  sd["std_ded_single"],
                "married": sd["std_ded_mfj"],
            },
        }

    print(f"  → {len(states)} states loaded for {year}")
    return {"federal": federal, "states": states}


def build_all() -> dict:
    data: dict = {"years": {}}
    for year in SUPPORTED_YEARS:
        print(f"\nProcessing {year}...")
        data["years"][year] = build_year_data(year)
    return data


# ---------------------------------------------------------------------------
# YAML serialisation
# ---------------------------------------------------------------------------

def _yaml_representer_none(dumper: yaml.Dumper, _data: None) -> yaml.Node:
    """Represent Python None as YAML null (infinity sentinel)."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


def write_yaml(data: dict, output_path: str) -> None:
    dumper = yaml.Dumper
    dumper.add_representer(type(None), _yaml_representer_none)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=dumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=True)
    print(f"\nWrote: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tax_data.yaml from source CSVs."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "tax_data.yaml"),
        help="Output YAML path (default: tax_data.yaml in the same directory)",
    )
    args = parser.parse_args()

    print("Building tax data from CSVs...")
    data = build_all()
    write_yaml(data, args.output)

    # Quick summary
    print("\nSummary:")
    for year in SUPPORTED_YEARS:
        states = data["years"][year]["states"]
        no_tax = [c for c, s in states.items() if s["no_income_tax"]]
        cg_only = [c for c, s in states.items() if s["capital_gains_only"]]
        normal = [c for c, s in states.items()
                  if not s["no_income_tax"] and not s["capital_gains_only"]]
        print(f"  {year}: {len(normal)} normal states, "
              f"{len(no_tax)} no-income-tax, {len(cg_only)} CG-only")


if __name__ == "__main__":
    main()
