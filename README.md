# Stock Tax Estimator (2025 & 2026)

> **Disclaimer:** This program is for **educational purposes only** and is not intended to replace professional tax advice or the services of a CPA or qualified tax professional. Tax laws are complex, subject to change, and vary by individual circumstance. The calculations here are estimates based on published tax brackets and may not reflect your actual tax liability. Always consult a licensed tax professional before making financial decisions.

## Description

This Python program provides **estimated** federal and state income tax calculations for salary and long-term capital gains from stock sales, supporting both 2025 and 2026 tax years.

**For 2026, the program was massively overhauled** to support:

- **Multi-state support** — covers all U.S. states with income taxes, including Washington state's capital-gains-only tax regime and no-income-tax states (TX, FL, NV, etc.)
- **All three federal filing statuses** — Single, Married Filing Jointly, and Head of Household
- **Alternative Minimum Tax (AMT)** — basic AMT calculation including AMTI, exemption phaseout, the 26%/28% rate split, and LTCG preferential rates under AMT. Note: 2026 AMT is significantly more aggressive due to the OBBBA (phaseout rate increases from 25¢/$ to 50¢/$)
- **Data-driven architecture** — tax bracket data is parsed from CSV files (Tax Foundation data) into a unified `tax_data.yaml`, making it easy to update when new brackets are published

**Key Features:**

- Federal income tax using progressive brackets
- Federal long-term capital gains tax (stacked on top of ordinary income)
- Net Investment Income Tax (NIIT) at 3.8%
- Alternative Minimum Tax (AMT) with full phaseout calculation
- State income tax for all 50 states (where applicable)
- Effective tax rate and estimated take-home pay

**Limitations / Simplified Assumptions:**

- Assumes standard deduction (not itemized)
- Assumes all stock sales qualify for long-term capital gains rates
- Does not consider:
  - Social Security, Medicare, or payroll taxes
  - Charitable donations or other itemized deductions
  - 401(k), IRA, or other tax-deferred contributions
  - Other income sources (interest, dividends, rental income) — include them in your salary number
  - Dependents or other credits
  - State AMT (some states have their own AMT)
  - Married Filing Separately status
  - Alternative minimum tax for states

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and requires Python 3.12+.

```bash
poetry install
```

## How to Use

### Step 1 (Optional): Build the tax data (run once, or when CSVs are updated)

```bash
poetry run python build_tax_data.py
```

This reads all the tax bracket CSVs and writes `tax_data.yaml`.

### Step 2: Run the estimator

```bash
poetry run python stock_tax_estimator.py <salary> <stock_sales> [--year YEAR] [--state STATE] [--filing-status {single,married,head_of_household}]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `salary` | required | W-2 or ordinary income |
| `stock_sales` | required | Long-term capital gains from stock sales |
| `--year` | current year | Tax year: `2025` or `2026` |
| `--state` | (none) | 2-letter postal code, e.g. `CA`, `NY`, `TX` |
| `--filing-status` | `single` | `single`, `married`, or `head_of_household` |

**Examples:**

```bash
# California single filer, 2025
poetry run python stock_tax_estimator.py 200000 100000 --year 2025 --state CA

# New York married filing jointly, 2026
poetry run python stock_tax_estimator.py 300000 500000 --year 2026 --state NY --filing-status married

# Federal only (no state), head of household
poetry run python stock_tax_estimator.py 150000 50000 --year 2025 --filing-status head_of_household

# Texas (no state income tax)
poetry run python stock_tax_estimator.py 500000 1000000 --year 2026 --state TX
```

## Testing

The project includes a pytest test suite in [tests/test_tax_data.py](tests/test_tax_data.py) that covers both the integrity of the generated `tax_data.yaml` and the correctness of the calculation functions.

### Running the tests

```bash
poetry run pytest tests/ -v
```

### What the tests cover

The suite is organized into nine test classes:

**YAML data integrity**

| Test class | What it checks |
|---|---|
| `TestYamlTopLevel` | Both years (2025, 2026) are present; each has `federal` and `states` sections |
| `TestFederalStructure` | All required federal keys exist; all three filing statuses are present in every sub-section (brackets, LTCG brackets, standard deduction, AMT) |
| `TestBracketIntegrity` | Every bracket list starts at $0, is contiguous (no gaps or overlaps between adjacent brackets), ends with `None` (infinity), has rates between 0–100%, and has the expected count (7 ordinary brackets, 3 LTCG brackets) |
| `TestKnownFederalValues` | Exact spot-checks against published IRS / Tax Foundation tables: standard deductions, LTCG thresholds (0%/15%/20% breakpoints), AMT exemptions and phaseout thresholds, the 26%/28% rate breakpoints, and NIIT thresholds for both years |
| `TestKnownStateValues` | At least 40 states loaded per year; key states present (CA, TX, NY, WA, FL); TX and FL flagged as no-income-tax; CA deduction values and 13.3% top rate; WA capital-gains-only flag, $270K/$278K deductions, single 7% bracket in 2025 and two-bracket (7%/9%) structure in 2026; CA and NY bracket continuity |

**Calculation correctness**

| Test class | What it checks |
|---|---|
| `TestApplyBrackets` | Core bracket engine edge cases: zero income, income within one bracket, income spanning two brackets, `None` end treated as infinity |
| `TestFederalIncomeTax` | Standard deduction is subtracted before computing tax; income below the deduction yields $0; MFJ deduction is double the single deduction |
| `TestNiit` | No NIIT below threshold; partial NIIT when gains push MAGI over the limit; NIIT capped at gains amount; MFJ's higher $250K threshold |
| `TestLtcgTax` | LTCG stacking rule: gains entirely in 0% zone, entirely in 15% zone, straddling both; zero gains returns zero |
| `TestAmt` | No additional AMT at low income; AMTI equals salary + stock sales (no standard deduction); exemption fully phases out at very high AMTI; 2026 50¢/$ phaseout rate leaves a smaller exemption than 2025's 25¢/$ at the same income level |
| `TestStateTax` | TX returns zero state tax; CA returns positive tax; WA applies 7% only above the CG deduction threshold; unknown state codes return zero; Head of Household uses single-filer state brackets |
| `TestCalculateTaxesOrchestrator` | All expected output keys are present; take-home pay equals gross income minus total tax; no-state omits state tax; invalid year raises an error; effective rate is between 0% and 100% |
