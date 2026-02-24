"""
tests/test_tax_data.py
----------------------
Tests for tax_data.yaml structure and integrity, plus smoke tests for the
calculation functions in stock_tax_estimator.py.

Run with:
    poetry run pytest tests/ -v
"""

import os
import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(DATA_DIR, "tax_data.yaml")
SUPPORTED_YEARS = [2025, 2026]
FILING_STATUSES = ["single", "married", "head_of_household"]


@pytest.fixture(scope="session")
def tax_data():
    """Load tax_data.yaml once for the entire test session."""
    assert os.path.isfile(YAML_PATH), (
        f"tax_data.yaml not found at {YAML_PATH}. "
        "Run:  poetry run python build_tax_data.py"
    )
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def estimator():
    """Import the estimator module (loads tax_data.yaml internally)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stock_tax_estimator",
        os.path.join(DATA_DIR, "stock_tax_estimator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Top-level YAML structure
# ---------------------------------------------------------------------------

class TestYamlTopLevel:
    def test_years_key_exists(self, tax_data):
        assert "years" in tax_data

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    def test_year_present(self, tax_data, year):
        assert year in tax_data["years"], f"Year {year} missing from tax_data.yaml"

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    def test_year_has_federal_and_states(self, tax_data, year):
        year_data = tax_data["years"][year]
        assert "federal" in year_data
        assert "states" in year_data


# ---------------------------------------------------------------------------
# 2. Federal section structure
# ---------------------------------------------------------------------------

REQUIRED_FEDERAL_KEYS = [
    "brackets",
    "standard_deduction",
    "ltcg_brackets",
    "amt",
    "amt_rate_breakpoint",
    "amt_phaseout_rate",
    "niit_threshold",
]


class TestFederalStructure:
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("key", REQUIRED_FEDERAL_KEYS)
    def test_federal_key_present(self, tax_data, year, key):
        assert key in tax_data["years"][year]["federal"], (
            f"Missing federal key '{key}' for {year}"
        )

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_all_filing_statuses_in_brackets(self, tax_data, year, fs):
        assert fs in tax_data["years"][year]["federal"]["brackets"]

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_all_filing_statuses_in_ltcg_brackets(self, tax_data, year, fs):
        assert fs in tax_data["years"][year]["federal"]["ltcg_brackets"]

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_all_filing_statuses_in_standard_deduction(self, tax_data, year, fs):
        assert fs in tax_data["years"][year]["federal"]["standard_deduction"]

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_all_filing_statuses_in_amt(self, tax_data, year, fs):
        amt = tax_data["years"][year]["federal"]["amt"]
        assert fs in amt
        assert "exemption" in amt[fs]
        assert "phaseout_threshold" in amt[fs]


# ---------------------------------------------------------------------------
# 3. Bracket integrity — format, ordering, continuity
# ---------------------------------------------------------------------------

def _check_bracket_list(brackets, label):
    """Shared assertions for any [[start, end, rate], ...] list."""
    assert isinstance(brackets, list), f"{label}: expected list"
    assert len(brackets) > 0, f"{label}: bracket list is empty"

    for i, bracket in enumerate(brackets):
        assert len(bracket) == 3, f"{label}[{i}]: expected [start, end, rate]"
        start, end, rate = bracket
        assert start >= 0, f"{label}[{i}]: start must be non-negative"
        assert 0.0 <= rate <= 1.0, f"{label}[{i}]: rate {rate} out of [0, 1]"
        if end is not None:
            assert end > start, f"{label}[{i}]: end must be > start"

    # First bracket must start at 0
    assert brackets[0][0] == 0.0, f"{label}: first bracket must start at 0"

    # Brackets must be contiguous: end[i] == start[i+1]
    for i in range(len(brackets) - 1):
        _, end_i, _ = brackets[i]
        start_next, _, _ = brackets[i + 1]
        assert end_i == start_next, (
            f"{label}: gap/overlap between bracket {i} and {i+1}: "
            f"end={end_i}, next start={start_next}"
        )

    # Last bracket must have None end (infinity)
    assert brackets[-1][1] is None, f"{label}: last bracket end must be None (infinity)"


class TestBracketIntegrity:
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_federal_ordinary_brackets(self, tax_data, year, fs):
        brackets = tax_data["years"][year]["federal"]["brackets"][fs]
        _check_bracket_list(brackets, f"{year} federal ordinary {fs}")

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_federal_ltcg_brackets(self, tax_data, year, fs):
        brackets = tax_data["years"][year]["federal"]["ltcg_brackets"][fs]
        _check_bracket_list(brackets, f"{year} federal LTCG {fs}")

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_federal_ltcg_has_three_brackets(self, tax_data, year, fs):
        """Federal LTCG always has exactly three brackets: 0%, 15%, 20%."""
        brackets = tax_data["years"][year]["federal"]["ltcg_brackets"][fs]
        assert len(brackets) == 3, (
            f"{year} LTCG {fs}: expected 3 brackets (0/15/20%), got {len(brackets)}"
        )
        assert brackets[0][2] == 0.0,   "First LTCG bracket must be 0%"
        assert brackets[1][2] == 0.15,  "Second LTCG bracket must be 15%"
        assert brackets[2][2] == 0.20,  "Third LTCG bracket must be 20%"

    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("fs", FILING_STATUSES)
    def test_federal_ordinary_has_seven_brackets(self, tax_data, year, fs):
        """Federal ordinary income always has exactly 7 brackets (10–37%)."""
        brackets = tax_data["years"][year]["federal"]["brackets"][fs]
        assert len(brackets) == 7, (
            f"{year} federal {fs}: expected 7 ordinary brackets, got {len(brackets)}"
        )


# ---------------------------------------------------------------------------
# 4. Known federal values from published IRS / Tax Foundation tables
# ---------------------------------------------------------------------------

class TestKnownFederalValues:
    # Standard deductions
    def test_2025_std_ded_single(self, tax_data):
        assert tax_data["years"][2025]["federal"]["standard_deduction"]["single"] == 15000.0

    def test_2025_std_ded_married(self, tax_data):
        assert tax_data["years"][2025]["federal"]["standard_deduction"]["married"] == 30000.0

    def test_2025_std_ded_hoh(self, tax_data):
        assert tax_data["years"][2025]["federal"]["standard_deduction"]["head_of_household"] == 22500.0

    def test_2026_std_ded_single(self, tax_data):
        assert tax_data["years"][2026]["federal"]["standard_deduction"]["single"] == 16100.0

    def test_2026_std_ded_married(self, tax_data):
        assert tax_data["years"][2026]["federal"]["standard_deduction"]["married"] == 32200.0

    # LTCG breakpoints
    def test_2025_ltcg_single_15pct_threshold(self, tax_data):
        brackets = tax_data["years"][2025]["federal"]["ltcg_brackets"]["single"]
        assert brackets[1][0] == 48350.0, "2025 single LTCG 15% starts at $48,350"

    def test_2025_ltcg_single_20pct_threshold(self, tax_data):
        brackets = tax_data["years"][2025]["federal"]["ltcg_brackets"]["single"]
        assert brackets[2][0] == 533400.0, "2025 single LTCG 20% starts at $533,400"

    def test_2025_ltcg_married_15pct_threshold(self, tax_data):
        brackets = tax_data["years"][2025]["federal"]["ltcg_brackets"]["married"]
        assert brackets[1][0] == 96700.0, "2025 MFJ LTCG 15% starts at $96,700"

    def test_2026_ltcg_single_15pct_threshold(self, tax_data):
        brackets = tax_data["years"][2026]["federal"]["ltcg_brackets"]["single"]
        assert brackets[1][0] == 49450.0, "2026 single LTCG 15% starts at $49,450"

    def test_2026_ltcg_single_20pct_threshold(self, tax_data):
        brackets = tax_data["years"][2026]["federal"]["ltcg_brackets"]["single"]
        assert brackets[2][0] == 545500.0, "2026 single LTCG 20% starts at $545,500"

    # AMT exemptions and phaseout thresholds
    def test_2025_amt_single_exemption(self, tax_data):
        assert tax_data["years"][2025]["federal"]["amt"]["single"]["exemption"] == 88100.0

    def test_2025_amt_single_phaseout_threshold(self, tax_data):
        assert tax_data["years"][2025]["federal"]["amt"]["single"]["phaseout_threshold"] == 626350.0

    def test_2025_amt_married_exemption(self, tax_data):
        assert tax_data["years"][2025]["federal"]["amt"]["married"]["exemption"] == 137000.0

    def test_2026_amt_single_phaseout_threshold_more_aggressive(self, tax_data):
        """2026 AMT phaseout starts at $500K for single filers (OBBBA — more aggressive than 2025)."""
        assert tax_data["years"][2026]["federal"]["amt"]["single"]["phaseout_threshold"] == 500000.0

    def test_2026_amt_single_phaseout_lower_than_2025(self, tax_data):
        t2025 = tax_data["years"][2025]["federal"]["amt"]["single"]["phaseout_threshold"]
        t2026 = tax_data["years"][2026]["federal"]["amt"]["single"]["phaseout_threshold"]
        assert t2026 < t2025, "2026 AMT phaseout threshold should be lower (more aggressive)"

    # AMT rate breakpoints
    def test_2025_amt_rate_breakpoint(self, tax_data):
        assert tax_data["years"][2025]["federal"]["amt_rate_breakpoint"] == 239100

    def test_2026_amt_rate_breakpoint(self, tax_data):
        assert tax_data["years"][2026]["federal"]["amt_rate_breakpoint"] == 244500

    # AMT phaseout rates
    def test_2025_amt_phaseout_rate(self, tax_data):
        assert tax_data["years"][2025]["federal"]["amt_phaseout_rate"] == 0.25

    def test_2026_amt_phaseout_rate_more_aggressive(self, tax_data):
        """2026 phaseout rate is 50¢/$ (OBBBA) vs 25¢/$ in 2025."""
        assert tax_data["years"][2026]["federal"]["amt_phaseout_rate"] == 0.50

    # NIIT thresholds
    def test_niit_threshold_single(self, tax_data):
        for year in SUPPORTED_YEARS:
            assert tax_data["years"][year]["federal"]["niit_threshold"]["single"] == 200000

    def test_niit_threshold_married(self, tax_data):
        for year in SUPPORTED_YEARS:
            assert tax_data["years"][year]["federal"]["niit_threshold"]["married"] == 250000

    # 2025 federal ordinary bracket spot-checks
    def test_2025_federal_top_rate_37pct(self, tax_data):
        brackets = tax_data["years"][2025]["federal"]["brackets"]["single"]
        assert brackets[-1][2] == 0.37, "Top federal rate must be 37%"

    def test_2025_federal_single_10pct_ceiling(self, tax_data):
        brackets = tax_data["years"][2025]["federal"]["brackets"]["single"]
        assert brackets[0][1] == 11925.0, "2025 10% bracket ends at $11,925"


# ---------------------------------------------------------------------------
# 5. Known state values
# ---------------------------------------------------------------------------

class TestKnownStateValues:
    def test_states_dict_not_empty(self, tax_data):
        for year in SUPPORTED_YEARS:
            states = tax_data["years"][year]["states"]
            assert len(states) > 40, f"{year}: expected 40+ states, got {len(states)}"

    # Key states present in both years
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("code", ["CA", "TX", "NY", "WA", "FL"])
    def test_key_states_present(self, tax_data, year, code):
        assert code in tax_data["years"][year]["states"], (
            f"{code} missing from {year} state data"
        )

    # Texas — no income tax
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    def test_texas_no_income_tax(self, tax_data, year):
        tx = tax_data["years"][year]["states"]["TX"]
        assert tx["no_income_tax"] is True
        assert tx["name"] == "Texas"

    # Florida — no income tax
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    def test_florida_no_income_tax(self, tax_data, year):
        fl = tax_data["years"][year]["states"]["FL"]
        assert fl["no_income_tax"] is True

    # California — has brackets, correct deduction
    def test_california_2025_std_ded_single(self, tax_data):
        ca = tax_data["years"][2025]["states"]["CA"]
        assert ca["no_income_tax"] is False
        assert ca["standard_deduction"]["single"] == 5540.0

    def test_california_2025_std_ded_married(self, tax_data):
        ca = tax_data["years"][2025]["states"]["CA"]
        assert ca["standard_deduction"]["married"] == 11080.0

    def test_california_top_rate_13_3_pct(self, tax_data):
        """CA top marginal rate is 13.3% (including mental health surcharge)."""
        for year in SUPPORTED_YEARS:
            ca = tax_data["years"][year]["states"]["CA"]
            top_rate = ca["brackets"]["single"][-1][2]
            assert abs(top_rate - 0.133) < 1e-6, (
                f"{year} CA top rate expected 13.3%, got {top_rate:.4%}"
            )

    def test_california_has_multiple_brackets(self, tax_data):
        ca = tax_data["years"][2025]["states"]["CA"]
        assert len(ca["brackets"]["single"]) >= 7

    # Washington — capital gains only
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    def test_washington_capital_gains_only(self, tax_data, year):
        wa = tax_data["years"][year]["states"]["WA"]
        assert wa["capital_gains_only"] is True
        assert wa["no_income_tax"] is False

    def test_washington_2025_cg_deduction(self, tax_data):
        wa = tax_data["years"][2025]["states"]["WA"]
        assert wa["capital_gains_deduction"]["single"] == 270000.0

    def test_washington_2025_single_cg_bracket(self, tax_data):
        """2025 WA has a single 7% bracket with no upper limit."""
        wa = tax_data["years"][2025]["states"]["WA"]
        assert len(wa["capital_gains_brackets"]) == 1
        assert wa["capital_gains_brackets"][0][2] == pytest.approx(0.07)
        assert wa["capital_gains_brackets"][0][1] is None

    def test_washington_2026_has_two_cg_brackets(self, tax_data):
        """2026 WA adds a 9% bracket above $1M."""
        wa = tax_data["years"][2026]["states"]["WA"]
        brackets = wa["capital_gains_brackets"]
        assert len(brackets) == 2
        assert brackets[0][2] == pytest.approx(0.07)
        assert brackets[1][0] == 1_000_000.0
        assert brackets[1][2] == pytest.approx(0.09)
        assert brackets[1][1] is None

    def test_washington_2026_cg_deduction_higher_than_2025(self, tax_data):
        ded_2025 = tax_data["years"][2025]["states"]["WA"]["capital_gains_deduction"]["single"]
        ded_2026 = tax_data["years"][2026]["states"]["WA"]["capital_gains_deduction"]["single"]
        assert ded_2026 > ded_2025, "2026 WA CG deduction should be inflation-adjusted upward"

    # State bracket integrity for CA and NY
    @pytest.mark.parametrize("year", SUPPORTED_YEARS)
    @pytest.mark.parametrize("code", ["CA", "NY"])
    @pytest.mark.parametrize("fs", ["single", "married"])
    def test_normal_state_bracket_integrity(self, tax_data, year, code, fs):
        state = tax_data["years"][year]["states"][code]
        brackets = state["brackets"][fs]
        _check_bracket_list(brackets, f"{year} {code} {fs}")


# ---------------------------------------------------------------------------
# 6. Calculation smoke tests (via stock_tax_estimator.py functions)
# ---------------------------------------------------------------------------

class TestApplyBrackets:
    def test_zero_income(self, estimator):
        brackets = [[0, 10000, 0.10], [10000, None, 0.20]]
        assert estimator._apply_brackets(0, brackets) == 0.0

    def test_within_first_bracket(self, estimator):
        brackets = [[0, 10000, 0.10], [10000, None, 0.20]]
        assert estimator._apply_brackets(5000, brackets) == pytest.approx(500.0)

    def test_spans_two_brackets(self, estimator):
        brackets = [[0, 10000, 0.10], [10000, None, 0.20]]
        # 10000 * 0.10 + 5000 * 0.20 = 1000 + 1000 = 2000
        assert estimator._apply_brackets(15000, brackets) == pytest.approx(2000.0)

    def test_none_end_treated_as_infinity(self, estimator):
        brackets = [[0, None, 0.10]]
        assert estimator._apply_brackets(1_000_000, brackets) == pytest.approx(100_000.0)


class TestFederalIncomeTax:
    def test_zero_salary_zero_tax(self, estimator):
        result = estimator._calculate_federal_income_tax(0, 2025, "single")
        assert result["tax"] == 0.0

    def test_below_standard_deduction_zero_tax(self, estimator):
        """Income below the std deduction should yield $0 tax."""
        result = estimator._calculate_federal_income_tax(14000, 2025, "single")
        assert result["tax"] == 0.0
        assert result["taxable_income"] == 0.0

    def test_standard_deduction_applied(self, estimator):
        result = estimator._calculate_federal_income_tax(50000, 2025, "single")
        assert result["standard_deduction"] == 15000.0
        assert result["taxable_income"] == 35000.0

    def test_married_double_deduction(self, estimator):
        single = estimator._calculate_federal_income_tax(100000, 2025, "single")
        married = estimator._calculate_federal_income_tax(100000, 2025, "married")
        assert married["standard_deduction"] == 30000.0
        assert married["taxable_income"] < single["taxable_income"]


class TestNiit:
    def test_below_threshold_no_niit(self, estimator):
        # salary + stock_sales < $200K — no NIIT
        niit = estimator._calculate_niit(100000, 50000, 2025, "single")
        assert niit == 0.0

    def test_stock_sales_push_over_threshold(self, estimator):
        # salary $190K, stock_sales $20K → MAGI $210K; excess over $200K = $10K
        # niit_base = min(20000, 10000) = 10000; NIIT = 10000 * 0.038 = 380
        niit = estimator._calculate_niit(190000, 20000, 2025, "single")
        assert niit == pytest.approx(380.0)

    def test_large_gains_niit_capped_at_gains(self, estimator):
        # salary $300K (already over threshold), stock_sales $50K
        # niit_base = min(50000, 350000 - 200000) = min(50000, 150000) = 50000
        niit = estimator._calculate_niit(300000, 50000, 2025, "single")
        assert niit == pytest.approx(50000 * 0.038)

    def test_married_higher_threshold(self, estimator):
        # salary $240K, gains $10K — over single $200K threshold but under MFJ $250K
        niit_single  = estimator._calculate_niit(240000, 10000, 2025, "single")
        niit_married = estimator._calculate_niit(240000, 10000, 2025, "married")
        assert niit_single > 0
        assert niit_married == 0.0


class TestLtcgTax:
    def test_gains_entirely_in_0pct_zone(self, estimator):
        # Ordinary base $0, gains $30,000 — all below $48,350 → 0% tax
        tax = estimator._calculate_ltcg_tax(0, 30000, 2025, "single")
        assert tax == 0.0

    def test_gains_entirely_in_15pct_zone(self, estimator):
        # Ordinary base $60,000 (already above 0% zone), gains $10,000 → all at 15%
        tax = estimator._calculate_ltcg_tax(60000, 10000, 2025, "single")
        assert tax == pytest.approx(10000 * 0.15)

    def test_gains_straddle_0_and_15_pct(self, estimator):
        # Ordinary base $0, gains $100,000
        # $48,350 at 0%; remaining $51,650 at 15%
        tax = estimator._calculate_ltcg_tax(0, 100000, 2025, "single")
        assert tax == pytest.approx(51650 * 0.15)

    def test_zero_gains_zero_tax(self, estimator):
        assert estimator._calculate_ltcg_tax(200000, 0, 2025, "single") == 0.0


class TestAmt:
    def test_no_amt_at_low_income(self, estimator):
        """Low income: regular tax > TMT, so additional AMT should be $0."""
        result = estimator._calculate_amt(
            salary=100000, stock_sales=0,
            federal_income_tax=17000, federal_capital_gains_tax=0,
            year=2025, filing_status="single",
        )
        assert result["additional_amt"] == 0.0
        assert result["amt_applies"] is False

    def test_amti_equals_salary_plus_gains(self, estimator):
        """AMTI = salary + stock_sales (no standard deduction)."""
        result = estimator._calculate_amt(
            salary=200000, stock_sales=50000,
            federal_income_tax=40000, federal_capital_gains_tax=7500,
            year=2025, filing_status="single",
        )
        assert result["amti"] == 250000.0

    def test_exemption_fully_phased_out_at_very_high_income(self, estimator):
        """At AMTI far above phaseout threshold, exemption should reduce to $0."""
        # 2025 single: exemption=$88,100, phaseout starts $626,350, rate=0.25
        # Full phaseout at $626,350 + $88,100/0.25 = $626,350 + $352,400 = $978,750
        result = estimator._calculate_amt(
            salary=1_500_000, stock_sales=0,
            federal_income_tax=500000, federal_capital_gains_tax=0,
            year=2025, filing_status="single",
        )
        assert result["reduced_exemption"] == 0.0

    def test_2026_phaseout_more_aggressive(self, estimator):
        """
        At the same AMTI, the 2026 50¢/$ phaseout should leave a smaller exemption
        than the 2025 25¢/$ phaseout.
        """
        kwargs = dict(salary=700000, stock_sales=0,
                      federal_income_tax=220000, federal_capital_gains_tax=0,
                      filing_status="single")
        r2025 = estimator._calculate_amt(year=2025, **kwargs)
        r2026 = estimator._calculate_amt(year=2026, **kwargs)
        assert r2026["reduced_exemption"] < r2025["reduced_exemption"]


class TestStateTax:
    def test_texas_no_state_tax(self, estimator):
        result = estimator._calculate_state_tax(200000, 100000, 2025, "TX", "single")
        assert result["total_state_tax"] == 0.0
        assert result["no_income_tax"] is True

    def test_california_state_tax_positive(self, estimator):
        result = estimator._calculate_state_tax(200000, 100000, 2025, "CA", "single")
        assert result["total_state_tax"] > 0
        assert result["no_income_tax"] is False

    def test_washington_capital_gains_only(self, estimator):
        # $500K gains, $270K deduction → $230K taxable at 7% = $16,100
        result = estimator._calculate_state_tax(200000, 500000, 2025, "WA", "single")
        assert result["capital_gains_only"] is True
        assert result["total_state_tax"] == pytest.approx(230000 * 0.07)

    def test_washington_below_deduction_no_tax(self, estimator):
        result = estimator._calculate_state_tax(100000, 200000, 2025, "WA", "single")
        assert result["total_state_tax"] == 0.0

    def test_unknown_state_returns_zero(self, estimator):
        result = estimator._calculate_state_tax(200000, 100000, 2025, "ZZ", "single")
        assert result["total_state_tax"] == 0.0

    def test_head_of_household_uses_single_state_brackets(self, estimator):
        """HoH uses single brackets at the state level (no HoH in state data)."""
        single = estimator._calculate_state_tax(200000, 50000, 2025, "CA", "single")
        hoh    = estimator._calculate_state_tax(200000, 50000, 2025, "CA", "head_of_household")
        assert single["total_state_tax"] == hoh["total_state_tax"]


class TestCalculateTaxesOrchestrator:
    def test_returns_all_expected_keys(self, estimator):
        result = estimator.calculate_taxes(200000, 100000, 2025, "CA", "single")
        for key in ["total_tax", "total_federal_tax", "total_state_tax",
                    "effective_tax_rate", "take_home_pay", "federal_income_tax",
                    "federal_capital_gains_tax", "niit", "amt"]:
            assert key in result, f"Missing key: {key}"

    def test_take_home_pay_equals_gross_minus_total_tax(self, estimator):
        r = estimator.calculate_taxes(200000, 100000, 2025, "CA", "single")
        assert r["take_home_pay"] == pytest.approx(r["gross_income"] - r["total_tax"])

    def test_no_state_omits_state_tax(self, estimator):
        r = estimator.calculate_taxes(200000, 100000, 2025, None, "single")
        assert r["total_state_tax"] == 0.0

    def test_invalid_year_raises(self, estimator):
        with pytest.raises((ValueError, SystemExit)):
            estimator.calculate_taxes(200000, 100000, 2000, "CA", "single")

    def test_effective_rate_between_0_and_100(self, estimator):
        for state in ["CA", "TX", None]:
            r = estimator.calculate_taxes(300000, 200000, 2025, state, "single")
            assert 0 < r["effective_tax_rate"] < 100
