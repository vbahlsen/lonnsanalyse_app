from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

import analysis


# --- name_matches ---


@pytest.mark.parametrize(
    "query",
    ["ingrid", "ING", "ing hau", "hau ing", "Ingrid Haug", "mar", "sol"],
)
def test_name_matches_prefix_of_any_name_part(query):
    assert analysis.name_matches("Ingrid Marie Haug-Solberg", query)


@pytest.mark.parametrize("query", ["grid", "aug", "ingrid x", "", "   "])
def test_name_matches_rejects_non_prefix_and_empty(query):
    assert not analysis.name_matches("Ingrid Marie Haug-Solberg", query)


def test_name_matches_is_not_fuzzy():
    # Streamlits innebygde søk gir treff her - vårt skal ikke.
    assert not analysis.name_matches("Tram Vuong", "Hauge")


# --- parse_date_flexible ---


@pytest.mark.parametrize(
    "value",
    ["15.03.2020", " 15.03.2020 ", "2020-03-15", "15/03/2020", datetime(2020, 3, 15), pd.Timestamp("2020-03-15")],
)
def test_parse_date_flexible_formats(value):
    assert analysis.parse_date_flexible(value).date() == date(2020, 3, 15)


@pytest.mark.parametrize("value", ["03/04/2020", "03.04.2020", "03-04-2020", "2020-04-03", "2020-04-03 00:00:00"])
def test_parse_date_flexible_ambiguous_dates_are_day_first(value):
    assert analysis.parse_date_flexible(value).date() == date(2020, 4, 3)


@pytest.mark.parametrize("value", [None, np.nan, pd.NaT, "", "ikke en dato"])
def test_parse_date_flexible_missing_or_invalid(value):
    assert analysis.parse_date_flexible(value) is None


# --- calculate_years_of_service ---


def test_years_of_service_missing_is_nan_not_zero():
    assert np.isnan(analysis.calculate_years_of_service(None))
    assert np.isnan(analysis.calculate_years_of_service("ikke en dato"))


def test_years_of_service_approx():
    start = datetime.now() - timedelta(days=round(10 * 365.25))
    assert analysis.calculate_years_of_service(start) == pytest.approx(10, abs=0.01)


# --- is_new_hire ---


def test_is_new_hire_cutoff_is_inclusive():
    cutoff = date(2025, 1, 1)
    assert analysis.is_new_hire("01.01.2025", cutoff)
    assert analysis.is_new_hire("02.01.2025", cutoff)
    assert not analysis.is_new_hire("31.12.2024", cutoff)


def test_is_new_hire_missing_values():
    assert not analysis.is_new_hire("01.01.2025", None)
    assert not analysis.is_new_hire(None, date(2025, 1, 1))


# --- kolonnegjenkjenning ---


def test_detect_column_mapping_exact_and_case_insensitive():
    cols = ["Etternavn", "fornavn", "STILLINGSKODE", "Tiltredelsesdato", "Annet"]
    mapping = analysis.detect_column_mapping(cols)
    assert mapping == {
        "Etternavn": "Etternavn",
        "Fornavn": "fornavn",
        "Stillingskode": "STILLINGSKODE",
        "Tiltredelsesdato": "Tiltredelsesdato",
        "Årslønn": None,
    }


def test_detect_column_mapping_saved_choice_wins_over_name_match():
    cols = ["Etternavn", "Fornavn", "Stillingskode", "Tiltredelsesdato", "Årslønn", "Lønn 100%"]
    mapping = analysis.detect_column_mapping(cols, {"Årslønn": "Lønn 100%"})
    assert mapping["Årslønn"] == "Lønn 100%"


def test_detect_column_mapping_ignores_saved_column_that_no_longer_exists():
    cols = ["Etternavn", "Fornavn", "Stillingskode", "Tiltredelsesdato", "Årslønn"]
    mapping = analysis.detect_column_mapping(cols, {"Årslønn": "Borte"})
    assert mapping["Årslønn"] == "Årslønn"


def test_detect_optional_columns():
    cols = ["avdeling", "Stillingsansiennitet", "FAGFORENING"]
    assert analysis.detect_unit_column(cols) == "avdeling"
    assert analysis.detect_position_seniority_column(cols) == "Stillingsansiennitet"
    assert analysis.detect_union_column(cols) == "FAGFORENING"
    assert analysis.detect_unit_column(["Navn"]) is None


# --- regresjon ---


def test_fit_linear_regression_perfect_line():
    x = pd.Series([0.0, 1, 2, 3, 4])
    fit = analysis.fit_linear_regression(x, 500_000 + 10_000 * x)
    assert fit["slope"] == pytest.approx(10_000)
    assert fit["intercept"] == pytest.approx(500_000)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["n"] == 5
    assert fit["dof"] == 3


def test_fit_linear_regression_matches_polyfit_and_drops_nan():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 30, 40))
    y = pd.Series(450_000 + 8_000 * x + rng.normal(0, 30_000, 40))
    x_with_nan = pd.concat([x, pd.Series([np.nan, 5.0])], ignore_index=True)
    y_with_nan = pd.concat([y, pd.Series([600_000.0, np.nan])], ignore_index=True)

    fit = analysis.fit_linear_regression(x_with_nan, y_with_nan)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit["n"] == 40
    assert fit["slope"] == pytest.approx(slope)
    assert fit["intercept"] == pytest.approx(intercept)


@pytest.mark.parametrize(
    "x, y",
    [
        ([1.0], [500_000.0]),  # for få punkter
        ([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]),  # ingen variasjon i x
        ([1.0, np.nan], [1.0, 2.0]),  # for få etter at NaN er fjernet
    ],
)
def test_fit_linear_regression_returns_none_when_impossible(x, y):
    assert analysis.fit_linear_regression(pd.Series(x), pd.Series(y)) is None


# --- konfidensintervall ---


def _noisy_fit():
    x = pd.Series([1.0, 2, 4, 5, 7, 9, 12, 15])
    y = pd.Series([510, 525, 540, 548, 575, 590, 610, 650], dtype=float) * 1000
    return analysis.fit_linear_regression(x, y)


def test_confidence_band_matches_textbook_formula():
    fit = _noisy_fit()
    xs = np.array([0.0, fit["x_mean"], 20.0])
    lower, upper = analysis.confidence_band(fit, xs)

    t_crit = t_dist.ppf(0.975, fit["dof"])
    se = fit["residual_se"] * np.sqrt(1 / fit["n"] + (xs - fit["x_mean"]) ** 2 / fit["sxx"])
    center = fit["intercept"] + fit["slope"] * xs
    np.testing.assert_allclose(lower, center - t_crit * se)
    np.testing.assert_allclose(upper, center + t_crit * se)


def test_confidence_band_is_narrowest_at_mean():
    fit = _noisy_fit()
    xs = np.linspace(0, 20, 201)
    lower, upper = analysis.confidence_band(fit, xs)
    width = upper - lower
    assert xs[np.argmin(width)] == pytest.approx(fit["x_mean"], abs=0.1)
    assert width[0] > width.min() and width[-1] > width.min()


def test_confidence_band_none_without_degrees_of_freedom():
    fit = analysis.fit_linear_regression(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
    assert analysis.confidence_band(fit, [1.0]) == (None, None)
    assert analysis.confidence_band(None, [1.0]) == (None, None)


# --- regresjon per stillingskode ---


def _salary_df():
    return pd.DataFrame(
        {
            "Stillingskode": ["A"] * 5 + ["B"],
            "Ansiennitet (År)": [0.0, 1, 2, 3, 10, 5],
            "Årslønn": [500_000.0, 510_000, 520_000, 530_000, 900_000, 600_000],
            "_hash": ["a0", "a1", "a2", "a3", "a_outlier", "b0"],
        }
    )


def test_run_regression_excludes_outliers_but_writes_deviation_for_all_rows():
    df = _salary_df()
    results = analysis.run_regression_per_code(df, {"A": ["a_outlier"]})

    assert results["A"]["slope"] == pytest.approx(10_000)
    assert results["A"]["n"] == 4
    a_rows = df[df["Stillingskode"] == "A"]
    # Også outlieren får forventet lønn/avvik, beregnet fra trendlinjen uten den.
    assert a_rows["Forventet Lønn"].tolist() == pytest.approx([500_000, 510_000, 520_000, 530_000, 600_000])
    assert df.loc[4, "Lønnsavvik (Kr)"] == pytest.approx(300_000)


def test_run_regression_code_with_too_few_points():
    df = _salary_df()
    results = analysis.run_regression_per_code(df, {})
    assert results["B"]["slope"] is None
    assert results["B"]["n"] == 1
    assert pd.isna(df.loc[5, "Forventet Lønn"])


def test_run_regression_without_new_hires_and_without_writing():
    df = _salary_df()
    new_hire = pd.Series([True, False, False, False, False, False], index=df.index)
    results = analysis.run_regression_per_code(
        df, {"A": ["a_outlier"]}, new_hire_mask=new_hire, exclude_new_hires=True, write_columns=False
    )
    assert results["A"]["n"] == 3
    assert "Forventet Lønn" not in df.columns


# --- nøkkeltall ---


def test_summary_stats():
    stats = analysis.summary_stats(pd.DataFrame({"Årslønn": [100.0, 200, 300, 400]}))
    assert stats["min"] == 100 and stats["max"] == 400
    assert stats["median"] == 250 and stats["mean"] == 250
    assert stats["q1"] == 175 and stats["q3"] == 325
    assert stats["n"] == 4


def test_non_outlier_rows():
    df = _salary_df()
    rows = analysis.non_outlier_rows(df, "A", {"A": ["a_outlier"]})
    assert rows["_hash"].tolist() == ["a0", "a1", "a2", "a3"]
