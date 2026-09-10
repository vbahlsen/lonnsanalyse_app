"""Analyse-logikk: ansiennitet, kolonne-mapping og regresjon per stillingskode."""

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress

REQUIRED_COLUMNS = ["Etternavn", "Fornavn", "Stillingskode", "Tiltredelsesdato", "Årslønn"]
OPTIONAL_UNIT_SYNONYMS = ["Ansattenhet", "Enhet", "Avdeling"]


def calculate_years_of_service(start_date):
    """Beregner ansiennitet i år fra en startdato.

    Godtar ekte dato-/tidsstempelverdier og tekst i flere vanlige formater
    (dd.mm.åååå, åååå-mm-dd, dd/mm/åååå, osv.). Returnerer np.nan (ikke 0.0)
    når datoen mangler eller ikke lar seg tolke, slik at slike rader kan
    ekskluderes fra analysen i stedet for å bli feilaktig talt som "nyansatt"
    - noe som tidligere kunne kollapse hele stillingskoder til identisk
    ansiennitet og krasje regresjonen.
    """
    if pd.isna(start_date):
        return np.nan

    date_obj = None
    if isinstance(start_date, datetime):
        date_obj = start_date
    elif isinstance(start_date, str):
        text = start_date.strip()
        try:
            date_obj = datetime.strptime(text, "%d.%m.%Y")
        except ValueError:
            # Prøv pandas' formatgjenkjenning først (håndterer bl.a. åååå-mm-dd
            # uten tvetydighet), fall tilbake på dayfirst=True for dd/mm/åååå-stil.
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.isna(parsed):
                parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
            if pd.notna(parsed):
                date_obj = parsed.to_pydatetime()
    else:
        parsed = pd.to_datetime(start_date, dayfirst=True, errors="coerce")
        if pd.notna(parsed):
            date_obj = parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed

    if date_obj is None:
        return np.nan

    diff = datetime.now() - date_obj
    return round(diff.days / 365.25, 2)


def detect_column_mapping(df_columns: list[str], saved_mapping: dict | None = None) -> dict[str, str | None]:
    """Forsøker å matche faktiske kolonnenavn mot de påkrevde logiske navnene.

    Prioriterer eksakt (case-insensitive) match, deretter en tidligere lagret
    mapping fra oppgjøret (hvis den faktiske kolonnen fortsatt finnes).
    Manglende felter mappes til None og må fylles inn manuelt av brukeren.
    """
    lower_lookup = {col.lower(): col for col in df_columns}
    mapping: dict[str, str | None] = {}

    for logical in REQUIRED_COLUMNS:
        if logical in df_columns:
            mapping[logical] = logical
        elif logical.lower() in lower_lookup:
            mapping[logical] = lower_lookup[logical.lower()]
        elif saved_mapping and saved_mapping.get(logical) in df_columns:
            mapping[logical] = saved_mapping[logical]
        else:
            mapping[logical] = None

    return mapping


def detect_unit_column(df_columns: list[str]) -> str | None:
    lower_lookup = {col.lower(): col for col in df_columns}
    for synonym in OPTIONAL_UNIT_SYNONYMS:
        if synonym.lower() in lower_lookup:
            return lower_lookup[synonym.lower()]
    return None


def run_regression_per_code(df: pd.DataFrame, outliers_by_code: dict[str, list[str]], hash_col: str = "_hash") -> dict:
    """Kjører separat lineær regresjon (Årslønn vs. Ansiennitet) per stillingskode.

    Outliers (identifisert via hash_col mot outliers_by_code) ekskluderes fra
    regresjonen for sin stillingskode. Returnerer et dict per kode med
    slope/intercept/r_squared/n, og skriver 'Forventet Lønn' / 'Lønnsavvik (Kr)'
    inn i df for radene som inngikk i en gyldig regresjon.
    """
    results = {}
    df["Forventet Lønn"] = pd.NA
    df["Lønnsavvik (Kr)"] = pd.NA

    for kode, group in df.groupby("Stillingskode"):
        outlier_hashes = set(outliers_by_code.get(str(kode), []))
        clean = group[~group[hash_col].isin(outlier_hashes)]

        if len(clean) < 2 or clean["Ansiennitet (År)"].nunique() < 2:
            # For få datapunkter, eller alle har samme ansiennitet (f.eks. tilfeldig
            # identisk tiltredelsesdato i en liten gruppe) - kan ikke regne trendlinje
            # for denne stillingskoden, men resten av appen skal fortsatt fungere.
            results[kode] = {"slope": None, "intercept": None, "r_squared": None, "n": len(clean)}
            continue

        slope, intercept, r_value, _, _ = linregress(clean["Ansiennitet (År)"], clean["Årslønn"])
        results[kode] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "n": len(clean),
        }

        expected = intercept + slope * group["Ansiennitet (År)"]
        df.loc[group.index, "Forventet Lønn"] = expected
        df.loc[group.index, "Lønnsavvik (Kr)"] = group["Årslønn"] - expected

    return results


def summary_stats(code_df: pd.DataFrame) -> dict:
    """Nøkkeltall (min/Q1/median/mean/Q3/max) for Årslønn i en gruppe."""
    salaries = code_df["Årslønn"]
    return {
        "min": salaries.min(),
        "q1": salaries.quantile(0.25),
        "median": salaries.median(),
        "mean": salaries.mean(),
        "q3": salaries.quantile(0.75),
        "max": salaries.max(),
        "n": len(code_df),
    }


def non_outlier_rows(df: pd.DataFrame, stillingskode, outliers_by_code: dict, hash_col: str = "_hash") -> pd.DataFrame:
    outlier_hashes = set(outliers_by_code.get(str(stillingskode), []))
    code_df = df[df["Stillingskode"] == stillingskode]
    return code_df[~code_df[hash_col].isin(outlier_hashes)]
