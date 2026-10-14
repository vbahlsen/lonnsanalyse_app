"""Analyse-logikk: ansiennitet, kolonne-mapping og regresjon per stillingskode."""

import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import t as t_dist

REQUIRED_COLUMNS = ["Etternavn", "Fornavn", "Stillingskode", "Tiltredelsesdato", "Årslønn"]
OPTIONAL_UNIT_SYNONYMS = ["Ansattenhet", "Enhet", "Avdeling"]
OPTIONAL_POSITION_SENIORITY_SYNONYMS = ["Stillingsansiennitet"]
OPTIONAL_UNION_SYNONYMS = ["Fagforening"]


def name_matches(full_name: str, query: str) -> bool:
    """Presis navnesøk: hvert søkeord må matche starten av et av navnene (fornavn,
    mellomnavn eller etternavn, også hver del av dobbeltnavn med bindestrek).
    Rekkefølgen på søkeordene spiller ingen rolle, og store/små bokstaver ignoreres.
    """
    words = [w for w in re.split(r"[\s\-]+", str(full_name).casefold()) if w]
    tokens = [t for t in re.split(r"[\s\-]+", query.casefold()) if t]
    return bool(tokens) and all(any(w.startswith(t) for w in words) for t in tokens)


def parse_date_flexible(value):
    """Tolker en dato fra typiske Excel-verdier (ekte dato-/tidsstempel eller
    tekst i vanlige formater: dd.mm.åååå, åååå-mm-dd, dd/mm/åååå, osv.).

    Returnerer et datetime-objekt, eller None hvis verdien mangler eller ikke
    lar seg tolke.
    """
    if pd.isna(value):
        return None

    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        text = value.strip()
        try:
            return datetime.strptime(text, "%d.%m.%Y")
        except ValueError:
            # Prøv pandas' formatgjenkjenning først (håndterer bl.a. åååå-mm-dd
            # uten tvetydighet), fall tilbake på dayfirst=True for dd/mm/åååå-stil.
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.isna(parsed):
                parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        return None
    else:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.notna(parsed):
            return parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed
        return None


def calculate_years_of_service(start_date):
    """Beregner ansiennitet i år fra en startdato.

    Returnerer np.nan (ikke 0.0) når datoen mangler eller ikke lar seg tolke,
    slik at slike rader kan ekskluderes fra analysen i stedet for å bli
    feilaktig talt som "nyansatt" - noe som tidligere kunne kollapse hele
    stillingskoder til identisk ansiennitet og krasje regresjonen.
    """
    date_obj = parse_date_flexible(start_date)
    if date_obj is None:
        return np.nan

    diff = datetime.now() - date_obj
    return round(diff.days / 365.25, 2)


def is_new_hire(start_date, cutoff_date) -> bool:
    """True hvis start_date er på eller etter cutoff_date.

    cutoff_date er et date-objekt (f.eks. fra st.date_input). Returnerer
    False hvis cutoff_date mangler eller start_date ikke lar seg tolke.
    """
    if cutoff_date is None:
        return False
    date_obj = parse_date_flexible(start_date)
    if date_obj is None:
        return False
    return date_obj.date() >= cutoff_date


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


def detect_position_seniority_column(df_columns: list[str]) -> str | None:
    """Finner en egen 'Stillingsansiennitet'-kolonne (dato), hvis den finnes.

    Dette er IKKE det samme som Tiltredelsesdato - en ansatt kan ha vært i
    virksomheten lenge, men hatt kortere ansiennitet i nåværende stilling
    etter et opprykk. Kolonnen forventes å inneholde en dato (samme formater
    som Tiltredelsesdato), som konverteres til år på samme måte.
    """
    lower_lookup = {col.lower(): col for col in df_columns}
    for synonym in OPTIONAL_POSITION_SENIORITY_SYNONYMS:
        if synonym.lower() in lower_lookup:
            return lower_lookup[synonym.lower()]
    return None


def detect_union_column(df_columns: list[str]) -> str | None:
    """Finner en 'Fagforening'-kolonne, hvis den finnes. Tom celle = ikke angitt."""
    lower_lookup = {col.lower(): col for col in df_columns}
    for synonym in OPTIONAL_UNION_SYNONYMS:
        if synonym.lower() in lower_lookup:
            return lower_lookup[synonym.lower()]
    return None


def fit_linear_regression(x: pd.Series, y: pd.Series) -> dict | None:
    """Kjører OLS-regresjon og returnerer slope/intercept/r_squared/std_residual/n,
    samt grunnlaget (residual_se/x_mean/sxx/dof) for å beregne et korrekt,
    innsnevrende 95%-konfidensintervall langs trendlinjen via confidence_band().

    Rader hvor x eller y mangler ekskluderes automatisk. Returnerer None hvis
    det er for få gjenværende punkter eller ingen variasjon i x (f.eks. alle
    har identisk ansiennitet) - da kan ingen trendlinje beregnes.
    """
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    if len(x) < 2 or x.nunique() < 2:
        return None

    slope, intercept, r_value, _, _ = linregress(x, y)
    residuals = y - (intercept + slope * x)
    n = len(x)
    dof = n - 2
    residual_se = np.sqrt((residuals**2).sum() / dof) if dof > 0 else None
    x_mean = x.mean()
    sxx = ((x - x_mean) ** 2).sum()
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "std_residual": residuals.std(),
        "residual_se": residual_se,
        "x_mean": x_mean,
        "sxx": sxx,
        "dof": dof,
        "n": n,
    }


def confidence_band(fit: dict | None, x_values, confidence: float = 0.95):
    """Beregner et 95%-konfidensintervall for FORVENTET (gjennomsnittlig) lønn
    langs trendlinjen ved de gitte x-verdiene.

    I motsetning til et enkelt bånd med konstant bredde (±1,96×std.avvik) tar
    dette hensyn til at usikkerheten i en OLS-regresjon er lavest nær
    gjennomsnittlig ansiennitet i datagrunnlaget og øker jo lenger unna man
    kommer - båndet snevrer seg dermed inn på midten og videre ut mot
    ytterpunktene, slik en statistisk korrekt fremstilling skal se ut.

    Returnerer (lower, upper) som numpy-arrays, eller (None, None) hvis det
    ikke er nok datagrunnlag (færre enn 3 punkter) til å beregne et intervall.
    """
    if not fit or not fit.get("residual_se") or not fit.get("dof") or fit["dof"] <= 0 or fit.get("sxx", 0) <= 0:
        return None, None

    t_crit = t_dist.ppf(1 - (1 - confidence) / 2, fit["dof"])
    x_arr = np.asarray(x_values, dtype=float)
    se = fit["residual_se"] * np.sqrt(1.0 / fit["n"] + (x_arr - fit["x_mean"]) ** 2 / fit["sxx"])
    y_center = fit["intercept"] + fit["slope"] * x_arr
    half_width = t_crit * se
    return y_center - half_width, y_center + half_width


def run_regression_per_code(
    df: pd.DataFrame,
    outliers_by_code: dict[str, list[str]],
    x_col: str = "Ansiennitet (År)",
    hash_col: str = "_hash",
    new_hire_mask: pd.Series | None = None,
    exclude_new_hires: bool = False,
    write_columns: bool = True,
) -> dict:
    """Kjører separat lineær regresjon (Årslønn vs. x_col) per stillingskode.

    Outliers (identifisert via hash_col mot outliers_by_code) ekskluderes alltid
    fra regresjonen for sin stillingskode. Hvis exclude_new_hires er True,
    ekskluderes i tillegg ansatte markert i new_hire_mask (bruken av dette er
    å kunne sammenligne trenden med/uten nyansatte, f.eks. i koder med høy
    turnover hvor ansiennitet før tiltredelse ikke er kjent).

    Returnerer et dict per kode med slope/intercept/r_squared/std_residual/n.
    Hvis write_columns er True (standard) skrives også 'Forventet Lønn' /
    'Lønnsavvik (Kr)' inn i df for radene som inngikk i en gyldig regresjon
    (NaN for rader uten gyldig verdi i x_col, f.eks. manglende Stillingsansiennitet)
    - denne baseres alltid på HELE datagrunnlaget (med nyansatte), uavhengig av
    exclude_new_hires, slik at avviksvisningen for én enkelt ansatt forblir
    konsistent uansett hvilke sammenligningslinjer som vises i plottet.
    """
    results = {}
    if write_columns:
        df["Forventet Lønn"] = pd.NA
        df["Lønnsavvik (Kr)"] = pd.NA

    for kode, group in df.groupby("Stillingskode"):
        outlier_hashes = set(outliers_by_code.get(str(kode), []))
        clean = group[~group[hash_col].isin(outlier_hashes)]
        if exclude_new_hires and new_hire_mask is not None:
            clean = clean[~new_hire_mask.loc[clean.index]]

        fit = fit_linear_regression(clean[x_col], clean["Årslønn"])
        if fit is None:
            # For få datapunkter, eller alle har samme verdi på x-aksen (f.eks.
            # tilfeldig identisk startdato i en liten gruppe) - kan ikke regne
            # trendlinje for denne stillingskoden, men resten av appen fungerer.
            results[kode] = {
                "slope": None, "intercept": None, "r_squared": None, "std_residual": None,
                "residual_se": None, "x_mean": None, "sxx": None, "dof": None, "n": len(clean),
            }
            continue

        results[kode] = fit
        if write_columns:
            expected = fit["intercept"] + fit["slope"] * group[x_col]
            df.loc[group.index, "Forventet Lønn"] = expected
            df.loc[group.index, "Lønnsavvik (Kr)"] = group["Årslønn"] - expected

    return results


def summary_stats(code_df: pd.DataFrame) -> dict:
    """Nøkkeltall (min/Q1/median/mean/Q3/max/std) for Årslønn i en gruppe."""
    salaries = code_df["Årslønn"]
    return {
        "min": salaries.min(),
        "q1": salaries.quantile(0.25),
        "median": salaries.median(),
        "mean": salaries.mean(),
        "q3": salaries.quantile(0.75),
        "max": salaries.max(),
        "std": salaries.std(),
        "n": len(code_df),
    }


def non_outlier_rows(df: pd.DataFrame, stillingskode, outliers_by_code: dict, hash_col: str = "_hash") -> pd.DataFrame:
    outlier_hashes = set(outliers_by_code.get(str(stillingskode), []))
    code_df = df[df["Stillingskode"] == stillingskode]
    return code_df[~code_df[hash_col].isin(outlier_hashes)]
