from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis
import settlement_store as store
from pdf_export import KEY_LABELS, build_export_zip

UNION_HIGHLIGHT_COLORS = ["#17BECF", "#2CA8A4", "#00CED1", "#20B2AA", "#48D1CC"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _guess_initial_sidebar_state() -> str:
    """Sidebaren skal starte lukket når vi allerede har lastet data denne
    økten, eller når det aktive (eller sist brukte) oppgjøret har en kjent
    filsti som faktisk finnes på disk - ellers åpen, slik at brukeren finner
    fil-/oppgjør-kontrollene."""
    if st.session_state.get("data_loaded_ok"):
        return "collapsed"
    name = st.session_state.get("settlement_name")
    if not name:
        existing = store.list_settlements()
        name = existing[0] if existing else None
    if name:
        settlement = store.load_settlement(name)
        if store.resolve_data_file(settlement.get("file_path")):
            return "collapsed"
    return "expanded"


st.set_page_config(
    layout="wide", page_title="Lønnsnivåanalyse mot Ansiennitet", initial_sidebar_state=_guess_initial_sidebar_state()
)
st.markdown(
    "<style>h3 {font-size: 1.3rem !important;} h4 {font-size: 1.1rem !important;}</style>",
    unsafe_allow_html=True,
)
st.title("💰 Lønnsnivåanalyse mot Ansiennitet")

# --- OPPGJØR-VELGER ---

st.sidebar.header("📁 Oppgjør")

existing = store.list_settlements()
options = ["-- Nytt oppgjør --"] + existing
current_name = st.session_state.get("settlement_name")
default_index = options.index(current_name) if current_name in options else (1 if existing else 0)

picked = st.sidebar.selectbox("Velg oppgjør", options, index=default_index, key="settlement_picker")

if picked == "-- Nytt oppgjør --":
    def _create_settlement():
        clean_name = st.session_state.get("new_settlement_name_input", "").strip()
        if not clean_name:
            return
        store.save_settlement(clean_name, store.default_settlement(clean_name))
        # NB: session_state.settlement_name settes bevisst IKKE her. Den skal fortsatt
        # peke på forrige oppgjør (eller mangle) slik at "current_name != picked" lenger
        # ned i scriptet oppdager byttet og rydder unna gammel widget-tilstand.
        st.session_state["settlement_picker"] = clean_name

    new_name = st.sidebar.text_input(
        "Navn på nytt oppgjør", placeholder="f.eks. 2026 oppgjøret", key="new_settlement_name_input"
    )
    st.sidebar.button("Opprett oppgjør", disabled=not new_name.strip(), on_click=_create_settlement)
    st.sidebar.info("Opprett eller velg et oppgjør for å starte. Et oppgjør husker filsti, kolonnevalg og outlier-vurderinger mellom økter, uten å lagre personopplysninger.")
    st.stop()

if current_name != picked:
    # Fjern widget-tilstand fra forrige oppgjør. Streamlit gjenbruker en widgets verdi
    # på tvers av script-kjøringer så lenge nøkkelen finnes i session_state, uavhengig av
    # hvilken "value"/"index" vi sender inn - uten dette ville f.eks. filsti, valgt ansatt,
    # stillingskode-filter og outlier-avkrysninger fra forrige oppgjør lekke inn i det nye.
    stale_prefixes = ("outlier_toggle_", "map_", "union_highlight_", "union_export_btn_", "union_dl_")
    stale_keys = (
        "file_path_widget", "employee_selector", "employee_search", "column_selector", "export_selection",
        "stillingskode_filter", "multi_code_mode", "x_axis_radio", "pdf_stats_fields", "pdf_show_axis_values", "pdf_show_avvik_text",
        "pdf_x_axis_mode_radio", "pdf_report_date", "pdf_colleague_display", "pdf_show_mean_line",
        "new_hire_cutoff_widget", "show_regression_with_new_hires", "show_regression_without_new_hires",
        "hide_new_hires_live", "pdf_fig_tiltredelse_with", "pdf_fig_tiltredelse_without",
        "pdf_fig_stilling_with", "pdf_fig_stilling_without",
    )
    for key in list(st.session_state.keys()):
        if key in stale_keys or key.startswith(stale_prefixes) or key.startswith("_union_zip_"):
            del st.session_state[key]

    settlement = store.load_settlement(picked)
    st.session_state.settlement_name = picked
    st.session_state.outliers_by_code = settlement.get("outliers") or {}
    st.session_state.column_mapping = settlement.get("column_mapping")
    st.session_state.file_path_input = settlement.get("file_path") or ""
    st.session_state.display_columns = settlement.get("display_columns") or analysis.REQUIRED_COLUMNS
    st.session_state.x_axis_choice = settlement.get("x_axis_choice") or "Ansiennitet (År)"
    st.session_state.pdf_options = settlement.get("pdf_options") or dict(store.DEFAULT_PDF_OPTIONS)
    new_hire_settings = {**store.DEFAULT_NEW_HIRE_SETTINGS, **(settlement.get("new_hire_settings") or {})}
    st.session_state["new_hire_cutoff_widget"] = (
        date.fromisoformat(new_hire_settings["cutoff_date"]) if new_hire_settings.get("cutoff_date") else date.today()
    )
    st.session_state["show_regression_with_new_hires"] = new_hire_settings["show_regression_with_new_hires"]
    st.session_state["show_regression_without_new_hires"] = new_hire_settings["show_regression_without_new_hires"]
    st.session_state["hide_new_hires_live"] = new_hire_settings["hide_new_hires_live"]
    st.session_state.pending_selected_employee_hash = settlement.get("last_selected_employee_hash")
    st.session_state.pending_selected_codes = settlement.get("selected_codes")
    st.session_state["multi_code_mode"] = settlement.get("multi_code_mode", True)
    st.session_state.selected_employee = None
    st.session_state.data_loaded_ok = False
    st.session_state.pop("_export_zip_bytes", None)
    st.rerun()

settlement_name = st.session_state.settlement_name
st.sidebar.caption(f"Aktivt oppgjør: **{settlement_name}**")

st.session_state.setdefault("outliers_by_code", {})
st.session_state.setdefault("column_mapping", None)
st.session_state.setdefault("file_path_input", "")
st.session_state.setdefault("display_columns", analysis.REQUIRED_COLUMNS)
st.session_state.setdefault("x_axis_choice", "Ansiennitet (År)")
st.session_state.setdefault("pdf_options", dict(store.DEFAULT_PDF_OPTIONS))
st.session_state.setdefault("selected_employee", None)
st.session_state.setdefault("new_hire_cutoff_widget", date.today())
st.session_state.setdefault("show_regression_with_new_hires", True)
st.session_state.setdefault("show_regression_without_new_hires", False)
st.session_state.setdefault("hide_new_hires_live", False)
st.session_state.setdefault("multi_code_mode", True)

# --- DATAFIL ---

st.sidebar.header("📄 Datafil")
path_input = st.sidebar.text_input(
    "Filsti til lønnsdata (Excel)",
    value=st.session_state.file_path_input,
    key="file_path_widget",
    help="Siden appen kjører lokalt kan den lese filen direkte fra disk og huske stien til neste økt. "
    "Fungerer også med sti limt inn via 'Kopier som bane' i Utforsker.",
)

df_raw = None

cleaned_path = store.clean_path_string(path_input)
if cleaned_path:
    resolved = store.resolve_data_file(cleaned_path)
    if resolved:
        try:
            df_raw = pd.read_excel(resolved)
            st.session_state.file_path_input = str(resolved)
            if path_input.strip() != str(resolved):
                # Feltet viser fortsatt det brukeren limte inn (f.eks. med anførselstegn
                # fra "Kopier som bane") siden en widgets verdi ikke kan overstyres etter at
                # den er instansiert i samme kjøring - rydd opp visningen på neste kjøring.
                st.session_state.pop("file_path_widget", None)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Klarte ikke å lese filen: {e}")
    else:
        st.sidebar.warning(f"Finner ikke filen på oppgitt sti:\n\n`{cleaned_path}`\n\nSjekk stien, eller last opp filen manuelt under.")

if df_raw is None:
    uploaded_file = st.sidebar.file_uploader("...eller last opp Excel-fil manuelt:", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file)
        st.sidebar.info("Lastet opp manuelt. Oppgi filstien i feltet over for at appen skal huske filen til neste økt.")

if df_raw is None:
    st.session_state.data_loaded_ok = False
    st.info("Oppgi filsti eller last opp Excel-fil for å starte analysen.")
    st.stop()

try:
    # --- DATAPREPARASJON ---

    df_raw.columns = [str(col).strip() for col in df_raw.columns]
    column_mapping = analysis.detect_column_mapping(df_raw.columns.tolist(), st.session_state.column_mapping)

    if any(v is None for v in column_mapping.values()):
        st.warning("Fant ikke alle forventede kolonner automatisk. Angi hvilken kolonne i filen som tilsvarer hvert felt.")
        with st.form("column_mapping_form"):
            new_mapping = {}
            for logical in analysis.REQUIRED_COLUMNS:
                col_options = ["-- velg kolonne --"] + df_raw.columns.tolist()
                current = column_mapping.get(logical) or "-- velg kolonne --"
                idx = col_options.index(current) if current in col_options else 0
                new_mapping[logical] = st.selectbox(f"{logical}:", col_options, index=idx, key=f"map_{logical}")
            submitted = st.form_submit_button("Bekreft kolonner")
        if submitted and all(v != "-- velg kolonne --" for v in new_mapping.values()):
            st.session_state.column_mapping = new_mapping
            st.rerun()
        st.stop()

    rename_dict = {actual: logical for logical, actual in column_mapping.items() if actual != logical}
    df = df_raw.rename(columns=rename_dict)
    unit_col = analysis.detect_unit_column(df.columns.tolist())
    position_seniority_col = analysis.detect_position_seniority_column(df.columns.tolist())
    union_col = analysis.detect_union_column(df.columns.tolist())

    df["Stillingskode"] = df["Stillingskode"].astype(str).str.strip()
    df["Ansiennitet (År)"] = df["Tiltredelsesdato"].apply(analysis.calculate_years_of_service)
    if position_seniority_col:
        # Stillingsansiennitet er IKKE det samme som år siden Tiltredelsesdato - den kommer
        # fra en egen kolonne, siden en ansatt kan ha vært lenger i virksomheten enn i
        # nåværende stilling (f.eks. etter et opprykk).
        df["Stillingsansiennitet (År)"] = df[position_seniority_col].apply(analysis.calculate_years_of_service)
    df["Fullt Navn"] = df["Fornavn"].astype(str) + " " + df["Etternavn"].astype(str)
    df["Årslønn"] = pd.to_numeric(df["Årslønn"], errors="coerce")

    rows_before = len(df)
    missing_mask = df["Årslønn"].isna() | df["Ansiennitet (År)"].isna()
    if missing_mask.any():
        excluded_names = df.loc[missing_mask, "Fullt Navn"].tolist()
        st.warning(
            f"{missing_mask.sum()} av {rows_before} rader mangler gyldig Årslønn og/eller "
            f"tolkbar Tiltredelsesdato, og er utelatt fra analysen: {', '.join(excluded_names[:15])}"
            + (" ..." if len(excluded_names) > 15 else "")
        )
    df = df.dropna(subset=["Årslønn", "Ansiennitet (År)"])
    df["_hash"] = df.apply(
        lambda r: store.employee_hash(r["Fornavn"], r["Etternavn"], r["Stillingskode"], r["Tiltredelsesdato"]),
        axis=1,
    )

    # --- LAYOUT MED KOLONNER ---
    # col_plot rendres FØR col_stats' ansattvelger i koden (selv om col_stats vises til venstre)
    # slik at et klikk i grafen kan sette "employee_selector" trygt før den widgeten
    # instansieres denne kjøringen - Streamlit tillater ikke å endre en widgets
    # session_state-verdi etter at den allerede er instansiert i samme kjøring.

    all_codes = sorted(df["Stillingskode"].unique(), key=str)

    col_stats, col_plot = st.columns([1, 2])

    with col_stats:
        # Gjenopprett lagret stillingskode-filter FØR "stillingskode_filter" instansieres
        # denne kjøringen. Å kun sende inn "default=" er upålitelig hvis widget-nøkkelen av
        # noen grunn allerede eksisterer (f.eks. etter en mellomliggende rerun) - Streamlit
        # ignorerer da "default=" og beholder sin egen, gamle verdi. Å skrive direkte inn i
        # nøkkelen her er den eneste garanterte måten å sette startverdien på.
        pending_codes = st.session_state.pop("pending_selected_codes", None)
        if pending_codes and "stillingskode_filter" not in st.session_state:
            valid_codes = [c for c in pending_codes if c in all_codes]
            if valid_codes:
                st.session_state["stillingskode_filter"] = valid_codes

        # Valgt ansatt fra en stillingskode som ikke er med i filteret: legg koden til
        # (satt som "pending" i forrige kjøring, siden filteret da allerede var instansiert).
        # Mangler nøkkelen helt bruker widgeten default=all_codes, som alt inneholder koden.
        # I enkeltkode-modus erstatter koden i stedet det som var valgt fra før.
        pending_add_code = st.session_state.pop("pending_add_code", None)
        if pending_add_code in all_codes:
            if not st.session_state.multi_code_mode:
                st.session_state["stillingskode_filter"] = [pending_add_code]
            elif "stillingskode_filter" in st.session_state:
                current_codes = list(st.session_state["stillingskode_filter"])
                if pending_add_code not in current_codes:
                    st.session_state["stillingskode_filter"] = current_codes + [pending_add_code]

        selected_employee_code = None
        if st.session_state.get("selected_employee"):
            emp_match = df.loc[df["Fullt Navn"] == st.session_state.selected_employee, "Stillingskode"]
            if not emp_match.empty:
                selected_employee_code = emp_match.iloc[0]

        def _on_multi_code_mode_change(preferred_code):
            # Byttet til enkeltkode-modus med flere koder valgt: behold koden til valgt
            # ansatt hvis den er blant dem, ellers den første. Callbacks kjører før
            # widgetene instansieres, så her er det lov å skrive til filterets nøkkel.
            if st.session_state.multi_code_mode:
                return
            current = list(st.session_state.get("stillingskode_filter", all_codes))
            if len(current) > 1:
                keep = preferred_code if preferred_code in current else current[0]
                st.session_state["stillingskode_filter"] = [keep]

        def _on_code_filter_change():
            # Enkeltkode-modus: nylig valgt kode (lagt til sist av multiselect) erstatter forrige.
            codes = st.session_state.get("stillingskode_filter", [])
            if not st.session_state.multi_code_mode and len(codes) > 1:
                st.session_state["stillingskode_filter"] = [codes[-1]]

        st.checkbox(
            "Tillat flere aktive stillingskoder samtidig",
            key="multi_code_mode",
            on_change=_on_multi_code_mode_change,
            args=(selected_employee_code,),
            help="Når denne er av kan kun én stillingskode være aktiv om gangen - valg av ny kode "
            "(i menyen eller ved å velge en ansatt i en annen kode) erstatter den forrige.",
        )

        if not st.session_state.multi_code_mode and len(st.session_state.get("stillingskode_filter", all_codes)) > 1:
            # F.eks. lagret oppgjør med flere koder, åpnet i enkeltkode-modus.
            current = list(st.session_state.get("stillingskode_filter", all_codes))
            st.session_state["stillingskode_filter"] = (
                [selected_employee_code] if selected_employee_code in current else current[:1]
            )

        selected_codes = st.multiselect(
            "Stillingskode(r) for visning (regresjon kjøres alltid per kode, på hele datasettet):"
            if st.session_state.multi_code_mode
            else "Stillingskode for visning (regresjon kjøres alltid per kode, på hele datasettet):",
            options=all_codes,
            default=all_codes,
            key="stillingskode_filter",
            on_change=_on_code_filter_change,
        )

        x_col = "Ansiennitet (År)"
        x_label = "Ansiennitet (år) siden tiltredelse"
        if position_seniority_col:
            x_choice_options = ["Ansiennitet (År)", "Stillingsansiennitet (År)"]
            if "x_axis_radio" not in st.session_state and st.session_state.x_axis_choice in x_choice_options:
                st.session_state["x_axis_radio"] = st.session_state.x_axis_choice
            x_col = st.radio(
                "X-akse:",
                options=x_choice_options,
                format_func=lambda v: "Ansiennitet siden tiltredelse" if v == "Ansiennitet (År)" else "Stillingsansiennitet",
                horizontal=True,
                key="x_axis_radio",
            )
            x_label = "Ansiennitet (år) siden tiltredelse" if x_col == "Ansiennitet (År)" else "Stillingsansiennitet (år)"
            st.session_state.x_axis_choice = x_col

        st.markdown("---")
        st.caption(
            "🆕 Nyansatte (høy turnover i enkelte koder kan gi en frontlasting som skjevfordeler trenden):"
        )
        cutoff_date_value = st.date_input(
            "Nyansatt-grense (tiltrådt på/etter denne datoen regnes som nyansatt):",
            format="DD/MM/YYYY",
            key="new_hire_cutoff_widget",
        )
        reg_col_a, reg_col_b = st.columns(2)
        with reg_col_a:
            show_regression_with_new_hires = st.checkbox(
                "Regresjon med nyansatte", key="show_regression_with_new_hires"
            )
        with reg_col_b:
            show_regression_without_new_hires = st.checkbox(
                "Regresjon uten nyansatte", key="show_regression_without_new_hires"
            )
        hide_new_hires_live = st.checkbox(
            "Skjul nyansatte i visningen",
            key="hide_new_hires_live",
            help="Fjerner kun nyansatte fra selve grafen - de inngår fortsatt i statistikk og "
            "regresjon i henhold til avkrysningene over.",
        )

    df["Er Nyansatt"] = df["Tiltredelsesdato"].apply(lambda d: analysis.is_new_hire(d, cutoff_date_value))

    # --- REGRESJON PER STILLINGSKODE (alltid på hele datasettet) ---

    reg_results = analysis.run_regression_per_code(df, st.session_state.outliers_by_code, x_col=x_col, hash_col="_hash")
    reg_results_excl_new = analysis.run_regression_per_code(
        df, st.session_state.outliers_by_code, x_col=x_col, hash_col="_hash",
        new_hire_mask=df["Er Nyansatt"], exclude_new_hires=True, write_columns=False,
    )

    filtered_df = df[df["Stillingskode"].isin(selected_codes)].copy()

    # Bygges med listeuttrykk i stedet for DataFrame.apply, som returnerer en DataFrame
    # (ikke en Series) når filtered_df er tom - f.eks. når ingen stillingskode er valgt.
    # Eksplisitt bool-dtype: en tom object-kolonne brukt som maske under ville ellers blitt
    # tolket som kolonnevalg og fjernet alle kolonnene.
    filtered_df["Is Outlier"] = pd.Series(
        [
            h in set(st.session_state.outliers_by_code.get(str(k), []))
            for h, k in zip(filtered_df["_hash"], filtered_df["Stillingskode"])
        ],
        index=filtered_df.index,
        dtype=bool,
    )
    clean_filtered_df = filtered_df[~filtered_df["Is Outlier"]]

    union_values = []
    if union_col:
        union_values = sorted(v for v in df[union_col].dropna().unique() if str(v).strip())

    with col_plot:
        union_highlight_selection = {}
        if union_values:
            st.caption("Uthev fagforeningsmedlemmer:")
            union_widget_cols = st.columns(min(len(union_values), 4))
            for i, uv in enumerate(union_values):
                with union_widget_cols[i % len(union_widget_cols)]:
                    union_highlight_selection[uv] = st.checkbox(
                        f"Uthev {uv}-medlemmer", key=f"union_highlight_{uv}"
                    )

        plot_df = filtered_df.copy()
        if hide_new_hires_live:
            plot_df = plot_df[~plot_df["Er Nyansatt"]]
        plot_df["Farge"] = plot_df["Stillingskode"].astype(str)
        plot_df["Størrelse"] = plot_df["Is Outlier"].map({True: 5, False: 7})
        plot_df["Symbol"] = plot_df["Is Outlier"].map({True: "circle-open", False: "circle"})

        palette = px.colors.qualitative.Dark24
        code_color_map = {str(kode): palette[i % len(palette)] for i, kode in enumerate(selected_codes)}

        union_color_map = {}
        if union_col:
            for i, uv in enumerate(union_values):
                if union_highlight_selection.get(uv):
                    mask = plot_df[union_col] == uv
                    plot_df.loc[mask, "Farge"] = f"🟦 {uv}-medlem"
                    union_color_map[f"🟦 {uv}-medlem"] = UNION_HIGHLIGHT_COLORS[i % len(UNION_HIGHLIGHT_COLORS)]

        currently_selected = st.session_state.get("selected_employee")
        if currently_selected:
            sel_mask = plot_df["Fullt Navn"] == currently_selected
            plot_df.loc[sel_mask, "Farge"] = "🔴 VALGT ANSATT"
            plot_df.loc[sel_mask, "Størrelse"] = 13

        if plot_df.empty:
            # Ingen kode valgt (eller alle skjult) - vis et tomt diagram i stedet for å
            # stoppe siden, slik at resten av grensesnittet fortsatt er tilgjengelig.
            fig = go.Figure()
            fig.update_layout(title="Årslønn mot ansiennitet (ingen data å vise)", yaxis_title="Årslønn")
        else:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y="Årslønn",
                color="Farge",
                size="Størrelse",
                symbol="Symbol",
                symbol_map={"circle-open": "circle-open", "circle": "circle"},
                custom_data=["Fullt Navn"],
                hover_data=["Fullt Navn", "Stillingskode", "Is Outlier", "Lønnsavvik (Kr)"],
                title="Årslønn mot ansiennitet (åpen sirkel = markert outlier)",
                color_discrete_map={"🔴 VALGT ANSATT": "red", **code_color_map, **union_color_map},
            )

        highlighted_mask = plot_df["Farge"].astype(str).str.startswith("🟦")
        if highlighted_mask.any():
            # Liten svart prikk i midten av hvert uthevet fagforeningspunkt, slik at de er
            # lettere å skille fra andre punkter i samme turkis-fargefamilie.
            fig.add_scatter(
                x=plot_df.loc[highlighted_mask, x_col],
                y=plot_df.loc[highlighted_mask, "Årslønn"],
                mode="markers",
                marker=dict(size=3, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )

        new_hire_mask_plot = plot_df["Er Nyansatt"]
        if new_hire_mask_plot.any():
            # Ring rundt punktet i stedet for stripete fylling (Plotly støtter ikke
            # stripete/hatch-mønster på scatter-markører, kun på stolpediagram) - samme
            # teknikk som brukes i PDF-eksporten, slik at nyansatte er tydelig merket
            # uten at de forsvinner inn i fargen til sin stillingskode.
            fig.add_scatter(
                x=plot_df.loc[new_hire_mask_plot, x_col],
                y=plot_df.loc[new_hire_mask_plot, "Årslønn"],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2.5, color="#e07b00")),
                name=f"🟠 Nyansatt (fra {cutoff_date_value.strftime('%d.%m.%Y')})",
                showlegend=True,
                hoverinfo="skip",
            )

        for kode in selected_codes:
            line_color = code_color_map[str(kode)]

            if show_regression_with_new_hires:
                r = reg_results.get(kode)
                code_points = clean_filtered_df[clean_filtered_df["Stillingskode"] == kode]
                if r and r["slope"] is not None and not code_points.empty:
                    x_range = np.linspace(code_points[x_col].min(), code_points[x_col].max(), 50)
                    y_range = r["intercept"] + r["slope"] * x_range

                    ci_lower, ci_upper = analysis.confidence_band(r, x_range)
                    if ci_lower is not None:
                        fig.add_scatter(
                            x=x_range, y=ci_upper, mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip",
                        )
                        fig.add_scatter(
                            x=x_range, y=ci_lower, mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_hex_to_rgba(line_color, 0.12),
                            name=f"95% konfidensintervall {kode}", showlegend=False, hoverinfo="skip",
                        )

                    fig.add_scatter(
                        x=x_range,
                        y=y_range,
                        mode="lines",
                        name=f"Trend {kode} (R²={r['r_squared']:.2f})",
                        line=dict(width=2, color=line_color),
                    )

            if show_regression_without_new_hires:
                r2 = reg_results_excl_new.get(kode)
                code_points_excl = clean_filtered_df[
                    (clean_filtered_df["Stillingskode"] == kode) & (~clean_filtered_df["Er Nyansatt"])
                ]
                if r2 and r2["slope"] is not None and not code_points_excl.empty:
                    x_range2 = np.linspace(code_points_excl[x_col].min(), code_points_excl[x_col].max(), 50)
                    y_range2 = r2["intercept"] + r2["slope"] * x_range2

                    ci_lower2, ci_upper2 = analysis.confidence_band(r2, x_range2)
                    if ci_lower2 is not None:
                        fig.add_scatter(
                            x=x_range2, y=ci_upper2, mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip",
                        )
                        fig.add_scatter(
                            x=x_range2, y=ci_lower2, mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_hex_to_rgba(line_color, 0.08),
                            name=f"95% konfidensintervall {kode} (uten nyansatte)", showlegend=False, hoverinfo="skip",
                        )

                    fig.add_scatter(
                        x=x_range2,
                        y=y_range2,
                        mode="lines",
                        name=f"Trend {kode} uten nyansatte (R²={r2['r_squared']:.2f})",
                        line=dict(width=2, color=line_color, dash="dash"),
                    )

        fig.update_layout(height=750, xaxis_title=x_label, clickmode="event+select")
        click_event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", selection_mode=("points",), key="main_scatter_chart"
        )

        clicked_name = None
        if click_event and click_event.get("selection") and click_event["selection"].get("points"):
            customdata = click_event["selection"]["points"][0].get("customdata")
            if customdata:
                clicked_name = customdata[0]

        if clicked_name and clicked_name != currently_selected:
            match = filtered_df[filtered_df["Fullt Navn"] == clicked_name]
            if not match.empty:
                # "employee_selector" er IKKE instansiert ennå denne kjøringen (col_stats
                # kjører etter col_plot i koden), så det er trygt å skrive rett inn i widgetens
                # egen session_state-nøkkel her. Å kun fjerne nøkkelen og stole på "index=" (som
                # tidligere) var upålitelig - nedtrekksmenyen kunne fortsatt vise sin gamle,
                # persisterte verdi og til og med overskrive valget igjen ved neste rerun.
                st.session_state["employee_selector"] = clicked_name
                st.session_state.selected_employee = clicked_name
                st.rerun()

    with col_stats:
        st.markdown("##### 📊 Statistisk Sammendrag")
        codes_text = ", ".join(str(c) for c in selected_codes) if selected_codes else "ingen"
        st.caption(f"Valgte koder: {codes_text} · {len(clean_filtered_df)} ansatte (ekskl. outliers)")

        if not selected_codes:
            st.info("Velg én eller flere stillingskoder for å fylle inn statistikk og graf.")
        elif clean_filtered_df.empty:
            st.info("Alle valgte ansatte er markert som outliers - ingen datagrunnlag for statistikk.")

        salaries = clean_filtered_df["Årslønn"]

        def _fmt_kr(value):
            return "–" if pd.isna(value) else f"kr {value:,.0f}".replace(",", " ")

        # Tabellen vises alltid, med "–" som verdi når det ikke er datagrunnlag.
        stats_rows = [
            {"Nøkkeltall": "Median", "Verdi": _fmt_kr(salaries.median())},
            {"Nøkkeltall": "Gjennomsnitt", "Verdi": _fmt_kr(salaries.mean())},
            {"Nøkkeltall": "Standardavvik", "Verdi": _fmt_kr(salaries.std())},
            {"Nøkkeltall": "Kvartil 1 (25%)", "Verdi": _fmt_kr(salaries.quantile(0.25))},
            {"Nøkkeltall": "Kvartil 3 (75%)", "Verdi": _fmt_kr(salaries.quantile(0.75))},
        ]
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True, height=210)

        with st.popover("📈 Regresjon per stillingskode", use_container_width=True):
            st.caption(
                "Outliers ekskludert. Std.avvik = spredningen (residualene) rundt trendlinjen. Det skraverte "
                "området i grafen er et 95% konfidensintervall for forventet lønn - det snevres inn mot "
                "gjennomsnittlig ansiennitet i datagrunnlaget (der presisjonen er størst) og videre ut mot "
                "ytterpunktene."
            )
            reg_table = []
            for kode in selected_codes:
                r = reg_results.get(kode)
                if r and r["slope"] is not None:
                    reg_table.append(
                        {
                            "Stillingskode": kode,
                            "N": r["n"],
                            "Stigning (kr/år)": f"{r['slope']:,.0f}".replace(",", " "),
                            "R²": f"{r['r_squared']:.2f}",
                            "Std.avvik (kr)": f"{r['std_residual']:,.0f}".replace(",", " "),
                        }
                    )
                else:
                    reg_table.append(
                        {
                            "Stillingskode": kode,
                            "N": r["n"] if r else 0,
                            "Stigning (kr/år)": "–",
                            "R²": "–",
                            "Std.avvik (kr)": "–",
                        }
                    )
            st.dataframe(
                pd.DataFrame(reg_table, columns=["Stillingskode", "N", "Stigning (kr/år)", "R²", "Std.avvik (kr)"]),
                hide_index=True,
                use_container_width=True,
            )

        all_employee_names = df.sort_values(by="Etternavn")["Fullt Navn"].drop_duplicates().tolist()
        code_by_name = df.drop_duplicates("Fullt Navn").set_index("Fullt Navn")["Stillingskode"].to_dict()

        # Samme prinsipp som for stillingskode-filteret over: skriv den lagrede verdien
        # direkte inn i "employee_selector" sin egen nøkkel FØR widgeten instansieres,
        # i stedet for å stole på "index=" (som ignoreres hvis nøkkelen alt finnes).
        pending_hash = st.session_state.pop("pending_selected_employee_hash", None)
        if pending_hash and "employee_selector" not in st.session_state:
            match = df[df["_hash"] == pending_hash]
            if not match.empty:
                st.session_state["employee_selector"] = match.iloc[0]["Fullt Navn"]
                st.session_state.selected_employee = match.iloc[0]["Fullt Navn"]

        # Streamlits innebygde søk i nedtrekksmenyen er "fuzzy" (f.eks. gir "Hauge" også
        # treff på "Tram Vuong") og kan ikke styres. Derfor et eget søkefelt med presis
        # matching som søker i hele datasettet, mens menyen uten søk kun viser de valgte kodene.
        search_query = st.text_input(
            "🔍 Søk etter ansatt (alle stillingskoder):",
            key="employee_search",
            placeholder="Fornavn, mellomnavn og/eller etternavn",
            help="Hvert søkeord må matche starten av et av navnene, f.eks. finner \"ing hau\" "
            "Ingrid Haug. Uten søk viser menyen under kun ansatte i valgte stillingskode(r).",
        )
        if search_query.strip():
            employee_names = [n for n in all_employee_names if analysis.name_matches(n, search_query)]
            if not employee_names:
                st.caption("Ingen ansatte matcher søket.")
        elif selected_codes:
            in_codes = set(filtered_df["Fullt Navn"])
            employee_names = [n for n in all_employee_names if n in in_codes]
        else:
            employee_names = all_employee_names

        # Nåværende valg må alltid finnes blant alternativene, ellers nullstiller Streamlit menyen.
        for current in (st.session_state.get("employee_selector"), st.session_state.get("selected_employee")):
            if current and current in code_by_name and current not in employee_names:
                employee_names = [current] + employee_names
        employee_options = [None] + employee_names

        # Streamlit regner en selectbox med endrede alternativer (nytt søk / nye koder) som
        # en ny widget og nullstiller verdien. Skriv nåværende verdi eksplisitt tilbake før
        # widgeten instansieres, slik at valget overlever at listen endres.
        current_selection = st.session_state.get("employee_selector")
        if current_selection in employee_options:
            st.session_state["employee_selector"] = current_selection

        def _clear_employee_search():
            # Tøm søket når en ansatt er valgt, slik at menyen igjen viser valgt(e) kode(r).
            st.session_state["employee_search"] = ""

        new_selected_employee = st.selectbox(
            "👤 Velg ansatt for detaljvisning:" if not search_query.strip() else "👤 Velg blant søketreff:",
            options=employee_options,
            format_func=lambda x: "Ingen valgt" if x is None else f"{x} ({code_by_name.get(x, '?')})",
            key="employee_selector",
            on_change=_clear_employee_search,
        )

        if new_selected_employee != st.session_state.get("selected_employee"):
            st.session_state.selected_employee = new_selected_employee
            new_code = code_by_name.get(new_selected_employee)
            if new_code is not None and new_code not in selected_codes:
                # Filteret er allerede instansiert denne kjøringen - koden legges til
                # i neste kjøring (se "pending_add_code" over stillingskode-filteret).
                st.session_state["pending_add_code"] = new_code
            st.rerun()

    # --- DETALJVISNING FOR ANSATT ---

    if st.session_state.selected_employee:
        st.markdown("---")
        st.markdown(f"#### Detaljvisning: {st.session_state.selected_employee}")

        col_settings, col_details = st.columns([1, 2])

        selected_row = df[df["Fullt Navn"] == st.session_state.selected_employee].iloc[0]
        stillingskode = selected_row["Stillingskode"]
        is_outlier = selected_row["_hash"] in set(st.session_state.outliers_by_code.get(str(stillingskode), []))

        with col_settings:
            st.markdown("**Velg kolonner for visning:**")
            cols_to_exclude = ["Fullt Navn", "Ansiennitet (År)", "Stillingsansiennitet (År)", "Forventet Lønn", "Lønnsavvik (Kr)", "_hash", "Is Outlier", "Er Nyansatt"]
            available_cols = [c for c in df.columns if c not in cols_to_exclude]

            st.session_state.display_columns = st.multiselect(
                "Velg hvilke rådata-kolonner som skal vises:",
                options=available_cols,
                default=[c for c in st.session_state.display_columns if c in available_cols] or available_cols[:5],
                key="column_selector",
            )

            st.markdown("---")
            outlier_toggle = st.checkbox(
                f"Marker som outlier for stillingskode {stillingskode}",
                value=is_outlier,
                key=f"outlier_toggle_{selected_row['_hash']}",
                help="Outliers ekskluderes fra regresjon og nøkkeltall for denne stillingskoden, og vises aldri i PDF-eksport.",
            )
            code_key = str(stillingskode)
            current_outliers = set(st.session_state.outliers_by_code.get(code_key, []))
            state_changed = False
            if outlier_toggle and selected_row["_hash"] not in current_outliers:
                current_outliers.add(selected_row["_hash"])
                state_changed = True
            elif not outlier_toggle and selected_row["_hash"] in current_outliers:
                current_outliers.discard(selected_row["_hash"])
                state_changed = True
            st.session_state.outliers_by_code[code_key] = sorted(current_outliers)
            if state_changed:
                # Regresjon/nøkkeltall lenger opp i scriptet ble beregnet med forrige
                # outlier-status - kjør på nytt slik at hele siden reflekterer endringen.
                st.rerun()

        with col_details:
            display_df = selected_row[st.session_state.display_columns].to_frame().T
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Lønnsavviksanalyse")

            if is_outlier:
                st.warning("Ansatt er markert som outlier for sin stillingskode og inngår ikke i regresjons- eller nøkkeltallsberegningen.")
            else:
                avvik = selected_row["Lønnsavvik (Kr)"]
                std_residual = reg_results.get(stillingskode, {}).get("std_residual")
                z_suffix = ""
                if pd.notna(avvik) and std_residual and std_residual > 1e-9:
                    z_suffix = f" (tilsvarer {abs(avvik) / std_residual:.1f} standardavvik)"

                if pd.isna(avvik):
                    st.info("For få datapunkter, eller ingen verdi for valgt x-akse, til å beregne forventet lønn.")
                elif avvik > 0:
                    st.success(f"Ansatt ligger **{avvik:,.0f} kr** over trendlinjen for sin {x_label.lower()} og stillingskode{z_suffix}.".replace(",", " "))
                elif avvik < 0:
                    st.error(f"Ansatt ligger **{abs(avvik):,.0f} kr** under trendlinjen for sin {x_label.lower()} og stillingskode{z_suffix}.".replace(",", " "))
                else:
                    st.info("Ansatt ligger nøyaktig på trendlinjen.")

                st.markdown("Dette avviket indikerer hvor langt personens lønn er fra det statistisk forventede lønnsnivået for stillingskoden (basert på OLS-regresjon, uten outliers).")

    # --- PDF-INNSTILLINGER ---

    st.markdown("---")
    st.markdown("### 📄 Eksporter anonymiserte PDF-rapporter")

    with st.expander("⚙️ Innhold i PDF-rapportene"):
        stats_options = [key for key, _ in KEY_LABELS]
        stats_labels = dict(KEY_LABELS)
        selected_stats = st.multiselect(
            "Nøkkeltall som skal vises:",
            options=stats_options,
            default=[k for k in st.session_state.pdf_options.get("stats_fields", stats_options) if k in stats_options],
            format_func=lambda k: stats_labels[k],
            key="pdf_stats_fields",
        )
        show_axis_values = st.checkbox(
            "Vis tallverdier på aksene i figuren",
            value=st.session_state.pdf_options.get("show_axis_values", True),
            key="pdf_show_axis_values",
        )
        show_avvik_text = st.checkbox(
            "Vis avviksforklaring (kr og standardavvik fra trendlinjen)",
            value=st.session_state.pdf_options.get("show_avvik_text", True),
            key="pdf_show_avvik_text",
        )

        saved_report_date = st.session_state.pdf_options.get("report_date")
        try:
            default_report_date = date.fromisoformat(saved_report_date) if saved_report_date else date.today()
        except ValueError:
            default_report_date = date.today()
        report_date_value = st.date_input(
            "Lønnsdata per dato (vises i PDF-en):",
            value=default_report_date,
            format="DD/MM/YYYY",
            key="pdf_report_date",
        )
        report_date_str = report_date_value.strftime("%d/%m-%Y")

        st.markdown("**Figurer i PDF (kryss av for de du vil ha med - hver gir én egen figur):**")
        if position_seniority_col:
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.caption("Ansiennitet siden tiltredelse")
                pdf_fig_tiltredelse_with = st.checkbox(
                    "Med nyansatte",
                    value=st.session_state.pdf_options.get("pdf_fig_tiltredelse_with", True),
                    key="pdf_fig_tiltredelse_with",
                )
                pdf_fig_tiltredelse_without = st.checkbox(
                    "Uten nyansatte",
                    value=st.session_state.pdf_options.get("pdf_fig_tiltredelse_without", False),
                    key="pdf_fig_tiltredelse_without",
                )
            with fig_col2:
                st.caption("Stillingsansiennitet")
                pdf_fig_stilling_with = st.checkbox(
                    "Med nyansatte",
                    value=st.session_state.pdf_options.get("pdf_fig_stilling_with", False),
                    key="pdf_fig_stilling_with",
                )
                pdf_fig_stilling_without = st.checkbox(
                    "Uten nyansatte",
                    value=st.session_state.pdf_options.get("pdf_fig_stilling_without", False),
                    key="pdf_fig_stilling_without",
                )
        else:
            pdf_fig_tiltredelse_with = st.checkbox(
                "Med nyansatte",
                value=st.session_state.pdf_options.get("pdf_fig_tiltredelse_with", True),
                key="pdf_fig_tiltredelse_with",
            )
            pdf_fig_tiltredelse_without = st.checkbox(
                "Uten nyansatte",
                value=st.session_state.pdf_options.get("pdf_fig_tiltredelse_without", False),
                key="pdf_fig_tiltredelse_without",
            )
            pdf_fig_stilling_with = False
            pdf_fig_stilling_without = False

        colleague_display_options = ["points", "heatmap", "hidden"]
        colleague_display_labels = {
            "points": "Enkeltpunkter (som i dag)",
            "heatmap": "Diffus tetthetssky (heatmap)",
            "hidden": "Skjul helt (vis kun deg selv)",
        }
        saved_colleague_display = st.session_state.pdf_options.get("colleague_display", "points")
        colleague_display = st.radio(
            "Kollegaer i figuren:",
            options=colleague_display_options,
            index=colleague_display_options.index(saved_colleague_display) if saved_colleague_display in colleague_display_options else 0,
            format_func=lambda v: colleague_display_labels[v],
            help="Trendlinje, konfidensintervall og eventuell snittlinje beregnes fortsatt på hele "
            "stillingskoden uansett valg - dette styrer kun hvordan kollegaenes lønnsdata vises i figuren.",
            horizontal=True,
            key="pdf_colleague_display",
        )
        show_mean_line = st.checkbox(
            "Vis linje for gjennomsnittlig lønn i stillingskoden",
            value=st.session_state.pdf_options.get("show_mean_line", False),
            key="pdf_show_mean_line",
        )

        st.session_state.pdf_options = {
            "stats_fields": selected_stats,
            "show_axis_values": show_axis_values,
            "show_avvik_text": show_avvik_text,
            "report_date": report_date_value.isoformat(),
            "colleague_display": colleague_display,
            "show_mean_line": show_mean_line,
            "pdf_fig_tiltredelse_with": pdf_fig_tiltredelse_with,
            "pdf_fig_tiltredelse_without": pdf_fig_tiltredelse_without,
            "pdf_fig_stilling_with": pdf_fig_stilling_with,
            "pdf_fig_stilling_without": pdf_fig_stilling_without,
        }

    pdf_options = st.session_state.pdf_options
    new_hire_cutoff_str = cutoff_date_value.strftime("%d.%m.%Y")

    TILTREDELSE_AXIS = ("Ansiennitet (År)", "Ansiennitet (år) siden tiltredelse")
    STILLING_AXIS = ("Stillingsansiennitet (År)", "Stillingsansiennitet (år)")
    pdf_x_axes = []
    if pdf_options.get("pdf_fig_tiltredelse_with", True):
        pdf_x_axes.append((*TILTREDELSE_AXIS, False))
    if pdf_options.get("pdf_fig_tiltredelse_without", False):
        pdf_x_axes.append((*TILTREDELSE_AXIS, True))
    if position_seniority_col:
        if pdf_options.get("pdf_fig_stilling_with", False):
            pdf_x_axes.append((*STILLING_AXIS, False))
        if pdf_options.get("pdf_fig_stilling_without", False):
            pdf_x_axes.append((*STILLING_AXIS, True))
    if not pdf_x_axes:
        # Ingen avkrysset - fall tilbake på standardfiguren slik at eksporten aldri blir tom.
        pdf_x_axes = [(*TILTREDELSE_AXIS, False)]

    if union_values:
        st.markdown("**Hurtigeksport per fagforening:**")
        for uv in union_values:
            members = sorted(df[df[union_col] == uv]["Fullt Navn"].tolist())
            btn_col, dl_col = st.columns([2, 2])
            with btn_col:
                if st.button(f"📄 Generer for alle {uv}-medlemmer ({len(members)} stk)", key=f"union_export_btn_{uv}"):
                    with st.spinner(f"Genererer PDF-rapporter for {uv}..."):
                        zip_bytes = build_export_zip(
                            members, df, st.session_state.outliers_by_code, "_hash", unit_col,
                            pdf_x_axes, pdf_options, report_date_str, new_hire_cutoff_str,
                        )
                    st.session_state[f"_union_zip_{uv}"] = zip_bytes
            with dl_col:
                if st.session_state.get(f"_union_zip_{uv}"):
                    st.download_button(
                        f"⬇️ Last ned {uv}-rapporter",
                        data=st.session_state[f"_union_zip_{uv}"],
                        file_name=f"{settlement_name}_{uv}_eksport.zip",
                        mime="application/zip",
                        key=f"union_dl_{uv}",
                    )
        st.markdown("---")

    st.caption(
        "Velg ansatte manuelt (f.eks. på tvers av stillingskoder). Hver PDF navngis etter og "
        "inneholder kun personopplysninger for den aktuelle ansatte. Analysen for hver ansatt baseres alltid "
        "på deres egen stillingskode og ekskluderer outliers - uavhengig av hvem som er valgt for eksport. "
        "Fagforeningstilhørighet sendes aldri med i PDF-en."
    )

    all_names_in_dataset = sorted(df["Fullt Navn"].tolist())
    export_selection = st.multiselect(
        "Velg ansatte for eksport:",
        options=all_names_in_dataset,
        key="export_selection",
    )

    if st.button("Generer PDF-rapporter", disabled=not export_selection):
        with st.spinner("Genererer PDF-rapporter..."):
            zip_bytes = build_export_zip(
                export_selection, df, st.session_state.outliers_by_code, "_hash", unit_col,
                pdf_x_axes, pdf_options, report_date_str, new_hire_cutoff_str,
            )
        st.session_state["_export_zip_bytes"] = zip_bytes

    if st.session_state.get("_export_zip_bytes"):
        st.download_button(
            "⬇️ Last ned zip med PDF-rapporter",
            data=st.session_state["_export_zip_bytes"],
            file_name=f"{settlement_name}_anonymisert_eksport.zip",
            mime="application/zip",
        )

    # --- LAGRE OPPGJØR ---

    st.session_state.data_loaded_ok = True

    selected_hash = None
    if st.session_state.selected_employee:
        match = df[df["Fullt Navn"] == st.session_state.selected_employee]
        if not match.empty:
            selected_hash = match.iloc[0]["_hash"]

    store.save_settlement(
        settlement_name,
        {
            "file_path": st.session_state.file_path_input,
            "column_mapping": column_mapping,
            "outliers": st.session_state.outliers_by_code,
            "last_selected_employee_hash": selected_hash,
            "selected_codes": list(selected_codes),
            "multi_code_mode": st.session_state.multi_code_mode,
            "display_columns": st.session_state.display_columns,
            "x_axis_choice": x_col,
            "pdf_options": pdf_options,
            "new_hire_settings": {
                "cutoff_date": cutoff_date_value.isoformat(),
                "show_regression_with_new_hires": show_regression_with_new_hires,
                "show_regression_without_new_hires": show_regression_without_new_hires,
                "hide_new_hires_live": hide_new_hires_live,
            },
        },
    )

except Exception as e:
    st.error(f"En feil oppstod under lasting eller behandling av filen: {e}")
    st.info("Vennligst sjekk at filen er en gyldig Excel-arbeidsbok (.xlsx/.xls) og at kolonnenavnene er riktige.")
