import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import analysis
import settlement_store as store
from pdf_export import build_export_zip

st.set_page_config(layout="wide", page_title="Lønnsnivåanalyse mot Ansiennitet")
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
    # hvilken "value"/"index" vi sender inn - uten dette ville f.eks. filsti, valgt ansatt
    # og outlier-avkrysninger fra forrige oppgjør lekke inn i det nye.
    stale_prefixes = ("outlier_toggle_", "map_")
    stale_keys = ("file_path_widget", "employee_selector", "column_selector", "export_selection")
    for key in list(st.session_state.keys()):
        if key in stale_keys or key.startswith(stale_prefixes):
            del st.session_state[key]

    settlement = store.load_settlement(picked)
    st.session_state.settlement_name = picked
    st.session_state.outliers_by_code = settlement.get("outliers") or {}
    st.session_state.column_mapping = settlement.get("column_mapping")
    st.session_state.file_path_input = settlement.get("file_path") or ""
    st.session_state.display_columns = settlement.get("display_columns") or analysis.REQUIRED_COLUMNS
    st.session_state.pending_selected_employee_hash = settlement.get("last_selected_employee_hash")
    st.session_state.pending_selected_codes = settlement.get("selected_codes")
    st.session_state.selected_employee = None
    st.session_state.pop("_export_zip_bytes", None)
    st.rerun()

settlement_name = st.session_state.settlement_name
st.sidebar.caption(f"Aktivt oppgjør: **{settlement_name}**")

st.session_state.setdefault("outliers_by_code", {})
st.session_state.setdefault("column_mapping", None)
st.session_state.setdefault("file_path_input", "")
st.session_state.setdefault("display_columns", analysis.REQUIRED_COLUMNS)
st.session_state.setdefault("selected_employee", None)

# --- DATAFIL ---

st.sidebar.header("📄 Datafil")
path_input = st.sidebar.text_input(
    "Filsti til lønnsdata (Excel)",
    value=st.session_state.file_path_input,
    key="file_path_widget",
    help="Siden appen kjører lokalt kan den lese filen direkte fra disk og huske stien til neste økt.",
)

df_raw = None

if path_input.strip():
    resolved = store.resolve_data_file(path_input.strip())
    if resolved:
        try:
            df_raw = pd.read_excel(resolved)
            st.session_state.file_path_input = str(resolved)
        except Exception as e:
            st.sidebar.error(f"Klarte ikke å lese filen: {e}")
    else:
        st.sidebar.warning(f"Finner ikke filen på oppgitt sti:\n\n`{path_input}`\n\nSjekk stien, eller last opp filen manuelt under.")

if df_raw is None:
    uploaded_file = st.sidebar.file_uploader("...eller last opp Excel-fil manuelt:", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file)
        st.sidebar.info("Lastet opp manuelt. Oppgi filstien i feltet over for at appen skal huske filen til neste økt.")

if df_raw is None:
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

    df["Stillingskode"] = df["Stillingskode"].astype(str).str.strip()
    df["Ansiennitet (År)"] = df["Tiltredelsesdato"].apply(analysis.calculate_years_of_service)
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

    # --- REGRESJON PER STILLINGSKODE (alltid på hele datasettet) ---

    reg_results = analysis.run_regression_per_code(df, st.session_state.outliers_by_code, hash_col="_hash")

    # --- SIDEPANEL FILTER ---

    st.sidebar.header("Filter & Innstillinger")

    all_codes = sorted(df["Stillingskode"].unique(), key=str)
    pending_codes = st.session_state.get("pending_selected_codes")
    default_codes = [c for c in pending_codes if c in all_codes] if pending_codes else all_codes
    if not default_codes:
        default_codes = all_codes

    selected_codes = st.sidebar.multiselect(
        "Velg Stillingskode(r) for analyse (kun visning - regresjon kjøres alltid per kode):",
        options=all_codes,
        default=default_codes,
    )
    st.session_state.pending_selected_codes = None

    filtered_df = df[df["Stillingskode"].isin(selected_codes)].copy()

    if filtered_df.empty:
        st.warning("Ingen ansatte funnet for de valgte stillingskodene.")
        st.stop()

    def _is_outlier(row):
        return row["_hash"] in set(st.session_state.outliers_by_code.get(str(row["Stillingskode"]), []))

    filtered_df["Is Outlier"] = filtered_df.apply(_is_outlier, axis=1)
    clean_filtered_df = filtered_df[~filtered_df["Is Outlier"]]

    # --- LAYOUT MED KOLONNER ---

    col_stats, col_plot = st.columns([1, 2])

    with col_stats:
        st.subheader("📊 Statistisk Sammendrag")

        if clean_filtered_df.empty:
            st.info("Alle valgte ansatte er markert som outliers - ingen datagrunnlag for statistikk.")
        else:
            median_lonn = clean_filtered_df["Årslønn"].median()
            mean_lonn = clean_filtered_df["Årslønn"].mean()
            q1 = clean_filtered_df["Årslønn"].quantile(0.25)
            q3 = clean_filtered_df["Årslønn"].quantile(0.75)

            st.markdown(f"**Valgte koder:** {', '.join(str(c) for c in selected_codes)}")
            st.markdown(f"**Antall ansatte (ekskl. outliers):** {len(clean_filtered_df)}")
            st.markdown(f"**Median Årslønn:** kr {median_lonn:,.0f}".replace(",", " "))
            st.markdown(f"**Gjennomsnitt:** kr {mean_lonn:,.0f}".replace(",", " "))
            st.markdown(f"**Kvartil 1 (25%):** kr {q1:,.0f}".replace(",", " "))
            st.markdown(f"**Kvartil 3 (75%):** kr {q3:,.0f}".replace(",", " "))

        st.markdown("---")
        st.markdown("**Regresjon per stillingskode (outliers ekskludert):**")
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
                    }
                )
            else:
                reg_table.append({"Stillingskode": kode, "N": r["n"] if r else 0, "Stigning (kr/år)": "–", "R²": "–"})
        st.dataframe(pd.DataFrame(reg_table), hide_index=True, use_container_width=True)

        employee_names = filtered_df.sort_values(by="Etternavn")["Fullt Navn"].tolist()

        st.markdown("---")
        st.subheader("👤 Ansatt-oversikt")

        pending_hash = st.session_state.get("pending_selected_employee_hash")
        default_employee = None
        if pending_hash:
            match = filtered_df[filtered_df["_hash"] == pending_hash]
            if not match.empty:
                default_employee = match.iloc[0]["Fullt Navn"]

        employee_options = [None] + employee_names
        default_idx = employee_options.index(default_employee) if default_employee in employee_options else 0

        st.session_state.selected_employee = st.selectbox(
            "Velg ansatt for detaljvisning:",
            options=employee_options,
            index=default_idx,
            format_func=lambda x: "Ingen valgt" if x is None else x,
            key="employee_selector",
        )
        st.session_state.pending_selected_employee_hash = None

    with col_plot:
        st.subheader("📈 Lønn vs. Ansiennitet med Regresjon per Stillingskode")

        plot_df = filtered_df.copy()
        plot_df["Farge"] = plot_df["Stillingskode"].astype(str)
        plot_df["Størrelse"] = plot_df["Is Outlier"].map({True: 8, False: 12})
        plot_df["Symbol"] = plot_df["Is Outlier"].map({True: "circle-open", False: "circle"})

        if st.session_state.selected_employee:
            sel_mask = plot_df["Fullt Navn"] == st.session_state.selected_employee
            plot_df.loc[sel_mask, "Farge"] = "🔴 VALGT ANSATT"
            plot_df.loc[sel_mask, "Størrelse"] = 20

        fig = px.scatter(
            plot_df,
            x="Ansiennitet (År)",
            y="Årslønn",
            color="Farge",
            size="Størrelse",
            symbol="Symbol",
            symbol_map={"circle-open": "circle-open", "circle": "circle"},
            hover_data=["Fullt Navn", "Stillingskode", "Is Outlier", "Lønnsavvik (Kr)"],
            title="Årslønn mot Ansiennitet (åpen sirkel = markert outlier)",
            color_discrete_map={"🔴 VALGT ANSATT": "red"},
        )

        for kode in selected_codes:
            r = reg_results.get(kode)
            if not r or r["slope"] is None:
                continue
            code_points = clean_filtered_df[clean_filtered_df["Stillingskode"] == kode]
            if code_points.empty:
                continue
            x_range = np.linspace(code_points["Ansiennitet (År)"].min(), code_points["Ansiennitet (År)"].max(), 50)
            y_range = r["intercept"] + r["slope"] * x_range
            fig.add_scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name=f"Trend {kode} (R²={r['r_squared']:.2f})",
                line=dict(width=2),
            )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # --- DETALJVISNING FOR ANSATT ---

    if st.session_state.selected_employee:
        st.markdown("---")
        st.subheader(f"Detaljvisning: {st.session_state.selected_employee}")

        col_settings, col_details = st.columns([1, 2])

        selected_row = filtered_df[filtered_df["Fullt Navn"] == st.session_state.selected_employee].iloc[0]
        stillingskode = selected_row["Stillingskode"]
        is_outlier = selected_row["_hash"] in set(st.session_state.outliers_by_code.get(str(stillingskode), []))

        with col_settings:
            st.markdown("**Velg kolonner for visning:**")
            cols_to_exclude = ["Fullt Navn", "Ansiennitet (År)", "Forventet Lønn", "Lønnsavvik (Kr)", "_hash", "Is Outlier"]
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
                if pd.isna(avvik):
                    st.info("For få datapunkter i denne stillingskoden til å beregne forventet lønn.")
                elif avvik > 0:
                    st.success(f"Ansatt ligger **{avvik:,.0f} kr** over trendlinjen for sin ansiennitet og stillingskode.".replace(",", " "))
                elif avvik < 0:
                    st.error(f"Ansatt ligger **{abs(avvik):,.0f} kr** under trendlinjen for sin ansiennitet og stillingskode.".replace(",", " "))
                else:
                    st.info("Ansatt ligger nøyaktig på trendlinjen.")

                st.markdown("Dette avviket indikerer hvor langt personens lønn er fra det statistisk forventede lønnsnivået for stillingskoden (basert på OLS-regresjon, uten outliers).")

    # --- ANONYMISERT PDF-EKSPORT ---

    st.markdown("---")
    st.subheader("📄 Eksporter anonymiserte PDF-rapporter")
    st.caption(
        "Velg ansatte (f.eks. fagforeningsmedlemmer på tvers av stillingskoder). Hver PDF navngis etter og "
        "inneholder kun personopplysninger for den aktuelle ansatte. Analysen for hver ansatt baseres alltid "
        "på deres egen stillingskode og ekskluderer outliers - uavhengig av hvem som er valgt for eksport."
    )

    all_names_in_dataset = sorted(df["Fullt Navn"].tolist())
    export_selection = st.multiselect(
        "Velg ansatte for eksport:",
        options=all_names_in_dataset,
        key="export_selection",
    )

    if st.button("Generer PDF-rapporter", disabled=not export_selection):
        with st.spinner("Genererer PDF-rapporter..."):
            zip_bytes = build_export_zip(export_selection, df, st.session_state.outliers_by_code, "_hash", unit_col)
        st.session_state["_export_zip_bytes"] = zip_bytes

    if st.session_state.get("_export_zip_bytes"):
        st.download_button(
            "⬇️ Last ned zip med PDF-rapporter",
            data=st.session_state["_export_zip_bytes"],
            file_name=f"{settlement_name}_anonymisert_eksport.zip",
            mime="application/zip",
        )

    # --- LAGRE OPPGJØR ---

    selected_hash = None
    if st.session_state.selected_employee:
        match = filtered_df[filtered_df["Fullt Navn"] == st.session_state.selected_employee]
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
            "display_columns": st.session_state.display_columns,
        },
    )

except Exception as e:
    st.error(f"En feil oppstod under lasting eller behandling av filen: {e}")
    st.info("Vennligst sjekk at filen er en gyldig Excel-arbeidsbok (.xlsx/.xls) og at kolonnenavnene er riktige.")
