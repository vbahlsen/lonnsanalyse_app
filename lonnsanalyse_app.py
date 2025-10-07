import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from scipy.stats import linregress

# --- FUNKSJONER ---

def calculate_years_of_service(start_date):
    """Beregner ansiennitet i år fra en gitt startdato (dd.mm.åååå)."""
    try:
        if pd.isna(start_date):
            return 0.0
        
        # Forsøk å konvertere til datetime-objekt
        if isinstance(start_date, str):
            # Antar formatet dd.mm.åååå for strenger
            date_obj = datetime.strptime(start_date, '%d.%m.%Y')
        elif isinstance(start_date, datetime):
            date_obj = start_date
        else:
            return 0.0 # Ukjent format

        today = datetime.now()
        # Beregner forskjellen og konverterer til antall år
        diff = today - date_obj
        return round(diff.days / 365.25, 2)
    except Exception:
        return 0.0 # Returner 0.0 ved feil i datoformat

def run_regression_analysis(df):
    """Utfører lineær regresjon (Årslønn vs. Ansiennitet) og beregner avvik."""
    if df.empty or len(df) < 2:
        return None

    # Bruk linregress fra scipy for OLS (Ordinary Least Squares)
    slope, intercept, r_value, p_value, std_err = linregress(df['Ansiennitet (År)'], df['Årslønn'])

    # Beregn forventet lønn og avvik
    df['Forventet Lønn'] = intercept + slope * df['Ansiennitet (År)']
    df['Lønnsavvik (Kr)'] = df['Årslønn'] - df['Forventet Lønn']
    
    # R-kvadrert
    r_squared = r_value**2
    
    return slope, intercept, r_squared

# --- INITIALISERING AV SESJONSSTATE ---

# Initialiserer tilstander som brukes til å huske brukerinnstillinger (som synlige kolonner)
if 'display_columns' not in st.session_state:
    st.session_state.display_columns = ['Fornavn', 'Etternavn', 'Stillingskode', 'Tiltredelsesdato', 'Årslønn']
if 'selected_employee' not in st.session_state:
    st.session_state.selected_employee = None

# --- HOVEDAPPLIKASJON ---

st.set_page_config(layout="wide", page_title="Lønnsnivåanalyse mot Ansiennitet")

# Viser en melding for å bekrefte at appen har startet
st.sidebar.info("Last opp filen for å starte analysen.")
st.title("💰 Lønnsnivåanalyse mot Ansiennitet")

uploaded_file = st.file_uploader("Last opp Excel-fil (.xlsx, .xls) med lønnsdata:", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Last inn data
        df = pd.read_excel(uploaded_file)
        
        # --- DATAPREPPERASJON ---
        
        # Rensing og beregning av Ansiennitet
        df.columns = [col.strip() for col in df.columns]
        
        # Sikkerhetsjekk på påkrevde kolonner
        required_cols = ['Etternavn', 'Fornavn', 'Stillingskode', 'Tiltredelsesdato', 'Årslønn']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Excel-filen mangler følgende påkrevde kolonner: {', '.join(missing_cols)}. Vennligst sjekk kolonnenavnene.")
            st.stop()
            
        df['Ansiennitet (År)'] = df['Tiltredelsesdato'].apply(calculate_years_of_service)
        df['Fullt Navn'] = df['Fornavn'] + ' ' + df['Etternavn']
        
        # Filtrerer ut rader med ugyldig eller manglende lønn/ansiennitet for analyse
        df = df.dropna(subset=['Årslønn', 'Ansiennitet (År)'])
        df['Årslønn'] = pd.to_numeric(df['Årslønn'], errors='coerce')
        
        # --- SIDEPANEL FILTER ---
        
        st.sidebar.header("Filter & Innstillinger")
        
        # Velger Stillingskoder
        all_codes = sorted(df['Stillingskode'].unique())
        selected_codes = st.sidebar.multiselect(
            "Velg Stillingskode(r) for analyse:",
            options=all_codes,
            default=all_codes # Velg alle som standard
        )

        # Filtrer data basert på valgte koder
        filtered_df = df[df['Stillingskode'].isin(selected_codes)].copy()
        
        if filtered_df.empty:
            st.warning("Ingen ansatte funnet for de valgte stillingskodene.")
            st.stop()
        
        # --- DATAANALYSE ---
        
        # Utfør regresjonsanalyse og beregn avvik
        reg_results = run_regression_analysis(filtered_df)
        if reg_results:
            slope, intercept, r_squared = reg_results
        
        # --- LAYOUT MED KOLONNER ---
        
        col_stats, col_plot = st.columns([1, 2])

        # KOLONNE 1: STATISTIKK OG ANSATT-LISTE
        with col_stats:
            st.subheader("📊 Statistisk Sammendrag")
            
            # Beregn statistikk
            median_lonn = filtered_df['Årslønn'].median()
            mean_lonn = filtered_df['Årslønn'].mean()
            q1 = filtered_df['Årslønn'].quantile(0.25)
            q3 = filtered_df['Årslønn'].quantile(0.75)

            st.markdown(f"**Valgte koder:** {', '.join(selected_codes)}")
            st.markdown(f"**Antall ansatte:** {len(filtered_df)}")
            st.markdown(f"**Median Årslønn:** kr {median_lonn:,.0f}".replace(",", " "))
            st.markdown(f"**Gjennomsnitt:** kr {mean_lonn:,.0f}".replace(",", " "))
            st.markdown(f"**Kvartil 1 (25%):** kr {q1:,.0f}".replace(",", " "))
            st.markdown(f"**Kvartil 3 (75%):** kr {q3:,.0f}".replace(",", " "))

            if reg_results:
                st.markdown("---")
                st.markdown(f"**Regresjon (Ansiennitet vs. Lønn):**")
                
                # FIKS: Bruker rå strenger (r"...") og separerer f-string interpolasjon for å unngå
                # SyntaxError pga. Escaping av backslash i LaTeX-formelen under deployering.
                # Setter variabelen i en separat f-string med dollartegn for å sikre riktig LaTeX-formatering.
                st.markdown(r"- $\text{Stigning (kr/år):}$ $\text{kr}$ " + f"${slope:,.0f}$".replace(",", " "))
                st.markdown(r"- $\text{R}^{2} \text{(Forklart variasjon):}$ " + f"${r_squared:.2f}$")
                
                # Sorter listen etter navn
                employee_names = filtered_df.sort_values(by='Etternavn')['Fullt Navn'].tolist()
                
                st.markdown("---")
                st.subheader("👤 Ansatt-oversikt")
                
                # Ansattvalg
                st.session_state.selected_employee = st.selectbox(
                    "Velg ansatt for detaljvisning:",
                    options=[None] + employee_names, # Legg til 'None' for ingen valg
                    index=0,
                    format_func=lambda x: "Ingen valgt" if x is None else x
                )

        # KOLONNE 2: SCATTERPLOT
        with col_plot:
            st.subheader("📈 Lønn vs. Ansiennitet med Regresjon")
            
            # Sett farge og størrelse basert på valg
            filtered_df['Color'] = filtered_df['Stillingskode']
            filtered_df['Size'] = 10 
            
            if st.session_state.selected_employee:
                filtered_df.loc[filtered_df['Fullt Navn'] == st.session_state.selected_employee, 'Size'] = 20
                filtered_df.loc[filtered_df['Fullt Navn'] == st.session_state.selected_employee, 'Color'] = '🔴 VALGT ANSATT'

            # Sikkerhetskopierer de valgte kodene før vi legger til "VALGT ANSATT"
            plot_codes = selected_codes + ['🔴 VALGT ANSATT']
            
            # Plotly scatterplot
            fig = px.scatter(
                filtered_df,
                x='Ansiennitet (År)',
                y='Årslønn',
                color='Color',
                size='Size', # Bruker 'Size' for å fremheve valgt ansatt
                hover_data=['Fullt Navn', 'Stillingskode', 'Lønnsavvik (Kr)'],
                title=f"Årslønn mot Ansiennitet for {', '.join(selected_codes)}",
                color_discrete_map={'🔴 VALGT ANSATT': 'red'},
                category_orders={"Color": [c for c in plot_codes if c != '🔴 VALGT ANSATT'] + ['🔴 VALGT ANSATT']}
            )

            # Legg til regresjonslinje (Lønn = intercept + slope * Ansiennitet)
            if reg_results:
                x_range = np.linspace(filtered_df['Ansiennitet (År)'].min(), filtered_df['Ansiennitet (År)'].max(), 100)
                y_range = intercept + slope * x_range
                
                # Legger til OLS linje (trendlinje)
                fig.add_scatter(x=x_range, y=y_range, mode='lines', 
                                name=f'Trendlinje (R²={r_squared:.2f})', 
                                line=dict(color='black', width=2))
                
                # Legg til konfidensintervall (Dette er en forenklet visualisering for illustrasjon)
                # For enkelhet skyld bruker vi standardavviket til residualene (Lønnsavvik) for CI-bånd.
                std_residual = filtered_df['Lønnsavvik (Kr)'].std()
                
                fig.add_scatter(x=x_range, y=y_range + 1.96 * std_residual, mode='lines', 
                                name='95% Konfidensbånd', line=dict(color='gray', width=1, dash='dash'), 
                                showlegend=False)
                fig.add_scatter(x=x_range, y=y_range - 1.96 * std_residual, mode='lines', 
                                name='95% Konfidensbånd', line=dict(color='gray', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)', showlegend=False)

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

        # --- DETALJVISNING FOR ANSATT OG INNSTILLINGER ---

        if st.session_state.selected_employee:
            st.markdown("---")
            st.subheader(f"Detaljvisning: {st.session_state.selected_employee}")

            # Splitt i to kolonner: Valg av kolonner og Ansatt-data
            col_settings, col_details = st.columns([1, 2])

            with col_settings:
                st.markdown("**Velg kolonner for visning:**")
                # Hent alle tilgjengelige kolonner
                all_raw_cols = df.columns.tolist()
                
                # Fjern automatisk genererte kolonner
                cols_to_exclude = ['Fullt Navn', 'Ansiennitet (År)', 'Forventet Lønn', 'Lønnsavvik (Kr)', 'Color', 'Size']
                available_cols = [col for col in all_raw_cols if col not in cols_to_exclude]

                # Multiselect for å velge synlige kolonner (lagres i session_state)
                st.session_state.display_columns = st.multiselect(
                    "Velg hvilke rådata-kolonner som skal vises:",
                    options=available_cols,
                    default=st.session_state.display_columns,
                    key='column_selector'
                )

            with col_details:
                selected_data = filtered_df[filtered_df['Fullt Navn'] == st.session_state.selected_employee].iloc[0]
                
                # Vis data
                display_df = selected_data[st.session_state.display_columns].to_frame().T
                st.dataframe(display_df, hide_index=True, use_container_width=True)

                # Forslag til relevant analyse: Lønnsavvik
                st.markdown("---")
                st.markdown("#### Lønnsavviksanalyse")
                
                avvik = selected_data['Lønnsavvik (Kr)']

                if avvik > 0:
                    st.success(f"Ansatt ligger **{avvik:,.0f} kr** over trendlinjen for sin ansiennitet og stillingskode.")
                elif avvik < 0:
                    st.error(f"Ansatt ligger **{abs(avvik):,.0f} kr** under trendlinjen for sin ansiennitet og stillingskode.")
                else:
                    st.info("Ansatt ligger nøyaktig på trendlinjen.")

                st.markdown("Dette avviket indikerer hvor langt personens lønn er fra det statistisk forventede lønnsnivået for den ansienniteten (basert på OLS-regresjonen).")

    except Exception as e:
        st.error(f"En feil oppstod under lasting eller behandling av filen: {e}")
        st.info("Vennligst sjekk at filen er en gyldig Excel-arbeidsbok (.xlsx/.xls) og at kolonnenavnene er riktige.")
        
else:
    # Vises når filen ikke er lastet opp
    st.info("Last opp din Excel-fil for å starte analysen og visualiseringen.")
