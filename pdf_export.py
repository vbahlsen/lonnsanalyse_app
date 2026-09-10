"""Anonymisert PDF-eksport: én PDF per ansatt, navngitt etter dem, med kun
deres egne personopplysninger. Andre ansattes navn vises aldri i en PDF."""

import io
import re
import zipfile

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analysis import fit_linear_regression, summary_stats

KEY_LABELS = [
    ("min", "Min"),
    ("q1", "1. kvartil (25%)"),
    ("median", "Median"),
    ("mean", "Gjennomsnitt"),
    ("q3", "3. kvartil (75%)"),
    ("max", "Max"),
    ("std", "Standardavvik"),
]


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-æøåÆØÅ ]", "", name).strip().replace(" ", "_")
    return cleaned or "ansatt"


def _format_kr(value) -> str:
    return f"kr {value:,.0f}".replace(",", " ")


def _render_chart(employee_row, code_clean_df, is_outlier: bool, fit: dict | None) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.scatter(
        code_clean_df["Ansiennitet (År)"],
        code_clean_df["Årslønn"],
        color="#7a8a99",
        s=40,
        label="Kollegaer i samme stillingskode",
        zorder=2,
    )

    if fit:
        x_range = [code_clean_df["Ansiennitet (År)"].min(), code_clean_df["Ansiennitet (År)"].max()]
        y_range = [fit["intercept"] + fit["slope"] * x for x in x_range]
        ax.plot(x_range, y_range, color="black", linewidth=2, label=f"Trendlinje (R²={fit['r_squared']:.2f})")

        std = fit["std_residual"]
        if std and std > 1e-9:
            upper = [y + 1.96 * std for y in y_range]
            lower = [y - 1.96 * std for y in y_range]
            ax.fill_between(x_range, lower, upper, color="black", alpha=0.08, label="95% referanseintervall")

    ax.scatter(
        [employee_row["Ansiennitet (År)"]],
        [employee_row["Årslønn"]],
        color="#d1495b",
        s=140,
        edgecolor="black",
        zorder=3,
        label="Deg",
    )
    ax.annotate(
        employee_row["Fullt Navn"],
        (employee_row["Ansiennitet (År)"], employee_row["Årslønn"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=11,
        fontweight="bold",
    )

    if is_outlier:
        ax.set_title("Ekskludert fra trendlinjeberegning (markert som outlier)", fontsize=11, color="#d1495b")

    ax.set_xlabel("Ansiennitet (år)")
    ax.set_ylabel("Årslønn (kr)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_employee_pdf(employee_row, code_clean_df, unit_col: str | None, is_outlier: bool) -> bytes:
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    elements = []

    elements.append(Paragraph(f"Lønnsanalyse: {employee_row['Fullt Navn']}", styles["Title"]))
    elements.append(Spacer(1, 0.4 * cm))

    personalia_rows = [
        ["Navn", employee_row["Fullt Navn"]],
        ["Årslønn", _format_kr(employee_row["Årslønn"])],
        ["Stillingskode", str(employee_row["Stillingskode"])],
        ["Tiltredelsesdato", str(employee_row["Tiltredelsesdato"])],
    ]
    if unit_col and unit_col in employee_row and not employee_row.isna().get(unit_col, True):
        personalia_rows.insert(2, ["Ansattenhet", str(employee_row[unit_col])])
    if "Stillingsansiennitet (År)" in employee_row.index and pd.notna(employee_row["Stillingsansiennitet (År)"]):
        personalia_rows.append(["Stillingsansiennitet", f"{employee_row['Stillingsansiennitet (År)']:.1f} år"])

    personalia_table = Table(personalia_rows, colWidths=[5 * cm, 10 * cm])
    personalia_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]
        )
    )
    elements.append(personalia_table)
    elements.append(Spacer(1, 0.6 * cm))

    fit = fit_linear_regression(code_clean_df["Ansiennitet (År)"], code_clean_df["Årslønn"])

    elements.append(Paragraph("Lønn vs. ansiennitet for stillingskoden", styles["Heading2"]))
    chart_buf = _render_chart(employee_row, code_clean_df, is_outlier, fit)
    elements.append(Image(chart_buf, width=16 * cm, height=9.14 * cm))
    elements.append(Spacer(1, 0.4 * cm))

    if fit and not is_outlier:
        avvik = employee_row["Årslønn"] - (fit["intercept"] + fit["slope"] * employee_row["Ansiennitet (År)"])
        retning = "over" if avvik > 0 else "under" if avvik < 0 else "på"
        z_text = ""
        if fit["std_residual"] and fit["std_residual"] > 1e-9:
            z = abs(avvik) / fit["std_residual"]
            z_text = f", tilsvarende {z:.1f} standardavvik {retning} trendlinjen"
        elements.append(
            Paragraph(
                f"Avvik fra trendlinje: {_format_kr(abs(avvik))} {retning} forventet lønnsnivå for "
                f"ansiennitet og stillingskode{z_text}.",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph(f"Nøkkeltall for stillingskode {employee_row['Stillingskode']}", styles["Heading2"]))
    stats = summary_stats(code_clean_df)
    stats_rows = [["Nøkkeltall", "Verdi"]] + [
        [label, _format_kr(stats[key])] for key, label in KEY_LABELS
    ]
    stats_rows.append(["Antall i grunnlaget (uten outliers)", str(stats["n"])])
    stats_table = Table(stats_rows, colWidths=[8 * cm, 7 * cm])
    stats_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(stats_table)

    if is_outlier:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(
            Paragraph(
                "Merk: denne ansatte er markert som outlier for sin stillingskode og "
                "inngår derfor ikke i trendlinje- eller nøkkeltallsberegningen over.",
                styles["Italic"],
            )
        )

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


def build_export_zip(selected_employees: list[str], df, outliers_by_code: dict, hash_col: str, unit_col: str | None) -> bytes:
    from analysis import non_outlier_rows

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names = {}
        for full_name in selected_employees:
            employee_row = df[df["Fullt Navn"] == full_name].iloc[0]
            stillingskode = employee_row["Stillingskode"]
            code_clean_df = non_outlier_rows(df, stillingskode, outliers_by_code, hash_col)
            is_outlier = employee_row[hash_col] in set(outliers_by_code.get(str(stillingskode), []))

            pdf_bytes = build_employee_pdf(employee_row, code_clean_df, unit_col, is_outlier)

            base_name = _sanitize_filename(full_name)
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            filename = f"{base_name}.pdf" if count == 0 else f"{base_name}_{count + 1}.pdf"

            zf.writestr(filename, pdf_bytes)

    zip_buf.seek(0)
    return zip_buf.getvalue()
