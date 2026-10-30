"""Røyktest: kjører hele Streamlit-appen mot en liten testfil med to ansatte som heter det samme."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import settlement_store as store

APP = str(Path(__file__).parent.parent / "lonnsanalyse_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    excel = tmp_path / "lonn.xlsx"
    pd.DataFrame(
        {
            "Fornavn": ["Ola", "Ola", "Ola", "Kari", "Per", "Lise", "Anne"],
            "Etternavn": ["Nordmann", "Nordmann", "Nordmann", "Hansen", "Olsen", "Berg", "Dahl"],
            "Stillingskode": ["1065", "1065", "1066", "1065", "1065", "1066", "1066"],
            "Tiltredelsesdato": [
                "01.01.2015", "01.06.2019", "01.01.2020", "01.01.2018", "01.01.2010", "01.01.2012", "01.01.2022",
            ],
            "Årslønn": [600_000, 530_000, 480_000, 560_000, 650_000, 540_000, 470_000],
        }
    ).to_excel(excel, index=False)

    monkeypatch.setattr(store, "SETTLEMENTS_DIR", tmp_path / "oppgjor")
    store.save_settlement("Test", {**store.default_settlement("Test"), "file_path": str(excel)})

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def _hash(fornavn, etternavn, kode, dato):
    return store.employee_hash(fornavn, etternavn, kode, dato)


def test_namesakes_are_separate_options(app):
    # AppTest gir alternativene slik de vises (etter format_func).
    labels = app.selectbox(key="employee_selector").options
    assert "Ola Nordmann (1066)" in labels
    # Samme navn i samme kode skilles med tiltredelsesdato.
    assert "Ola Nordmann (1065, tiltrådt 01.01.2015)" in labels
    assert "Ola Nordmann (1065, tiltrådt 01.06.2019)" in labels


def test_selecting_namesake_shows_the_right_person(app):
    ola_1066 = _hash("Ola", "Nordmann", "1066", "01.01.2020")
    app.selectbox(key="employee_selector").set_value(ola_1066).run()
    assert not app.exception, app.exception

    assert any(m.value == "#### Detaljvisning: Ola Nordmann" for m in app.markdown)
    details = app.dataframe[-1].value
    assert details["Stillingskode"].iloc[0] == "1066"
    assert details["Årslønn"].iloc[0] == 480_000
    assert store.load_settlement("Test")["last_selected_employee_hash"] == ola_1066


def test_export_both_namesakes(app):
    hashes = [_hash("Ola", "Nordmann", "1065", "01.01.2015"), _hash("Ola", "Nordmann", "1066", "01.01.2020")]
    app.multiselect(key="export_selection").set_value(hashes).run()
    next(b for b in app.button if b.label == "Generer PDF-rapporter").click().run()
    assert not app.exception, app.exception
    assert app.session_state["_export_zip_bytes"]
