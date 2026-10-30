import io
import zipfile

import pandas as pd

from pdf_export import build_export_zip


def _df_with_namesakes():
    return pd.DataFrame(
        {
            "Fornavn": ["Ola", "Ola", "Kari", "Per"],
            "Etternavn": ["Nordmann", "Nordmann", "Hansen", "Olsen"],
            "Fullt Navn": ["Ola Nordmann", "Ola Nordmann", "Kari Hansen", "Per Olsen"],
            "Stillingskode": ["1065", "1066", "1065", "1065"],
            "Tiltredelsesdato": ["01.01.2015", "01.01.2020", "01.01.2018", "01.01.2010"],
            "Ansiennitet (År)": [10.0, 5.0, 7.0, 15.0],
            "Årslønn": [600_000.0, 480_000.0, 560_000.0, 650_000.0],
            "_hash": ["ola_1065", "ola_1066", "kari", "per"],
        }
    )


def _pdf_texts(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def test_export_uses_hash_so_namesakes_get_their_own_report():
    df = _df_with_namesakes()
    files = _pdf_texts(build_export_zip(["ola_1066", "ola_1065"], df, {}, "_hash", None))

    assert sorted(files) == ["Ola_Nordmann.pdf", "Ola_Nordmann_2.pdf"]
    # Rekkefølgen følger valget: første fil er Ola i 1066, andre er Ola i 1065.
    assert files["Ola_Nordmann.pdf"] != files["Ola_Nordmann_2.pdf"]


def test_export_single_employee():
    files = _pdf_texts(build_export_zip(["kari"], _df_with_namesakes(), {}, "_hash", None))
    assert list(files) == ["Kari_Hansen.pdf"]
    assert files["Kari_Hansen.pdf"].startswith(b"%PDF")
