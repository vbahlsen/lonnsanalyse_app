import pytest

import settlement_store as store


def test_employee_hash_is_stable_and_normalized():
    h = store.employee_hash("Ingrid", "Haug", "1065", "15.03.2020")
    assert h == store.employee_hash(" ingrid ", "HAUG", "1065", "15.03.2020")
    assert len(h) == 16


def test_employee_hash_differs_for_same_name_in_other_code_or_date():
    base = store.employee_hash("Ola", "Nordmann", "1065", "15.03.2020")
    assert base != store.employee_hash("Ola", "Nordmann", "1066", "15.03.2020")
    assert base != store.employee_hash("Ola", "Nordmann", "1065", "16.03.2020")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('"C:\\data\\lønn.xlsx"', "C:\\data\\lønn.xlsx"),
        ("  “C:\\data\\lønn.xlsx”  ", "C:\\data\\lønn.xlsx"),
        ("C:\\data\\lønn.xlsx", "C:\\data\\lønn.xlsx"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_path_string(raw, expected):
    assert store.clean_path_string(raw) == expected


def test_settlement_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SETTLEMENTS_DIR", tmp_path)
    store.save_settlement("2026 oppgjøret", {**store.default_settlement("2026 oppgjøret"), "outliers": {"A": ["x"]}})
    store.update_settlement("2026 oppgjøret", file_path="C:\\lønn.xlsx")

    loaded = store.load_settlement("2026 oppgjøret")
    assert loaded["outliers"] == {"A": ["x"]}
    assert loaded["file_path"] == "C:\\lønn.xlsx"
    assert store.list_settlements() == ["2026 oppgjøret"]

    store.delete_settlement("2026 oppgjøret")
    assert store.list_settlements() == []
