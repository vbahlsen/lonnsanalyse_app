"""Lagring av "oppgjør" - navngitte økter som husker filsti, kolonne-mapping,
outlier-erklæringer og navigasjonstilstand mellom Streamlit-økter.

Konfigurasjonen inneholder aldri personopplysninger - kun filstier og
ikke-reversible hasher som pseudonymer for ansatte.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

SETTLEMENTS_DIR = Path(__file__).parent / "oppgjor"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]+", "_", name.strip()).strip("_")
    return slug or "oppgjor"


def _settlement_path(name: str) -> Path:
    return SETTLEMENTS_DIR / f"{_slugify(name)}.json"


def list_settlements() -> list[str]:
    if not SETTLEMENTS_DIR.exists():
        return []
    names = []
    for path in sorted(SETTLEMENTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            names.append(data.get("name", path.stem))
        except (json.JSONDecodeError, OSError):
            continue
    return names


def default_settlement(name: str) -> dict:
    return {
        "name": name,
        "file_path": None,
        "column_mapping": None,
        "outliers": {},
        "last_selected_employee_hash": None,
        "selected_codes": None,
        "display_columns": ["Fornavn", "Etternavn", "Stillingskode", "Tiltredelsesdato", "Årslønn"],
        "updated_at": None,
    }


def load_settlement(name: str) -> dict:
    path = _settlement_path(name)
    if not path.exists():
        return default_settlement(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_settlement(name)
    merged = default_settlement(name)
    merged.update(data)
    return merged


def save_settlement(name: str, data: dict) -> None:
    SETTLEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["name"] = name
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _settlement_path(name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def employee_hash(fornavn: str, etternavn: str, stillingskode: str, tiltredelsesdato) -> str:
    key = "|".join(
        str(part).strip().lower()
        for part in (fornavn, etternavn, stillingskode, tiltredelsesdato)
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def resolve_data_file(file_path: str | None) -> Path | None:
    if not file_path:
        return None
    path = Path(file_path)
    return path if path.exists() and path.is_file() else None
