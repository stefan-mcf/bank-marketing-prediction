from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root directory.

    Uses the location of this file to walk up until a directory that contains
    either a `.git` folder or both `notebooks/` and `data/`.
    """
    start = Path(__file__).resolve().parent
    for p in [*start.parents]:
        if (p / ".git").exists() or ((p / "notebooks").exists() and (p / "data").exists()):
            return p
    return start


def data_path(*parts: str) -> Path:
    """Build a path under `data/`.

    Example: `data_path('bank-full.csv')`.
    """
    return project_root() / "data" / Path(*parts)


def reports_path(*parts: str) -> Path:
    """Build a path under `reports/`.

    Example: `reports_path('figures', 'roc_curve.png')`.
    """
    return project_root() / "reports" / Path(*parts)

