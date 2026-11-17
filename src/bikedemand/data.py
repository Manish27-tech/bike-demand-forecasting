"""Dataset acquisition and integrity checks for the bike-sharing extract."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from bikedemand.config import (
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    CSV_NAME,
    CSV_SHA256,
    DATA_RAW,
    DATASET_URL,
)

REQUIRED_COLUMNS = (
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> Path:
    """Raise when ``path`` does not match the pinned digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path.name}: {actual} != {expected}")
    return path


def download(force: bool = False, target_dir: Path = DATA_RAW) -> Path:
    """Fetch and extract the dataset, returning the path to ``hour.csv``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / CSV_NAME
    if csv_path.exists() and not force:
        return verify(csv_path, CSV_SHA256)

    archive_path = target_dir / ARCHIVE_NAME
    if force or not archive_path.exists():
        urlretrieve(DATASET_URL, archive_path)  # noqa: S310 - pinned https URL
    verify(archive_path, ARCHIVE_SHA256)

    with zipfile.ZipFile(archive_path) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith(CSV_NAME)),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"{CSV_NAME} not found inside {ARCHIVE_NAME}")
        with archive.open(member) as source, open(csv_path, "wb") as sink:
            sink.write(source.read())

    return verify(csv_path, CSV_SHA256)


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Load the hourly table with the published dtypes."""
    path = download(force=force_download)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"unexpected schema, missing columns: {missing}")
    return frame


def describe(frame: pd.DataFrame) -> dict[str, object]:
    """Provenance record used by the report."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "date_min": str(frame["dteday"].min()),
        "date_max": str(frame["dteday"].max()),
        "total_rentals": int(frame["cnt"].sum()),
        "mean_hourly_demand": round(float(frame["cnt"].mean()), 2),
        "csv_sha256": CSV_SHA256,
        "source": DATASET_URL,
    }
