"""Fail loudly if committed artefacts are missing or suspiciously small."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bikedemand.config import REPORTS  # noqa: E402

FILES = [
    REPORTS / "REPORT.md",
    REPORTS / "model_card.md",
    REPORTS / "metrics.json",
    REPORTS / "eda_summary.json",
    REPORTS / "diagnostics_summary.json",
    REPORTS / "clean_report.json",
]
FIGURES = sorted((REPORTS / "figures").glob("*.png"))
MIN_FIGURES = 15
MIN_BYTES = 300


def main() -> int:
    problems: list[str] = []
    for path in FILES:
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
        elif path.stat().st_size < MIN_BYTES:
            problems.append(f"suspiciously small {path.relative_to(ROOT)}")

    if len(FIGURES) < MIN_FIGURES:
        problems.append(f"expected at least {MIN_FIGURES} figures, found {len(FIGURES)}")
    for figure in FIGURES:
        if figure.stat().st_size < 5_000:
            problems.append(f"figure too small: {figure.name}")

    metrics_path = REPORTS / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in (
            "selected_model",
            "cv",
            "test",
            "test_all_models",
            "split",
            "feature_importance",
        ):
            if key not in metrics:
                problems.append(f"metrics.json missing '{key}'")
        if "test" in metrics and "seasonal_naive" not in metrics.get("test_all_models", {}):
            problems.append("metrics.json is missing the seasonal-naive baseline")

    if problems:
        print("artefact check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"artefacts OK: {len(FILES)} files, {len(FIGURES)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
