"""Command line interface for the pipeline stages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bikedemand.config import ARTIFACTS, DATA_PROCESSED, FIGURES, REPORTS


def _log(message: str) -> None:
    print(f"[bikedemand] {message}", flush=True)


def stage_download(force: bool = False) -> dict[str, object]:
    from bikedemand.data import describe, download, load_raw

    path = download(force=force)
    _log(f"dataset ready at {path}")
    summary = describe(load_raw())
    _log(f"{summary['rows']:,} hours, {summary['total_rentals']:,} rentals")
    return summary


def _cleaned_frame():
    from bikedemand.clean import clean_hourly
    from bikedemand.data import load_raw

    frame, report = clean_hourly(load_raw())
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "clean_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def stage_clean() -> dict[str, object]:
    frame = _cleaned_frame()
    _log(f"gridded table: {len(frame):,} hours")
    return {"rows": int(len(frame))}


def stage_eda() -> dict[str, object]:
    from bikedemand.eda import generate_eda

    summary = generate_eda(_cleaned_frame())
    _log(f"figures written to {FIGURES}")
    return summary


def stage_train() -> dict[str, object]:
    from bikedemand.model import train_and_evaluate

    metrics = train_and_evaluate(_cleaned_frame())
    _log(
        f"selected {metrics['selected_model']} | RMSE {metrics['test']['rmse']:.2f} "
        f"| R2 {metrics['test']['r2']:.4f}"
    )
    _log(f"model artefact written to {ARTIFACTS}")
    return metrics


def stage_diagnostics() -> dict[str, object]:
    from bikedemand.diagnostics import generate_diagnostics

    summary = generate_diagnostics()
    _log("diagnostic figures and error breakdown written")
    return summary


def stage_report() -> dict[str, Path]:
    from bikedemand.report import generate_report

    report, card = generate_report()
    _log(f"report written to {report}")
    _log(f"model card written to {card}")
    return {"report": report, "model_card": card}


STAGES = {
    "download": stage_download,
    "clean": stage_clean,
    "eda": stage_eda,
    "train": stage_train,
    "diagnostics": stage_diagnostics,
    "report": stage_report,
}


def _run(name: str, force: bool = False) -> object:
    stage = STAGES[name]
    if name == "download":
        return stage(force=force)
    return stage()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bikedemand", description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=[*STAGES, "all"],
        help="pipeline stage to run (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="re-download the dataset")
    args = parser.parse_args(argv)

    started = time.time()
    try:
        if args.stage == "all":
            for name in ("download", "clean", "eda", "train", "diagnostics", "report"):
                _log(f"--- {name} ---")
                _run(name, force=args.force)
        else:
            _run(args.stage, force=args.force)
    except Exception as error:  # noqa: BLE001 - surface a clean CLI failure
        _log(f"failed: {error}")
        return 1

    _log(f"done in {time.time() - started:.1f}s")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
