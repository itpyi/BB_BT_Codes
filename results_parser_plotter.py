"""Parse resume/summary CSVs and plot logical error rates.

Supports:
- Parsing resume CSV rows: shots,errors,discards,seconds,decoder,json_metadata,custom_counts
- Parsing summary CSV rows: p,rounds,shots,errors,ler,seconds

Aggregates results across rows by (decoder, l, m, rounds, p) and plots
LER vs p grouped by rounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

from typing import Any
import re
from qec_simulation_core import ResultPoint, plot_points


def _try_load_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        # Attempt to unescape common quoting; fall back to empty
        try:
            return json.loads(s.replace("''", '"').replace('""', '"'))
        except Exception:
            return {}


def load_resume_csv(paths: Iterable[str]) -> List[ResultPoint]:
    by_key: Dict[Tuple[str, str, int, int, int, float], ResultPoint] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    shots = int(row["shots"])
                    errors = int(row["errors"])
                    seconds = float(row.get("seconds", 0.0) or 0.0)
                    decoder = row.get("decoder", "bposd")
                    meta = _try_load_json(row.get("json_metadata", "{}"))
                except Exception:
                    continue

                p = float(meta.get("p", math.nan))
                rounds = int(meta.get("rounds", -1))
                # Prefer explicit l,m if present; else fall back to code_l, code_m
                l = int(meta.get("l", meta.get("code_l", -1)))
                m = int(meta.get("m", meta.get("code_m", -1)))
                code_type = str(meta.get("code_type", "BB")).upper()
                # Prefer TT dimension 'n' if present under code_n/code_n_dim
                n_dim = int(meta.get("n", meta.get("code_n", -1)))
                key = (decoder, code_type, l, m, rounds, p)
                if key not in by_key:
                    by_key[key] = ResultPoint(
                        decoder=decoder,
                        l=l,
                        m=m,
                        rounds=rounds,
                        p=p,
                        shots=0,
                        errors=0,
                        seconds=0.0,
                        code_type=code_type,
                        n=n_dim,
                    )
                pt = by_key[key]
                pt.shots += shots
                pt.errors += errors
                pt.seconds += seconds
    return list(by_key.values())


def infer_K_from_csvs(paths: Iterable[str]) -> int | None:
    """Infer logical operator count K from CSV metadata if present.

    Returns K if a consistent value is found across files, else None.
    """
    Ks = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get("json_metadata", "{}")
                    meta = _try_load_json(raw)
                    if isinstance(meta, dict) and "code_k" in meta:
                        try:
                            Ks.add(int(meta["code_k"]))
                            continue
                        except Exception:
                            pass
                    # Fallback: regex search code_k from raw string if CSV/JSON quoting is malformed
                    m = re.search(r"code_k\"?\s*:\s*(\d+)", str(raw))
                    if m:
                        try:
                            Ks.add(int(m.group(1)))
                        except Exception:
                            pass
        except Exception:
            continue
    if len(Ks) == 1:
        return next(iter(Ks))
    return None


def load_summary_csv(
    paths: Iterable[str], *, decoder: str = "bposd"
) -> List[ResultPoint]:
    out: List[ResultPoint] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not {"p", "rounds", "shots", "errors"} <= row.keys():
                    continue
                p = float(row["p"]) if row["p"] != "" else float("nan")
                rounds = int(row["rounds"]) if row["rounds"] != "" else -1
                shots = int(row["shots"]) if row["shots"] != "" else 0
                errors = int(row["errors"]) if row["errors"] != "" else 0
                seconds = float(row.get("seconds", 0.0) or 0.0)
                meta = _try_load_json(row.get("json_metadata", "{}"))
                l_eff = int((meta.get("l") if meta else -1) or -1)
                m_eff = int((meta.get("m") if meta else -1) or -1)
                code_type = str((meta.get("code_type") if meta else "BB") or "BB").upper()
                n_dim = int((meta.get("code_n") if meta else -1) or -1)
                out.append(
                    ResultPoint(
                        decoder=decoder,
                        l=l_eff,
                        m=m_eff,
                        rounds=rounds,
                        p=p,
                        shots=shots,
                        errors=errors,
                        seconds=seconds,
                        code_type=code_type,
                        n=n_dim,
                    )
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse BB resume/summary CSVs and plot error rates vs p."
    )
    parser.add_argument(
        "--resume",
        nargs="*",
        default=[],
        help="Paths to resume CSV files to aggregate.",
    )
    parser.add_argument(
        "--summary", nargs="*", default=[], help="Paths to summary CSV files to load."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output plot path. For multiple modes, suffixes are added unless separate calls are made.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    # l, m, K are inferred from CSV metadata
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["ler"],
        choices=["ler", "per_logical", "per_round", "all"],
        help="Which modes to plot: ler, per_logical, per_round, or all.",
    )
    args = parser.parse_args()

    points: List[ResultPoint] = []
    inferred_K = None
    if args.resume:
        points.extend(load_resume_csv(args.resume))
        inferred_K = infer_K_from_csvs(args.resume)
    if args.summary:
        points.extend(load_summary_csv(args.summary))
        if inferred_K is None:
            inferred_K = infer_K_from_csvs(args.summary)
    if not points:
        print("No points loaded. Provide --resume or --summary CSV paths.")
        return

    # Determine modes
    modes = args.modes
    if "all" in modes:
        modes = ["ler", "per_logical", "per_round"]

    # For multiple modes with a single --out, append suffixes
    def out_with_suffix(base: str | None, suffix: str | None) -> str | None:
        if base is None:
            return None
        if not suffix or suffix == "":
            return base
        root, ext = os.path.splitext(base)
        return f"{root}_{suffix}{ext}"

    for mode in modes:
        out_path = args.out
        suffix = None
        if args.out is not None and len(modes) > 1:
            suffix = {
                "ler": "ler",
                "per_logical": "per_logical",
                "per_round": "per_round",
            }.get(mode, mode)
            out_path = out_with_suffix(args.out, suffix)
        try:
            plot_points(
                points, out_png=out_path, show=args.show, y_mode=mode, K=inferred_K
            )
        except ValueError as e:
            # Likely missing K for per_logical
            print(f"Skipping mode '{mode}': {e}")


if __name__ == "__main__":
    main()
