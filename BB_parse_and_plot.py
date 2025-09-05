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
from bb_common import ResultPoint, plot_points


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
    by_key: Dict[Tuple[str, int, int, int, float], ResultPoint] = {}
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
                l = int(meta.get("l", -1))
                m = int(meta.get("m", -1))
                key = (decoder, l, m, rounds, p)
                if key not in by_key:
                    by_key[key] = ResultPoint(decoder=decoder, l=l, m=m, rounds=rounds, p=p, shots=0, errors=0, seconds=0.0)
                pt = by_key[key]
                pt.shots += shots
                pt.errors += errors
                pt.seconds += seconds
    return list(by_key.values())


def load_summary_csv(paths: Iterable[str], *, l: int | None = None, m: int | None = None, decoder: str = "bposd") -> List[ResultPoint]:
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
                out.append(ResultPoint(decoder=decoder, l=int(l or -1), m=int(m or -1), rounds=rounds, p=p, shots=shots, errors=errors, seconds=seconds))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse BB resume/summary CSVs and plot LER vs p.")
    parser.add_argument("--resume", nargs="*", default=[], help="Paths to resume CSV files to aggregate.")
    parser.add_argument("--summary", nargs="*", default=[], help="Paths to summary CSV files to load.")
    parser.add_argument("--out", default=None, help="Output plot path. If not set, use Data/bb_l_m_parsed_results.png per group.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--l", type=int, default=None, help="Override l for summary CSV points.")
    parser.add_argument("--m", type=int, default=None, help="Override m for summary CSV points.")
    args = parser.parse_args()

    points: List[ResultPoint] = []
    if args.resume:
        points.extend(load_resume_csv(args.resume))
    if args.summary:
        points.extend(load_summary_csv(args.summary, l=args.l, m=args.m))
    if not points:
        print("No points loaded. Provide --resume or --summary CSV paths.")
        return

    plot_points(points, out_png=args.out, show=args.show)


if __name__ == "__main__":
    main()
