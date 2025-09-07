"""Search BT codes over c_poly for finite-weight logical operators.

Given fixed A(x,y) and B(x,y) polynomials, enumerate candidate C(x,y)
polynomials up to a bounded degree and number of monomials. For each
candidate, build the BT CSS code at specified (l, m) sizes and flag
those where a logical-Z generator has Hamming weight <= threshold.

Usage examples:

  # Quick scan with defaults (l=6, m=8, deg<=3, up to 2 terms in C):
  python search_bt_c_poly.py

  # Wider scan, verify across sizes, stop on first hit:
  python search_bt_c_poly.py --l 6 --m 8 \
      --deg-x-max 4 --deg-y-max 4 --max-terms 3 \
      --sizes 6x8,8x10,10x12 --threshold 10 --stop-on-first

Notes:
  - This check inspects the provided logical-Z generator basis (code.lz).
    It is a sufficient but not necessary indicator of finite-weight logicals.
    For stronger evidence, verify across multiple sizes.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
import itertools as it
from typing import List, Tuple

import numpy as np
from scipy import sparse as sp

from bivariate_tricycle_codes import get_BT_Hx_Hz
from bposd.css import css_code


def parse_sizes(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "x" not in tok:
            raise argparse.ArgumentTypeError(f"Invalid size token '{tok}', expected like 6x8")
        l_s, m_s = tok.split("x", 1)
        out.append((int(l_s), int(m_s)))
    return out


def hamming_weight_rows(mat) -> List[int]:
    if mat is None:
        return []
    # Handle scipy sparse matrices
    if sp.issparse(mat):
        # Ensure CSR for fast row sums
        csr = mat.tocsr().astype(np.uint8)
        # Row sums as a 2D matrix (n, 1); flatten to list
        rs = np.asarray(csr.sum(axis=1)).ravel()
        return [int(x) for x in rs]
    # Handle numpy arrays
    arr = np.asarray(mat, dtype=np.uint8)
    if arr.ndim == 1:
        return [int(arr.sum())]
    return [int(int(x)) for x in arr.sum(axis=1)]


def _min_weight_in_span(lz_mat, max_k: int = 12) -> int:
    """Return minimum Hamming weight among nonzero combinations of lz rows.

    Enumerates all 2^K - 1 combinations if K <= max_k, otherwise returns
    min row weight as a safe lower bound.
    """
    if lz_mat is None:
        return -1
    # Convert to dense boolean for simplicity (sizes are modest here)
    if sp.issparse(lz_mat):
        L = lz_mat.toarray().astype(np.uint8)
    else:
        L = np.asarray(lz_mat, dtype=np.uint8)
    if L.ndim == 1:
        return int(L.sum())
    K = L.shape[0]
    # Quick lower bound: min row weight
    best = int(L.sum(axis=1).min()) if K > 0 else -1
    if K == 0:
        return -1
    if K > max_k:
        return best
    # Enumerate all nonzero masks
    N = L.shape[1]
    acc = np.zeros(N, dtype=np.uint8)
    # Gray-code enumeration to flip one row at a time efficiently
    prev_mask = 0
    for mask in range(1, 1 << K):
        diff = mask ^ prev_mask
        row = (diff & -diff).bit_length() - 1  # index of flipped bit
        # XOR toggle the row
        acc ^= L[row]
        w = int(acc.sum())
        if w < best:
            best = w
            if best == 0:
                break
        prev_mask = mask
    return best


def try_code(a_poly: List[Tuple[int, int]], b_poly: List[Tuple[int, int]], c_poly: List[Tuple[int, int]], l: int, m: int) -> Tuple[int, int, int]:
    """Build BT code and return (K, N, min_lz_weight_in_span)."""
    Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BT_{l}x{m}")
    K = int(getattr(code, "K", 0))
    N = int(getattr(code, "N", Hx.shape[1] if hasattr(Hx, "shape") else 0))
    lz = getattr(code, "lz", None)
    min_w = _min_weight_in_span(lz)
    return K, N, min_w


def enumerate_c_candidates(dx: int, dy: int, max_terms: int):
    """Yield candidate c_poly lists without materializing all combinations."""
    mons = [(i, j) for i in range(dx + 1) for j in range(dy + 1)]
    for k in range(1, max(1, max_terms) + 1):
        for comb in it.combinations(mons, k):
            yield list(comb)


def sample_c_candidates_random(
    dx: int,
    dy: int,
    max_terms: int,
    *,
    samples: int,
    seed: int | None = None,
    terms_exact: bool = False,
):
    """Yield random candidate c_poly lists.

    - Picks k uniformly from [1, max_terms] unless terms_exact is True, in which case k=max_terms.
    - Samples without replacement within a candidate; deduplicates across candidates.
    - Yields at most `samples` unique candidates.
    """
    rng = np.random.default_rng(seed)
    mons = [(i, j) for i in range(dx + 1) for j in range(dy + 1)]
    M = len(mons)
    if M == 0:
        return
    seen: set[tuple[tuple[int, int], ...]] = set()
    tries = 0
    goal = max(0, int(samples))
    while len(seen) < goal:
        tries += 1
        k = max_terms if terms_exact else int(rng.integers(1, max(1, max_terms) + 1))
        k = max(1, min(k, M))
        idxs = rng.choice(M, size=k, replace=False)
        cand = tuple(sorted((mons[int(i)] for i in idxs)))
        if cand in seen:
            # Avoid infinite loops in tiny spaces
            if tries > goal * 10:
                break
            continue
        seen.add(cand)
        yield [list(p) for p in cand]


def main() -> None:
    p = argparse.ArgumentParser(description="Search BT c_poly for finite-weight logical operators")
    p.add_argument("--l", type=int, default=6, help="BT l dimension")
    p.add_argument("--m", type=int, default=8, help="BT m dimension")
    p.add_argument("--deg-x-max", type=int, default=3, help="Max exponent for x in c(x,y)")
    p.add_argument("--deg-y-max", type=int, default=3, help="Max exponent for y in c(x,y)")
    p.add_argument("--max-terms", type=int, default=2, help="Max number of monomials in c(x,y)")
    p.add_argument("--sizes", type=str, default="", help="Optional extra sizes to verify, e.g. '6x8,8x10'")
    p.add_argument("--scan-sizes", type=str, default="", help="Optional sizes to actively scan for hits, e.g. '4x6,6x8,8x10'")
    # For this task the primary criterion is K>0 (any nonzero logicals).
    # Threshold kept only for reporting/sorting context, not as a filter.
    p.add_argument("--threshold", type=int, default=-1, help="Deprecated: no longer filters; kept for info only")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on number of candidates to test (0=all)")
    p.add_argument("--stop-on-first", action="store_true", help="Stop after the first hit")
    # Random search options
    p.add_argument("--random", action="store_true", help="Use random sampling instead of exhaustive enumeration")
    p.add_argument("--samples", type=int, default=10000, help="Number of random candidates to test when --random is set")
    p.add_argument("--seed", type=int, default=None, help="Random seed for --random mode")
    p.add_argument("--terms-exact", action="store_true", help="In --random mode, use exactly max_terms monomials per candidate")
    p.add_argument("--log-csv", type=str, default="Data/bt_c_search_log.csv", help="CSV file to append per-candidate results")
    args = p.parse_args()

    # Fixed A, B from prompt
    a_poly = [(3, 0), (0, 1), (0, 2)]
    b_poly = [(0, 3), (1, 0), (2, 0)]

    verify_sizes: List[Tuple[int, int]] = []
    if args.sizes:
        verify_sizes = parse_sizes(args.sizes)
    scan_sizes: List[Tuple[int, int]] = []
    if args.scan_sizes:
        scan_sizes = parse_sizes(args.scan_sizes)

    hits = []
    best_k = 5
    best: List[Tuple[int, List[Tuple[int,int]]]] = []  # (min_w, c_poly)
    tested = 0
    # Prepare CSV logging
    log_path = args.log_csv
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    header = [
        "ts",
        "a_poly",
        "b_poly",
        "c_poly",
        "primary_l",
        "primary_m",
        "primary_K",
        "primary_N",
        "primary_min_lz_weight",
        "hit_any_size",
        "first_hit_l",
        "first_hit_m",
        "verify_all_sizes",
        "scan_sizes",
    ]
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    cand_iter = (
        sample_c_candidates_random(
            args.deg_x_max,
            args.deg_y_max,
            args.max_terms,
            samples=(args.limit if (args.random and args.limit and args.limit > 0) else args.samples),
            seed=args.seed,
            terms_exact=args.terms_exact,
        )
        if args.random
        else enumerate_c_candidates(args.deg_x_max, args.deg_y_max, args.max_terms)
    )

    for c_poly in cand_iter:
        tested += 1
        if not args.random:
            if args.limit and args.limit > 0 and tested > args.limit:
                break
        # Test primary size and optionally a scan list of sizes; hit if any has K>0
        ok = False
        primary_res = None
        first_hit = None
        sizes_to_try = [(args.l, args.m)] + scan_sizes
        seen = set()
        uniq_sizes = []
        for s in sizes_to_try:
            if s not in seen:
                seen.add(s)
                uniq_sizes.append(s)
        for (l0, m0) in uniq_sizes:
            K, N, min_w = try_code(a_poly, b_poly, c_poly, l0, m0)
            if primary_res is None and (l0, m0) == (args.l, args.m):
                primary_res = (K, N, min_w)
            if K > 0:
                ok = True
                if first_hit is None:
                    first_hit = (l0, m0, K, N, min_w)
                if primary_res is None:
                    primary_res = (K, N, min_w)
                break
        if not ok:
            # Track best few even if not hits (based on primary if available)
            if primary_res is None:
                Kp, Np, min_wp = try_code(a_poly, b_poly, c_poly, args.l, args.m)
                mw = int(min_wp) if min_wp is not None and min_wp >= 0 else 10**9
            else:
                mw = int(primary_res[2]) if primary_res[2] is not None and primary_res[2] >= 0 else 10**9
            best.append((mw, c_poly))
            if len(best) > best_k:
                best.sort(key=lambda t: t[0])
                best = best[:best_k]
            # Log CSV row for this non-hit
            ts = datetime.utcnow().isoformat()
            if primary_res is not None:
                Kp, Np, min_wp = primary_res
            else:
                Kp, Np, min_wp = (0, 0, -1)
            row = [
                ts,
                str(a_poly),
                str(b_poly),
                str(c_poly),
                args.l,
                args.m,
                int(Kp),
                int(Np),
                int(min_wp) if (min_wp is not None and min_wp >= 0) else -1,
                0,
                "",
                "",
                "",
                ",".join([f"{l}x{m}" for (l, m) in scan_sizes]) if scan_sizes else "",
            ]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            continue

        # Optionally verify across extra sizes (keeps same a,b,c)
        verify_ok = True
        if ok and verify_sizes:
            for l2, m2 in verify_sizes:
                K2, N2, w2 = try_code(a_poly, b_poly, c_poly, l2, m2)
                if K2 <= 0:
                    verify_ok = False
                    break

        if ok:
            Kp, Np, min_wp = primary_res if primary_res is not None else (K, N, min_w)
            hit = {
                "c_poly": [list(p) for p in c_poly],
                "primary": {"l": args.l, "m": args.m, "K": int(Kp), "N": int(Np), "min_lz_weight": int(min_wp) if (min_wp is not None and min_wp >= 0) else -1},
            }
            if verify_sizes:
                hit["verify"] = []
                for l2, m2 in verify_sizes:
                    K2, N2, w2 = try_code(a_poly, b_poly, c_poly, l2, m2)
                    hit["verify"].append({
                        "l": l2,
                        "m": m2,
                        "K": K2,
                        "N": N2,
                        "min_lz_weight": int(w2) if w2 >= 0 else -1,
                    })
            hits.append(hit)
            print(f"HIT: c_poly={hit['c_poly']} min_lz_weight={min_w} (K={K})")
            if args.stop_on_first:
                break
        else:
            # Track best few even if not hits
            best.append((int(min_w), c_poly))
            if len(best) > best_k:
                best.sort(key=lambda t: t[0])
                best = best[:best_k]
        # Log CSV row for this candidate
        ts = datetime.utcnow().isoformat()
        Kp, Np, min_wp = primary_res if primary_res is not None else (K, N, min_w)
        row = [
            ts,
            str(a_poly),
            str(b_poly),
            str(c_poly),
            args.l,
            args.m,
            int(Kp),
            int(Np),
            int(min_wp) if (min_wp is not None and min_wp >= 0) else -1,
            1 if ok else 0,
            int(first_hit[0]) if first_hit else "",
            int(first_hit[1]) if first_hit else "",
            (1 if (ok and verify_sizes and verify_ok) else ("" if not verify_sizes else 0)),
            ",".join([f"{l}x{m}" for (l, m) in scan_sizes]) if scan_sizes else "",
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if tested % 1000 == 0:
            print(f"Tested {tested} candidates… hits={len(hits)}")

    print(f"Done. Tested {tested} candidates. Found {len(hits)} hits.")
    if hits:
        import json
        print(json.dumps(hits, indent=2))
    if best:
        print("Top candidates by min logical-Z weight (primary size):")
        for w, c in sorted(best, key=lambda t: t[0])[:best_k]:
            print(f"  min_w={w}  c_poly={[list(p) for p in c]}")


if __name__ == "__main__":
    main()
