# std_monomial_basis_f2.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Standard monomial basis over GF(2) with optional periodic constraints.
# Port of the Mathematica function `StandardMonomialBasisF2Simple` to Python/SymPy.
#
# Usage examples:
#
#   # Example 1 (same as in the WL snippet):
#   python std_monomial_basis_f2.py \
#       --vars x y \
#       --gens "y + y**2 + x**3" "x + x**2 + y**3" \
#       --periods "x:12" "y:6" \
#       --order lex
#
#   # Example 2: also reduce an expression to its normal form:
#   python std_monomial_basis_f2.py \
#       --vars x y \
#       --gens "y + y**2 + x**3" "x + x**2 + y**3" \
#       --periods "x:12" "y:6" \
#       --nf "1 + x**7 * y**2"
#
# Notes:
#   * All arithmetic is done over GF(2).
#   * Monomial order defaults to 'lex' (lexicographic).
#   * Period constraints add relations of the form 1 + var**L to the ideal.
#   * Output contains the Groebner basis, leading monomials, bounds, the
#     standard-monomial basis, its dimension, and the "LogicalOperator" count.

import argparse
from typing import Dict, List, Tuple, Any
import sympy as sp

def standard_monomial_basis_f2_simple(
    gens: List[sp.Expr],
    vars_symbols: List[sp.Symbol],
    periods: Dict[sp.Symbol, int] = None,
    order: str = "lex"
) -> Dict[str, Any]:
    """
    Compute the standard monomial basis of F2[vars]/<gens, {1+v**L for v->L in periods}>.

    Returns a dict with:
      - "GroebnerBasis"     : list of Expr
      - "LeadingMonomials"  : list of Expr
      - "Bounds"            : dict {Symbol: int}
      - "StandardMonomials" : list of Expr
      - "Dimension"         : int
      - "LogicalOperator"   : int
      - "NormalForm"        : callable (Expr -> Expr)
    """
    if periods is None:
        periods = {}
    dom = sp.GF(2)

    # 1) Build the full generating set with period polynomials 1 + var**L
    period_polys = [1 + (var**L) for var, L in periods.items()]
    J_exprs = list(gens) + period_polys

    # Convert to Polys over GF(2)
    polys = [sp.Poly(p, *vars_symbols, domain=dom) for p in J_exprs]

    # 2) Groebner basis
    gb = sp.groebner(polys, *vars_symbols, order=order, domain=dom)

    # Leading monomials and their exponent vectors
    lt_exps: List[Tuple[int, ...]] = []
    lm_exprs: List[sp.Expr] = []
    for g in gb.polys:
        exp_t = g.monoms()[0]  # leading monomial exponents under the chosen order
        lt_exps.append(exp_t)
        lm_exprs.append(sp.prod(v**e for v, e in zip(vars_symbols, exp_t)))

    # 3) Determine bounds for each variable
    bounds: List[int] = []
    for i, var in enumerate(vars_symbols):
        # (a) Period bound
        if var in periods:
            bounds.append(int(periods[var]))
            continue

        # (b) Elimination: others first, then var last
        others = [v for v in vars_symbols if v != var]
        elim_order = [*others, var]
        elim_gb = sp.groebner(polys, *elim_order, order="lex", domain=dom)

        # keep polynomials depending only on 'var'
        rels = []
        for g in elim_gb.polys:
            if all(all(m[j] == 0 for j in range(len(elim_order) - 1)) for m in g.monoms()):
                rels.append(g)

        if rels:
            # smallest positive exponent of var among univariate relations
            min_exp = min(
                min(m[-1] for m in r.monoms() if m[-1] > 0)
                for r in rels
            )
            bounds.append(int(min_exp))
            continue

        # (c) Fall back to pure-power leading monomials in the original GB
        candidates = []
        for exp_t in lt_exps:
            if all((e == 0) for j, e in enumerate(exp_t) if j != i) and exp_t[i] > 0:
                candidates.append(exp_t[i])
        if candidates:
            bounds.append(int(min(candidates)))
        else:
            # Not zero-dimensional
            return {
                "Error": "Ideal is not zero-dimensional (infinite basis). "
                         "Add periods {var: L, ...} or include bounding relations like 1+var**L."
            }

    # 4) Candidate exponent vectors in the finite box
    ranges = [range(0, b) for b in bounds]
    from itertools import product
    exps_all = list(product(*ranges))

    # Divisibility test: e >= d component-wise
    def divisible_by_some_lm(e: Tuple[int, ...]) -> bool:
        for d in lt_exps:
            if all(ej >= dj for ej, dj in zip(e, d)):
                return True
        return False

    # Standard monomials are those NOT divisible by any leading monomial
    std_exps = [e for e in exps_all if not divisible_by_some_lm(e)]
    mon_basis = [sp.prod(v**e for v, e in zip(vars_symbols, evec)) for evec in std_exps]

    # 5) Normal form reduction mod the Groebner basis over GF(2)
    def nf(expr: sp.Expr) -> sp.Expr:
        p = sp.Poly(expr, *vars_symbols, domain=dom)
        _, r = gb.reduce(p)  # remainder as Poly
        return sp.expand(r.as_expr())

    return {
        "GroebnerBasis":     [sp.expand(g.as_expr()) for g in gb.polys],
        "LeadingMonomials":  lm_exprs,
        "Bounds":            dict(zip(vars_symbols, bounds)),
        "StandardMonomials": mon_basis,
        "Dimension":         len(mon_basis),
        "LogicalOperator":   len(mon_basis) * len(gens),
        "NormalForm":        nf
    }

def parse_periods(pairs: List[str], symmap: Dict[str, sp.Symbol]) -> Dict[sp.Symbol, int]:
    out = {}
    for s in pairs or []:
        if ":" not in s:
            raise ValueError(f"Malformed period '{s}', expected 'var:L'")
        name, L = s.split(":", 1)
        name = name.strip()
        L = int(L.strip())
        if name not in symmap:
            raise ValueError(f"Unknown variable in periods: '{name}'")
        out[symmap[name]] = L
    return out

def main():
    ap = argparse.ArgumentParser(description="Standard monomial basis over GF(2).")
    ap.add_argument("--vars", nargs="+", required=True, help="Variable names, e.g. --vars x y z")
    ap.add_argument("--gens", nargs="+", required=True, help='Generator polynomials, e.g. "y + y**2 + x**3"')
    ap.add_argument("--periods", nargs="*", default=None, help="Optional period constraints like x:12 y:6")
    ap.add_argument("--order", default="lex", help="Monomial order for Groebner basis (default: lex)")
    ap.add_argument("--nf", default=None, help='Optional expression to reduce to normal form, e.g. "1 + x**7*y**2"')
    args = ap.parse_args()

    # Build symbols in the given order
    vars_symbols = [sp.symbols(v) for v in args.vars]
    symmap = {str(v): v for v in vars_symbols}

    # Parse generators and optional periods
    gens = [sp.sympify(g, locals=symmap) for g in args.gens]
    periods = parse_periods(args.periods, symmap) if args.periods else {}

    res = standard_monomial_basis_f2_simple(gens, vars_symbols, periods, order=args.order)

    if "Error" in res:
        print(res["Error"])
        return

    print("=== Groebner basis (over GF(2)) ===")
    for g in res["GroebnerBasis"]:
        print(" ", g)

    print("\n=== Leading monomials ===")
    for m in res["LeadingMonomials"]:
        print(" ", m)

    print("\n=== Bounds ===")
    for k, v in res["Bounds"].items():
        print(f"  {k}: {v}")

    print("\n=== Standard monomial basis ===")
    print(" ", res["StandardMonomials"])

    print("\nDimension:", res["Dimension"])
    print("LogicalOperator:", res["LogicalOperator"])

    if args.nf is not None:
        expr = sp.sympify(args.nf, locals=symmap)
        reduced = res["NormalForm"](expr)
        print("\n=== Normal form of", args.nf, "===")
        print(" ", reduced)

if __name__ == "__main__":
    main()