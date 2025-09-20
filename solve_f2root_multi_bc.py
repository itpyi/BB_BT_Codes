#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve polynomial systems over F2 with optional boundary conditions:
    x^m - 1 = 0,  y^l - 1 = 0    (over F2 these are x^m+1=0, y^l+1=0)

- Works for 2 or 3+ variables/equations.
- Computes lex Groebner bases over F2:
    (i) without BCs,
    (ii) with BCs  (adds x^m+1, y^l+1 to the ideal if provided)
- Reports if a COMMON ROOT exists under the BCs (i.e., whether 1 ∈ <polys, BCs>).
- Shows univariate eliminants and factors; also gcd with x^m+1 and y^l+1.

Usage (CLI):
  python solve_f2root_multi_bc.py \
      --vars x,y \
      --poly "x^3 + y + y^2" --poly "y^3 + x + x^2" \
      --m 6 --l 6

  python solve_f2root_multi_bc.py \
      --vars x,y,z \
      --poly "x^3 + x + 1" --poly "y + x^2" --poly "z + x + y" \
      --m 7 --l 3

Author: (you)
"""
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import sympy as sp

# ---------------- Utilities ----------------

import sympy as sp

def all_c_variants_over_F2(a_str, b_str, m=None, l=None):
    x, y = sp.symbols('x y')
    F2 = dict(modulus=2)

    def P(s): return sp.Poly(sp.sympify(s.replace('^','**'), locals={'x':x,'y':y}), x, y, **F2)

    aP, bP = P(a_str), P(b_str)

    # Groebner both orders
    Gxy = sp.groebner([aP,bP], x,y, order='lex', modulus=2)
    Gyx = sp.groebner([aP,bP], y,x, order='lex', modulus=2)

    # Extract f_y, x+k(y) from Gxy; f_x, y+h(x) from Gyx
    fy = next((sp.Poly(g,y, modulus=2) for g in Gxy if g.free_symbols <= {y}), None)
    fx = next((sp.Poly(g,x, modulus=2) for g in Gyx if g.free_symbols <= {x}), None)

    def find_x_eq_ky(G):
        for g in G:
            Pg = sp.Poly(g, x,y, modulus=2)
            if Pg.degree(x)==1 and all((ex==1 and ey==0) or ey==0 for (ex,ey) in Pg.monoms()):
                k = sp.expand(g.as_expr().subs(x,0))
                if x not in k.free_symbols:
                    return sp.Poly(k, y, modulus=2)
        return None

    def find_y_eq_hx(G):
        for g in G:
            Pg = sp.Poly(g, x,y, modulus=2)
            if Pg.degree(y)==1 and all((ey==1 and ex==0) or ex==0 for (ex,ey) in Pg.monoms()):
                h = sp.expand(g.as_expr().subs(y,0))
                if y not in h.free_symbols:
                    return sp.Poly(h, x, modulus=2)
        return None

    k_of_y = find_x_eq_ky(Gxy)
    h_of_x = find_y_eq_hx(Gyx)

    # BC reductions
    if fx is not None and m:
        fx_bc = sp.gcd(fx, sp.Poly(x**m + 1, x, modulus=2))
    else:
        fx_bc = fx

    if fy is not None and l:
        fy_bc = sp.gcd(fy, sp.Poly(y**l + 1, y, modulus=2))
    else:
        fy_bc = fy

    cs = {}

    if fx_bc is not None and h_of_x is not None and not fx_bc.is_zero():
        cs['shape_y_of_x'] = sp.Poly(sp.expand(fx_bc.as_expr() * (y + h_of_x.as_expr())), x,y, modulus=2)

    if fy_bc is not None and k_of_y is not None and not fy_bc.is_zero():
        cs['shape_x_of_y'] = sp.Poly(sp.expand(fy_bc.as_expr() * (x + k_of_y.as_expr())), x,y, modulus=2)

    if k_of_y is not None and h_of_x is not None:
        cs['two_order_product'] = sp.Poly(sp.expand((x + k_of_y.as_expr()) * (y + h_of_x.as_expr())), x,y, modulus=2)

    if fx_bc is not None:
        cs['fx_univariate'] = sp.Poly(fx_bc.as_expr(), x,y, modulus=2)
    if fy_bc is not None:
        cs['fy_univariate'] = sp.Poly(fy_bc.as_expr(), x,y, modulus=2)

    # smallest mixed Groebner element with BCs (if any)
    polys_bc = [aP,bP]
    if m: polys_bc.append(sp.Poly(x**m + 1, x,y, modulus=2))
    if l: polys_bc.append(sp.Poly(y**l + 1, x,y, modulus=2))
    Gbc = sp.groebner(polys_bc, x,y, order='lex', modulus=2)
    mixed = [sp.Poly(g, x,y, modulus=2) for g in Gbc
             if not (g.free_symbols <= {x} or g.free_symbols <= {y}) and sp.Poly(g, x,y, modulus=2).total_degree() > 0]
    if mixed:
        cs['mixed_groebner_bc'] = min(mixed, key=lambda p: p.total_degree())

    return {name: poly.as_expr() for name, poly in cs.items()}


def _sanitize_poly_string(s: str) -> str:
    """Allow caret '^' for powers; keep Python '**' too."""
    return s.replace("^", "**")

def _mk_symbols(var_names: List[str]) -> Tuple[List[sp.Symbol], Dict[str, sp.Symbol]]:
    vars_syms = [sp.symbols(v) for v in var_names]
    return vars_syms, {s.name: s for s in vars_syms}

def _parse_polys(poly_strs: List[str], syms_map: Dict[str, sp.Symbol], vars_order: List[sp.Symbol]) -> List[sp.Poly]:
    polys = []
    for s in poly_strs:
        expr = sp.sympify(_sanitize_poly_string(s), locals=syms_map)
        polys.append(sp.Poly(expr, *vars_order, modulus=2))
    return polys

def _extract_univariate_from_G(G: sp.GroebnerBasis, var: sp.Symbol) -> Optional[sp.Poly]:
    """Return a polynomial from G that involves only 'var', if one exists."""
    for g in G:
        if g.free_symbols <= {var}:
            return sp.Poly(g, var, modulus=2)
    return None

def _detect_assignment_for_var(G: sp.GroebnerBasis, vars_order: List[sp.Symbol], j: int) -> Optional[sp.Expr]:
    """
    Look for strict triangular relation  v_j + h(v_1,...,v_{j-1}) = 0,
    i.e., linear in v_j with unit coeff and no mixed v_j*... terms.
    """
    vj = vars_order[j]
    earlier = set(vars_order[:j])
    for g in G:
        P = sp.Poly(g, *vars_order, modulus=2)
        if P.degree(vj) != 1:
            continue
        ok = True
        for mono, coeff in zip(P.monoms(), P.coeffs()):
            exps = dict(zip(vars_order, mono))
            if exps.get(vj, 0) == 1:
                if any((vv != vj and ee != 0) for vv, ee in exps.items()):
                    ok = False
                    break
        if not ok:
            continue
        if any((sym not in earlier and sym != vj) for sym in g.free_symbols):
            continue
        h = sp.expand(g.as_expr().subs(vj, 0))  # char 2: g = vj + h
        if (vj not in h.free_symbols) and (h.free_symbols <= earlier):
            return sp.simplify(h)
    return None

def _poly_is_one(g, vars_syms) -> bool:
    """Check if Groebner basis element is the constant 1 over F2."""
    try:
        P = sp.Poly(g, *vars_syms, modulus=2)
        return P.is_one
    except Exception:
        return sp.simplify(g) == 1

# ---------------- Data structures ----------------

@dataclass
class MultiF2BCResult:
    vars: List[str]
    polys: List[str]
    groebner_order: str

    # Without BCs
    groebner_basis_no_bc: List[str]
    univariate_eliminants_no_bc: Dict[str, str]
    univariate_factors_no_bc: Dict[str, str]

    # With BCs
    m: Optional[int]
    l: Optional[int]
    bc_polys: List[str]
    groebner_basis_with_bc: List[str]
    has_common_root_with_bc: bool

    # GCD diagnostics against x^m+1, y^l+1
    gcd_fx_xm1: Optional[str]
    gcd_fy_yl1: Optional[str]

    # Optional triangular relations (from no-BC basis, informative)
    triangular_assignments: Dict[str, str]

    notes: str = ""

# ---------------- Core solver ----------------

def solve_common_roots_multi_over_F2_with_BC(
    poly_strs: List[str],
    var_names: List[str],
    m: Optional[int] = None,
    l: Optional[int] = None,
    order: str = "lex",
    try_swap_for_triangular: bool = True,
) -> MultiF2BCResult:
    """
    Describe common roots of a polynomial system over F̄2, optionally restricted by
    x^m+1=0 and y^l+1=0 (boundary conditions). Reports if a common root exists
    under the BCs.

    Parameters
    ----------
    poly_strs : list of str   (the system a(x,y,...) = 0, b(x,y,...) = 0, ...)
    var_names : list of str   (lex order v1 > v2 > ...; typically ["x","y"] or ["x","y","z"])
    m, l      : optional ints, boundary exponents for x and y (x^m+1, y^l+1). If None, omitted.
    order     : monomial order for Groebner basis ("lex" default)

    Returns
    -------
    MultiF2BCResult with Groebner bases, eliminants, gcd diagnostics, and a boolean
    has_common_root_with_bc.
    """
    # ---- setup
    vars_syms, syms_map = _mk_symbols(var_names)
    polys = _parse_polys(poly_strs, syms_map, vars_syms)
    x = syms_map.get("x", None)
    y = syms_map.get("y", None)

    # ---- Groebner basis WITHOUT BCs (for eliminants diagnostics)
    G0 = sp.groebner(polys, *vars_syms, order=order, modulus=2)
    # Elements of a GroebnerBasis may be Poly; expand on the Expr form.
    G0_list = [str(sp.expand(g.as_expr())) for g in G0]

    # Univariate eliminants (from G0)
    univar_no_bc: Dict[str, sp.Expr] = {}
    univar_fact_no_bc: Dict[str, sp.Expr] = {}
    for v in vars_syms:
        g_uni = _extract_univariate_from_G(G0, v)
        if g_uni is not None:
            expr = sp.expand(g_uni.as_expr())
            univar_no_bc[v.name] = expr
            univar_fact_no_bc[v.name] = sp.factor(expr, modulus=2)

    # Triangular (informational, no-BC)
    assignments: Dict[str, sp.Expr] = {}
    for j in range(1, len(vars_syms)):
        h = _detect_assignment_for_var(G0, vars_syms, j)
        if h is not None:
            assignments[vars_syms[j].name] = h
    if try_swap_for_triangular and len(assignments) < len(vars_syms)-1:
        swapped = list(reversed(vars_syms))
        Gs = sp.groebner(polys, *swapped, order=order, modulus=2)
        for j in range(1, len(swapped)):
            h_alt = _detect_assignment_for_var(Gs, swapped, j)
            if h_alt is not None:
                varname = swapped[j].name
                if varname not in assignments:
                    assignments[varname] = sp.simplify(h_alt)

    # ---- Add boundary conditions (BCs) if provided
    bc_polys = []
    if m is not None:
        if m <= 0:
            raise ValueError("m must be a positive integer.")
        if x is None:
            raise ValueError('Boundary uses x^m+1 but variable "x" was not declared in --vars.')
        bc_polys.append(sp.Poly(x**m + 1, *vars_syms, modulus=2))
    if l is not None:
        if l <= 0:
            raise ValueError("l must be a positive integer.")
        if y is None:
            raise ValueError('Boundary uses y^l+1 but variable "y" was not declared in --vars.')
        bc_polys.append(sp.Poly(y**l + 1, *vars_syms, modulus=2))

    polys_with_bc = polys + bc_polys

    # ---- Groebner WITH BCs: decide existence (1 ∈ ideal ?)
    if polys_with_bc:
        G1 = sp.groebner(polys_with_bc, *vars_syms, order=order, modulus=2)
        G1_list = [str(sp.expand(g.as_expr())) for g in G1]
        has_common_root_with_bc = not any(_poly_is_one(g, vars_syms) for g in G1)
    else:
        # no BCs provided -> trivially "yes" refers to the original system’s solvability; we don't test it here
        G1_list = []
        has_common_root_with_bc = False

    # ---- GCD diagnostics vs cyclotomic constraints (optional, helpful)
    gcd_fx_xm1 = None
    gcd_fy_yl1 = None
    if x is not None and m is not None:
        # get an x-only eliminant from NO-BC basis; otherwise use resultant eliminating y (if present)
        fx_poly = _extract_univariate_from_G(G0, x)
        if fx_poly is None:
            # Try a simple resultant using the first two polynomials, eliminating the other var.
            other = [v for v in vars_syms if v != x]
            if len(other) == 1 and len(polys) >= 2:
                elim = other[0]
                try:
                    fx_expr = sp.resultant(polys[0].as_expr(), polys[1].as_expr(), elim, modulus=2)
                    fx_poly = sp.Poly(sp.expand(fx_expr), x, modulus=2)
                except Exception:
                    fx_poly = None
        if fx_poly is not None:
            xm1 = sp.Poly(x**m + 1, x, modulus=2)
            gx = sp.gcd(sp.Poly(fx_poly.as_expr(), x, modulus=2), xm1)
            gcd_fx_xm1 = str(sp.factor(gx.as_expr(), modulus=2))

    if y is not None and l is not None:
        fy_poly = _extract_univariate_from_G(G0, y)
        if fy_poly is None:
            other = [v for v in vars_syms if v != y]
            if len(other) == 1 and len(polys) >= 2:
                elim = other[0]
                try:
                    fy_expr = sp.resultant(polys[0].as_expr(), polys[1].as_expr(), elim, modulus=2)
                    fy_poly = sp.Poly(sp.expand(fy_expr), y, modulus=2)
                except Exception:
                    fy_poly = None
        if fy_poly is not None:
            yl1 = sp.Poly(y**l + 1, y, modulus=2)
            gy = sp.gcd(sp.Poly(fy_poly.as_expr(), y, modulus=2), yl1)
            gcd_fy_yl1 = str(sp.factor(gy.as_expr(), modulus=2))

    return MultiF2BCResult(
        vars=var_names,
        polys=poly_strs,
        groebner_order=order,
        groebner_basis_no_bc=[str(g) for g in G0_list],
        univariate_eliminants_no_bc={k: str(v) for k, v in univar_no_bc.items()},
        univariate_factors_no_bc={k: str(v) for k, v in univar_fact_no_bc.items()},
        m=m,
        l=l,
        bc_polys=[str(P.as_expr()) for P in bc_polys],
        groebner_basis_with_bc=[str(g) for g in G1_list],
        has_common_root_with_bc=has_common_root_with_bc,
        gcd_fx_xm1=gcd_fx_xm1,
        gcd_fy_yl1=gcd_fy_yl1,
        triangular_assignments={k: str(v) for k, v in assignments.items()},
        notes="(Over F2, x^m-1 and y^l-1 are encoded as x^m+1 and y^l+1.)"
    )

# ---------------- CLI ----------------

def _print_result(res: MultiF2BCResult) -> None:
    print("=== Input (over F2) ===")
    print("vars:", ", ".join(res.vars))
    print("polys:")
    for p in res.polys:
        print("  ", p)

    if res.m is not None or res.l is not None:
        print("\n=== Boundary conditions ===")
        if res.m is not None: print(f"  x^{res.m} + 1 = 0   (i.e., x^{res.m}-1=0 over F2)")
        if res.l is not None: print(f"  y^{res.l} + 1 = 0   (i.e., y^{res.l}-1=0 over F2)")
        for bcp in res.bc_polys:
            print("   added to ideal:", bcp)

    print(f"\n=== Groebner basis (no BCs, order: {res.groebner_order}) ===")
    for g in res.groebner_basis_no_bc:
        print("  ", g)

    print("\nUnivariate eliminants (no BCs):")
    if res.univariate_eliminants_no_bc:
        for v, fx in res.univariate_eliminants_no_bc.items():
            print(f"  f_{v}({v}) =", fx)
            print("    factor over F2:", res.univariate_factors_no_bc.get(v, "<n/a>"))
    else:
        print("  (none detected)")

    if res.triangular_assignments:
        print("\nTriangular assignments (informational, no BCs):")
        for v, h in res.triangular_assignments.items():
            print(f"  {v} =", h)

    print(f"\n=== Groebner basis (WITH BCs, order: {res.groebner_order}) ===")
    if res.groebner_basis_with_bc:
        for g in res.groebner_basis_with_bc:
            print("  ", g)
    else:
        print("  (no BCs provided)")

    if res.m is not None:
        print("\nGCD diagnostic: gcd(f_x(x), x^m+1) =", res.gcd_fx_xm1)
    if res.l is not None:
        print("GCD diagnostic: gcd(f_y(y), y^l+1) =", res.gcd_fy_yl1)

    if res.m is not None or res.l is not None:
        print("\n=== Existence under boundary conditions ===")
        print("Common root satisfying BCs:",
              "YES" if res.has_common_root_with_bc else "NO")

    if res.notes:
        print("\nNotes:", res.notes)

def main(argv=None) -> int:
    import argparse, json
    p = argparse.ArgumentParser(
        description="Common roots over F̄2 with boundary conditions x^m-1=0, y^l-1=0."
    )
    
    # Add subparsers for different modes
    subparsers = p.add_subparsers(dest='mode', help='Operation mode')
    
    # Root analysis mode (original functionality)
    root_parser = subparsers.add_parser('roots', help='Find common roots of polynomial system')
    root_parser.add_argument("--vars", required=True,
                           help="Comma-separated variable names in lex order (e.g., x,y or x,y,z).")
    root_parser.add_argument("--poly", action="append", required=True,
                           help="A polynomial in the given variables (use '^' or '**'). Repeat for multiple.")
    root_parser.add_argument("--order", default="lex", choices=["lex", "grlex", "grevlex"],
                           help="Monomial order for Groebner basis (default: lex).")
    root_parser.add_argument("--m", type=int, default=None, help="Boundary exponent m for x^m-1=0 (i.e., x^m+1=0 over F2).")
    root_parser.add_argument("--l", type=int, default=None, help="Boundary exponent l for y^l-1=0 (i.e., y^l+1=0 over F2).")
    root_parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text.")
    
    # C-variants generation mode (new functionality)  
    c_parser = subparsers.add_parser('c-variants', help='Generate c polynomial variants from a,b system')
    c_parser.add_argument("--a", required=True, help="Polynomial a(x,y) as string (e.g., 'x^3 + y + y^2')")
    c_parser.add_argument("--b", required=True, help="Polynomial b(x,y) as string (e.g., 'y^3 + x + x^2')")
    c_parser.add_argument("--m", type=int, default=None, help="Boundary exponent m for x^m-1=0 (i.e., x^m+1=0 over F2).")
    c_parser.add_argument("--l", type=int, default=None, help="Boundary exponent l for y^l-1=0 (i.e., y^l+1=0 over F2).")
    c_parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text.")
    
    args = p.parse_args(argv)
    
    # Handle the case where no subcommand is provided (backward compatibility)
    if args.mode is None:
        # Fall back to original behavior - assume root analysis
        if not hasattr(args, 'vars') or not hasattr(args, 'poly'):
            print("Error: Must specify either 'roots' or 'c-variants' mode")
            print("Usage examples:")
            print("  python solve_f2root_multi_bc.py roots --vars x,y --poly 'x^3 + y + y^2' --poly 'y^3 + x + x^2' --m 6 --l 6")
            print("  python solve_f2root_multi_bc.py c-variants --a 'x^3 + y + y^2' --b 'y^3 + x + x^2' --m 6 --l 6")
            return 1
    
    if args.mode == 'roots':
        var_names = [v.strip() for v in args.vars.split(",") if v.strip()]
        res = solve_common_roots_multi_over_F2_with_BC(
            poly_strs=args.poly,
            var_names=var_names,
            m=args.m, l=args.l,
            order=args.order,
        )

        if args.json:
            print(json.dumps(asdict(res), indent=2))
        else:
            _print_result(res)
        return 0
        
    elif args.mode == 'c-variants':
        # Generate c polynomial variants from a,b system
        try:
            c_variants = all_c_variants_over_F2(
                a_str=args.a,
                b_str=args.b, 
                m=args.m,
                l=args.l
            )
            
            if args.json:
                # Convert sympy expressions to strings for JSON serialization
                json_variants = {name: str(expr) for name, expr in c_variants.items()}
                print(json.dumps(json_variants, indent=2))
            else:
                print("=== C Polynomial Variants from a,b System ===")
                print(f"Input: a(x,y) = {args.a}")
                print(f"       b(x,y) = {args.b}")
                if args.m:
                    print(f"       x^{args.m} + 1 = 0")
                if args.l:
                    print(f"       y^{args.l} + 1 = 0")
                print()
                
                if c_variants:
                    print("Generated c polynomial variants:")
                    for name, expr in c_variants.items():
                        print(f"  {name:<25}: {expr}")
                else:
                    print("No c polynomial variants generated.")
                    
        except Exception as e:
            print(f"Error generating c variants: {e}")
            return 1
            
        return 0
    
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
