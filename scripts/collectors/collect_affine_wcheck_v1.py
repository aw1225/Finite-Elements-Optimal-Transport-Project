#!/usr/bin/env python3
import argparse, csv, pathlib, math

METRICS = [
    "full_residual_p", "full_residual_q",
    "source_marginal_l2_physical", "target_marginal_l2_physical",
    "map_l2_physical", "map_linf_physical",
    "jacobian_l2", "det_l2", "skew_l2",
    "det_min", "det_max", "tangled_area",
    "monge_defect_physical", "rms_conditional_spread_physical",
]
AUDIT = "run_affine_wcheck_v1_fine_to_tol_science_audit.csv"
HIER = "run_affine_wcheck_v1_hierarchy.csv"
TRACE = "run_affine_wcheck_v1_fine_to_tol_trace.csv"


def _one_csv(path):
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def _last_csv(path):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def read_one(run_dir):
    d=pathlib.Path(run_dir)
    a=_one_csv(d/AUDIT)
    h=_one_csv(d/HIER)
    t=_last_csv(d/TRACE)
    out={
        "run_dir": str(d),
        "W": float(a["bridge_width_physical"]),
        "bridge_n": int(float(a["bridge_n"])),
        "h_physical": float(h["h_physical"]),
        "tau_unit": float(h["tau_unit"]),
        "tau_over_h2": float(h["tau_over_h2"]),
        "heat_substeps": int(float(a["heat_substeps"])),
        "updates": int(float(t["updates"])),
        "reached_target": int(float(t["reached_target"])),
    }
    for k in METRICS:
        v=a.get(k, "")
        out[k]=float(v) if v not in ("", None) else math.nan
    return out


def rel(a,b):
    return abs(a-b)/max(abs(b),1e-300)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs=3, help="W=3,4,5 result directories in any order")
    ap.add_argument("--out", default="affine_wcheck_comparison.csv")
    args=ap.parse_args()
    rows=sorted((read_one(x) for x in args.run_dirs), key=lambda r:r["W"])
    byW={round(r["W"],12):r for r in rows}
    if set(byW)!={3.0,4.0,5.0}:
        raise SystemExit(f"Expected W=3,4,5; found {sorted(byW)}")
    expected={3.0:345,4.0:460,5.0:575}
    for W,n in expected.items():
        if byW[W]["bridge_n"] != n:
            raise SystemExit(f"W={W} should use n={n}, found {byW[W]['bridge_n']}")
    href=4.0/460.0
    for r in rows:
        if abs(r["h_physical"]-href)>1e-12:
            raise SystemExit(f"Physical spacing mismatch in {r['run_dir']}: {r['h_physical']} vs {href}")
        if r["heat_substeps"] != 16:
            raise SystemExit(f"Expected m=16 in {r['run_dir']}, found {r['heat_substeps']}")

    base=["W","bridge_n","h_physical","tau_unit","tau_over_h2","heat_substeps","updates","reached_target","run_dir"]
    fields=base+METRICS+[f"rel_to_W5__{k}" for k in METRICS]
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            out={k:r.get(k,"") for k in base+METRICS}
            for k in METRICS:
                out[f"rel_to_W5__{k}"] = rel(r[k],byW[5.0][k]) if math.isfinite(r[k]) and math.isfinite(byW[5.0][k]) else ""
            w.writerow(out)

    print("Matched physical spacing h =", href)
    print("Primary W=4 -> W=5 relative changes:")
    for k in ["map_l2_physical","jacobian_l2","monge_defect_physical","rms_conditional_spread_physical","full_residual_p","full_residual_q"]:
        print(f"  {k}: {rel(byW[4.0][k],byW[5.0][k]):.6e}")
    print("Smaller-domain W=3 -> W=5 relative changes:")
    for k in ["map_l2_physical","jacobian_l2","monge_defect_physical"]:
        print(f"  {k}: {rel(byW[3.0][k],byW[5.0][k]):.6e}")
    print(f"Wrote {args.out}")

if __name__=="__main__":
    main()
