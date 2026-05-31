#!/usr/bin/env python3
"""
Forward-accuracy comparison of the full dRoute routing catalogue (GMD section 4.2).

Routes one synthetic runoff series through every available routing scheme on the same
network and compares them, using the full Saint-Venant (SV) dynamic-wave solution as
the high-fidelity reference. Quantifies how the cheaper schemes approximate SV (KGE/NSE
against SV, peak ratio, peak-timing lag, volume ratio).

Schemes: Muskingum-Cunge (mc), Lag, IRF, soft-gated KWT, Diffusive wave, Saint-Venant.
SV requires a SUNDIALS-enabled build; if SV returns NaN the script reports it and
compares the remaining schemes to Muskingum-Cunge instead.

Usage:
    python experiments/method_catalogue.py --reaches 30 --n-time 1500 --sv-nodes 10
"""

import argparse
import json
from pathlib import Path

import numpy as np

import droute
from idealized_chain import build_chain, synthetic_runoff, true_manning
from bow_at_banff_dds_vs_adam import kge, nse, set_manning

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROUTERS = [
    ("mc", "MuskingumCungeRouter", "Muskingum-Cunge"),
    ("lag", "LagRouter", "Lag"),
    ("irf", "IRFRouter", "IRF (gamma UH)"),
    ("kwt", "SoftGatedKWT", "KWT (soft-gated)"),
    ("diffusive", "DiffusiveWaveIFT", "Diffusive wave"),
    ("sve", "SaintVenantRouter", "Saint-Venant (ref)"),
]


def route_method(net, lateral, outlet, key, cls_name, dt, sv_nodes):
    n_time, n_reach = lateral.shape
    if key == "sve":
        cfg = droute.SaintVenantConfig(); cfg.dt = dt
        for k, v in (("n_nodes", sv_nodes), ("initial_depth", 0.5),
                     ("initial_velocity", 0.1)):
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        cfg = droute.RouterConfig(); cfg.dt = dt
    rt = getattr(droute, cls_name)(net, cfg)
    Q = np.zeros(n_time)
    for t in range(n_time):
        for k in range(n_reach):
            rt.set_lateral_inflow(k, float(lateral[t, k]))
        rt.route_timestep()
        Q[t] = rt.get_discharge(outlet)
    return Q


def compare_metrics(sim, ref, warmup):
    s, r = sim[warmup:], ref[warmup:]
    m = ~(np.isnan(s) | np.isnan(r))
    s, r = s[m], r[m]
    if len(s) < 2:
        return {"kge": float("nan"), "nse": float("nan"),
                "peak_ratio": float("nan"), "peak_lag_h": float("nan"),
                "volume_ratio": float("nan")}
    lag = int(np.argmax(s) - np.argmax(r))
    return {"kge": kge(s, r), "nse": nse(s, r),
            "peak_ratio": float(s.max() / r.max()) if r.max() else float("nan"),
            "peak_lag_h": lag,
            "volume_ratio": float(s.sum() / r.sum()) if r.sum() else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reaches", type=int, default=30)
    ap.add_argument("--reach-len-km", type=float, default=20.0)
    ap.add_argument("--slope", type=float, default=0.0008)
    ap.add_argument("--n-time", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--dt", type=float, default=3600.0)
    ap.add_argument("--sv-nodes", type=int, default=10)
    ap.add_argument("--manning", type=float, default=0.0,
                    help="uniform n; 0 -> use heterogeneous true_manning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/results_catalogue")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = args.reaches; outlet = n - 1
    net = build_chain(n, args.reach_len_km * 1000.0, args.slope)
    n_vals = (np.full(n, args.manning) if args.manning > 0
              else true_manning(n, args.seed))
    set_manning(net, n_vals)
    lateral = synthetic_runoff(args.n_time, n, args.seed)
    print(f"Chain {n} reaches x {args.reach_len_km} km; routing {len(ROUTERS)} schemes")

    sims, timings = {}, {}
    import time as _t
    for key, cls_name, _ in ROUTERS:
        if not hasattr(droute, cls_name):
            print(f"  {key}: not in build, skip"); continue
        try:
            t0 = _t.time()
            sims[key] = route_method(net, lateral, outlet, key, cls_name, args.dt, args.sv_nodes)
            timings[key] = _t.time() - t0
            nanfrac = float(np.isnan(sims[key]).mean())
            print(f"  {key:10s} mean={np.nanmean(sims[key]):.1f} peak={np.nanmax(sims[key]):.1f} "
                  f"nan={nanfrac:.2f} ({args.n_time/timings[key]:.0f} st/s)")
        except Exception as e:
            print(f"  {key:10s} FAIL: {str(e)[:80]}")

    sv_ok = "sve" in sims and not np.all(np.isnan(sims["sve"]))
    ref_key = "sve" if sv_ok else "mc"
    print(f"\nReference = {ref_key}" + ("" if sv_ok else "  (SV unavailable/NaN -> using MC; "
          "rebuild with -DDMC_ENABLE_SUNDIALS=ON for real SV)"))
    ref = sims[ref_key]

    metrics = {k: compare_metrics(sims[k], ref, args.warmup)
               for k in sims if k != ref_key}
    summary = {"reference": ref_key, "sv_available": sv_ok, "n_reaches": n,
               "timings_steps_per_s": {k: args.n_time / timings[k] for k in timings},
               "metrics_vs_reference": metrics}
    (out / "catalogue_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nApproximation error vs reference:")
    for k, m in metrics.items():
        print(f"  {k:10s} KGE={m['kge']:.3f} NSE={m['nse']:.3f} "
              f"peakRatio={m['peak_ratio']:.2f} lag={m['peak_lag_h']}h "
              f"volRatio={m['volume_ratio']:.2f}")
    print(f"\nWrote {out/'catalogue_summary.json'}")

    if HAS_MPL:
        t = np.arange(args.n_time)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        colors = {"mc": "C0", "lag": "C1", "irf": "C2", "kwt": "C3",
                  "diffusive": "C4", "sve": "k"}
        for key, _, label in ROUTERS:
            if key not in sims:
                continue
            lw, ls = (2.2, "-") if key == ref_key else (1.3, "--")
            ax.plot(t[args.warmup:], sims[key][args.warmup:], color=colors[key],
                    lw=lw, ls=ls, alpha=0.85, label=label + (" [ref]" if key == ref_key else ""))
        ax.set_xlabel("time step (h)"); ax.set_ylabel("outlet discharge (m³/s)")
        ax.set_title(f"dRoute catalogue: outlet hydrographs ({n}-reach chain)")
        ax.legend(loc="upper right", ncol=2); ax.grid(alpha=0.3)
        fig.savefig(out / "catalogue_hydrographs.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out/'catalogue_hydrographs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
