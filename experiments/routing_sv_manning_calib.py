# SPDX-License-Identifier: Apache-2.0
"""Multi-gauge Manning's-n calibration of the FULL Saint-Venant solver on Bow at Calgary.

This is the differentiable-SV counterpart of the Muskingum-Cunge multi-gauge experiment: it
calibrates per-reach Manning's n on the real 107-reach Bow-at-Calgary network using the dRoute
SaintVenantEnzyme adjoint (CVODES-ASA + forward-mode Enzyme, graph-colored sparse Jacobian,
single-backward-solve multi-gauge source).

Recipe that makes full-eval-window SV calibration feasible (see notes below):
  * dt = 12 h (sub-daily for SV stability, but few enough steps to be affordable).
  * clip the SUMMA spin-up runoff artifact (early-2010 basin inflow spikes to ~34000 m^3/s,
    ~20x the record flood) to a physical per-reach cap so the forward solve stays stable.
  * route the 2010 spin-up UNRECORDED (state equilibrates), then start_recording for the
    2011-2012 evaluation window -- only the scored window carries the adjoint tape.
  * loss = mean over the nested WSC gauges of (1 - KGE); the analytical d(1-KGE)/dQ seeds the
    adjoint at every gauge in a SINGLE backward solve (compute_gradients_multigauge).

Cost (measured): ~1.7 s per recorded 12 h step at Calgary scale -> ~43 min/gradient for the full
730-day window; a coarse Adam calibration is an overnight run. Use --smoke for a fast end-to-end
check (short window, 2 iters) before the full run.
"""
import argparse
import io
import contextlib
import os
import time
import numpy as np
import pandas as pd
import droute

from reservoir_rules_adam import (
    D, OBS, RES, load_inputs, build_network, kge, dloss_dsim, ROUTE_START, CAL,
)

# The 5 nested Bow mainstem gauges (coarse, for the --gauges nested option).
NESTED_STATIONS = {"05BA001", "05BB001", "05BE004", "05BH005", "05BH004"}

N_LO, N_HI = 0.02, 0.08   # per-reach Manning's n bounds


def load_gauges(inp, rec_dates, which):
    """All WSC gauges that map into the network (observations/streamflow/gauge_seg_mapping.csv).

    Because the adjoint injects every gauge's loss source into a SINGLE backward solve
    (compute_gradients_multigauge), using all 10 gauges costs the same as one -- and the extra
    tributary gauges constrain parts of the per-reach roughness field the 5 mainstem gauges
    cannot, which is the whole point of multi-gauge calibration (identifiability)."""
    m = pd.read_csv(f"{OBS}/gauge_seg_mapping.csv")
    gauges = []
    for _, r in m.iterrows():
        sid = str(r["station"]); seg = int(r["seg"])
        if which == "nested" and sid not in NESTED_STATIONS:
            continue
        ridx = inp["id_to_idx"].get(seg)
        if ridx is None:
            continue
        obs = pd.read_csv(f"{OBS}/wsc_{sid}_daily.csv", parse_dates=["date"]).set_index("date")["q_cms"]
        obs = obs.reindex(rec_dates).values.astype(float)
        if np.isfinite(obs).sum() < 10:          # skip gauges with no usable obs in the window
            continue
        gauges.append(dict(name=sid, station=sid, ridx=int(ridx), obs=obs))
    return gauges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt-hours", type=float, default=12.0)
    ap.add_argument("--cap", type=float, default=50.0, help="per-reach lateral inflow clip [m3/s]")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05, help="Adam step in log(n) space")
    ap.add_argument("--n-nodes", type=int, default=4)
    ap.add_argument("--gauges", choices=["all", "nested"], default="all",
                    help="all WSC gauges in the network (10) or just the 5 nested mainstem")
    ap.add_argument("--min-default-kge", type=float, default=-2.0,
                    help="drop gauges whose DEFAULT-run KGE is below this (severe volume bias that "
                         "Manning's n cannot fix); set to -inf to keep all")
    ap.add_argument("--smoke", action="store_true", help="short window + 2 iters for validation")
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    DT = a.dt_hours * 3600.0
    sub = int(round(86400.0 / DT))            # recorded steps per day
    if a.smoke:
        a.iters = 2

    inp = load_inputs()
    runoff = np.clip(inp["daily_runoff"], 0.0, a.cap)
    dates = inp["dates"]
    n_seg = len(inp["seg_ids"])
    order = None  # set after network build

    # spin-up = everything before the eval window; record the eval window (or a short smoke slice)
    i_eval0 = int(np.argmax(dates >= pd.Timestamp(CAL[0])))
    rec_days_total = int((pd.Timestamp(CAL[1]) - pd.Timestamp(CAL[0])).days) + 1
    rec_days = min(60, rec_days_total) if a.smoke else rec_days_total
    rec_dates = dates[i_eval0:i_eval0 + rec_days]
    n_rec = rec_days * sub                    # recorded 12h steps

    # gauges: all WSC gauges in the network (or just the nested mainstem), with daily obs
    gauges = load_gauges(inp, rec_dates, a.gauges)
    gauge_ids = [g["ridx"] for g in gauges]

    print(f"SV multi-gauge calibration | dt={a.dt_hours}h ({sub}/day) | cap={a.cap} | "
          f"record {rec_days}d ({n_rec} steps) from {rec_dates[0].date()} | {len(gauges)} gauges | "
          f"{n_seg} reaches | iters={a.iters}{'  [SMOKE]' if a.smoke else ''}", flush=True)

    # --- forward+gradient closure -----------------------------------------------------------
    def route_and_grad(n_vec, want_grad):
        net = build_network(inp["seg_ids"], inp["downstream_idx"], inp["lengths"], inp["slopes"])
        for i in range(n_seg):
            net.get_reach(i).manning_n = float(n_vec[i])
        topo = np.asarray(net.topological_order(), dtype=int)
        c = droute.SaintVenantEnzymeConfig()
        c.dt = DT; c.n_nodes = a.n_nodes
        c.enable_adjoint = want_grad; c.use_enzyme_adjoint = want_grad
        c.use_colored_jacobian = True
        rt = droute.SaintVenantEnzyme(net, c)
        with contextlib.redirect_stderr(io.StringIO()):   # suppress recoverable spin-up warnings
            # spin-up, unrecorded
            for d in range(0, i_eval0):
                for s in range(sub):
                    for idx in topo:
                        rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
                    rt.route_timestep()
            if want_grad:
                rt.reset_gradients(); rt.start_recording()
            # recorded eval window: store daily (end-of-day) Q at every gauge
            Qd = {g["ridx"]: np.zeros(rec_days) for g in gauges}
            for k in range(rec_days):
                for s in range(sub):
                    for idx in topo:
                        rt.set_lateral_inflow(int(idx), float(runoff[i_eval0 + k, idx]))
                    rt.route_timestep()
                for g in gauges:
                    Qd[g["ridx"]][k] = rt.get_discharge(g["ridx"])
            if want_grad:
                rt.stop_recording()
        grad = None
        if want_grad:
            # per-gauge dL/dQ at the end-of-day recorded steps (loss = mean_g (1-KGE_g))
            dL_per_gauge = []
            for g in gauges:
                dday = dloss_dsim(Qd[g["ridx"]], g["obs"]) / len(gauges)
                series = np.zeros(n_rec)
                for k in range(rec_days):
                    series[(k + 1) * sub - 1] = dday[k]   # end-of-day step
                dL_per_gauge.append(series.tolist())
            with contextlib.redirect_stderr(io.StringIO()):
                rt.compute_gradients_multigauge(gauge_ids, dL_per_gauge)
            gd = rt.get_gradients()
            grad = np.array([gd[f"reach_{i}_manning_n"] for i in range(n_seg)])
        return Qd, grad

    def metrics(Qd):
        kges = {g["name"]: kge(Qd[g["ridx"]], g["obs"]) for g in gauges}
        loss = np.mean([1.0 - (k if np.isfinite(k) else -1.0) for k in kges.values()])
        return kges, loss

    # --- Adam in log(n) space ---------------------------------------------------------------
    n = np.full(n_seg, 0.035)
    u = np.log(n)
    mt = np.zeros(n_seg); vt = np.zeros(n_seg); b1, b2, eps = 0.9, 0.999, 1e-8
    Qd0, _ = route_and_grad(n, want_grad=False)
    kges0_all, _ = metrics(Qd0)
    print(f"  it 0 (all {len(gauges)} gauges): meanKGE={np.mean(list(kges0_all.values())):.4f}  "
          + " ".join(f"{g['name'][-4:]}={kges0_all[g['name']]:.3f}" for g in gauges), flush=True)

    # Drop gauges Manning's-n calibration cannot help: a severely negative default KGE here is
    # volume bias (KGE's beta = sim_mean/obs_mean), a runoff problem -- routing conserves volume,
    # so n moves timing/attenuation, not the mean. Keeping them would let unfixable terms dominate
    # the loss and waste the gradient. (Filter is data-driven on the default-run KGE.)
    keep = [g for g in gauges
            if np.isfinite(kges0_all[g["name"]]) and kges0_all[g["name"]] >= a.min_default_kge]
    dropped = [g["name"] for g in gauges if g not in keep]
    if dropped:
        print(f"  dropped {len(dropped)} gauge(s) with default KGE < {a.min_default_kge} "
              f"(volume bias, unfixable by routing): {', '.join(dropped)}", flush=True)
    gauges = keep
    gauge_ids = [g["ridx"] for g in gauges]
    kges0 = {g["name"]: kges0_all[g["name"]] for g in gauges}
    loss0 = metrics(Qd0)[1]                       # loss on the kept calibration set
    best = dict(loss=loss0, n=n.copy(), it=0, kges=kges0)
    print(f"  calibrating on {len(gauges)} gauges: meanKGE={np.mean(list(kges0.values())):.4f} "
          f"loss={loss0:.4f}", flush=True)

    for it in range(1, a.iters + 1):
        t0 = time.time()
        Qd, grad = route_and_grad(n, want_grad=True)
        kges, loss = metrics(Qd)
        # chain rule to log space: dL/du = dL/dn * n
        g_u = grad * n
        mt = b1 * mt + (1 - b1) * g_u
        vt = b2 * vt + (1 - b2) * g_u * g_u
        mh = mt / (1 - b1 ** it); vh = vt / (1 - b2 ** it)
        u = u - a.lr * mh / (np.sqrt(vh) + eps)
        n = np.clip(np.exp(u), N_LO, N_HI); u = np.log(n)
        if loss < best["loss"]:
            best = dict(loss=loss, n=n.copy(), it=it, kges=kges)
        print(f"  it {it}: meanKGE={np.mean([v for v in kges.values()]):.4f} loss={loss:.4f}  "
              + " ".join(f"{g['name'][-4:]}={kges[g['name']]:.3f}" for g in gauges)
              + f"  |g|={np.abs(grad).max():.1e}  {time.time()-t0:.0f}s", flush=True)

    # --- report + save ----------------------------------------------------------------------
    Qdf, _ = route_and_grad(best["n"], want_grad=False)
    kf, lf = metrics(Qdf)
    print(f"\nBEST it {best['it']}: loss {loss0:.4f} -> {best['loss']:.4f}  "
          f"meanKGE {np.mean(list(kges0.values())):.4f} -> {np.mean(list(kf.values())):.4f}", flush=True)
    rows = []
    for g in gauges:
        rows.append((g["name"], g["station"], round(kges0[g["name"]], 3), round(kf[g["name"]], 3)))
    tbl = pd.DataFrame(rows, columns=["gauge", "station", "KGE_default", "KGE_calibrated"])
    tag = "smoke" if a.smoke else "full"
    tbl.to_csv(f"{RES}/sv_manning_calib_{tag}_metrics.csv", index=False)
    np.savetxt(f"{RES}/sv_manning_calib_{tag}_n.csv", best["n"], delimiter=",")
    print(tbl.to_string(index=False))
    print(f"\nsaved -> {RES}/sv_manning_calib_{tag}_metrics.csv")


if __name__ == "__main__":
    main()
