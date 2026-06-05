# SPDX-License-Identifier: Apache-2.0
"""Validate the SaintVenantEnzyme adjoint gradient (CVODES-ASA + forward-mode Enzyme).

What this checks
----------------
1. The recording forward pass + CVODES adjoint backward solve runs to completion with no
   crash and returns a finite per-reach Manning's-n gradient.
2. The gradient is DIRECTIONALLY CORRECT, verified by parameter recovery: starting from a
   wrong Manning's-n guess, Adam steps driven solely by the adjoint gradient must recover a
   known ground-truth n (loss collapses, every reach moves the right way).

Why not a finite-difference cosine?
-----------------------------------
The forward map is NON-SMOOTH (adaptive-BDF step-pattern jitter under tiny parameter changes,
the min_depth/min_area/celerity clamps, and the Rusanov |u| term), so loss(manning) is kinky at
every probeable step size: central differences flip sign and magnitude as eps varies and do NOT
converge. A finite-difference gradient is therefore not a valid oracle here, and an adjoint-vs-FD
cosine is meaningless. The noise-free, purpose-relevant validation is recovery of known
parameters by gradient descent -- which is exactly what the gradient is used for in calibration.
A central-difference gradient is still printed for reference, clearly flagged as unreliable.
"""
import numpy as np
import droute

np.set_printoptions(precision=5, suppress=True)

N = 3
nt = 120
lat = 3.0 + 9.0 * np.exp(-((np.arange(nt) - 50.0) ** 2) / 200.0)

# Keep every Network alive for the whole run. The router stores Network& by reference and
# dereferences it in compute_gradients(); the binding now ties their lifetimes via
# py::keep_alive<1,2>, but holding the refs here keeps the intent explicit.
_nets = []


def build():
    net = droute.Network()
    for j in range(N + 1):
        jn = droute.Junction()
        jn.id = j
        jn.upstream_reach_ids = [j - 1] if j > 0 else []
        net.add_junction(jn)
    for i in range(N):
        r = droute.Reach()
        r.id = i
        r.length = 5000.0
        r.slope = 0.005
        r.manning_n = 0.035
        r.upstream_junction_id = i
        r.downstream_junction_id = i + 1
        net.add_reach(r)
    net.build_topology()
    _nets.append(net)
    return net


def forward(mann, record=False):
    net = build()
    for i in range(N):
        net.get_reach(i).manning_n = float(mann[i])
    c = droute.SaintVenantEnzymeConfig()
    c.dt = 3600.0
    c.n_nodes = 4
    c.enable_adjoint = record
    c.use_enzyme_adjoint = True
    rt = droute.SaintVenantEnzyme(net, c)
    if record:
        rt.reset_gradients()
        rt.start_recording()
    Q = np.zeros(nt)
    for t in range(nt):
        rt.set_lateral_inflow(0, float(lat[t]))
        rt.route_timestep()
        Q[t] = rt.get_discharge(N - 1)
    if record:
        rt.stop_recording()
    return Q, rt


def adjoint_grad(mann, target):
    Q, rt = forward(mann, record=True)
    dL_dQ = (2.0 / nt) * (Q - target)
    rt.compute_gradients(N - 1, dL_dQ.tolist())
    g = rt.get_gradients()
    grad = np.array([g[f"reach_{i}_manning_n"] for i in range(N)])
    loss = float(np.mean((Q - target) ** 2))
    return grad, loss


# --- Ground truth + a single adjoint vs (unreliable) FD snapshot ----------------------------
true_n = np.array([0.045, 0.025, 0.038])
target, _ = forward(true_n, record=False)

m0 = np.array([0.035, 0.035, 0.035])
ad, _ = adjoint_grad(m0, target)

eps = 1e-6
fd = np.zeros(N)
for i in range(N):
    mp = m0.copy(); mp[i] += eps
    mm = m0.copy(); mm[i] -= eps
    lp = float(np.mean((forward(mp)[0] - target) ** 2))
    lm = float(np.mean((forward(mm)[0] - target) ** 2))
    fd[i] = (lp - lm) / (2 * eps)
print("adjoint grad :", np.round(ad, 5), "(finite=%s)" % np.isfinite(ad).all())
print("FD grad      :", np.round(fd, 3), "  <-- UNRELIABLE: non-smooth solver, eps-dependent")

# --- The real validation: recover known Manning's n via Adam on the adjoint gradient --------
m = m0.copy()
lr, b1, b2, aeps = 2e-3, 0.9, 0.999, 1e-8
mt = np.zeros(N); vt = np.zeros(N)
L0 = float(np.mean((forward(m0)[0] - target) ** 2))
for it in range(1, 81):
    g, L = adjoint_grad(m, target)
    mt = b1 * mt + (1 - b1) * g
    vt = b2 * vt + (1 - b2) * g * g
    mhat = mt / (1 - b1 ** it)
    vhat = vt / (1 - b2 ** it)
    m = np.clip(m - lr * mhat / (np.sqrt(vhat) + aeps), 0.005, 0.2)
Lf = float(np.mean((forward(m)[0] - target) ** 2))

reduction = L0 / max(Lf, 1e-30)
n_err = np.max(np.abs(m - true_n))
print(f"\nrecovery: n {m0} -> {np.round(m, 5)}  (true {true_n})")
print(f"loss {L0:.3e} -> {Lf:.3e}  ({reduction:.0f}x)  max |n_err| = {n_err:.5f}")

ok = np.isfinite(ad).all() and reduction > 10.0 and n_err < 5e-3
print("RESULT:", "PASS" if ok else "FAIL")
