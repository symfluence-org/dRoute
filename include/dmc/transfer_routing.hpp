#ifndef DMC_ROUTE_TRANSFER_ROUTING_HPP
#define DMC_ROUTE_TRANSFER_ROUTING_HPP

/**
 * @file transfer_routing.hpp
 * @brief Per-timestep application of transfer edges and level-pool lake nodes,
 *        plus a mass-balance tracker. Shared by MuskingumCungeRouter and
 *        IRFRouter.
 *
 * Step protocol used by the routers (reaches visited in the topological order
 * of the combined tree + transfer graph, see Network::build_topology):
 *
 *   idx = transfer_ops::begin_step(net, dt, losses)
 *       zero transfer_in/out, release groundwater stores (lagged edges)
 *   for reach in order:
 *       inflow = sum(up.outflow_curr - up.transfer_out) + reach.transfer_in
 *       outflow = channel routing, or route_level_pool() for lake nodes
 *                 (route_level_pool also takes operated withdrawals from storage)
 *       transfer_ops::apply_outflow_transfers(...)
 *           withdraw from the routed outflow and add to the target's
 *           transfer_in (or to a groundwater store, or to declared losses)
 *
 * The reach keeps its gross outflow in outflow_curr (this is what a gauge at
 * the reach outlet, upstream of the offtake, measures); downstream reaches
 * receive outflow_curr - transfer_out.
 *
 * All parameter arithmetic uses Real so fractions, capacities, groundwater
 * residence times and weir parameters carry CoDiPack gradients. min/max are
 * the hard (piecewise) versions, as in the existing lake routine, which keeps
 * the water balance exact.
 */

#include "network.hpp"
#include "transfers.hpp"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace dmc {

/**
 * Cumulative water balance since the last reset [m^3].
 *
 *   input = terminal_outflow + declared_losses
 *         + lake_storage_change + groundwater_storage_change
 *         + channel_storage_change + residual
 *
 * lake_storage_change includes level-pool, rating-curve and subgrid lake
 * stores. channel_storage_change is exact for the IRF router; for
 * Muskingum-Cunge it uses the Muskingum storage K(XI + (1-X)O), which is exact
 * within a step but not across steps when K and X change, so expect a small
 * transient residual that vanishes at steady state.
 */
struct MassBalance {
    double input = 0.0;
    double terminal_outflow = 0.0;
    double declared_losses = 0.0;
    double lake_storage_change = 0.0;
    double groundwater_storage_change = 0.0;
    double channel_storage_change = 0.0;
    double residual = 0.0;
    double relative_residual = 0.0;
    double elapsed_time = 0.0;
};

namespace transfer_ops {

/** transfers indexed by source reach; empty when the network has no transfers */
using TransferIndex = std::unordered_map<int, std::vector<size_t>>;

inline double lake_storage(const Network& net) {
    double s = 0.0;
    for (int id : net.topological_order()) {
        const Reach& r = net.get_reach(id);
        if (r.is_lake) s += to_double(r.storage);
        if (r.has_subgrid_lake) s += to_double(r.subgrid_storage);
    }
    return s;
}

inline double groundwater_storage(const Network& net) {
    double s = 0.0;
    for (const auto& t : net.transfers()) s += to_double(t.gw_store);
    return s;
}

/**
 * Deliver a withdrawal: to the target's inflow, into the edge's groundwater
 * store, or out of the system (declared loss) when there is no target.
 */
inline void deliver(Network& net, Transfer& t, const Real& w, double dt, double& loss_volume) {
    t.last_flow = w;
    if (t.is_groundwater()) {
        t.gw_store = t.gw_store + w * Real(dt);
    } else if (t.to_reach >= 0) {
        Reach& to = net.get_reach(t.to_reach);
        to.transfer_in = to.transfer_in + w;
    } else {
        loss_volume += to_double(w) * dt;
    }
}

/**
 * Start a timestep: clear per-step transfer fluxes and release the
 * groundwater stores (linear reservoir, exact exponential decay over dt, so
 * leaked water reaches the target at the earliest one step later).
 * Returns the source-reach index of the transfers.
 */
inline TransferIndex begin_step(Network& net, double dt, double& loss_volume) {
    TransferIndex index;
    auto& transfers = net.transfers();
    if (transfers.empty()) return index;

    for (int id : net.topological_order()) {
        Reach& r = net.get_reach(id);
        r.transfer_in = Real(0.0);
        r.transfer_out = Real(0.0);
    }
    for (size_t i = 0; i < transfers.size(); ++i) {
        Transfer& t = transfers[i];
        index[t.from_reach].push_back(i);
        t.last_flow = Real(0.0);
        if (t.is_groundwater() && to_double(t.gw_store) > 0.0) {
            Real k = safe_max(t.gw_k, Real(1.0));
            Real released = t.gw_store * (Real(1.0) - exp(Real(-dt) / k));
            t.gw_store = t.gw_store - released;
            Real q = released / Real(dt);
            if (t.to_reach >= 0) {
                Reach& to = net.get_reach(t.to_reach);
                to.transfer_in = to.transfer_in + q;
            } else {
                loss_volume += to_double(released);
            }
        }
    }
    return index;
}

/** Passive cap from an operating rule (NaN = no cap). */
inline double rule_cap(const Transfer& t, double year, double time_s, double q_available) {
    if (!t.rule) return std::numeric_limits<double>::quiet_NaN();
    double r = t.rule(year, time_s, q_available);
    if (std::isnan(r)) return r;
    return r < 0.0 ? 0.0 : r;
}

/** Demand of one edge given its reference flow (before availability limits). */
inline Real edge_demand(const Transfer& t, const Real& q_ref, double cap) {
    Real demand = t.fraction * q_ref;
    if (t.is_operated_type() && t.has_capacity) {
        demand = safe_min(demand, Real(t.capacity));
    }
    if (!std::isnan(cap)) {
        demand = safe_min(demand, Real(cap));
    }
    return demand;
}

/**
 * Is this transfer taken from the storage of its source lake?
 */
inline bool is_storage_withdrawal(const Transfer& t, const Reach& from) {
    return t.from_storage && from.is_lake && from.lake_model == LAKE_LEVEL_POOL;
}

/**
 * Level-pool lake node.
 *
 *   dS/dt = I - spill(S) - sum_i W_i
 *   spill  = weir_coef * h^weir_exp,  h = max(S, 0) / A   (S relative to the crest)
 *   W_i    = min(fraction_i * I, capacity_i, rule_i(t), available)
 *            for operated edges flagged from_storage (reservoir withdrawals),
 *            drawing the lake down to lake_min_storage at most.
 *
 * The spill uses a linearised implicit update (stable for any dt, since the
 * weir curve is convex) and the storage update is exact in volume, so the
 * node conserves mass to round-off. Returns the gross outflow (spill + all
 * storage withdrawals) averaged over the step; withdrawals are recorded in
 * reach.transfer_out and delivered to their targets.
 */
inline Real route_level_pool(Network& net, Reach& reach, const Real& inflow_total,
                             double dt, int nsub, const TransferIndex& index,
                             double year, double time_s, double& loss_volume) {
    nsub = nsub > 0 ? nsub : 1;
    const double sub_dt = dt / nsub;
    const Real A = Real(reach.lake_area > 1.0 ? reach.lake_area : 1.0);
    const Real Smin = Real(reach.lake_min_storage < 0.0 ? reach.lake_min_storage : 0.0);
    const Real I = safe_max(inflow_total, Real(0.0));

    // Active storage withdrawals for this lake
    std::vector<size_t> edges;
    std::vector<double> caps;
    auto it = index.find(reach.id);
    if (it != index.end()) {
        const TransferOptions& opt = net.transfer_options();
        for (size_t ti : it->second) {
            Transfer& t = net.transfers()[ti];
            if (!is_storage_withdrawal(t, reach)) continue;
            if (!transfer_active(t, opt, year)) continue;
            edges.push_back(ti);
            double q_av = to_double(I) + std::max(to_double(reach.storage) - to_double(Smin), 0.0) / dt;
            caps.push_back(rule_cap(t, year, time_s, q_av));
        }
    }

    Real S = reach.storage;
    Real spill_sum = Real(0.0);
    std::vector<Real> w_sum(edges.size(), Real(0.0));

    for (int k = 0; k < nsub; ++k) {
        // Withdrawals, limited by what the lake can deliver this substep
        Real avail = safe_max(Real((S - Smin) / Real(sub_dt) + I), Real(0.0));
        Real W_tot = Real(0.0);
        for (size_t e = 0; e < edges.size(); ++e) {
            const Transfer& t = net.transfers()[edges[e]];
            Real demand = edge_demand(t, I, caps[e]);
            Real w = safe_max(safe_min(demand, Real(avail - W_tot)), Real(0.0));
            W_tot = W_tot + w;
            w_sum[e] = w_sum[e] + w;
        }

        // Weir spill over the crest, linearised implicit in S
        Real net_in = I - W_tot;
        Real Q = Real(0.0);
        Real dQdS = Real(0.0);
        if (to_double(S) > 0.0) {
            Real h = S / A;
            Q = reach.lake_weir_coef * safe_pow_real(h, reach.lake_weir_exp);
            dQdS = reach.lake_weir_exp * Q / S;
        }
        Real dS = Real(sub_dt) * (net_in - Q) / (Real(1.0) + Real(sub_dt) * dQdS);
        Real spill = net_in - dS / Real(sub_dt);
        spill = safe_max(spill, Real(0.0));
        // cannot spill water that is below the crest after this substep
        spill = safe_min(spill, Real(safe_max(Real(S + Real(sub_dt) * net_in), Real(0.0)) / Real(sub_dt)));
        S = S + Real(sub_dt) * (net_in - spill);
        spill_sum = spill_sum + spill;
    }
    reach.storage = S;

    Real W_mean_tot = Real(0.0);
    for (size_t e = 0; e < edges.size(); ++e) {
        Transfer& t = net.transfers()[edges[e]];
        Real w = w_sum[e] / Real(nsub);
        W_mean_tot = W_mean_tot + w;
        reach.transfer_out = reach.transfer_out + w;
        deliver(net, t, w, dt, loss_volume);
    }
    return spill_sum / Real(nsub) + W_mean_tot;
}

/**
 * Apply the outflow transfers of a reach whose outflow_curr has just been
 * computed. Edges are applied in input order; each takes
 *   min(demand, what is left of the outflow)
 * where demand = fraction * Q (headrace/diversion additionally capped by
 * capacity and the rule hook; natural_bifurcation and groundwater are a
 * constant share of Q). Storage withdrawals of level-pool lakes were already
 * handled in route_level_pool and are skipped here.
 */
inline void apply_outflow_transfers(Network& net, Reach& reach, const TransferIndex& index,
                                    double year, double time_s, double dt, double& loss_volume) {
    auto it = index.find(reach.id);
    if (it == index.end()) return;
    const TransferOptions& opt = net.transfer_options();
    // reference flow: what leaves the reach after any storage withdrawals
    const Real Q0 = safe_max(Real(reach.outflow_curr - reach.transfer_out), Real(0.0));
    Real remaining = Q0;
    for (size_t ti : it->second) {
        Transfer& t = net.transfers()[ti];
        if (is_storage_withdrawal(t, reach)) continue;
        if (!transfer_active(t, opt, year)) continue;
        double cap = t.is_operated_type() ? rule_cap(t, year, time_s, to_double(remaining))
                                          : std::numeric_limits<double>::quiet_NaN();
        Real demand = edge_demand(t, Q0, cap);
        Real w = safe_max(safe_min(demand, remaining), Real(0.0));
        remaining = remaining - w;
        reach.transfer_out = reach.transfer_out + w;
        deliver(net, t, w, dt, loss_volume);
    }
}

/**
 * Accumulates the water balance over routed timesteps.
 */
class MassBalanceTracker {
public:
    bool initialized() const { return initialized_; }

    void reset(const Network& net, double channel_storage) {
        initialized_ = true;
        input_ = terminal_ = losses_ = elapsed_ = 0.0;
        lake0_ = lake_storage(net);
        gw0_ = groundwater_storage(net);
        chan0_ = channel_storage;
        prev_terminal_.clear();
        for (int id : net.terminal_reaches()) {
            prev_terminal_[id] = to_double(net.get_reach(id).net_outflow());
        }
    }

    void invalidate() { initialized_ = false; }

    void add_input(const Network& net, double dt) {
        double s = 0.0;
        for (int id : net.topological_order()) s += to_double(net.get_reach(id).lateral_inflow);
        input_ += s * dt;
        elapsed_ += dt;
    }

    /**
     * Add the terminal outflow volume of the step just routed.
     * trapezoid=true integrates channel outflow with the trapezoidal rule
     * (Muskingum-Cunge); lake nodes and the IRF router use the step value.
     */
    void add_terminal(const Network& net, double dt, bool trapezoid) {
        for (int id : net.terminal_reaches()) {
            const Reach& r = net.get_reach(id);
            double q = to_double(r.net_outflow());
            double& prev = prev_terminal_[id];
            if (trapezoid && !r.is_lake) {
                terminal_ += 0.5 * (prev + q) * dt;
            } else {
                terminal_ += q * dt;
            }
            prev = q;
        }
    }

    void add_losses(double v) { losses_ += v; }

    MassBalance report(const Network& net, double channel_storage) const {
        MassBalance mb;
        mb.input = input_;
        mb.terminal_outflow = terminal_;
        mb.declared_losses = losses_;
        mb.lake_storage_change = lake_storage(net) - lake0_;
        mb.groundwater_storage_change = groundwater_storage(net) - gw0_;
        mb.channel_storage_change = channel_storage - chan0_;
        mb.residual = mb.input - mb.terminal_outflow - mb.declared_losses
                    - mb.lake_storage_change - mb.groundwater_storage_change
                    - mb.channel_storage_change;
        mb.relative_residual = mb.input > 0.0 ? mb.residual / mb.input : 0.0;
        mb.elapsed_time = elapsed_;
        return mb;
    }

private:
    bool initialized_ = false;
    double input_ = 0.0, terminal_ = 0.0, losses_ = 0.0, elapsed_ = 0.0;
    double lake0_ = 0.0, gw0_ = 0.0, chan0_ = 0.0;
    std::unordered_map<int, double> prev_terminal_;
};

} // namespace transfer_ops
} // namespace dmc

#endif // DMC_ROUTE_TRANSFER_ROUTING_HPP
