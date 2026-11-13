#ifndef DMC_ROUTE_TRANSFERS_HPP
#define DMC_ROUTE_TRANSFERS_HPP

/**
 * @file transfers.hpp
 * @brief Non-tree edges of a river network: transfers, bifurcations and
 *        groundwater leaks.
 *
 * A surface river network is a tree (each reach drains to one downstream
 * reach). Real hydrofabrics also contain directed extra edges that move water
 * from one reach's outflow to another reach's inflow:
 *
 *   headrace             operated withdrawal to a power plant tailrace
 *   diversion            operated withdrawal without a plant
 *   natural_bifurcation  passive split of a river (constant fraction)
 *   groundwater          lagged leak through a linear groundwater store
 *
 * The routing logic that applies these edges lives in transfer_routing.hpp;
 * this header only holds the data types so that network.hpp can own the edge
 * list without a circular include.
 *
 * Learnable parameters (registered on the AD tape with the channel parameters):
 *   fraction  share of the source outflow (or lake inflow) that is withdrawn
 *   capacity  maximum withdrawal [m^3/s] (headrace/diversion only)
 *   gw_k      residence time of the groundwater store [s] (groundwater only)
 */

#include "types.hpp"
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace dmc {

enum class TransferType {
    Headrace = 0,
    Diversion = 1,
    NaturalBifurcation = 2,
    Groundwater = 3
};

inline TransferType parse_transfer_type(const std::string& s) {
    if (s == "headrace") return TransferType::Headrace;
    if (s == "diversion") return TransferType::Diversion;
    if (s == "natural_bifurcation" || s == "bifurcation") return TransferType::NaturalBifurcation;
    if (s == "groundwater") return TransferType::Groundwater;
    throw std::invalid_argument("Unknown transfer type '" + s +
        "' (expected headrace, diversion, natural_bifurcation or groundwater)");
}

inline std::string transfer_type_name(TransferType t) {
    switch (t) {
        case TransferType::Headrace: return "headrace";
        case TransferType::Diversion: return "diversion";
        case TransferType::NaturalBifurcation: return "natural_bifurcation";
        case TransferType::Groundwater: return "groundwater";
    }
    return "unknown";
}

/**
 * Operating-rule hook for operated edges.
 *
 * Called once per edge and timestep as rule(year, time_s, q_available) and
 * returns the maximum withdrawal [m^3/s] allowed at that time. A NaN return
 * value means "no constraint". The rule is treated as a passive (non
 * differentiated) cap; fraction and capacity remain differentiable.
 */
using TransferRule = std::function<double(double year, double time_s, double q_available)>;

/**
 * One directed transfer edge.
 */
struct Transfer {
    std::string id;
    TransferType type = TransferType::Diversion;
    bool operated = false;

    int from_reach = -1;      // reach whose outflow (or storage) is tapped
    int to_reach = -1;        // reach whose inflow receives the water; -1 = leaves the system (declared loss)

    // Learnable parameters
    Real fraction = 1.0;      // [0,1] share of the reference flow
    Real capacity = 0.0;      // [m^3/s], only used when has_capacity
    bool has_capacity = false;
    Real gw_k = 30.0 * 86400.0;  // groundwater residence time [s]

    // Activity window in (fractional) calendar years; NaN = open-ended
    double commissioned = std::numeric_limits<double>::quiet_NaN();
    double decommissioned = std::numeric_limits<double>::quiet_NaN();

    // When true the edge withdraws from the source lake's storage (reservoir
    // operation) instead of from the routed outflow. Only honoured when the
    // source reach is a level-pool lake.
    bool from_storage = false;

    // Manual switch (e.g. to drop hypothesis edges without deleting them)
    bool enabled = true;

    // Optional operating rule (operated edges)
    TransferRule rule;

    // ---- runtime state ----
    Real gw_store = 0.0;      // groundwater store [m^3]
    Real last_flow = 0.0;     // withdrawal in the last step [m^3/s]

    // ---- gradients ----
    double grad_fraction = 0.0;
    double grad_capacity = 0.0;
    double grad_gw_k = 0.0;

    bool is_groundwater() const { return type == TransferType::Groundwater; }
    bool is_operated_type() const {
        return type == TransferType::Headrace || type == TransferType::Diversion;
    }
};

/**
 * Global switches for transfer processing.
 */
struct TransferOptions {
    // Calendar year at simulation time 0 (fractional years allowed). When NaN,
    // commissioned/decommissioned windows are ignored and every edge is active.
    double sim_start_year = std::numeric_limits<double>::quiet_NaN();

    // Groundwater edges are hypotheses; they are skipped unless enabled.
    bool enable_groundwater = false;
};

/**
 * Is the edge active at the given (fractional) year?
 */
inline bool transfer_active(const Transfer& t, const TransferOptions& opt, double year) {
    if (!t.enabled) return false;
    if (t.is_groundwater() && !opt.enable_groundwater) return false;
    if (std::isnan(year)) return true;
    if (!std::isnan(t.commissioned) && year < t.commissioned) return false;
    if (!std::isnan(t.decommissioned) && year >= t.decommissioned) return false;
    return true;
}

inline double time_to_year(const TransferOptions& opt, double time_s) {
    if (std::isnan(opt.sim_start_year)) return std::numeric_limits<double>::quiet_NaN();
    return opt.sim_start_year + time_s / (365.2425 * 86400.0);
}

} // namespace dmc

#endif // DMC_ROUTE_TRANSFERS_HPP
