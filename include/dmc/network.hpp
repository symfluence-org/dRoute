#ifndef DMC_ROUTE_NETWORK_HPP
#define DMC_ROUTE_NETWORK_HPP

#include "types.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <queue>
#include <algorithm>

namespace dmc {

/**
 * Channel geometry parameters using power-law relationships:
 *   Width = width_coef * Q^width_exp
 *   Depth = depth_coef * Q^depth_exp
 * 
 * These can be calibrated/learned via AD.
 * 
 * IMPORTANT: Uses exp(exp * log(Q)) form instead of pow(Q, exp) to ensure
 * gradients flow through the exponent parameters (width_exp, depth_exp).
 */
struct ChannelGeometry {
    Real width_coef = 7.2;    // Leopold & Maddock (1953) defaults
    Real width_exp = 0.5;
    Real depth_coef = 0.27;
    Real depth_exp = 0.3;
    
    // AD-safe power function: x^y = exp(y * log(x))
    // This ensures gradients propagate through both x and y
    static Real ad_safe_pow(const Real& base, const Real& exponent) {
        // Ensure base > 0 for log
        Real safe_base = safe_max(base, Real(1e-6));
        return exp(exponent * log(safe_base));
    }
    
    // Compute hydraulic properties at given discharge
    Real width(const Real& Q) const {
        Real Q_safe = safe_max(Q, Real(0.01));
        return width_coef * ad_safe_pow(Q_safe, width_exp);
    }
    
    Real depth(const Real& Q) const {
        Real Q_safe = safe_max(Q, Real(0.01));
        return depth_coef * ad_safe_pow(Q_safe, depth_exp);
    }
    
    Real area(const Real& Q) const {
        return width(Q) * depth(Q);
    }
    
    Real wetted_perimeter(const Real& Q) const {
        Real w = width(Q);
        Real d = depth(Q);
        return w + Real(2.0) * d;  // Rectangular approximation
    }
    
    Real hydraulic_radius(const Real& Q) const {
        return area(Q) / wetted_perimeter(Q);
    }
};

/**
 * A single river reach with routing parameters.
 */
struct Reach {
    int id;                     // Unique identifier
    std::string name;           // Optional name
    
    // Fixed geometry
    double length;              // Δx [m]
    double slope;               // S₀ [m/m] - bed slope
    
    // Learnable parameter
    Real manning_n = 0.035;     // Manning's roughness coefficient
    
    // Channel geometry (learnable)
    ChannelGeometry geometry;
    
    // Topology
    int upstream_junction_id = -1;   // -1 for headwater
    int downstream_junction_id = -1; // -1 for outlet
    
    // State variables (for current timestep)
    Real inflow_prev = 0.0;     // I(t)
    Real inflow_curr = 0.0;     // I(t+Δt)
    Real outflow_prev = 0.0;    // Q(t)
    Real outflow_curr = 0.0;    // Q(t+Δt)
    Real lateral_inflow = 0.0;  // Lateral inflow from catchment [m³/s]
    
    // Computed quantities (stored for output/debugging)
    Real velocity = 0.0;
    Real celerity = 0.0;
    Real K = 0.0;               // Muskingum K
    Real X = 0.0;               // Muskingum X
    
    // Gradient storage
    double grad_manning_n = 0.0;
    double grad_width_coef = 0.0;
    double grad_width_exp = 0.0;
    double grad_depth_coef = 0.0;
    double grad_depth_exp = 0.0;

    // ===== Lake / reservoir extension =====
    // When is_lake is true this reach routes via a storage-discharge rating curve
    // (dS/dt = inflow - Q(S)) instead of the Muskingum-Cunge channel update.
    bool is_lake = false;
    int  lake_type = 0;             // 0 = natural lake, 1 = reservoir (regulated)
    // Fixed geometry (from HydroLAKES: Lake_area, Vol_total/Depth_avg)
    double lake_area = 0.0;         // surface area A [m^2]
    double storage_max = 0.0;       // S_max [m^3] (full-supply / total volume)
    // Storage state
    Real storage = 0.0;             // S [m^3]
    Real storage_prev = 0.0;
    // Learnable rating-curve parameters (Q_out = q_min + (q_ref-q_min)*frac^exp + spill):
    //   natural lake  -> q_min=0, exp~1.5 (weir-like)
    //   reservoir     -> q_min=min release, q_ref=target release, all learnable
    Real lake_q_ref = 0.0;          // reference/target outflow [m^3/s]  (init: HydroLAKES Dis_avg)
    Real lake_exp = 1.5;            // rating exponent
    Real lake_q_min = 0.0;          // minimum (regulated) release [m^3/s]
    Real lake_s_dead = 0.0;         // dead storage [m^3] (no release below this)
    Real lake_spill_coef = 1.0;     // fraction of above-full storage spilled per step
    // Gradient storage for lake params
    double grad_lake_q_ref = 0.0;
    double grad_lake_exp = 0.0;
    double grad_lake_q_min = 0.0;
    double grad_lake_spill_coef = 0.0;

    // ===== Subgrid (off-network) lake store =====
    // For a CHANNEL reach: a fraction of the catchment's lateral inflow first passes
    // through aggregated small lakes (storage-discharge) before entering the channel,
    // representing off-network HydroLAKES that attenuate local runoff. Differentiable;
    // q_ref/exp are learnable.
    bool has_subgrid_lake = false;
    double subgrid_lake_frac = 0.0;      // fraction of lateral inflow routed through lakes [0,1]
    double subgrid_storage_max = 0.0;    // S_max of the aggregate subgrid store [m^3]
    Real subgrid_storage = 0.0;          // state [m^3]
    Real subgrid_storage_prev = 0.0;
    Real subgrid_q_ref = 0.0;            // reference outflow [m^3/s] (learnable)
    Real subgrid_exp = 1.0;              // rating exponent (learnable)
    double grad_subgrid_q_ref = 0.0;
    double grad_subgrid_exp = 0.0;
};

/**
 * A junction where reaches meet.
 */
struct Junction {
    int id;
    std::string name;
    
    std::vector<int> upstream_reach_ids;    // Reaches flowing into this junction
    std::vector<int> downstream_reach_ids;  // Reaches flowing out (usually 1)
    
    // For boundary conditions
    bool is_headwater = false;
    bool is_outlet = false;
    
    // External inflow (e.g., point source, tributary not explicitly modeled)
    Real external_inflow = 0.0;
};

/**
 * River network topology with efficient traversal.
 */
class Network {
public:
    // Add components
    void add_reach(Reach reach);
    void add_junction(Junction junction);
    
    // Build traversal order (call after adding all components)
    void build_topology();
    
    // Access
    Reach& get_reach(int id);
    const Reach& get_reach(int id) const;
    Junction& get_junction(int id);
    const Junction& get_junction(int id) const;
    
    // Iterators in topological order (upstream to downstream)
    const std::vector<int>& topological_order() const { return topo_order_; }
    
    // Network properties
    size_t num_reaches() const { return reaches_.size(); }
    size_t num_junctions() const { return junctions_.size(); }
    
    // Parameter access for AD
    std::vector<Real*> get_all_parameters();
    std::vector<std::string> get_parameter_names();
    void collect_gradients();
    void zero_gradients();
    
    // I/O
    static Network load_from_geojson(const std::string& filepath);
    static Network load_from_csv(const std::string& reaches_file, 
                                  const std::string& junctions_file);
    void save_state(const std::string& filepath) const;
    
private:
    std::unordered_map<int, Reach> reaches_;
    std::unordered_map<int, Junction> junctions_;
    std::vector<int> topo_order_;  // Reach IDs in topological order
    
    void compute_topological_order();
};

// Implementation of key methods (inline for header-only convenience)

inline void Network::add_reach(Reach reach) {
    reaches_[reach.id] = std::move(reach);
}

inline void Network::add_junction(Junction junction) {
    junctions_[junction.id] = std::move(junction);
}

inline Reach& Network::get_reach(int id) {
    return reaches_.at(id);
}

inline const Reach& Network::get_reach(int id) const {
    return reaches_.at(id);
}

inline Junction& Network::get_junction(int id) {
    return junctions_.at(id);
}

inline const Junction& Network::get_junction(int id) const {
    return junctions_.at(id);
}

inline std::vector<Real*> Network::get_all_parameters() {
    std::vector<Real*> params;
    for (auto& [id, reach] : reaches_) {
        if (reach.is_lake) {
            // Learnable lake/reservoir rating-curve parameters
            params.push_back(&reach.lake_q_ref);
            params.push_back(&reach.lake_exp);
            params.push_back(&reach.lake_q_min);
            params.push_back(&reach.lake_spill_coef);
        } else {
            params.push_back(&reach.manning_n);
            params.push_back(&reach.geometry.width_coef);
            params.push_back(&reach.geometry.width_exp);
            params.push_back(&reach.geometry.depth_coef);
            params.push_back(&reach.geometry.depth_exp);
            if (reach.has_subgrid_lake) {
                params.push_back(&reach.subgrid_q_ref);
                params.push_back(&reach.subgrid_exp);
            }
        }
    }
    return params;
}

inline std::vector<std::string> Network::get_parameter_names() {
    std::vector<std::string> names;
    for (auto& [id, reach] : reaches_) {
        std::string prefix = "reach_" + std::to_string(id) + "_";
        if (reach.is_lake) {
            names.push_back(prefix + "lake_q_ref");
            names.push_back(prefix + "lake_exp");
            names.push_back(prefix + "lake_q_min");
            names.push_back(prefix + "lake_spill_coef");
        } else {
            names.push_back(prefix + "manning_n");
            names.push_back(prefix + "width_coef");
            names.push_back(prefix + "width_exp");
            names.push_back(prefix + "depth_coef");
            names.push_back(prefix + "depth_exp");
            if (reach.has_subgrid_lake) {
                names.push_back(prefix + "subgrid_q_ref");
                names.push_back(prefix + "subgrid_exp");
            }
        }
    }
    return names;
}

inline void Network::collect_gradients() {
    for (auto& [id, reach] : reaches_) {
        if (reach.is_lake) {
            reach.grad_lake_q_ref = get_gradient(reach.lake_q_ref);
            reach.grad_lake_exp = get_gradient(reach.lake_exp);
            reach.grad_lake_q_min = get_gradient(reach.lake_q_min);
            reach.grad_lake_spill_coef = get_gradient(reach.lake_spill_coef);
        } else {
            reach.grad_manning_n = get_gradient(reach.manning_n);
            reach.grad_width_coef = get_gradient(reach.geometry.width_coef);
            reach.grad_width_exp = get_gradient(reach.geometry.width_exp);
            reach.grad_depth_coef = get_gradient(reach.geometry.depth_coef);
            reach.grad_depth_exp = get_gradient(reach.geometry.depth_exp);
            if (reach.has_subgrid_lake) {
                reach.grad_subgrid_q_ref = get_gradient(reach.subgrid_q_ref);
                reach.grad_subgrid_exp = get_gradient(reach.subgrid_exp);
            }
        }
    }
}

inline void Network::zero_gradients() {
    for (auto& [id, reach] : reaches_) {
        reach.grad_manning_n = 0.0;
        reach.grad_width_coef = 0.0;
        reach.grad_width_exp = 0.0;
        reach.grad_depth_coef = 0.0;
        reach.grad_depth_exp = 0.0;
    }
}

inline void Network::build_topology() {
    compute_topological_order();
}

inline void Network::compute_topological_order() {
    topo_order_.clear();
    
    if (reaches_.empty()) return;
    
    // Build adjacency from junctions
    std::unordered_map<int, std::vector<int>> upstream_reaches;
    std::unordered_map<int, int> in_degree;
    
    // Initialize in-degree for all reaches
    for (const auto& [reach_id, reach] : reaches_) {
        in_degree[reach_id] = 0;
    }
    
    // Build dependency graph from junction connectivity
    for (const auto& [junc_id, junc] : junctions_) {
        for (int downstream_reach : junc.downstream_reach_ids) {
            for (int upstream_reach : junc.upstream_reach_ids) {
                upstream_reaches[downstream_reach].push_back(upstream_reach);
                in_degree[downstream_reach]++;
            }
        }
    }
    
    // Handle direct reach-to-reach connections via junction IDs
    for (auto& [reach_id, reach] : reaches_) {
        if (reach.upstream_junction_id >= 0 && junctions_.count(reach.upstream_junction_id)) {
            const Junction& upstream_junc = junctions_.at(reach.upstream_junction_id);
            for (int upstream_reach : upstream_junc.upstream_reach_ids) {
                auto& deps = upstream_reaches[reach_id];
                if (std::find(deps.begin(), deps.end(), upstream_reach) == deps.end()) {
                    deps.push_back(upstream_reach);
                    in_degree[reach_id]++;
                }
            }
        }
    }
    
    // Kahn's algorithm for topological sort
    std::queue<int> queue;
    
    for (const auto& [reach_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(reach_id);
        }
    }
    
    while (!queue.empty()) {
        int reach_id = queue.front();
        queue.pop();
        topo_order_.push_back(reach_id);
        
        for (auto& [downstream_id, upstream_list] : upstream_reaches) {
            auto it = std::find(upstream_list.begin(), upstream_list.end(), reach_id);
            if (it != upstream_list.end()) {
                in_degree[downstream_id]--;
                if (in_degree[downstream_id] == 0) {
                    queue.push(downstream_id);
                }
            }
        }
    }
    
    // Fall back to simple ordering if cycle detected
    if (topo_order_.size() != reaches_.size()) {
        topo_order_.clear();
        for (const auto& [reach_id, reach] : reaches_) {
            topo_order_.push_back(reach_id);
        }
        std::sort(topo_order_.begin(), topo_order_.end());
    }
}

} // namespace dmc

#endif // DMC_ROUTE_NETWORK_HPP
