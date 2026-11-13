#ifndef DMC_ROUTE_NETWORK_HPP
#define DMC_ROUTE_NETWORK_HPP

#include "types.hpp"
#include "transfers.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <queue>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

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

    // ===== Level-pool lake node (lake_model == LAKE_LEVEL_POOL) =====
    // Storage S [m^3] is measured relative to the outlet crest, so the stage
    // above the crest is h = max(S, 0) / lake_area and the outflow over the
    // crest is a weir relation Q = lake_weir_coef * h^lake_weir_exp.
    // Operated withdrawals (transfers with from_storage) may draw the lake
    // down to lake_min_storage (<= 0, i.e. below the crest).
    int  lake_model = 0;            // 0 = storage-fraction rating curve, 1 = level-pool weir
    Real lake_weir_coef = 1.7 * 10.0;  // C_w * crest width [m^1.5/s] (learnable)
    Real lake_weir_exp = 1.5;          // weir exponent (learnable)
    double lake_min_storage = 0.0;     // lowest storage reachable by withdrawals [m^3] (<= 0)
    double grad_lake_weir_coef = 0.0;
    double grad_lake_weir_exp = 0.0;

    // ===== Transfer bookkeeping (per timestep) =====
    // transfer_out: total withdrawn from this reach's outflow/storage [m^3/s];
    // the flow passed to the downstream reach is outflow_curr - transfer_out.
    // transfer_in: total delivered to this reach's inflow by transfer edges.
    Real transfer_in = 0.0;
    Real transfer_out = 0.0;

    Real net_outflow() const { return outflow_curr - transfer_out; }
};

constexpr int LAKE_RATING_CURVE = 0;
constexpr int LAKE_LEVEL_POOL = 1;

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
    
    // Reaches whose outflow is not routed into any other reach (tree outlets:
    // sea or sink). Computed by build_topology().
    const std::vector<int>& terminal_reaches() const { return terminal_reaches_; }

    // ===== Transfers (non-tree edges) =====
    void add_transfer(Transfer transfer);
    std::vector<Transfer>& transfers() { return transfers_; }
    const std::vector<Transfer>& transfers() const { return transfers_; }
    Transfer& get_transfer(const std::string& id);
    void clear_transfers() { transfers_.clear(); }
    TransferOptions& transfer_options() { return transfer_options_; }
    const TransferOptions& transfer_options() const { return transfer_options_; }

    // True when the network uses transfers or level-pool lake nodes, which only
    // the Muskingum-Cunge and IRF routers support.
    bool has_extensions() const;

    // Throw a clear error from routers that do not support the extensions.
    void require_tree_only(const std::string& router_name) const;

    // Check transfer parameters: every fraction in [0, 1], and the passive
    // shares (natural_bifurcation + groundwater) leaving one reach sum to <= 1
    // (e.g. Skafta SKA-1 + SKA-2). Throws std::invalid_argument. Called by
    // build_topology(); at run time withdrawals are additionally capped by the
    // flow left at the source, so an over-allocated source can never go negative.
    void validate_transfers() const;

    bool has_reach(int id) const { return reaches_.count(id) > 0; }

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
    std::vector<int> terminal_reaches_;
    std::vector<Transfer> transfers_;
    TransferOptions transfer_options_;
    
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
        if (reach.is_lake && reach.lake_model == LAKE_LEVEL_POOL) {
            params.push_back(&reach.lake_weir_coef);
            params.push_back(&reach.lake_weir_exp);
        } else if (reach.is_lake) {
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
    for (auto& t : transfers_) {
        params.push_back(&t.fraction);
        if (t.has_capacity) params.push_back(&t.capacity);
        if (t.is_groundwater()) params.push_back(&t.gw_k);
    }
    return params;
}

inline std::vector<std::string> Network::get_parameter_names() {
    std::vector<std::string> names;
    for (auto& [id, reach] : reaches_) {
        std::string prefix = "reach_" + std::to_string(id) + "_";
        if (reach.is_lake && reach.lake_model == LAKE_LEVEL_POOL) {
            names.push_back(prefix + "lake_weir_coef");
            names.push_back(prefix + "lake_weir_exp");
        } else if (reach.is_lake) {
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
    for (const auto& t : transfers_) {
        std::string prefix = "transfer_" + t.id + "_";
        names.push_back(prefix + "fraction");
        if (t.has_capacity) names.push_back(prefix + "capacity");
        if (t.is_groundwater()) names.push_back(prefix + "gw_k");
    }
    return names;
}

inline void Network::collect_gradients() {
    for (auto& [id, reach] : reaches_) {
        if (reach.is_lake && reach.lake_model == LAKE_LEVEL_POOL) {
            reach.grad_lake_weir_coef = get_gradient(reach.lake_weir_coef);
            reach.grad_lake_weir_exp = get_gradient(reach.lake_weir_exp);
        } else if (reach.is_lake) {
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
    for (auto& t : transfers_) {
        t.grad_fraction = get_gradient(t.fraction);
        t.grad_capacity = t.has_capacity ? get_gradient(t.capacity) : 0.0;
        t.grad_gw_k = t.is_groundwater() ? get_gradient(t.gw_k) : 0.0;
    }
}

inline void Network::zero_gradients() {
    for (auto& [id, reach] : reaches_) {
        reach.grad_manning_n = 0.0;
        reach.grad_width_coef = 0.0;
        reach.grad_width_exp = 0.0;
        reach.grad_depth_coef = 0.0;
        reach.grad_depth_exp = 0.0;
        reach.grad_lake_weir_coef = 0.0;
        reach.grad_lake_weir_exp = 0.0;
    }
    for (auto& t : transfers_) {
        t.grad_fraction = 0.0;
        t.grad_capacity = 0.0;
        t.grad_gw_k = 0.0;
    }
}

inline void Network::build_topology() {
    validate_transfers();
    compute_topological_order();
}

inline void Network::validate_transfers() const {
    std::unordered_map<int, double> passive_share;
    std::unordered_map<int, std::string> passive_ids;
    for (const auto& t : transfers_) {
        if (!t.enabled) continue;
        const double f = to_double(t.fraction);
        if (!(f >= 0.0 && f <= 1.0)) {
            throw std::invalid_argument("Transfer '" + t.id + "': fraction " + std::to_string(f) +
                                        " is outside [0, 1]");
        }
        if (t.type == TransferType::NaturalBifurcation || t.type == TransferType::Groundwater) {
            passive_share[t.from_reach] += f;
            passive_ids[t.from_reach] += (passive_ids[t.from_reach].empty() ? "" : " + ") + t.id;
        }
    }
    for (const auto& [reach_id, share] : passive_share) {
        if (share > 1.0 + 1e-12) {
            throw std::invalid_argument("Transfers " + passive_ids[reach_id] + " leaving reach " +
                                        std::to_string(reach_id) + " take fractions summing to " +
                                        std::to_string(share) + " > 1");
        }
    }
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
    
    // Terminal reaches: not routed into any other reach by the tree.
    {
        std::unordered_set<int> has_downstream;
        for (const auto& [down_id, ups] : upstream_reaches) {
            for (int u : ups) has_downstream.insert(u);
        }
        terminal_reaches_.clear();
        for (const auto& [reach_id, reach] : reaches_) {
            if (!has_downstream.count(reach_id)) terminal_reaches_.push_back(reach_id);
        }
        std::sort(terminal_reaches_.begin(), terminal_reaches_.end());
    }

    // Transfer edges: the receiving reach must be processed after the source
    // reach in the same timestep, so they join the dependency graph.
    for (const auto& t : transfers_) {
        if (t.to_reach < 0 || !reaches_.count(t.to_reach) || !reaches_.count(t.from_reach)) continue;
        auto& deps = upstream_reaches[t.to_reach];
        if (std::find(deps.begin(), deps.end(), t.from_reach) == deps.end()) {
            deps.push_back(t.from_reach);
            in_degree[t.to_reach]++;
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
    
    // With transfer edges a cycle is a modelling error: reject it.
    if (topo_order_.size() != reaches_.size() && !transfers_.empty()) {
        // Peel off reaches that are merely downstream of the cycle so the
        // message names the reaches that actually form it.
        std::unordered_set<int> remaining;
        for (const auto& [rid, deg] : in_degree) if (deg > 0) remaining.insert(rid);
        bool changed = true;
        while (changed) {
            changed = false;
            std::unordered_set<int> feeds;
            for (int rid : remaining) {
                auto it = upstream_reaches.find(rid);
                if (it == upstream_reaches.end()) continue;
                for (int u : it->second) if (remaining.count(u)) feeds.insert(u);
            }
            for (auto it = remaining.begin(); it != remaining.end();) {
                if (!feeds.count(*it)) { it = remaining.erase(it); changed = true; }
                else ++it;
            }
        }
        std::vector<int> cyc(remaining.begin(), remaining.end());
        std::sort(cyc.begin(), cyc.end());
        std::ostringstream msg;
        msg << "Network contains a cycle once transfer edges are added (tree + transfers must be acyclic). "
            << cyc.size() << " reaches involved:";
        for (size_t i = 0; i < cyc.size() && i < 20; ++i) msg << " " << cyc[i];
        if (cyc.size() > 20) msg << " ...";
        msg << ". Transfers on the cycle:";
        for (const auto& t : transfers_) {
            if (remaining.count(t.from_reach) && remaining.count(t.to_reach)) msg << " " << t.id;
        }
        topo_order_.clear();
        throw std::runtime_error(msg.str());
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

inline void Network::add_transfer(Transfer transfer) {
    if (!reaches_.count(transfer.from_reach)) {
        throw std::invalid_argument("Transfer '" + transfer.id + "': from_reach " +
                                    std::to_string(transfer.from_reach) + " is not in the network");
    }
    if (transfer.to_reach >= 0 && !reaches_.count(transfer.to_reach)) {
        throw std::invalid_argument("Transfer '" + transfer.id + "': to_reach " +
                                    std::to_string(transfer.to_reach) + " is not in the network");
    }
    if (transfer.to_reach == transfer.from_reach) {
        throw std::invalid_argument("Transfer '" + transfer.id + "' starts and ends at the same reach");
    }
    for (const auto& t : transfers_) {
        if (t.id == transfer.id) {
            throw std::invalid_argument("Duplicate transfer id '" + transfer.id + "'");
        }
    }
    transfers_.push_back(std::move(transfer));
}

inline Transfer& Network::get_transfer(const std::string& id) {
    for (auto& t : transfers_) {
        if (t.id == id) return t;
    }
    throw std::out_of_range("No transfer with id '" + id + "'");
}

inline bool Network::has_extensions() const {
    if (!transfers_.empty()) return true;
    for (const auto& [id, reach] : reaches_) {
        if (reach.is_lake && reach.lake_model == LAKE_LEVEL_POOL) return true;
    }
    return false;
}

inline void Network::require_tree_only(const std::string& router_name) const {
    if (has_extensions()) {
        throw std::runtime_error(router_name +
            " is not supported with transfers or level-pool lake nodes; "
            "use MuskingumCungeRouter or IRFRouter");
    }
}

} // namespace dmc

#endif // DMC_ROUTE_NETWORK_HPP
