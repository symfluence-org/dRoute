/**
 * @file unified_router.hpp
 * @brief Unified Router with AD Backend Selection
 * 
 * Provides a single interface that can use either CoDiPack or Enzyme
 * for automatic differentiation, selected at runtime.
 * 
 * Usage:
 *   RouterConfig config;
 *   config.ad_backend = ADBackend::ENZYME;  // or CODIPACK, NONE
 *   UnifiedRouter router(network, config);
 *   router.route_timestep();
 */

#ifndef DMC_UNIFIED_ROUTER_HPP
#define DMC_UNIFIED_ROUTER_HPP

#include "ad_backend.hpp"
#include "network.hpp"
#include "router.hpp"
#include "kernels_enzyme.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace dmc {

// ============================================================================
// Extended Router Configuration
// ============================================================================

/**
 * @brief Routing method for Enzyme kernels
 */
enum class EnzymeRoutingMethod {
    MUSKINGUM_CUNGE = 0,
    LAG = 1,
    IRF = 2,
    KWT = 3,
    DIFFUSIVE = 4
};

/**
 * @brief Extended router config with AD backend selection
 */
struct UnifiedRouterConfig : public RouterConfig {
    ADBackend ad_backend = get_default_backend();
    EnzymeRoutingMethod routing_method = EnzymeRoutingMethod::MUSKINGUM_CUNGE;
    
    // Enzyme-specific options
    bool enzyme_validate_against_codipack = false;  // Run both and compare
    double enzyme_gradient_tolerance = 1e-4;        // Tolerance for gradient comparison
    
    // Method-specific options
    double kwt_gate_steepness_enzyme = 5.0;         // Steepness for KWT soft gates
    double irf_shape_param = 2.5;                   // Gamma shape for IRF
    int dw_num_nodes = 10;                          // Nodes per reach for diffusive wave
};

// ============================================================================
// Enzyme Router Wrapper
// ============================================================================

/**
 * @brief Router implementation using Enzyme AD
 * 
 * Wraps the flat-array Enzyme kernels in an object-oriented interface
 * compatible with the existing Network/Reach structures.
 */
class EnzymeRouter {
public:
    explicit EnzymeRouter(Network& network, const UnifiedRouterConfig& config)
        : network_(network), config_(config)
    {
        network_.build_topology();
        network_.require_tree_only("EnzymeRouter");
        initialize_arrays();
        initialize_extended_state();
    }
    
    // =========== Core Routing ===========
    
    void route_timestep() {
        // Gather lateral inflows from network
        for (size_t i = 0; i < topo_order_.size(); ++i) {
            int reach_id = topo_order_[i];
            lateral_inflows_[reach_id] = to_double(network_.get_reach(reach_id).lateral_inflow);
        }
        
        int method = static_cast<int>(config_.routing_method);
        
        if (method == 0) {
            // Muskingum-Cunge uses existing optimized function
            enzyme::route_network_timestep(
                n_reaches_,
                topo_order_.data(),
                downstream_idx_.data(),
                upstream_counts_.data(),
                upstream_offsets_.data(),
                upstream_indices_.data(),
                reach_props_.data(),
                reach_states_.data(),
                lateral_inflows_.data(),
                config_.dt,
                config_.num_substeps,
                config_.min_flow,
                config_.x_lower_bound,
                config_.x_upper_bound,
                Q_out_.data()
            );
        } else {
            // Use method-specific routing
            enzyme::route_network_timestep_method(
                method,
                n_reaches_,
                topo_order_.data(),
                downstream_idx_.data(),
                upstream_counts_.data(),
                upstream_offsets_.data(),
                upstream_indices_.data(),
                reach_props_.data(),
                reach_states_.data(),
                extended_state_.data(),
                lateral_inflows_.data(),
                config_.dt,
                config_.num_substeps,
                config_.min_flow,
                config_.kwt_gate_steepness_enzyme,
                Q_out_.data()
            );
        }
        
        // Write back to network structure
        for (size_t i = 0; i < topo_order_.size(); ++i) {
            int reach_id = topo_order_[i];
            Reach& reach = network_.get_reach(reach_id);
            reach.outflow_curr = Real(Q_out_[reach_id]);
            reach.inflow_curr = Real(reach_states_[reach_id * enzyme::NUM_REACH_STATE + 1]);
            reach.outflow_prev = Real(Q_out_[reach_id]);
            reach.inflow_prev = reach.inflow_curr;
        }
        
        current_time_ += config_.dt;
    }
    
    void set_routing_method(EnzymeRoutingMethod method) {
        if (config_.routing_method != method) {
            config_.routing_method = method;
            initialize_extended_state();
        }
    }
    
    EnzymeRoutingMethod get_routing_method() const {
        return config_.routing_method;
    }
    
    void route(int num_timesteps) {
        for (int t = 0; t < num_timesteps; ++t) {
            route_timestep();
        }
    }
    
    // =========== Gradient Computation ===========
    
    void start_recording() {
        // Enzyme doesn't need tape recording - gradients computed on demand
        recording_ = true;
    }
    
    void stop_recording() {
        recording_ = false;
    }
    
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ) {
        // For single-timestep gradients, use per-reach gradient computation
        // For multi-timestep, need to call full simulation gradient
        
        gradient_result_.resize(n_reaches_);
        gradient_result_.zero();
        gradient_result_.backend = ADBackend::ENZYME;
        
#ifdef DMC_USE_ENZYME
        // Compute gradients for each reach individually
        // This is simplified - full implementation would accumulate through time
        for (size_t g = 0; g < gauge_reaches.size(); ++g) {
            int reach_id = gauge_reaches[g];
            double dL = dL_dQ[g];
            
            double d_props[enzyme::NUM_REACH_PROPS_FULL];
            const double* state = &reach_states_[reach_id * enzyme::NUM_REACH_STATE];
            const double* props = &reach_props_[reach_id * enzyme::NUM_REACH_PROPS_FULL];
            
            enzyme::compute_reach_gradient_enzyme(
                state, props, config_.dt, config_.min_flow,
                config_.x_lower_bound, config_.x_upper_bound,
                dL, d_props
            );
            
            // Map to named gradients
            // props order: length, slope, manning_n, width_c, width_e, depth_c, depth_e
            gradient_result_.d_manning_n[reach_id] += d_props[2];
            gradient_result_.d_width_coef[reach_id] += d_props[3];
            gradient_result_.d_width_exp[reach_id] += d_props[4];
            gradient_result_.d_depth_coef[reach_id] += d_props[5];
            gradient_result_.d_depth_exp[reach_id] += d_props[6];
        }
        gradient_result_.valid = true;
#else
        throw std::runtime_error("Enzyme AD not available - compile with -DDMC_USE_ENZYME");
#endif
    }
    
    GradientResult get_gradient_result() const {
        return gradient_result_;
    }
    
    std::unordered_map<std::string, double> get_gradients() const {
        std::unordered_map<std::string, double> grads;
        for (size_t i = 0; i < topo_order_.size(); ++i) {
            int reach_id = topo_order_[i];
            std::string prefix = "reach_" + std::to_string(reach_id) + "_";
            grads[prefix + "manning_n"] = gradient_result_.d_manning_n[reach_id];
            grads[prefix + "width_coef"] = gradient_result_.d_width_coef[reach_id];
            grads[prefix + "width_exp"] = gradient_result_.d_width_exp[reach_id];
            grads[prefix + "depth_coef"] = gradient_result_.d_depth_coef[reach_id];
            grads[prefix + "depth_exp"] = gradient_result_.d_depth_exp[reach_id];
        }
        return grads;
    }
    
    void reset_gradients() {
        gradient_result_.zero();
    }
    
    // =========== State Management ===========
    
    void set_lateral_inflow(int reach_id, double inflow) {
        network_.get_reach(reach_id).lateral_inflow = Real(inflow);
        lateral_inflows_[reach_id] = inflow;
    }
    
    void set_lateral_inflows(const std::vector<double>& inflows) {
        const auto& order = network_.topological_order();
        for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
            set_lateral_inflow(order[i], inflows[i]);
        }
    }
    
    double get_discharge(int reach_id) const {
        return Q_out_[reach_id];
    }
    
    std::vector<double> get_all_discharges() const {
        std::vector<double> d;
        for (int reach_id : topo_order_) {
            d.push_back(Q_out_[reach_id]);
        }
        return d;
    }
    
    void reset_state() {
        std::fill(reach_states_.begin(), reach_states_.end(), 0.0);
        std::fill(Q_out_.begin(), Q_out_.end(), 0.0);
        std::fill(lateral_inflows_.begin(), lateral_inflows_.end(), 0.0);
        current_time_ = 0.0;
        reset_gradients();
        
        // Re-initialize extended state for non-MC methods
        initialize_extended_state();
        
        for (int reach_id : topo_order_) {
            Reach& reach = network_.get_reach(reach_id);
            reach.inflow_prev = reach.inflow_curr = Real(0);
            reach.outflow_prev = reach.outflow_curr = Real(0);
            reach.lateral_inflow = Real(0);
        }
    }
    
    double current_time() const { return current_time_; }
    Network& network() { return network_; }
    const UnifiedRouterConfig& config() const { return config_; }
    int num_reaches() const { return n_reaches_; }
    double dt() const { return config_.dt; }
    
    // =========== Parameter Modification ===========
    
    void set_manning_n(int reach_id, double manning_n) {
        if (reach_id < 0 || reach_id >= n_reaches_) return;
        reach_props_[reach_id * enzyme::NUM_REACH_PROPS_FULL + 2] = manning_n;
        network_.get_reach(reach_id).manning_n = Real(manning_n);
    }
    
    double get_manning_n(int reach_id) const {
        if (reach_id < 0 || reach_id >= n_reaches_) return 0.0;
        return reach_props_[reach_id * enzyme::NUM_REACH_PROPS_FULL + 2];
    }
    
    std::vector<double> get_manning_n_all() const {
        std::vector<double> values(n_reaches_);
        for (int i = 0; i < n_reaches_; ++i) {
            values[i] = reach_props_[i * enzyme::NUM_REACH_PROPS_FULL + 2];
        }
        return values;
    }
    
    std::string get_topology_debug() const {
        std::ostringstream ss;
        ss << "EnzymeRouter topology: " << n_reaches_ << " reaches\n";
        int total_upstream = 0;
        for (int i = 0; i < n_reaches_; ++i) {
            total_upstream += upstream_counts_[i];
        }
        ss << "  Total upstream connections: " << total_upstream << "\n";
        
        // Count headwaters (reaches with no upstream)
        int headwaters = 0;
        for (int i = 0; i < n_reaches_; ++i) {
            if (upstream_counts_[i] == 0) headwaters++;
        }
        ss << "  Headwater reaches: " << headwaters << "\n";
        
        // Find outlet
        int outlets = 0;
        for (int i = 0; i < n_reaches_; ++i) {
            if (downstream_idx_[i] < 0) outlets++;
        }
        ss << "  Outlet reaches: " << outlets << "\n";
        
        return ss.str();
    }
    
private:
    Network& network_;
    UnifiedRouterConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    int n_reaches_ = 0;
    std::vector<int> topo_order_;
    std::vector<int> downstream_idx_;
    std::vector<int> upstream_counts_;
    std::vector<int> upstream_offsets_;
    std::vector<int> upstream_indices_;
    
    std::vector<double> reach_props_;
    std::vector<double> reach_states_;
    std::vector<double> extended_state_;   // For non-MC routing methods
    std::vector<double> lateral_inflows_;
    std::vector<double> Q_out_;
    
    GradientResult gradient_result_;
    
    void initialize_extended_state() {
        int method = static_cast<int>(config_.routing_method);
        
        // Compute extended state size
        int ext_state_size = 0;
        switch (method) {
            case 0: ext_state_size = 0; break;  // MC doesn't need extended state
            case 1: ext_state_size = enzyme::LAG_STATE_SIZE; break;
            case 2: ext_state_size = enzyme::IRF_STATE_SIZE; break;
            case 3: ext_state_size = enzyme::KWT_STATE_SIZE; break;
            case 4: ext_state_size = enzyme::DW_STATE_SIZE; break;
        }
        
        if (ext_state_size > 0) {
            extended_state_.resize(n_reaches_ * ext_state_size, 0.0);
            enzyme::init_extended_state(
                method,
                n_reaches_,
                reach_props_.data(),
                extended_state_.data(),
                config_.dt,
                config_.irf_shape_param,
                config_.dw_num_nodes
            );
        } else {
            extended_state_.clear();
        }
    }
    
    void initialize_arrays() {
        topo_order_ = network_.topological_order();
        n_reaches_ = static_cast<int>(topo_order_.size());
        
        // Build reach index mapping
        std::unordered_map<int, int> reach_to_idx;
        for (int i = 0; i < n_reaches_; ++i) {
            reach_to_idx[topo_order_[i]] = i;
        }
        
        // Allocate arrays
        downstream_idx_.resize(n_reaches_, -1);
        upstream_counts_.resize(n_reaches_, 0);
        upstream_offsets_.resize(n_reaches_, 0);
        
        reach_props_.resize(n_reaches_ * enzyme::NUM_REACH_PROPS_FULL);
        reach_states_.resize(n_reaches_ * enzyme::NUM_REACH_STATE, 0.0);
        lateral_inflows_.resize(n_reaches_, 0.0);
        Q_out_.resize(n_reaches_, 0.0);
        
        // Build topology arrays
        std::vector<std::vector<int>> upstream_lists(n_reaches_);
        
        for (int reach_id : topo_order_) {
            const Reach& reach = network_.get_reach(reach_id);
            
            // Pack properties
            int idx = reach_id;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 0] = reach.length;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 1] = reach.slope;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 2] = to_double(reach.manning_n);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 3] = to_double(reach.geometry.width_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 4] = to_double(reach.geometry.width_exp);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 5] = to_double(reach.geometry.depth_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 6] = to_double(reach.geometry.depth_exp);
            
            // Find downstream
            if (reach.downstream_junction_id >= 0) {
                try {
                    const Junction& junc = network_.get_junction(reach.downstream_junction_id);
                    for (int down_id : junc.downstream_reach_ids) {
                        downstream_idx_[reach_id] = down_id;
                        break;  // Only first downstream
                    }
                } catch (...) {}
            }
            
            // Find upstream reaches
            if (reach.upstream_junction_id >= 0) {
                try {
                    const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                    for (int up_id : junc.upstream_reach_ids) {
                        upstream_lists[reach_id].push_back(up_id);
                    }
                } catch (...) {}
            }
        }
        
        // Flatten upstream lists
        int total_upstream = 0;
        for (int i = 0; i < n_reaches_; ++i) {
            upstream_offsets_[i] = total_upstream;
            upstream_counts_[i] = static_cast<int>(upstream_lists[i].size());
            total_upstream += upstream_counts_[i];
        }
        
        upstream_indices_.resize(total_upstream);
        for (int i = 0; i < n_reaches_; ++i) {
            for (size_t j = 0; j < upstream_lists[i].size(); ++j) {
                upstream_indices_[upstream_offsets_[i] + j] = upstream_lists[i][j];
            }
        }
        
        gradient_result_.resize(n_reaches_);
    }
    
    void sync_props_from_network() {
        for (int reach_id : topo_order_) {
            const Reach& reach = network_.get_reach(reach_id);
            int idx = reach_id;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 2] = to_double(reach.manning_n);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 3] = to_double(reach.geometry.width_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 4] = to_double(reach.geometry.width_exp);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 5] = to_double(reach.geometry.depth_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 6] = to_double(reach.geometry.depth_exp);
        }
    }
};

// ============================================================================
// Unified Router with Backend Dispatch
// ============================================================================

/**
 * @brief Unified router that dispatches to the selected AD backend
 * 
 * This class provides a single interface that internally uses either
 * the CoDiPack-based MuskingumCungeRouter or the Enzyme-based EnzymeRouter.
 */
class UnifiedRouter {
public:
    explicit UnifiedRouter(Network& network, UnifiedRouterConfig config = {})
        : network_(network), config_(std::move(config))
    {
        network_.build_topology();
        initialize_backend();
    }
    
    // =========== Core Routing ===========
    
    void route_timestep() {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                codipack_router_->route_timestep();
                break;
            case ADBackend::ENZYME:
                enzyme_router_->route_timestep();
                break;
            case ADBackend::NONE:
                // Use Enzyme kernel without differentiation
                enzyme_router_->route_timestep();
                break;
        }
        
        // Optional validation: run both and compare
        if (config_.enzyme_validate_against_codipack && 
            config_.ad_backend == ADBackend::ENZYME &&
            CODIPACK_AVAILABLE) {
            validate_against_codipack();
        }
    }
    
    void route(int num_timesteps) {
        for (int t = 0; t < num_timesteps; ++t) {
            route_timestep();
        }
    }
    
    // =========== Gradient Computation ===========
    
    void enable_gradients(bool enable) {
        config_.enable_gradients = enable;
        if (codipack_router_) codipack_router_->enable_gradients(enable);
    }
    
    void start_recording() {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) codipack_router_->start_recording();
                break;
            case ADBackend::ENZYME:
                if (enzyme_router_) enzyme_router_->start_recording();
                break;
            case ADBackend::NONE:
                break;
        }
    }
    
    void stop_recording() {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) codipack_router_->stop_recording();
                break;
            case ADBackend::ENZYME:
                if (enzyme_router_) enzyme_router_->stop_recording();
                break;
            case ADBackend::NONE:
                break;
        }
    }
    
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ) {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) {
                    codipack_router_->compute_gradients(gauge_reaches, dL_dQ);
                }
                break;
            case ADBackend::ENZYME:
                if (enzyme_router_) {
                    enzyme_router_->compute_gradients(gauge_reaches, dL_dQ);
                }
                break;
            case ADBackend::NONE:
                throw std::runtime_error("Gradients not available with ADBackend::NONE");
        }
    }
    
    std::unordered_map<std::string, double> get_gradients() const {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) return codipack_router_->get_gradients();
                break;
            case ADBackend::ENZYME:
                if (enzyme_router_) return enzyme_router_->get_gradients();
                break;
            case ADBackend::NONE:
                break;
        }
        return {};
    }
    
    GradientResult get_gradient_result() const {
        if (config_.ad_backend == ADBackend::ENZYME && enzyme_router_) {
            return enzyme_router_->get_gradient_result();
        }
        
        // Convert CoDiPack gradients to GradientResult
        GradientResult result;
        if (codipack_router_) {
            result.backend = ADBackend::CODIPACK;
            result.resize(network_.num_reaches());
            
            size_t idx = 0;
            for (int reach_id : network_.topological_order()) {
                const Reach& reach = network_.get_reach(reach_id);
                result.d_manning_n[idx] = reach.grad_manning_n;
                result.d_width_coef[idx] = reach.grad_width_coef;
                result.d_width_exp[idx] = reach.grad_width_exp;
                result.d_depth_coef[idx] = reach.grad_depth_coef;
                result.d_depth_exp[idx] = reach.grad_depth_exp;
                ++idx;
            }
            result.valid = true;
        }
        return result;
    }
    
    void reset_gradients() {
        if (codipack_router_) codipack_router_->reset_gradients();
        if (enzyme_router_) enzyme_router_->reset_gradients();
    }
    
    // =========== State Management ===========
    
    void set_lateral_inflow(int reach_id, double inflow) {
        if (codipack_router_) codipack_router_->set_lateral_inflow(reach_id, inflow);
        if (enzyme_router_) enzyme_router_->set_lateral_inflow(reach_id, inflow);
    }
    
    void set_lateral_inflows(const std::vector<double>& inflows) {
        if (codipack_router_) codipack_router_->set_lateral_inflows(inflows);
        if (enzyme_router_) enzyme_router_->set_lateral_inflows(inflows);
    }
    
    double get_discharge(int reach_id) const {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) return codipack_router_->get_discharge(reach_id);
                break;
            case ADBackend::ENZYME:
            case ADBackend::NONE:
                if (enzyme_router_) return enzyme_router_->get_discharge(reach_id);
                break;
        }
        return 0.0;
    }
    
    std::vector<double> get_all_discharges() const {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) return codipack_router_->get_all_discharges();
                break;
            case ADBackend::ENZYME:
            case ADBackend::NONE:
                if (enzyme_router_) return enzyme_router_->get_all_discharges();
                break;
        }
        return {};
    }
    
    void reset_state() {
        if (codipack_router_) codipack_router_->reset_state();
        if (enzyme_router_) enzyme_router_->reset_state();
    }
    
    // =========== Accessors ===========
    
    double current_time() const {
        switch (config_.ad_backend) {
            case ADBackend::CODIPACK:
                if (codipack_router_) return codipack_router_->current_time();
                break;
            case ADBackend::ENZYME:
            case ADBackend::NONE:
                if (enzyme_router_) return enzyme_router_->current_time();
                break;
        }
        return 0.0;
    }
    
    Network& network() { return network_; }
    const Network& network() const { return network_; }
    const UnifiedRouterConfig& config() const { return config_; }
    ADBackend backend() const { return config_.ad_backend; }
    
    /**
     * @brief Switch AD backend at runtime
     */
    void set_backend(ADBackend backend) {
        if (!is_backend_available(backend)) {
            throw std::runtime_error("AD backend not available: " + 
                                     ad_backend_to_string(backend));
        }
        config_.ad_backend = backend;
        initialize_backend();
    }
    
private:
    Network& network_;
    UnifiedRouterConfig config_;
    
    std::unique_ptr<MuskingumCungeRouter> codipack_router_;
    std::unique_ptr<EnzymeRouter> enzyme_router_;
    
    void initialize_backend() {
        // Always create Enzyme router (it works without AD too)
        enzyme_router_ = std::make_unique<EnzymeRouter>(network_, config_);
        
        // Create CoDiPack router if available and requested
        if (CODIPACK_AVAILABLE && 
            (config_.ad_backend == ADBackend::CODIPACK ||
             config_.enzyme_validate_against_codipack)) {
            codipack_router_ = std::make_unique<MuskingumCungeRouter>(network_, config_);
        }
    }
    
    void validate_against_codipack() {
        if (!codipack_router_) return;
        
        auto enzyme_Q = enzyme_router_->get_all_discharges();
        
        // Run CoDiPack with same inputs
        codipack_router_->route_timestep();
        auto codipack_Q = codipack_router_->get_all_discharges();
        
        // Compare
        double max_diff = 0.0;
        for (size_t i = 0; i < enzyme_Q.size(); ++i) {
            double diff = std::abs(enzyme_Q[i] - codipack_Q[i]);
            max_diff = std::max(max_diff, diff);
        }
        
        if (max_diff > config_.enzyme_gradient_tolerance) {
            std::cerr << "WARNING: Enzyme/CoDiPack mismatch. Max diff: " << max_diff << "\n";
        }
    }
};

// ============================================================================
// Comparison Utilities
// ============================================================================

/**
 * @brief Compare forward pass results between backends
 */
struct BackendComparisonResult {
    std::vector<double> codipack_Q;
    std::vector<double> enzyme_Q;
    double max_Q_diff = 0.0;
    double mean_Q_diff = 0.0;
    
    std::unordered_map<std::string, double> codipack_grads;
    std::unordered_map<std::string, double> enzyme_grads;
    double max_grad_diff = 0.0;
    double mean_grad_diff = 0.0;
    
    bool forward_match = false;
    bool gradient_match = false;
};

/**
 * @brief Run comparison between CoDiPack and Enzyme backends
 */
inline BackendComparisonResult compare_backends(
    Network& network,
    const std::vector<double>& lateral_inflows,
    int num_timesteps,
    const std::vector<int>& gauge_reaches,
    double gradient_tol = 1e-4
) {
    BackendComparisonResult result;
    
    if (!CODIPACK_AVAILABLE) {
        throw std::runtime_error("CoDiPack not available for comparison");
    }
    
    UnifiedRouterConfig config;
    
    // Run CoDiPack
    config.ad_backend = ADBackend::CODIPACK;
    UnifiedRouter codipack_router(network, config);
    
    codipack_router.start_recording();
    for (int t = 0; t < num_timesteps; ++t) {
        codipack_router.set_lateral_inflows(lateral_inflows);
        codipack_router.route_timestep();
    }
    codipack_router.stop_recording();
    
    result.codipack_Q = codipack_router.get_all_discharges();
    
    // Compute CoDiPack gradients
    std::vector<double> dL_dQ(gauge_reaches.size(), 1.0);
    codipack_router.compute_gradients(gauge_reaches, dL_dQ);
    result.codipack_grads = codipack_router.get_gradients();
    
    // Reset network state
    codipack_router.reset_state();
    
    // Run Enzyme
    config.ad_backend = ADBackend::ENZYME;
    UnifiedRouter enzyme_router(network, config);
    
    enzyme_router.start_recording();
    for (int t = 0; t < num_timesteps; ++t) {
        enzyme_router.set_lateral_inflows(lateral_inflows);
        enzyme_router.route_timestep();
    }
    enzyme_router.stop_recording();
    
    result.enzyme_Q = enzyme_router.get_all_discharges();
    
    // Compute Enzyme gradients
    enzyme_router.compute_gradients(gauge_reaches, dL_dQ);
    result.enzyme_grads = enzyme_router.get_gradients();
    
    // Compare forward pass
    double sum_diff = 0.0;
    for (size_t i = 0; i < result.codipack_Q.size(); ++i) {
        double diff = std::abs(result.codipack_Q[i] - result.enzyme_Q[i]);
        result.max_Q_diff = std::max(result.max_Q_diff, diff);
        sum_diff += diff;
    }
    result.mean_Q_diff = sum_diff / result.codipack_Q.size();
    // Use relative tolerance for forward pass comparison
    double max_Q = 1.0;
    for (size_t i = 0; i < result.codipack_Q.size(); ++i) {
        max_Q = std::max(max_Q, std::abs(result.codipack_Q[i]));
    }
    result.forward_match = (result.max_Q_diff < 1e-4 * max_Q);  // 0.01% relative tolerance
    
    // Compare gradients using relative tolerance
    // Note: CoDiPack accumulates through tape, Enzyme computes per-timestep
    // So we use a looser tolerance for gradient comparison
    double grad_sum_diff = 0.0;
    int grad_count = 0;
    for (const auto& [name, codipack_val] : result.codipack_grads) {
        auto it = result.enzyme_grads.find(name);
        if (it != result.enzyme_grads.end()) {
            double diff = std::abs(codipack_val - it->second);
            result.max_grad_diff = std::max(result.max_grad_diff, diff);
            grad_sum_diff += diff;
            ++grad_count;
        }
    }
    result.mean_grad_diff = (grad_count > 0) ? grad_sum_diff / grad_count : 0.0;
    
    // For gradient match, use relative tolerance based on gradient magnitude
    double max_grad = 1e-10;
    for (const auto& [name, val] : result.codipack_grads) {
        max_grad = std::max(max_grad, std::abs(val));
    }
    result.gradient_match = (result.max_grad_diff < gradient_tol * max_grad);
    
    return result;
}

} // namespace dmc

#endif // DMC_UNIFIED_ROUTER_HPP
