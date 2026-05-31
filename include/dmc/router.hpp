#ifndef DMC_ROUTE_ROUTER_HPP
#define DMC_ROUTE_ROUTER_HPP

#include "types.hpp"
#include "network.hpp"
#include <vector>
#include <deque>
#include <functional>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iostream>

namespace dmc {

/**
 * Configuration for the router.
 */
struct RouterConfig {
    double dt = 3600.0;                    // Timestep [s]
    bool enable_gradients = true;          // Record tape for AD
    double min_flow = 1e-6;                // Minimum flow [m³/s]
    double x_lower_bound = 0.0;            // Muskingum X bounds
    double x_upper_bound = 0.5;
    
    // === AD Safety Options ===
    bool use_smooth_bounds = true;         // Use smooth clamp/max/min for AD
    double smooth_epsilon = 1e-6;          // Epsilon for smooth functions
    
    // === Sub-stepping (AD-safe) ===
    bool fixed_substepping = true;         // Use fixed substep count (AD-safe)
    int num_substeps = 4;                  // Fixed number of substeps per reach
    bool adaptive_substepping = false;     // Dynamic substeps (breaks AD)
    int max_substeps = 10;
    
    // === IRF Options ===
    int irf_max_kernel_size = 500;          // Maximum kernel length [timesteps] - large for safety
    double irf_shape_param = 2.5;           // Gamma shape parameter (k)
    bool irf_soft_mask = true;              // Use sigmoid masking for differentiable kernel length
    double irf_mask_steepness = 10.0;       // Sigmoid steepness for soft mask
    
    // === Diffusive Wave Options ===
    int dw_num_nodes = 10;                  // Spatial nodes per reach - fewer = larger dx = more stable
    bool dw_use_ift_adjoint = true;        // Use Implicit Function Theorem for adjoints
    int dw_max_substeps = 500;             // Maximum sub-steps per timestep (stability requires many)
    
    // === Soft-Gated KWT Options ===
    double kwt_gate_steepness = 5.0;       // Soft gate sharpness (lower = smoother gradients)
    bool kwt_anneal_steepness = false;     // Anneal steepness during training
    double kwt_steepness_min = 1.0;        // Starting steepness (blurry)
    double kwt_steepness_max = 20.0;       // Ending steepness (sharp)
    
    // === Memory Management ===
    bool enable_checkpointing = false;     // Checkpoint for long simulations
    int checkpoint_interval = 1000;        // Timesteps between checkpoints
    
    // === Parallelization ===
    bool parallel_routing = false;         // Enable OpenMP parallel routing
    int num_threads = 4;                   // Thread count for parallel mode
    
    /**
     * Load RouterConfig from a mizuRoute-style control file.
     * Supports tags like <dt_routing>, <irf_shape_param>, etc.
     */
    static RouterConfig load_from_control_file(const std::string& filepath) {
        RouterConfig cfg;
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open config file: " << filepath << ", using defaults\n";
            return cfg;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '!') continue;
            
            // Parse <tag> value format
            size_t tag_start = line.find('<');
            size_t tag_end = line.find('>');
            if (tag_start == std::string::npos || tag_end == std::string::npos) continue;
            
            std::string tag = line.substr(tag_start + 1, tag_end - tag_start - 1);
            
            // Extract value (everything after '>' until '!' or end)
            std::string rest = line.substr(tag_end + 1);
            size_t comment_pos = rest.find('!');
            if (comment_pos != std::string::npos) {
                rest = rest.substr(0, comment_pos);
            }
            
            // Trim whitespace
            size_t start = rest.find_first_not_of(" \t");
            size_t end_pos = rest.find_last_not_of(" \t");
            if (start == std::string::npos) continue;
            std::string value = rest.substr(start, end_pos - start + 1);
            
            // Parse boolean helper
            auto parse_bool = [](const std::string& v) {
                return v == "true" || v == "yes" || v == "1" || v == "True" || v == "TRUE";
            };
            
            // AD backend selection
            if (tag == "ad_backend") {
                // Will be parsed by UnifiedRouterConfig if needed
                // Store for later use
            }
            
            // Core options
            if (tag == "dt_routing" || tag == "dt") cfg.dt = std::stod(value);
            else if (tag == "enable_gradients") cfg.enable_gradients = parse_bool(value);
            else if (tag == "min_flow") cfg.min_flow = std::stod(value);
            
            // Muskingum bounds
            else if (tag == "x_lower_bound") cfg.x_lower_bound = std::stod(value);
            else if (tag == "x_upper_bound") cfg.x_upper_bound = std::stod(value);
            
            // AD safety
            else if (tag == "use_smooth_bounds") cfg.use_smooth_bounds = parse_bool(value);
            else if (tag == "smooth_epsilon") cfg.smooth_epsilon = std::stod(value);
            
            // Sub-stepping
            else if (tag == "fixed_substepping") cfg.fixed_substepping = parse_bool(value);
            else if (tag == "num_substeps" || tag == "mc_num_substeps") cfg.num_substeps = std::stoi(value);
            else if (tag == "adaptive_substepping") cfg.adaptive_substepping = parse_bool(value);
            else if (tag == "max_substeps") cfg.max_substeps = std::stoi(value);
            
            // IRF options
            else if (tag == "irf_max_kernel_size") cfg.irf_max_kernel_size = std::stoi(value);
            else if (tag == "irf_shape_param") cfg.irf_shape_param = std::stod(value);
            else if (tag == "irf_soft_mask") cfg.irf_soft_mask = parse_bool(value);
            else if (tag == "irf_mask_steepness") cfg.irf_mask_steepness = std::stod(value);
            
            // Diffusive Wave options
            else if (tag == "dw_num_nodes") cfg.dw_num_nodes = std::stoi(value);
            else if (tag == "dw_use_ift_adjoint") cfg.dw_use_ift_adjoint = parse_bool(value);
            else if (tag == "dw_max_substeps") cfg.dw_max_substeps = std::stoi(value);
            
            // KWT options
            else if (tag == "kwt_gate_steepness") cfg.kwt_gate_steepness = std::stod(value);
            else if (tag == "kwt_anneal_steepness") cfg.kwt_anneal_steepness = parse_bool(value);
            else if (tag == "kwt_steepness_min") cfg.kwt_steepness_min = std::stod(value);
            else if (tag == "kwt_steepness_max") cfg.kwt_steepness_max = std::stod(value);
            
            // Memory management
            else if (tag == "enable_checkpointing") cfg.enable_checkpointing = parse_bool(value);
            else if (tag == "checkpoint_interval") cfg.checkpoint_interval = std::stoi(value);
            
            // Parallelization
            else if (tag == "parallel_routing") cfg.parallel_routing = parse_bool(value);
            else if (tag == "num_threads") cfg.num_threads = std::stoi(value);
        }
        
        return cfg;
    }
};

/**
 * Method capabilities descriptor
 */
struct MethodCapabilities {
    bool supports_gradients;               // Can compute gradients
    bool full_ad;                          // Uses full AD (not analytical approx)
    std::string gradient_params;           // Which parameters have gradients
    std::string numerical_scheme;          // Numerical method description
    std::string recommended_use;           // When to use this method
    
    static MethodCapabilities muskingum_cunge() {
        return {true, true, "manning_n, width_coef, width_exp, depth_coef, depth_exp",
                "Explicit Muskingum-Cunge with sub-stepping",
                "Production calibration, general routing"};
    }
    
    static MethodCapabilities irf() {
        return {true, true, "manning_n (via travel time)",
                "Gamma unit hydrograph convolution with soft masking",
                "Fast calibration when detailed wave physics not needed"};
    }
    
    static MethodCapabilities diffusive_wave() {
        return {true, false, "manning_n (analytical approximation)",
                "Explicit finite difference (upwind advection, central diffusion)",
                "High-accuracy physics, flood wave attenuation"};
    }
    
    static MethodCapabilities diffusive_wave_ift() {
        return {true, true, "manning_n (via IFT)",
                "Crank-Nicolson implicit with IFT adjoint",
                "When exact gradients through implicit solver needed"};
    }
    
    static MethodCapabilities lag() {
        return {false, false, "none (forward-only)",
                "FIFO buffer with integer lag",
                "Simple delay, baseline comparison, NOT for calibration"};
    }
    
    static MethodCapabilities kwt() {
        return {false, false, "none (forward-only)",
                "Lagrangian parcel tracking with continuous wave segments",
                "mizuRoute compatibility, diagnostic comparison"};
    }
    
    static MethodCapabilities kwt_soft() {
        return {true, true, "manning_n (via soft gate)",
                "Parcel tracking with sigmoid exit probability",
                "Differentiable Lagrangian routing, experimental"};
    }
};

/**
 * State snapshot for serialization and checkpointing
 */
struct RouterState {
    double time;
    std::unordered_map<int, double> inflows;
    std::unordered_map<int, double> outflows;
    std::unordered_map<int, std::vector<double>> buffers;  // For IRF/Lag history
    
    // Serialize to binary
    std::vector<char> serialize() const;
    
    // Deserialize from binary
    static RouterState deserialize(const std::vector<char>& data);
    
    // Save to file
    bool save(const std::string& filepath) const;
    
    // Load from file
    static RouterState load(const std::string& filepath);
};

/**
 * Core Muskingum-Cunge routing engine.
 * 
 * Supports automatic differentiation for parameter learning.
 */
class MuskingumCungeRouter {
public:
    explicit MuskingumCungeRouter(Network& network, RouterConfig config = {});
    
    // =========== Core Routing ===========
    
    /**
     * Route a single reach for one timestep.
     * Returns the outflow Q(t+Δt).
     */
    Real route_reach(Reach& reach, double dt);
    
    /**
     * Route entire network for one timestep.
     * Processes reaches in topological order.
     */
    void route_timestep();
    
    /**
     * Route for multiple timesteps.
     */
    void route(int num_timesteps);
    
    // =========== Gradient Computation ===========
    
    /**
     * Enable/disable gradient recording.
     */
    void enable_gradients(bool enable);
    
    /**
     * Start recording operations to tape.
     * Call before simulation loop.
     */
    void start_recording();
    
    /**
     * Stop recording and prepare for backward pass.
     * Call after simulation loop.
     */
    void stop_recording();
    
    /**
     * Record current discharge at a reach for gradient computation.
     * Call this after each route_timestep() for reaches where you have observations.
     * The recorded values are stored on the tape for backpropagation.
     */
    void record_output(int reach_id);
    
    /**
     * Record current discharge at multiple reaches.
     */
    void record_outputs(const std::vector<int>& reach_ids);
    
    /**
     * Clear recorded output history.
     */
    void clear_output_history();
    
    /**
     * Get number of recorded timesteps for a reach.
     */
    size_t get_output_history_size(int reach_id) const;
    
    /**
     * Get recorded output values (as doubles) for a reach.
     */
    std::vector<double> get_output_history(int reach_id) const;
    
    /**
     * Compute gradients via reverse AD for single point.
     * @param gauge_reaches Reach IDs where we have observations
     * @param dL_dQ Gradient of loss w.r.t. discharge at each gauge
     */
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    
    /**
     * Compute gradients via reverse AD for full timeseries.
     * This properly accumulates gradients across all timesteps.
     * 
     * @param reach_id Reach ID where we have observations
     * @param dL_dQ Gradient of loss w.r.t. discharge at each timestep
     *              Must have same length as number of recorded outputs
     * 
     * For MSE loss: dL_dQ[t] = 2 * (sim[t] - obs[t]) / n_timesteps
     */
    void compute_gradients_timeseries(int reach_id,
                                      const std::vector<double>& dL_dQ);
    
    /**
     * Compute gradients via reverse AD for full timeseries at multiple reaches.
     * 
     * @param reach_ids Reach IDs where we have observations
     * @param dL_dQ Gradients for each reach, each with length == recorded timesteps
     */
    void compute_gradients_timeseries(const std::vector<int>& reach_ids,
                                      const std::vector<std::vector<double>>& dL_dQ);
    
    /**
     * Get accumulated gradients for all parameters.
     */
    std::unordered_map<std::string, double> get_gradients() const;
    
    /**
     * Reset tape and gradients.
     */
    void reset_gradients();
    
    // =========== State Management ===========
    
    /**
     * Set lateral inflow for a reach (from rainfall-runoff model).
     */
    void set_lateral_inflow(int reach_id, double inflow);
    
    /**
     * Set lateral inflow for all reaches (same order as topological order).
     */
    void set_lateral_inflows(const std::vector<double>& inflows);
    
    /**
     * Get current discharge at a reach.
     */
    double get_discharge(int reach_id) const;
    
    /**
     * Get discharge at all reaches (same order as topological order).
     */
    std::vector<double> get_all_discharges() const;
    
    /**
     * Reset state to initial conditions.
     */
    void reset_state();
    
    /**
     * Save current router state for checkpointing.
     */
    RouterState save_state() const;
    
    /**
     * Load router state from checkpoint.
     */
    void load_state(const RouterState& state);
    
    /**
     * Save state to file.
     */
    bool save_state_to_file(const std::string& filepath) const;
    
    /**
     * Load state from file.
     */
    bool load_state_from_file(const std::string& filepath);
    
    // =========== Time Management ===========
    
    double current_time() const { return current_time_; }
    void set_time(double t) { current_time_ = t; }
    
    // =========== Access ===========
    
    Network& network() { return network_; }
    const Network& network() const { return network_; }
    const RouterConfig& config() const { return config_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    // Gauge output storage for gradient computation
    std::vector<int> gauge_reach_ids_;
    std::vector<Real> gauge_outputs_;
    
    // Timeseries output storage: reach_id -> vector of outputs per timestep
    // These Real values are recorded on the tape for backpropagation
    std::unordered_map<int, std::vector<Real>> output_history_;
    
    /**
     * Compute inflow to a reach from upstream junction.
     */
    Real compute_reach_inflow(const Reach& reach);
    
    /**
     * Compute Muskingum K parameter.
     */
    Real compute_K(const Reach& reach, const Real& Q_ref);
    
    /**
     * Compute Muskingum X parameter.
     */
    Real compute_X(const Reach& reach, const Real& Q_ref, const Real& K);
    
    /**
     * Compute routing coefficients C1, C2, C3, C4.
     */
    void compute_routing_coefficients(const Real& K, const Real& X, double dt,
                                       Real& C1, Real& C2, Real& C3, Real& C4);
};

// ==================== Implementation ====================

inline MuskingumCungeRouter::MuskingumCungeRouter(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
}

inline Real MuskingumCungeRouter::compute_K(const Reach& reach, const Real& Q_ref) {
    // Hydraulic radius
    Real R_h = reach.geometry.hydraulic_radius(Q_ref);
    
    // Velocity from Manning's equation: v = (1/n) * R^(2/3) * S^(1/2)
    Real velocity = (1.0 / reach.manning_n) * 
                    safe_pow(R_h, 2.0/3.0) * 
                    safe_sqrt(Real(reach.slope));
    
    // Wave celerity: c = (5/3) * v for wide rectangular channel
    Real celerity = (5.0 / 3.0) * velocity;
    celerity = safe_max(celerity, Real(0.1));  // Prevent division by zero
    
    // K = Δx / c
    Real K = Real(reach.length) / celerity;
    
    // Cap K to prevent extremely slow routing
    // K should be at most ~10 timesteps worth for reasonable mass throughput
    Real K_max = Real(10.0 * config_.dt);
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        K = smooth_min(K, K_max, config_.smooth_epsilon);
    } else {
        K = safe_min(K, K_max);
    }
    
    return K;
}

inline Real MuskingumCungeRouter::compute_X(const Reach& reach, const Real& Q_ref, 
                                             const Real& K) {
    Real width = reach.geometry.width(Q_ref);
    Real celerity = Real(reach.length) / K;
    
    // X = 0.5 - Q / (2 * c * B * S₀ * Δx)
    Real X = Real(0.5) - Q_ref / (Real(2.0) * celerity * width * 
                            Real(reach.slope) * Real(reach.length));
    
    // Clamp to valid range [0, 0.5]
    // Use smooth clamp when gradients are enabled to avoid discontinuous derivatives
    if (config_.enable_gradients) {
        return smooth_clamp(X, Real(config_.x_lower_bound), Real(config_.x_upper_bound));
    } else {
        return clamp(X, Real(config_.x_lower_bound), Real(config_.x_upper_bound));
    }
}

inline void MuskingumCungeRouter::compute_routing_coefficients(
    const Real& K, const Real& X, double dt,
    Real& C1, Real& C2, Real& C3, Real& C4) {
    
    Real denom = 2.0 * K * (1.0 - X) + dt;
    
    C1 = (dt - 2.0 * K * X) / denom;
    C2 = (dt + 2.0 * K * X) / denom;
    C3 = (2.0 * K * (1.0 - X) - dt) / denom;
    C4 = 2.0 * dt / denom;
}

inline Real MuskingumCungeRouter::compute_reach_inflow(const Reach& reach) {
    if (reach.upstream_junction_id < 0) {
        // Headwater - no upstream inflow
        return Real(0.0);
    }
    
    const Junction& junc = network_.get_junction(reach.upstream_junction_id);
    Real total_inflow = junc.external_inflow;
    
    for (int up_reach_id : junc.upstream_reach_ids) {
        const Reach& up_reach = network_.get_reach(up_reach_id);
        total_inflow = total_inflow + up_reach.outflow_curr;
    }
    
    return total_inflow;
}

inline Real MuskingumCungeRouter::route_reach(Reach& reach, double dt) {
    // Reference discharge for hydraulic calculations
    // Use max of previous outflow, current inflow, AND lateral inflow for better stability
    // This prevents extremely slow routing when starting from zero flow
    Real Q_ref = safe_max(reach.outflow_prev, Real(config_.min_flow));
    Q_ref = safe_max(Q_ref, reach.inflow_curr);
    Q_ref = safe_max(Q_ref, reach.lateral_inflow);
    
    // Compute Muskingum parameters
    Real K = compute_K(reach, Q_ref);
    Real X = compute_X(reach, Q_ref, K);
    
    // Store for diagnostics
    reach.K = K;
    reach.X = X;
    reach.celerity = Real(reach.length) / K;
    reach.velocity = reach.celerity * (3.0 / 5.0);
    
    // Fixed sub-stepping for AD-safe stability
    // This ensures mass conservation on short reaches without control-flow branching
    if (config_.fixed_substepping && config_.num_substeps > 1) {
        double sub_dt = dt / config_.num_substeps;
        
        // Sub-step state
        Real Q_in_prev = reach.inflow_prev;
        Real Q_in_curr = reach.inflow_curr;
        Real Q_out_prev = reach.outflow_prev;
        Real lateral = reach.lateral_inflow;
        
        // Linear interpolation increments for inflow
        Real dQ_in = (Q_in_curr - Q_in_prev) / Real(config_.num_substeps);
        
        Real Q_out = Q_out_prev;
        for (int s = 0; s < config_.num_substeps; ++s) {
            // Interpolated inflows for this substep
            Real Q_in_s_prev = Q_in_prev + dQ_in * Real(s);
            Real Q_in_s_curr = Q_in_prev + dQ_in * Real(s + 1);
            
            // Compute routing coefficients for sub-timestep
            // CRITICAL: All coefficients must use the same dt (sub_dt) for mass conservation!
            // At steady state: Q_out = C4/(1-C3) * lateral
            // This equals lateral only when C1+C2+C3=1 and C4/(1-C3)=1,
            // which requires all coefficients use consistent dt.
            Real C1, C2, C3, C4;
            compute_routing_coefficients(K, X, sub_dt, C1, C2, C3, C4);
            
            // Apply Muskingum-Cunge equation with consistent coefficients
            Real Q_out_new = C1 * Q_in_s_curr + 
                             C2 * Q_in_s_prev + 
                             C3 * Q_out +
                             C4 * lateral;
            
            Q_out = safe_max(Q_out_new, Real(config_.min_flow));
        }
        
        return Q_out;
    }
    
    // Standard single-step routing (no sub-stepping)
    Real C1, C2, C3, C4;
    compute_routing_coefficients(K, X, dt, C1, C2, C3, C4);
    
    // Apply Muskingum-Cunge equation
    Real outflow = C1 * reach.inflow_curr + 
                   C2 * reach.inflow_prev + 
                   C3 * reach.outflow_prev +
                   C4 * reach.lateral_inflow;
    
    // Ensure non-negative
    outflow = safe_max(outflow, Real(config_.min_flow));
    
    return outflow;
}

inline void MuskingumCungeRouter::route_timestep() {
    // Process reaches in topological order
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        
        // Update inflow from upstream
        reach.inflow_curr = compute_reach_inflow(reach);
        
        // Route through reach
        reach.outflow_curr = route_reach(reach, config_.dt);
    }
    
    // Advance state: current becomes previous
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = reach.inflow_curr;
        reach.outflow_prev = reach.outflow_curr;
    }
    
    current_time_ += config_.dt;
}

inline void MuskingumCungeRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) {
        route_timestep();
    }
}

inline void MuskingumCungeRouter::enable_gradients(bool enable) {
    config_.enable_gradients = enable;
}

inline void MuskingumCungeRouter::start_recording() {
    if (!config_.enable_gradients || !AD_ENABLED) return;
    
    reset_tape();  // clear the global CoDiPack tape so successive recording sessions
                   // (e.g. gradient-descent epochs) don't accumulate and exhaust memory
    activate_tape();
    recording_ = true;
    
    // Register all parameters as inputs
    for (Real* param : network_.get_all_parameters()) {
        register_input(*param);
    }
}

inline void MuskingumCungeRouter::stop_recording() {
    if (!recording_) return;
    
    // Don't deactivate tape yet - we need it active for gradient computation
    // Just mark that the forward pass is done
    recording_ = false;
}

inline void MuskingumCungeRouter::compute_gradients(
    const std::vector<int>& gauge_reaches,
    const std::vector<double>& dL_dQ) {
    
    if (!AD_ENABLED) return;
    
    // Register outputs and seed with loss gradients
    for (size_t i = 0; i < gauge_reaches.size(); ++i) {
        Reach& reach = network_.get_reach(gauge_reaches[i]);
        register_output(reach.outflow_curr);
        set_gradient(reach.outflow_curr, dL_dQ[i]);
    }
    
    // Reverse pass
    evaluate_tape();
    
    // Collect gradients from tape
    network_.collect_gradients();
    
    // NOW deactivate the tape
    deactivate_tape();
}

inline void MuskingumCungeRouter::record_output(int reach_id) {
    if (!recording_ || !AD_ENABLED) return;
    
    // Copy current discharge to history vector
    // This copy operation IS recorded on the tape
    Real Q_out = network_.get_reach(reach_id).outflow_curr;
    output_history_[reach_id].push_back(Q_out);
}

inline void MuskingumCungeRouter::record_outputs(const std::vector<int>& reach_ids) {
    for (int reach_id : reach_ids) {
        record_output(reach_id);
    }
}

inline void MuskingumCungeRouter::clear_output_history() {
    output_history_.clear();
}

inline size_t MuskingumCungeRouter::get_output_history_size(int reach_id) const {
    auto it = output_history_.find(reach_id);
    if (it != output_history_.end()) {
        return it->second.size();
    }
    return 0;
}

inline std::vector<double> MuskingumCungeRouter::get_output_history(int reach_id) const {
    std::vector<double> result;
    auto it = output_history_.find(reach_id);
    if (it != output_history_.end()) {
        result.reserve(it->second.size());
        for (const Real& val : it->second) {
            result.push_back(to_double(val));
        }
    }
    return result;
}

inline void MuskingumCungeRouter::compute_gradients_timeseries(
    int reach_id,
    const std::vector<double>& dL_dQ) {
    
    if (!AD_ENABLED) return;
    
    auto it = output_history_.find(reach_id);
    if (it == output_history_.end()) {
        std::cerr << "Warning: No recorded outputs for reach " << reach_id << std::endl;
        return;
    }
    
    std::vector<Real>& history = it->second;
    
    if (dL_dQ.size() != history.size()) {
        std::cerr << "Warning: dL_dQ size (" << dL_dQ.size() 
                  << ") != recorded timesteps (" << history.size() << ")" << std::endl;
        return;
    }
    
    // Register all outputs and seed their adjoints
    for (size_t t = 0; t < history.size(); ++t) {
        register_output(history[t]);
        set_gradient(history[t], dL_dQ[t]);
    }
    
    // Single reverse pass accumulates all gradients
    evaluate_tape();
    
    // Collect gradients from tape
    network_.collect_gradients();
    
    // Deactivate tape
    deactivate_tape();
}

inline void MuskingumCungeRouter::compute_gradients_timeseries(
    const std::vector<int>& reach_ids,
    const std::vector<std::vector<double>>& dL_dQ) {
    
    if (!AD_ENABLED) return;
    
    if (reach_ids.size() != dL_dQ.size()) {
        std::cerr << "Warning: reach_ids size != dL_dQ size" << std::endl;
        return;
    }
    
    // Register all outputs and seed their adjoints for all reaches
    for (size_t i = 0; i < reach_ids.size(); ++i) {
        int reach_id = reach_ids[i];
        auto it = output_history_.find(reach_id);
        if (it == output_history_.end()) {
            std::cerr << "Warning: No recorded outputs for reach " << reach_id << std::endl;
            continue;
        }
        
        std::vector<Real>& history = it->second;
        const std::vector<double>& grads = dL_dQ[i];
        
        if (grads.size() != history.size()) {
            std::cerr << "Warning: dL_dQ[" << i << "] size mismatch" << std::endl;
            continue;
        }
        
        for (size_t t = 0; t < history.size(); ++t) {
            register_output(history[t]);
            set_gradient(history[t], grads[t]);
        }
    }
    
    // Single reverse pass
    evaluate_tape();
    
    // Collect gradients
    network_.collect_gradients();
    
    // Deactivate tape
    deactivate_tape();
}

inline std::unordered_map<std::string, double> MuskingumCungeRouter::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        std::string prefix = "reach_" + std::to_string(reach_id) + "_";
        
        grads[prefix + "manning_n"] = reach.grad_manning_n;
        grads[prefix + "width_coef"] = reach.grad_width_coef;
        grads[prefix + "width_exp"] = reach.grad_width_exp;
        grads[prefix + "depth_coef"] = reach.grad_depth_coef;
        grads[prefix + "depth_exp"] = reach.grad_depth_exp;
    }
    
    return grads;
}

inline void MuskingumCungeRouter::reset_gradients() {
    reset_tape();
    network_.zero_gradients();
}

inline void MuskingumCungeRouter::set_lateral_inflow(int reach_id, double inflow) {
    network_.get_reach(reach_id).lateral_inflow = Real(inflow);
}

inline void MuskingumCungeRouter::set_lateral_inflows(const std::vector<double>& inflows) {
    const auto& order = network_.topological_order();
    for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
        network_.get_reach(order[i]).lateral_inflow = Real(inflows[i]);
    }
}

inline double MuskingumCungeRouter::get_discharge(int reach_id) const {
    return to_double(network_.get_reach(reach_id).outflow_curr);
}

inline std::vector<double> MuskingumCungeRouter::get_all_discharges() const {
    std::vector<double> discharges;
    for (int reach_id : network_.topological_order()) {
        discharges.push_back(to_double(network_.get_reach(reach_id).outflow_curr));
    }
    return discharges;
}

inline void MuskingumCungeRouter::reset_state() {
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = Real(0.0);
        reach.inflow_curr = Real(0.0);
        reach.outflow_prev = Real(0.0);
        reach.outflow_curr = Real(0.0);
        reach.lateral_inflow = Real(0.0);
    }
    current_time_ = 0.0;
    reset_gradients();
}

inline RouterState MuskingumCungeRouter::save_state() const {
    RouterState state;
    state.time = current_time_;
    
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        state.inflows[reach_id] = to_double(reach.inflow_curr);
        state.outflows[reach_id] = to_double(reach.outflow_curr);
    }
    
    return state;
}

inline void MuskingumCungeRouter::load_state(const RouterState& state) {
    current_time_ = state.time;
    
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        
        auto it_in = state.inflows.find(reach_id);
        if (it_in != state.inflows.end()) {
            reach.inflow_curr = Real(it_in->second);
            reach.inflow_prev = reach.inflow_curr;
        }
        
        auto it_out = state.outflows.find(reach_id);
        if (it_out != state.outflows.end()) {
            reach.outflow_curr = Real(it_out->second);
            reach.outflow_prev = reach.outflow_curr;
        }
    }
}

inline bool MuskingumCungeRouter::save_state_to_file(const std::string& filepath) const {
    return save_state().save(filepath);
}

inline bool MuskingumCungeRouter::load_state_from_file(const std::string& filepath) {
    try {
        RouterState state = RouterState::load(filepath);
        load_state(state);
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// IRF (Impulse Response Function) Router
// ============================================================================

/**
 * Impulse Response Function routing using gamma distribution.
 * 
 * Convolves inflows with a unit hydrograph derived from channel properties.
 * The IRF is a gamma distribution parameterized by shape and scale.
 * 
 * NEW: Soft-masked convolution for fully differentiable kernel length.
 * Uses fixed maximum kernel size with sigmoid-masked weights:
 *   effective_weight[i] = base_weight[i] * sigmoid((T_cutoff - t_i) * steepness)
 * 
 * This allows gradients to flow through changes in travel time continuously,
 * solving the integer loop-bound problem that breaks AD.
 */
class IRFRouter {
public:
    explicit IRFRouter(Network& network, RouterConfig config = {});
    
    // =========== Core Routing ===========
    void route_timestep();
    void route(int num_timesteps);
    
    // =========== Gradient Computation ===========
    void enable_gradients(bool enable);
    void start_recording();
    void stop_recording();
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    std::unordered_map<std::string, double> get_gradients() const;
    void reset_gradients();
    
    // =========== State Management ===========
    void set_lateral_inflow(int reach_id, double inflow);
    void set_lateral_inflows(const std::vector<double>& inflows);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    void reset_state();
    
    // =========== Time Management ===========
    double current_time() const { return current_time_; }
    void set_time(double t) { current_time_ = t; }
    
    // =========== Access ===========
    Network& network() { return network_; }
    const Network& network() const { return network_; }
    const RouterConfig& config() const { return config_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    bool initialized_ = false;
    
    // Soft-masked kernel configuration
    int max_kernel_size_;           // Fixed maximum kernel length
    double shape_param_ = 2.5;      // Gamma shape parameter (k)
    double mask_steepness_ = 10.0;  // Sigmoid steepness for soft mask
    
    // Time values for each kernel position [s]
    std::vector<double> kernel_times_;
    
    // Base gamma kernel weights (before masking) for each reach
    std::unordered_map<int, std::vector<double>> base_kernels_;
    
    // Inflow history for convolution (most recent first)
    std::unordered_map<int, std::deque<Real>> inflow_history_;
    
    // Reach-specific parameters for gradient computation
    struct IRFParams {
        double manning_n;
        double travel_time;
        double scale;
    };
    std::unordered_map<int, IRFParams> irf_params_;
    std::unordered_map<int, double> analytical_dQ_dn_;
    
    void initialize_kernels();
    void build_base_kernel(int reach_id, const Reach& reach);
    
    /**
     * Compute soft-masked kernel weight at position i
     * 
     * Uses sigmoid mask: w_masked[i] = w_base[i] * sigmoid((T_cutoff - t_i) * steepness)
     * 
     * T_cutoff = 3 * travel_time (99% of gamma mass is within 3*scale)
     */
    Real compute_masked_weight(int reach_id, int i, const Real& travel_time) const;
    
    Real compute_reach_inflow(const Reach& reach);
    void route_reach_irf(Reach& reach);
};

// ==================== IRFRouter Implementation ====================

inline IRFRouter::IRFRouter(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    
    // Configure from options
    max_kernel_size_ = config_.irf_max_kernel_size;
    shape_param_ = config_.irf_shape_param;
    mask_steepness_ = config_.irf_mask_steepness;
    
    // Pre-compute kernel time values
    kernel_times_.resize(max_kernel_size_);
    for (int i = 0; i < max_kernel_size_; ++i) {
        kernel_times_[i] = (i + 0.5) * config_.dt;  // Mid-point of each timestep
    }
}

inline void IRFRouter::initialize_kernels() {
    if (initialized_) return;
    
    base_kernels_.clear();
    inflow_history_.clear();
    irf_params_.clear();
    analytical_dQ_dn_.clear();
    
    // Track travel time statistics (for internal use)
    double max_travel_time = 0.0;
    double max_kernel_coverage_time = max_kernel_size_ * config_.dt;  // seconds
    int num_truncated = 0;
    
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        build_base_kernel(reach_id, reach);
        
        // Check for kernel truncation
        double n = to_double(reach.manning_n);
        double S = reach.slope;
        double velocity = (1.0 / n) * std::pow(1.0, 2.0/3.0) * std::sqrt(S);
        velocity = std::max(0.1, std::min(5.0, velocity));
        double travel_time = reach.length / velocity;
        
        // For gamma with k=2.5, 99% mass is within ~3 * travel_time
        double needed_coverage = 3.0 * travel_time;
        if (needed_coverage > max_kernel_coverage_time) {
            num_truncated++;
        }
        max_travel_time = std::max(max_travel_time, travel_time);
        
        // Initialize inflow history to fixed size
        inflow_history_[reach_id] = std::deque<Real>(max_kernel_size_, Real(0.0));
        analytical_dQ_dn_[reach_id] = 0.0;
    }
    
    // Store for potential diagnostics (don't print by default)
    (void)max_travel_time;  // Suppress unused warning
    (void)num_truncated;
    
    initialized_ = true;
}

/**
 * Build base gamma kernel (unnormalized weights)
 * 
 * The base kernel contains unnormalized gamma PDF values.
 * Actual normalization happens at runtime with soft masking.
 */
inline void IRFRouter::build_base_kernel(int reach_id, const Reach& reach) {
    // Compute initial travel time estimate
    double n = to_double(reach.manning_n);
    double S = reach.slope;
    double R_h = 1.0;
    
    double velocity = (1.0 / n) * std::pow(R_h, 2.0/3.0) * std::sqrt(S);
    velocity = std::max(0.1, std::min(5.0, velocity));
    
    double travel_time = reach.length / velocity;
    double scale = travel_time / shape_param_;
    
    // Store parameters
    irf_params_[reach_id] = {n, travel_time, scale};
    
    // Build unnormalized gamma kernel
    std::vector<double> kernel(max_kernel_size_);
    for (int i = 0; i < max_kernel_size_; ++i) {
        double t = kernel_times_[i];
        // Gamma PDF (unnormalized): t^(k-1) * exp(-t/θ)
        kernel[i] = std::pow(t, shape_param_ - 1.0) * std::exp(-t / scale);
    }
    
    base_kernels_[reach_id] = kernel;
}

/**
 * Compute soft-masked kernel weight
 * 
 * Applies sigmoid mask based on travel time:
 *   mask = sigmoid((T_cutoff - t_i) * steepness)
 *   
 * where T_cutoff = 5 * scale (covers 99%+ of gamma mass)
 */
inline Real IRFRouter::compute_masked_weight(int reach_id, int i, const Real& travel_time) const {
    const auto& base_kernel = base_kernels_.at(reach_id);
    
    // Scale parameter
    Real scale = travel_time / Real(shape_param_);
    
    // Cutoff time (5 * scale covers >99% of gamma distribution)
    Real T_cutoff = Real(5.0) * scale;
    
    // Time at this kernel position
    Real t_i = Real(kernel_times_[i]);
    
    // Sigmoid mask: 1 near zero, 0 beyond cutoff
    // sigmoid((cutoff - t) * steepness)
    Real z = (T_cutoff - t_i) * Real(mask_steepness_) / scale;  // Normalize by scale
    
    // Numerically stable sigmoid
    Real mask;
    double z_val = to_double(z);
    if (z_val > 20.0) {
        mask = Real(1.0);
    } else if (z_val < -20.0) {
        mask = Real(0.0);
    } else {
        mask = Real(1.0) / (Real(1.0) + exp(-z));
    }
    
    // Recompute kernel weight with current scale (for differentiability)
    // Use the scale from current manning_n, not stored scale
    Real weight = pow(t_i, Real(shape_param_ - 1.0)) * exp(-t_i / scale);
    
    return weight * mask;
}

inline Real IRFRouter::compute_reach_inflow(const Reach& reach) {
    Real Q_in = Real(0.0);
    
    if (reach.upstream_junction_id >= 0) {
        try {
            const Junction& junc = network_.get_junction(reach.upstream_junction_id);
            for (int up_id : junc.upstream_reach_ids) {
                const Reach& up_reach = network_.get_reach(up_id);
                Q_in = Q_in + up_reach.outflow_curr;
            }
        } catch (...) {}
    }
    
    return Q_in;
}

inline void IRFRouter::route_reach_irf(Reach& reach) {
    // Get upstream inflow
    Real Q_upstream = compute_reach_inflow(reach);
    
    // Total inflow = upstream + lateral
    Real Q_in = Q_upstream + reach.lateral_inflow;
    
    // Update inflow history (push front, pop back)
    auto& history = inflow_history_[reach.id];
    history.push_front(Q_in);
    while (static_cast<int>(history.size()) > max_kernel_size_) {
        history.pop_back();
    }
    
    // Compute travel time from current manning_n (differentiable)
    Real n = reach.manning_n;
    double S = reach.slope;
    Real R_h = Real(1.0);
    Real S_sqrt = Real(std::sqrt(S));
    
    Real velocity = (Real(1.0) / n) * pow(R_h, Real(2.0/3.0)) * S_sqrt;
    velocity = safe_max(velocity, Real(0.1));
    velocity = safe_min(velocity, Real(5.0));
    
    Real travel_time = Real(reach.length) / velocity;
    Real scale = travel_time / Real(shape_param_);
    
    // MASS-CONSERVING CONVOLUTION:
    // Pre-compute normalized kernel weights (unit hydrograph)
    // H[i] = w[i] / Σw[j] where Σ H[i] = 1 guarantees mass conservation
    
    std::vector<Real> kernel_weights(max_kernel_size_);
    Real weight_sum = Real(0.0);
    
    // Find where most of the kernel mass is (for diagnostics)
    double peak_weight = 0.0;
    int peak_idx = 0;
    
    for (int i = 0; i < max_kernel_size_; ++i) {
        Real t_i = Real(kernel_times_[i]);
        // Gamma PDF (unnormalized): t^(k-1) * exp(-t/θ)
        Real w = pow(t_i, Real(shape_param_ - 1.0)) * exp(-t_i / scale);
        kernel_weights[i] = w;
        weight_sum = weight_sum + w;
        
        if (to_double(w) > peak_weight) {
            peak_weight = to_double(w);
            peak_idx = i;
        }
    }
    
    // Check for kernel truncation (if peak is near the end, we might be losing mass)
    static bool warned_truncation = false;
    if (!warned_truncation && peak_idx > max_kernel_size_ * 0.7) {
        std::cerr << "WARNING: IRF kernel peak at position " << peak_idx 
                  << "/" << max_kernel_size_ << " for reach " << reach.id
                  << " (travel_time=" << to_double(travel_time)/3600.0 << " hours)"
                  << " - consider increasing irf_max_kernel_size\n";
        warned_truncation = true;
    }
    
    // Normalize to create unit hydrograph (sums to 1)
    if (to_double(weight_sum) > 1e-10) {
        for (int i = 0; i < max_kernel_size_; ++i) {
            kernel_weights[i] = kernel_weights[i] / weight_sum;
        }
    } else {
        // Emergency fallback: if kernel is all zeros, use delta function at position 0
        kernel_weights[0] = Real(1.0);
    }
    
    // Verify normalization (debug)
    static bool verified_normalization = false;
    if (!verified_normalization) {
        Real check_sum = Real(0.0);
        for (int i = 0; i < max_kernel_size_; ++i) {
            check_sum = check_sum + kernel_weights[i];
        }
        if (std::abs(to_double(check_sum) - 1.0) > 0.01) {
            std::cerr << "WARNING: IRF kernel does not sum to 1.0: " << to_double(check_sum) << "\n";
        }
        verified_normalization = true;
    }
    
    // Apply convolution: Q_out = Σ H[i] * Q_in[t-i]
    Real Q_out = Real(0.0);
    int n_hist = static_cast<int>(history.size());
    for (int i = 0; i < n_hist; ++i) {
        Q_out = Q_out + kernel_weights[i] * history[i];
    }
    
    // Store analytical gradient approximation for fallback
    if (recording_ && config_.enable_gradients) {
        double k = shape_param_;
        double theta = to_double(scale);
        double manning_n = to_double(n);
        
        // Compute analytical gradient approximation
        double dQ_dn = 0.0;
        double w_sum = to_double(weight_sum);
        for (int i = 0; i < n_hist; ++i) {
            double t_i = kernel_times_[i];
            double Q_in_i = to_double(history[i]);
            double w_i = std::pow(t_i, k - 1.0) * std::exp(-t_i / theta);
            double d_w_dn = w_i * (t_i - k * theta) / (theta * manning_n);
            dQ_dn += Q_in_i * d_w_dn;
        }
        if (w_sum > 1e-10) dQ_dn /= w_sum;
        analytical_dQ_dn_[reach.id] = dQ_dn;
    }
    
    // Ensure non-negative
    Q_out = safe_max(Q_out, Real(config_.min_flow));
    
    // Update reach state
    reach.inflow_curr = Q_in;
    reach.outflow_curr = Q_out;
}

inline void IRFRouter::route_timestep() {
    // Initialize kernels on first call
    if (!initialized_) {
        initialize_kernels();
    }
    
    // Route each reach in topological order
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        route_reach_irf(reach);
    }
    
    current_time_ += config_.dt;
}

inline void IRFRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) {
        route_timestep();
    }
}

inline void IRFRouter::enable_gradients(bool enable) {
    config_.enable_gradients = enable;
}

inline void IRFRouter::start_recording() {
    if (!config_.enable_gradients || !AD_ENABLED) return;
    
    reset_tape();  // clear the global CoDiPack tape so successive recording sessions
                   // (e.g. gradient-descent epochs) don't accumulate and exhaust memory
    activate_tape();
    recording_ = true;
    
    // Register all parameters as inputs (same as MC router)
    for (Real* param : network_.get_all_parameters()) {
        register_input(*param);
    }
    
    // Reset analytical gradient accumulators (kept for diagnostics)
    for (auto& [id, g] : analytical_dQ_dn_) {
        g = 0.0;
    }
}

inline void IRFRouter::stop_recording() {
    // Don't deactivate tape yet - we need it active for gradient computation
    recording_ = false;
}

inline void IRFRouter::compute_gradients(const std::vector<int>& gauge_reaches,
                                          const std::vector<double>& dL_dQ) {
    if (!AD_ENABLED || gauge_reaches.empty() || dL_dQ.empty()) return;
    
    // Register outputs and seed with loss gradients
    for (size_t i = 0; i < gauge_reaches.size(); ++i) {
        Reach& reach = network_.get_reach(gauge_reaches[i]);
        register_output(reach.outflow_curr);
        set_gradient(reach.outflow_curr, dL_dQ[i]);
    }
    
    // Reverse pass
    evaluate_tape();
    
    // Collect gradients from tape
    network_.collect_gradients();
    
    // Deactivate the tape
    deactivate_tape();
}

inline std::unordered_map<std::string, double> IRFRouter::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        std::string prefix = "reach_" + std::to_string(reach_id) + "_";
        grads[prefix + "manning_n"] = reach.grad_manning_n;
        grads[prefix + "width_coef"] = reach.grad_width_coef;
        grads[prefix + "depth_coef"] = reach.grad_depth_coef;
    }
    
    return grads;
}

inline void IRFRouter::reset_gradients() {
    reset_tape();
    network_.zero_gradients();
    for (auto& [id, g] : analytical_dQ_dn_) {
        g = 0.0;
    }
}

inline void IRFRouter::set_lateral_inflow(int reach_id, double inflow) {
    network_.get_reach(reach_id).lateral_inflow = Real(inflow);
}

inline void IRFRouter::set_lateral_inflows(const std::vector<double>& inflows) {
    const auto& order = network_.topological_order();
    for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
        network_.get_reach(order[i]).lateral_inflow = Real(inflows[i]);
    }
}

inline double IRFRouter::get_discharge(int reach_id) const {
    return to_double(network_.get_reach(reach_id).outflow_curr);
}

inline std::vector<double> IRFRouter::get_all_discharges() const {
    std::vector<double> discharges;
    for (int reach_id : network_.topological_order()) {
        discharges.push_back(to_double(network_.get_reach(reach_id).outflow_curr));
    }
    return discharges;
}

inline void IRFRouter::reset_state() {
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = Real(0.0);
        reach.inflow_curr = Real(0.0);
        reach.outflow_prev = Real(0.0);
        reach.outflow_curr = Real(0.0);
        reach.lateral_inflow = Real(0.0);
    }
    
    // Reset inflow histories
    for (auto& [id, history] : inflow_history_) {
        std::fill(history.begin(), history.end(), Real(0.0));
    }
    
    current_time_ = 0.0;
    reset_gradients();
}


// ============================================================================
// Diffusive Wave Router
// ============================================================================

/**
 * Diffusive Wave routing using finite difference solution.
 * 
 * Solves the diffusive wave equation:
 *   ∂Q/∂t + c·∂Q/∂x = D·∂²Q/∂x² + q_lat
 * 
 * where:
 *   c = wave celerity [m/s]
 *   D = diffusion coefficient [m²/s]
 *   q_lat = lateral inflow per unit length [m²/s]
 * 
 * Uses Crank-Nicolson scheme for numerical stability.
 * Fully differentiable via CoDiPack.
 */
class DiffusiveWaveRouter {
public:
    explicit DiffusiveWaveRouter(Network& network, RouterConfig config = {});
    
    void route_timestep();
    void route(int num_timesteps);
    
    void enable_gradients(bool enable);
    void start_recording();
    void stop_recording();
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    std::unordered_map<std::string, double> get_gradients() const;
    void reset_gradients();
    
    void set_lateral_inflow(int reach_id, double inflow);
    void set_lateral_inflows(const std::vector<double>& inflows);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    void reset_state();
    
    double current_time() const { return current_time_; }
    void set_time(double t) { current_time_ = t; }
    
    Network& network() { return network_; }
    const Network& network() const { return network_; }
    const RouterConfig& config() const { return config_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    // Number of computational nodes per reach (max/default)
    int nodes_per_reach_ = 10;
    
    // Actual node count per reach (adaptive based on reach length)
    std::unordered_map<int, int> reach_nodes_;
    
    // State: Q at each node for each reach
    std::unordered_map<int, std::vector<Real>> Q_nodes_;
    
    // For analytical gradients
    struct DWParams {
        double manning_n;
        double celerity;
        double Q_mean;
    };
    std::unordered_map<int, DWParams> dw_params_;
    std::unordered_map<int, double> analytical_dQ_dn_;
    
    void initialize_state();
    Real compute_celerity(const Reach& reach, Real Q);
    Real compute_diffusion_coef(const Reach& reach, Real Q);
    void route_reach_diffusive(Reach& reach);
};

// Diffusive Wave Implementation
inline DiffusiveWaveRouter::DiffusiveWaveRouter(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    nodes_per_reach_ = config_.dw_num_nodes;  // Use config value
    initialize_state();
}

inline void DiffusiveWaveRouter::initialize_state() {
    Q_nodes_.clear();
    dw_params_.clear();
    analytical_dQ_dn_.clear();
    reach_nodes_.clear();  // Store per-reach node count
    
    // Minimum dx for stability with explicit scheme
    // With dt=3600s, c_max=5m/s, D_max=500m²/s, max_substeps=500:
    // Need (Cr + 2*Df) / 500 <= 0.8, so Cr + 2*Df <= 400
    // Cr = c*dt/dx = 5*3600/dx = 18000/dx  →  for Cr=200: dx = 90m
    // Df = D*dt/dx² = 500*3600/dx² = 1.8e6/dx²  →  for Df=100: dx = 134m
    // Use 500m to be safe (Cr=36, Df=7.2, stability=50 → needs ~63 substeps)
    double dx_min = 500.0;  // meters
    
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        
        // Compute nodes needed for this reach (minimum 3 for numerical scheme)
        int nodes = std::max(3, static_cast<int>(reach.length / dx_min) + 1);
        // Cap at config value
        nodes = std::min(nodes, nodes_per_reach_);
        
        // For very short reaches, use minimum nodes and accept coarse resolution
        if (reach.length < dx_min * 2) {
            nodes = 3;  // Minimum for 2nd order scheme
        }
        
        reach_nodes_[reach_id] = nodes;
        Q_nodes_[reach_id] = std::vector<Real>(nodes, Real(0.0));
        
        dw_params_[reach_id] = {to_double(reach.manning_n), 1.0, 0.0};
        analytical_dQ_dn_[reach_id] = 0.0;
    }
}

inline Real DiffusiveWaveRouter::compute_celerity(const Reach& reach, Real Q) {
    // Celerity from Manning's equation: c ≈ 5/3 * v
    // v = (1/n) * R_h^(2/3) * S^(1/2)
    
    // Use smooth bounds for AD when enabled
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        Q = smooth_max(Q, Real(0.01), config_.smooth_epsilon);
    } else {
        Q = safe_max(Q, Real(0.01));
    }
    
    // AD-safe power: exp(exp * log(base))
    Real width = reach.geometry.width_coef * exp(reach.geometry.width_exp * log(safe_max(Q, Real(1e-6))));
    Real depth = reach.geometry.depth_coef * exp(reach.geometry.depth_exp * log(safe_max(Q, Real(1e-6))));
    
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        width = smooth_max(width, Real(1.0), config_.smooth_epsilon);
        depth = smooth_max(depth, Real(0.1), config_.smooth_epsilon);
    } else {
        width = safe_max(width, Real(1.0));
        depth = safe_max(depth, Real(0.1));
    }
    
    // Hydraulic radius for wide channel: R_h ≈ depth
    Real R_h = depth;
    
    // Manning's velocity: v = (1/n) * R_h^(2/3) * sqrt(S)
    Real slope = Real(reach.slope);
    slope = safe_max(slope, Real(1e-6));
    
    Real velocity = (Real(1.0) / reach.manning_n) * pow(R_h, Real(2.0/3.0)) * sqrt(slope);
    
    // Kinematic wave celerity: c = 5/3 * v
    Real celerity = Real(5.0/3.0) * velocity;
    
    // Bounds for numerical stability (use smooth when AD enabled)
    // Lower max for explicit scheme stability (5 m/s gives Cr=36 at dx=500m, dt=3600s)
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        celerity = smooth_max(celerity, Real(0.1), config_.smooth_epsilon);
        celerity = smooth_min(celerity, Real(3.0), config_.smooth_epsilon);  // Reduced from 5.0
    } else {
        celerity = safe_max(celerity, Real(0.1));
        celerity = safe_min(celerity, Real(3.0));  // Reduced from 5.0
    }
    
    return celerity;
}

inline Real DiffusiveWaveRouter::compute_diffusion_coef(const Reach& reach, Real Q) {
    // Diffusion coefficient: D = Q / (2 * B * S)
    // Also can be written as: D = v * h / (2 * S) where v depends on n
    
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        Q = smooth_max(Q, Real(0.01), config_.smooth_epsilon);
    } else {
        Q = safe_max(Q, Real(0.01));
    }
    
    // AD-safe power
    Real width = reach.geometry.width_coef * exp(reach.geometry.width_exp * log(safe_max(Q, Real(1e-6))));
    
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        width = smooth_max(width, Real(1.0), config_.smooth_epsilon);
    } else {
        width = safe_max(width, Real(1.0));
    }
    
    Real slope = Real(reach.slope);
    slope = safe_max(slope, Real(1e-6));
    
    // D = Q / (2 * B * S)
    // Since Q depends implicitly on n through the routing, this captures n-sensitivity
    Real D = Q / (Real(2.0) * width * slope);
    
    // Bounds for stability (use smooth when AD enabled)
    // With dx~500m, dt=3600s: Df = D * dt/dx² = D * 0.0144
    // For Df < 0.5 (stability), need D < 35 m²/s without sub-stepping
    // Allow up to 200 m²/s with sub-stepping (will require ~10-20 substeps)
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        D = smooth_max(D, Real(1.0), config_.smooth_epsilon);
        D = smooth_min(D, Real(200.0), config_.smooth_epsilon);  // Reduced from 500
    } else {
        D = safe_max(D, Real(1.0));
        D = safe_min(D, Real(200.0));  // Reduced from 500
    }
    
    return D;
}

inline void DiffusiveWaveRouter::route_reach_diffusive(Reach& reach) {
    auto& Q = Q_nodes_[reach.id];
    int n_nodes = reach_nodes_[reach.id];  // Use adaptive node count for this reach
    double dx = reach.length / (n_nodes - 1);
    double dt = config_.dt;
    
    // Get upstream boundary condition (from upstream reaches only)
    Real Q_upstream = Real(0.0);
    if (reach.upstream_junction_id >= 0) {
        try {
            const Junction& junc = network_.get_junction(reach.upstream_junction_id);
            for (int up_id : junc.upstream_reach_ids) {
                const Reach& up_reach = network_.get_reach(up_id);
                Q_upstream = Q_upstream + up_reach.outflow_curr;
            }
        } catch (...) {}
    }
    
    // For very short reaches, use instant routing (pass-through)
    // This avoids numerical instability from tiny dx
    if (reach.length < 200.0) {  // < 200m
        reach.inflow_curr = Q_upstream + reach.lateral_inflow;
        reach.outflow_curr = reach.inflow_curr;
        // Fill Q_nodes uniformly for consistency
        for (int i = 0; i < n_nodes; ++i) {
            Q[i] = reach.outflow_curr;
        }
        return;
    }
    
    // Lateral inflow distributed as source term [m³/s per m of reach]
    // Total lateral inflow spread across reach length
    Real q_lat = reach.lateral_inflow / Real(reach.length);  // [m³/s/m]
    
    // Reference Q for parameters (use mean, including boundary and lateral for better initialization)
    Real Q_ref = Real(0.0);
    for (int i = 0; i < n_nodes; ++i) Q_ref = Q_ref + Q[i];
    Q_ref = Q_ref / Real(n_nodes);
    
    // Include upstream and lateral inflow in Q_ref for better initialization
    Q_ref = safe_max(Q_ref, Q_upstream);
    Q_ref = safe_max(Q_ref, reach.lateral_inflow);
    
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        Q_ref = smooth_max(Q_ref, Real(0.1), config_.smooth_epsilon);
    } else {
        Q_ref = safe_max(Q_ref, Real(0.1));
    }
    
    // Compute celerity and diffusion coefficient (both are Real, AD-enabled)
    Real c = compute_celerity(reach, Q_ref);
    Real D = compute_diffusion_coef(reach, Q_ref);
    
    // Store parameters for gradient diagnostics
    dw_params_[reach.id] = {to_double(reach.manning_n), to_double(c), to_double(Q_ref)};
    
    // Courant and diffusion numbers (keep as Real for AD)
    Real Cr = c * Real(dt / dx);
    Real Df = D * Real(dt / (dx * dx));
    
    // Stability check and sub-stepping
    // For explicit scheme, need Cr + 2*Df <= 1 approximately
    double stability = to_double(Cr) + 2.0 * to_double(Df);
    int sub_steps = std::max(1, static_cast<int>(stability / 0.8) + 1);
    
    // Hard cap with instability warning
    if (sub_steps > config_.dw_max_substeps) {
        static bool warned_cap = false;
        if (!warned_cap) {
            std::cerr << "WARNING: DW sub-steps capped from " << sub_steps 
                      << " to " << config_.dw_max_substeps 
                      << " - results may be unstable (NaN)!\n";
            std::cerr << "  Increase dw_max_substeps or use diffusive-ift method.\n";
            warned_cap = true;
        }
        sub_steps = config_.dw_max_substeps;
    }
    
    Real sub_dt = Real(dt / sub_steps);
    
    // Recompute Courant numbers for sub-timestep
    Cr = c * sub_dt / Real(dx);
    Df = D * sub_dt / Real(dx * dx);
    
    // Sub-stepping loop
    // The diffusive wave PDE: ∂Q/∂t + c·∂Q/∂x = D·∂²Q/∂x²
    // Lateral inflow is added at upstream boundary (like MC approach)
    for (int s = 0; s < sub_steps; ++s) {
        // Store old values
        std::vector<Real> Q_old = Q;
        
        // Update interior nodes (no lateral source here - it enters at upstream)
        for (int i = 1; i < n_nodes - 1; ++i) {
            // Upwind advection (c > 0 assumed)
            Real advection = -Cr * (Q_old[i] - Q_old[i-1]);
            
            // Central diffusion
            Real diffusion = Df * (Q_old[i+1] - Real(2.0) * Q_old[i] + Q_old[i-1]);
            
            // Update (no distributed source - lateral enters at upstream)
            Q[i] = Q_old[i] + advection + diffusion;
            
            // NaN detection and recovery
            if (std::isnan(to_double(Q[i])) || std::isinf(to_double(Q[i]))) {
                static bool warned_nan = false;
                if (!warned_nan) {
                    std::cerr << "WARNING: NaN/Inf detected in DW routing at node " << i 
                              << " (substep " << s << "/" << sub_steps << ")\n";
                    std::cerr << "  Cr=" << to_double(Cr) << ", Df=" << to_double(Df) << "\n";
                    std::cerr << "  Recovering by using previous value.\n";
                    warned_nan = true;
                }
                Q[i] = Q_old[i];
            }
            
            // Ensure non-negative (use smooth version for AD)
            if (config_.enable_gradients && config_.use_smooth_bounds) {
                Q[i] = smooth_max(Q[i], Real(0.0), config_.smooth_epsilon);
            } else if (to_double(Q[i]) < 0.0) {
                Q[i] = Real(0.0);
            }
        }
        
        // Boundary conditions
        // Upstream: inflow from upstream + lateral inflow (enters at top of reach)
        Q[0] = Q_upstream + reach.lateral_inflow;
        
        // Downstream: zero gradient outflow
        Q[n_nodes-1] = Q[n_nodes-2];
    }
    
    // Update reach state (these are Real, so gradients flow through)
    reach.inflow_curr = Q_upstream + reach.lateral_inflow;
    reach.outflow_curr = Q[n_nodes-1];
    
    // Analytical gradient approximation for fallback
    // dc/dn = -c/n (from Manning's equation)
    // When n increases: c decreases -> wave slows -> peak arrives later
    // For mass conservation routing: higher n -> more attenuation -> lower peak
    // dQ_out/dn = -(Q_peak - Q_mean) * (c/n) * (dt/L) is negative when peak > mean
    if (recording_ && config_.enable_gradients) {
        double n_val = to_double(reach.manning_n);
        double c_val = to_double(c);
        double Q_out = to_double(Q[n_nodes-1]);
        double Q_in = to_double(Q_upstream);
        
        // Sensitivity: increasing n -> slower wave -> lower Q_out during rising limb
        // The gradient depends on the flow regime (rising vs falling limb)
        // Approximate: dQ_out/dn ≈ -(Q_out/n) * (L/c) / (L/c + dt) 
        double travel_time = reach.length / c_val;
        double attenuation_factor = travel_time / (travel_time + dt);
        analytical_dQ_dn_[reach.id] = -Q_out * attenuation_factor / n_val;
    }
}

inline void DiffusiveWaveRouter::route_timestep() {
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        route_reach_diffusive(reach);
    }
    current_time_ += config_.dt;
}

inline void DiffusiveWaveRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) {
        route_timestep();
    }
}

inline void DiffusiveWaveRouter::enable_gradients(bool enable) {
    config_.enable_gradients = enable;
}

inline void DiffusiveWaveRouter::start_recording() {
    if (!config_.enable_gradients || !AD_ENABLED) return;
    
    reset_tape();  // clear the global CoDiPack tape so successive recording sessions
                   // (e.g. gradient-descent epochs) don't accumulate and exhaust memory
    activate_tape();
    recording_ = true;
    
    // Register all parameters as inputs (same as MC router)
    for (Real* param : network_.get_all_parameters()) {
        register_input(*param);
    }
    
    // Reset analytical gradient accumulators (kept for diagnostics)
    for (auto& [id, g] : analytical_dQ_dn_) g = 0.0;
}

inline void DiffusiveWaveRouter::stop_recording() {
    // Don't deactivate tape yet - we need it active for gradient computation
    recording_ = false;
}

inline void DiffusiveWaveRouter::compute_gradients(const std::vector<int>& gauge_reaches,
                                                    const std::vector<double>& dL_dQ) {
    if (!AD_ENABLED || gauge_reaches.empty() || dL_dQ.empty()) return;
    
    // Register outputs and seed with loss gradients
    for (size_t i = 0; i < gauge_reaches.size(); ++i) {
        Reach& reach = network_.get_reach(gauge_reaches[i]);
        register_output(reach.outflow_curr);
        set_gradient(reach.outflow_curr, dL_dQ[i]);
    }
    
    // Reverse pass
    evaluate_tape();
    
    // Collect gradients from tape
    network_.collect_gradients();
    
    // Deactivate the tape
    deactivate_tape();
}

inline std::unordered_map<std::string, double> DiffusiveWaveRouter::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        std::string key = "manning_n_" + std::to_string(reach_id);
        grads[key] = reach.grad_manning_n;
        std::string prefix = "reach_" + std::to_string(reach_id) + "_";
        grads[prefix + "manning_n"] = reach.grad_manning_n;
        grads[prefix + "width_coef"] = reach.grad_width_coef;
        grads[prefix + "depth_coef"] = reach.grad_depth_coef;
    }
    return grads;
}

inline void DiffusiveWaveRouter::reset_gradients() {
    reset_tape();
    network_.zero_gradients();
    for (auto& [id, g] : analytical_dQ_dn_) g = 0.0;
}

inline void DiffusiveWaveRouter::set_lateral_inflow(int reach_id, double inflow) {
    network_.get_reach(reach_id).lateral_inflow = Real(inflow);
}

inline void DiffusiveWaveRouter::set_lateral_inflows(const std::vector<double>& inflows) {
    const auto& order = network_.topological_order();
    for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
        network_.get_reach(order[i]).lateral_inflow = Real(inflows[i]);
    }
}

inline double DiffusiveWaveRouter::get_discharge(int reach_id) const {
    return to_double(network_.get_reach(reach_id).outflow_curr);
}

inline std::vector<double> DiffusiveWaveRouter::get_all_discharges() const {
    std::vector<double> discharges;
    for (int reach_id : network_.topological_order()) {
        discharges.push_back(to_double(network_.get_reach(reach_id).outflow_curr));
    }
    return discharges;
}

inline void DiffusiveWaveRouter::reset_state() {
    initialize_state();
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = reach.inflow_curr = Real(0.0);
        reach.outflow_prev = reach.outflow_curr = Real(0.0);
        reach.lateral_inflow = Real(0.0);
    }
    current_time_ = 0.0;
    reset_gradients();
}


// ============================================================================
// Lag Router (Simple Time Delay)
// ============================================================================

/**
 * Simple lag routing with optional attenuation.
 * 
 * Each reach delays flow by travel_time = length / velocity.
 * Optionally applies exponential decay for storage effects.
 * 
 * Fully differentiable. Good baseline for comparison.
 */
class LagRouter {
public:
    explicit LagRouter(Network& network, RouterConfig config = {});
    
    void route_timestep();
    void route(int num_timesteps);
    
    void enable_gradients(bool enable);
    void start_recording();
    void stop_recording();
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    std::unordered_map<std::string, double> get_gradients() const;
    void reset_gradients();
    
    void set_lateral_inflow(int reach_id, double inflow);
    void set_lateral_inflows(const std::vector<double>& inflows);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    void reset_state();
    
    double current_time() const { return current_time_; }
    Network& network() { return network_; }
    const RouterConfig& config() const { return config_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    // Lag buffer for each reach (stores delayed inflows)
    std::unordered_map<int, std::deque<Real>> lag_buffer_;
    std::unordered_map<int, int> lag_steps_;  // Number of timesteps to delay
    
    // For analytical gradients
    std::unordered_map<int, double> reach_velocity_;
    std::unordered_map<int, double> analytical_dQ_dn_;
    
    void initialize_lags();
};

inline LagRouter::LagRouter(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    initialize_lags();
}

inline void LagRouter::initialize_lags() {
    lag_buffer_.clear();
    lag_steps_.clear();
    reach_velocity_.clear();
    analytical_dQ_dn_.clear();
    
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        
        // Compute velocity from Manning's equation
        double n = to_double(reach.manning_n);
        double S = reach.slope;
        double R_h = 1.0;
        
        double velocity = (1.0 / n) * std::pow(R_h, 2.0/3.0) * std::sqrt(S);
        velocity = std::max(0.1, std::min(5.0, velocity));
        reach_velocity_[reach_id] = velocity;
        
        // Compute lag in timesteps
        double travel_time = reach.length / velocity;
        int lag = std::max(1, static_cast<int>(travel_time / config_.dt));
        lag_steps_[reach_id] = lag;
        
        // Initialize buffer
        lag_buffer_[reach_id] = std::deque<Real>(lag, Real(0.0));
        analytical_dQ_dn_[reach_id] = 0.0;
    }
}

inline void LagRouter::route_timestep() {
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        
        // Get upstream inflow
        Real Q_in = Real(0.0);
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    Q_in = Q_in + network_.get_reach(up_id).outflow_curr;
                }
            } catch (...) {}
        }
        Q_in = Q_in + reach.lateral_inflow;
        
        // Get lagged outflow
        auto& buffer = lag_buffer_[reach_id];
        Real Q_out = buffer.back();
        buffer.pop_back();
        buffer.push_front(Q_in);
        
        // Update state
        reach.inflow_curr = Q_in;
        reach.outflow_curr = Q_out;
        
        // Compute analytical gradient: dQ_out/dn
        // For lag routing: Q_out depends on n through travel time
        // Approximate gradient using numerical differentiation of buffer
        if (recording_) {
            double n = to_double(reach.manning_n);
            int lag = lag_steps_[reach_id];
            
            // Better gradient estimate: use average slope in buffer
            double dQ_sum = 0.0;
            int count = 0;
            for (size_t i = 0; i + 1 < buffer.size(); ++i) {
                dQ_sum += to_double(buffer[i]) - to_double(buffer[i+1]);
                count++;
            }
            double dQ_avg = (count > 0) ? dQ_sum / count : 0.0;
            
            // dQ/dn ≈ -dQ_avg * lag / n (negative because increasing n increases lag)
            // Scale by travel time to get reasonable magnitude
            double travel_time = lag * config_.dt;
            analytical_dQ_dn_[reach_id] = -std::abs(dQ_avg) * travel_time / (n * config_.dt);
        }
    }
    current_time_ += config_.dt;
}

inline void LagRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) route_timestep();
}

inline void LagRouter::enable_gradients(bool enable) { config_.enable_gradients = enable; }
inline void LagRouter::start_recording() { recording_ = true; }
inline void LagRouter::stop_recording() { recording_ = false; }

inline void LagRouter::compute_gradients(const std::vector<int>& gauge_reaches,
                                          const std::vector<double>& dL_dQ) {
    if (gauge_reaches.empty()) return;
    double dL_dQ_out = dL_dQ[0];
    
    // Propagate gradients upstream (similar to IRF)
    auto topo_order = network_.topological_order();
    std::unordered_map<int, double> factor;
    factor[gauge_reaches[0]] = 1.0;
    
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        int reach_id = *it;
        if (factor.count(reach_id) == 0) continue;
        
        Reach& reach = network_.get_reach(reach_id);
        reach.grad_manning_n = dL_dQ_out * factor[reach_id] * analytical_dQ_dn_[reach_id];
        
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    factor[up_id] = factor.count(up_id) ? factor[up_id] + factor[reach_id] * 0.9 
                                                         : factor[reach_id] * 0.9;
                }
            } catch (...) {}
        }
    }
}

inline std::unordered_map<std::string, double> LagRouter::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        grads["manning_n_" + std::to_string(reach_id)] = reach.grad_manning_n;
    }
    return grads;
}

inline void LagRouter::reset_gradients() {
    network_.zero_gradients();
    for (auto& [id, g] : analytical_dQ_dn_) g = 0.0;
}

inline void LagRouter::set_lateral_inflow(int reach_id, double inflow) {
    network_.get_reach(reach_id).lateral_inflow = Real(inflow);
}

inline void LagRouter::set_lateral_inflows(const std::vector<double>& inflows) {
    const auto& order = network_.topological_order();
    for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
        network_.get_reach(order[i]).lateral_inflow = Real(inflows[i]);
    }
}

inline double LagRouter::get_discharge(int reach_id) const {
    return to_double(network_.get_reach(reach_id).outflow_curr);
}

inline std::vector<double> LagRouter::get_all_discharges() const {
    std::vector<double> d;
    for (int reach_id : network_.topological_order()) {
        d.push_back(to_double(network_.get_reach(reach_id).outflow_curr));
    }
    return d;
}

inline void LagRouter::reset_state() {
    initialize_lags();
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = reach.inflow_curr = Real(0.0);
        reach.outflow_prev = reach.outflow_curr = Real(0.0);
    }
    current_time_ = 0.0;
}


// ============================================================================
// KWT Router (Kinematic Wave Tracking) - Non-differentiable
// ============================================================================

/**
 * Kinematic Wave Tracking routing.
 * 
 * Implements the kinematic wave tracking method as used in mizuRoute.
 * Each wave parcel is treated as a continuous wave segment with spatial extent,
 * not a point mass. This eliminates artificial spikiness from discrete arrivals.
 * 
 * Physical basis:
 * - Wave celerity from kinematic wave theory: c = (5/3) * v = (5/3) * Q/A
 * - Each parcel has a spatial extent (wave_length = c * dt at creation)
 * - Outflow is proportional to how much of the wave has passed the outlet
 * - Parcels are removed only when fully past the outlet
 * 
 * Reference: Mizukami et al. (2016), mizuRoute technical documentation
 * 
 * NOTE: This method is NOT differentiable due to discrete parcel operations.
 */
class KWTRouter {
public:
    explicit KWTRouter(Network& network, RouterConfig config = {});
    
    void route_timestep();
    void route(int num_timesteps);
    
    void enable_gradients(bool) { /* Not supported */ }
    void start_recording() { /* Not supported */ }
    void stop_recording() { /* Not supported */ }
    void compute_gradients(const std::vector<int>&, const std::vector<double>&) {
        std::cerr << "Warning: KWT routing does not support gradients\n";
    }
    std::unordered_map<std::string, double> get_gradients() const {
        return {};  // Empty - no gradients
    }
    void reset_gradients() { network_.zero_gradients(); }
    
    void set_lateral_inflow(int reach_id, double inflow);
    void set_lateral_inflows(const std::vector<double>& inflows);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    void reset_state();
    
    double current_time() const { return current_time_; }
    Network& network() { return network_; }
    const RouterConfig& config() const { return config_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    
    /**
     * Wave parcel structure - represents a continuous wave segment
     * 
     * The wave extends from (position - wave_length) to (position)
     * where position is the leading edge.
     */
    struct WaveParcel {
        double volume;       // Total volume in parcel [m³]
        double position;     // Leading edge position [m from upstream]
        double wave_length;  // Spatial extent of wave [m]
        double celerity;     // Wave celerity [m/s]
        double rf;           // Remaining fraction (1.0 = full, 0.0 = fully exited)
    };
    
    // Active parcels in each reach
    std::unordered_map<int, std::vector<WaveParcel>> parcels_;
    
    // Downstream reach mapping for parcel transfer
    std::unordered_map<int, int> downstream_reach_;
    
    double compute_celerity(const Reach& reach, double Q);
    void initialize_topology();
    bool topology_initialized_ = false;
};

inline KWTRouter::KWTRouter(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    reset_state();
}

inline void KWTRouter::initialize_topology() {
    if (topology_initialized_) return;
    
    // Build downstream reach mapping
    downstream_reach_.clear();
    for (int reach_id : network_.topological_order()) {
        const Reach& reach = network_.get_reach(reach_id);
        downstream_reach_[reach_id] = -1;  // Default: no downstream
        
        // Find downstream reach through junction
        if (reach.downstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.downstream_junction_id);
                if (!junc.downstream_reach_ids.empty()) {
                    downstream_reach_[reach_id] = junc.downstream_reach_ids[0];
                }
            } catch (...) {}
        }
    }
    
    topology_initialized_ = true;
}

inline double KWTRouter::compute_celerity(const Reach& reach, double Q) {
    // Minimum flow for numerical stability
    if (Q < 0.001) Q = 0.001;
    
    // Get channel geometry
    double n = to_double(reach.manning_n);
    double S = reach.slope;
    if (S < 1e-6) S = 1e-6;
    
    // Hydraulic geometry: width and depth from power laws
    double width = to_double(reach.geometry.width_coef) * 
                   std::pow(Q, to_double(reach.geometry.width_exp));
    if (width < 0.5) width = 0.5;
    
    double depth = to_double(reach.geometry.depth_coef) * 
                   std::pow(Q, to_double(reach.geometry.depth_exp));
    if (depth < 0.05) depth = 0.05;
    
    // Cross-sectional area and hydraulic radius
    double area = width * depth;
    double wetted_perimeter = width + 2.0 * depth;
    double R_h = area / wetted_perimeter;
    
    // Velocity from Manning's equation
    double velocity = (1.0 / n) * std::pow(R_h, 2.0/3.0) * std::sqrt(S);
    
    // Kinematic wave celerity: c = (5/3) * v for wide rectangular channel
    // This comes from c = dQ/dA and the Manning equation
    double celerity = (5.0 / 3.0) * velocity;
    
    // Bound celerity for stability (0.1 - 5.0 m/s typical for rivers)
    return std::max(0.1, std::min(5.0, celerity));
}

inline void KWTRouter::route_timestep() {
    initialize_topology();
    
    double dt = config_.dt;
    
    // Storage for outflow rates (computed from exiting wave fractions)
    std::unordered_map<int, double> outflow_rate;
    
    // Storage for volumes to transfer downstream
    std::unordered_map<int, double> transfer_volume;
    
    for (int reach_id : network_.topological_order()) {
        outflow_rate[reach_id] = 0.0;
        transfer_volume[reach_id] = 0.0;
    }
    
    // Process reaches in topological order
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        double L = reach.length;
        
        // Collect inflow: upstream transfers + lateral inflow
        double inflow_vol = 0.0;
        
        // Get upstream contributions
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    inflow_vol += transfer_volume[up_id];
                }
            } catch (...) {}
        }
        
        // Add lateral inflow
        double lateral_vol = to_double(reach.lateral_inflow) * dt;
        inflow_vol += lateral_vol;
        
        // Create new parcel for inflow (if significant)
        if (inflow_vol > 1e-6) {
            double Q_est = inflow_vol / dt;
            double celerity = compute_celerity(reach, Q_est);
            double wave_length = celerity * dt;  // Wave extends over one timestep
            
            WaveParcel parcel;
            parcel.volume = inflow_vol;
            parcel.position = wave_length;  // Leading edge after one dt
            parcel.wave_length = wave_length;
            parcel.celerity = celerity;
            parcel.rf = 1.0;  // Fully present
            
            parcels_[reach_id].push_back(parcel);
        }
        
        // Advance existing parcels and compute outflow
        auto& reach_parcels = parcels_[reach_id];
        std::vector<WaveParcel> remaining;
        double total_outflow_vol = 0.0;
        
        for (auto& parcel : reach_parcels) {
            // Advance parcel position
            double old_position = parcel.position;
            parcel.position += parcel.celerity * dt;
            
            // Trailing edge positions
            double old_trailing = old_position - parcel.wave_length;
            double new_trailing = parcel.position - parcel.wave_length;
            
            // Compute fraction of wave that has exited the reach
            // The wave extends from trailing edge to leading edge
            
            if (new_trailing >= L) {
                // Entire wave has exited
                double exit_vol = parcel.rf * parcel.volume;
                total_outflow_vol += exit_vol;
                parcel.rf = 0.0;
                // Don't keep this parcel
            } else if (parcel.position > L) {
                // Partial exit: leading edge past outlet, trailing edge still in reach
                // Fraction exited = (position - L) / wave_length
                double fraction_past = (parcel.position - L) / parcel.wave_length;
                fraction_past = std::min(1.0, std::max(0.0, fraction_past));
                
                // Volume that exited this timestep
                // Need to track what already exited vs. what's new
                double old_fraction_past = 0.0;
                if (old_position > L) {
                    old_fraction_past = (old_position - L) / parcel.wave_length;
                    old_fraction_past = std::min(1.0, std::max(0.0, old_fraction_past));
                }
                
                double new_exit_fraction = fraction_past - old_fraction_past;
                double exit_vol = new_exit_fraction * parcel.volume;
                total_outflow_vol += exit_vol;
                
                // Update remaining fraction
                parcel.rf = 1.0 - fraction_past;
                
                if (parcel.rf > 1e-6) {
                    remaining.push_back(parcel);
                }
            } else {
                // Wave entirely within reach
                remaining.push_back(parcel);
            }
        }
        
        reach_parcels = remaining;
        
        // Compute outflow rate
        outflow_rate[reach_id] = total_outflow_vol / dt;
        
        // Store volume for downstream transfer
        transfer_volume[reach_id] = total_outflow_vol;
        
        // Update reach state
        reach.inflow_curr = Real(inflow_vol / dt);
        reach.outflow_curr = Real(outflow_rate[reach_id]);
    }
    
    current_time_ += dt;
}

inline void KWTRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) route_timestep();
}

inline void KWTRouter::set_lateral_inflow(int reach_id, double inflow) {
    network_.get_reach(reach_id).lateral_inflow = Real(inflow);
}

inline void KWTRouter::set_lateral_inflows(const std::vector<double>& inflows) {
    const auto& order = network_.topological_order();
    for (size_t i = 0; i < order.size() && i < inflows.size(); ++i) {
        network_.get_reach(order[i]).lateral_inflow = Real(inflows[i]);
    }
}

inline double KWTRouter::get_discharge(int reach_id) const {
    return to_double(network_.get_reach(reach_id).outflow_curr);
}

inline std::vector<double> KWTRouter::get_all_discharges() const {
    std::vector<double> d;
    for (int reach_id : network_.topological_order()) {
        d.push_back(to_double(network_.get_reach(reach_id).outflow_curr));
    }
    return d;
}

inline void KWTRouter::reset_state() {
    parcels_.clear();
    topology_initialized_ = false;
    for (int reach_id : network_.topological_order()) {
        parcels_[reach_id] = {};
        
        Reach& reach = network_.get_reach(reach_id);
        reach.inflow_prev = reach.inflow_curr = Real(0.0);
        reach.outflow_prev = reach.outflow_curr = Real(0.0);
    }
    current_time_ = 0.0;
}

// ============================================================================
// RouterState Implementation (Serialization/Checkpointing)
// ============================================================================

inline std::vector<char> RouterState::serialize() const {
    std::vector<char> data;
    
    // Helper to append data
    auto append = [&data](const void* ptr, size_t size) {
        const char* bytes = static_cast<const char*>(ptr);
        data.insert(data.end(), bytes, bytes + size);
    };
    
    // Write time
    append(&time, sizeof(time));
    
    // Write inflows count and data
    size_t n_inflows = inflows.size();
    append(&n_inflows, sizeof(n_inflows));
    for (const auto& [id, val] : inflows) {
        append(&id, sizeof(id));
        append(&val, sizeof(val));
    }
    
    // Write outflows count and data
    size_t n_outflows = outflows.size();
    append(&n_outflows, sizeof(n_outflows));
    for (const auto& [id, val] : outflows) {
        append(&id, sizeof(id));
        append(&val, sizeof(val));
    }
    
    // Write buffers count and data
    size_t n_buffers = buffers.size();
    append(&n_buffers, sizeof(n_buffers));
    for (const auto& [id, vec] : buffers) {
        append(&id, sizeof(id));
        size_t vec_size = vec.size();
        append(&vec_size, sizeof(vec_size));
        for (double v : vec) {
            append(&v, sizeof(v));
        }
    }
    
    return data;
}

inline RouterState RouterState::deserialize(const std::vector<char>& data) {
    RouterState state;
    size_t pos = 0;
    
    // Helper to read data
    auto read = [&data, &pos](void* ptr, size_t size) {
        std::memcpy(ptr, data.data() + pos, size);
        pos += size;
    };
    
    // Read time
    read(&state.time, sizeof(state.time));
    
    // Read inflows
    size_t n_inflows;
    read(&n_inflows, sizeof(n_inflows));
    for (size_t i = 0; i < n_inflows; ++i) {
        int id;
        double val;
        read(&id, sizeof(id));
        read(&val, sizeof(val));
        state.inflows[id] = val;
    }
    
    // Read outflows
    size_t n_outflows;
    read(&n_outflows, sizeof(n_outflows));
    for (size_t i = 0; i < n_outflows; ++i) {
        int id;
        double val;
        read(&id, sizeof(id));
        read(&val, sizeof(val));
        state.outflows[id] = val;
    }
    
    // Read buffers
    size_t n_buffers;
    read(&n_buffers, sizeof(n_buffers));
    for (size_t i = 0; i < n_buffers; ++i) {
        int id;
        read(&id, sizeof(id));
        size_t vec_size;
        read(&vec_size, sizeof(vec_size));
        std::vector<double> vec(vec_size);
        for (size_t j = 0; j < vec_size; ++j) {
            read(&vec[j], sizeof(double));
        }
        state.buffers[id] = vec;
    }
    
    return state;
}

inline bool RouterState::save(const std::string& filepath) const {
    auto data = serialize();
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;
    file.write(data.data(), data.size());
    return file.good();
}

inline RouterState RouterState::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open state file: " + filepath);
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> data(size);
    file.read(data.data(), size);
    
    return deserialize(data);
}

} // namespace dmc

#endif // DMC_ROUTE_ROUTER_HPP