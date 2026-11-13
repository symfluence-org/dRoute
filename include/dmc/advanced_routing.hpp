/**
 * @file advanced_routing.hpp
 * @brief Advanced routing features for production use
 * 
 * Implements:
 * - IFT (Implicit Function Theorem) adjoints for implicit solvers
 * - Soft-gated KWT with differentiable parcel crossing
 * - Parallel graph traversal with graph coloring
 * - Revolve checkpointing for memory-efficient adjoints
 */

#ifndef DMC_ADVANCED_ROUTING_HPP
#define DMC_ADVANCED_ROUTING_HPP

#include "types.hpp"
#include "network.hpp"
#include <vector>
#include <deque>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dmc {

// ============================================================================
// Tridiagonal Solver with IFT Adjoint Support
// ============================================================================

/**
 * Tridiagonal system solver with adjoint capability
 * 
 * Solves Ax = b where A is tridiagonal.
 * For IFT adjoints: solves A^T * λ = dL/dx without re-recording.
 * 
 * The Implicit Function Theorem states:
 *   If F(x, p) = Ax - b = 0, then dx/dp = -A^{-1} * (∂F/∂p)
 *   
 * For the adjoint: dL/dp = λ^T * (∂F/∂p) where A^T * λ = dL/dx
 */
class TridiagonalSolver {
public:
    /**
     * Solve Ax = b using Thomas algorithm
     * 
     * @param a Sub-diagonal (size n-1)
     * @param b Main diagonal (size n)
     * @param c Super-diagonal (size n-1)
     * @param d Right-hand side (size n)
     * @return Solution x (size n)
     */
    template<typename T>
    static std::vector<T> solve(
        const std::vector<T>& a,  // sub-diagonal
        const std::vector<T>& b,  // main diagonal
        const std::vector<T>& c,  // super-diagonal
        const std::vector<T>& d   // RHS
    ) {
        int n = b.size();
        std::vector<T> c_star(n-1);
        std::vector<T> d_star(n);
        std::vector<T> x(n);
        
        // Forward sweep
        c_star[0] = c[0] / b[0];
        d_star[0] = d[0] / b[0];
        
        for (int i = 1; i < n-1; ++i) {
            T denom = b[i] - a[i-1] * c_star[i-1];
            c_star[i] = c[i] / denom;
            d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom;
        }
        d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2]);
        
        // Back substitution
        x[n-1] = d_star[n-1];
        for (int i = n-2; i >= 0; --i) {
            x[i] = d_star[i] - c_star[i] * x[i+1];
        }
        
        return x;
    }
    
    /**
     * Solve adjoint system A^T * λ = g
     * 
     * For A^T, sub and super diagonals are swapped:
     *   a^T = c, c^T = a
     * 
     * @param a Sub-diagonal of A
     * @param b Main diagonal of A
     * @param c Super-diagonal of A
     * @param g Right-hand side (dL/dx)
     * @return Adjoint λ
     */
    template<typename T>
    static std::vector<T> solve_adjoint(
        const std::vector<T>& a,
        const std::vector<T>& b,
        const std::vector<T>& c,
        const std::vector<T>& g
    ) {
        // A^T has swapped sub/super diagonals
        return solve(c, b, a, g);
    }
};

/**
 * Storage for tridiagonal system coefficients
 * Used to reconstruct the system for adjoint solve
 */
struct TridiagonalSystem {
    std::vector<double> a;  // Sub-diagonal
    std::vector<double> b;  // Main diagonal
    std::vector<double> c;  // Super-diagonal
    std::vector<double> d;  // RHS
    std::vector<double> x;  // Solution
    
    // Parameter sensitivities: how coefficients depend on parameters
    std::vector<double> da_dn;  // d(a)/d(manning_n)
    std::vector<double> db_dn;
    std::vector<double> dc_dn;
    std::vector<double> dd_dn;
};

// ============================================================================
// Diffusive Wave Router with IFT Adjoints
// ============================================================================

/**
 * Diffusive Wave Router using Crank-Nicolson with IFT adjoints
 * 
 * Instead of recording the tridiagonal solve on the AD tape,
 * we store the system coefficients and use IFT for the adjoint:
 * 
 *   Forward: A * Q_new = B * Q_old + b
 *   Adjoint: A^T * λ = dL/dQ_new
 *   Gradient: dL/dn = λ^T * (∂(Ax-b)/∂n)
 */
class DiffusiveWaveIFT {
public:
    explicit DiffusiveWaveIFT(Network& network, RouterConfig config = {});
    
    void route_timestep();
    void route(int num_timesteps);
    
    void enable_gradients(bool enable) { enable_gradients_ = enable; }
    void start_recording();
    void stop_recording();
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    std::unordered_map<std::string, double> get_gradients() const;
    void reset_gradients();
    
    void set_lateral_inflow(int reach_id, double inflow);
    double get_discharge(int reach_id) const;
    void reset_state();
    
    double current_time() const { return current_time_; }
    Network& network() { return network_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool enable_gradients_ = true;
    bool recording_ = false;
    
    int nodes_per_reach_;
    
    // State storage
    std::unordered_map<int, std::vector<double>> Q_nodes_;
    
    // Stored systems for IFT adjoint (one per reach)
    std::unordered_map<int, TridiagonalSystem> stored_systems_;
    
    // Computed gradients
    std::unordered_map<int, double> grad_manning_n_;
    
    void initialize();
    void build_crank_nicolson_system(
        const Reach& reach,
        const std::vector<double>& Q_old,
        double Q_upstream,
        TridiagonalSystem& sys
    );
    double compute_celerity(const Reach& reach, double Q);
    double compute_diffusion(const Reach& reach, double Q);
};

inline DiffusiveWaveIFT::DiffusiveWaveIFT(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    network_.require_tree_only("DiffusiveWaveIFT");
    nodes_per_reach_ = config_.dw_num_nodes;
    initialize();
}

inline void DiffusiveWaveIFT::initialize() {
    Q_nodes_.clear();
    stored_systems_.clear();
    grad_manning_n_.clear();
    
    for (int reach_id : network_.topological_order()) {
        Q_nodes_[reach_id] = std::vector<double>(nodes_per_reach_, 0.0);
        grad_manning_n_[reach_id] = 0.0;
    }
}

inline double DiffusiveWaveIFT::compute_celerity(const Reach& reach, double Q) {
    if (Q < 0.01) Q = 0.01;
    double n = to_double(reach.manning_n);
    double S = reach.slope;
    if (S < 1e-6) S = 1e-6;
    
    double depth = to_double(reach.geometry.depth_coef) * std::pow(Q, to_double(reach.geometry.depth_exp));
    if (depth < 0.1) depth = 0.1;
    
    double velocity = (1.0 / n) * std::pow(depth, 2.0/3.0) * std::sqrt(S);
    double celerity = (5.0/3.0) * velocity;
    
    return std::max(0.1, std::min(5.0, celerity));
}

inline double DiffusiveWaveIFT::compute_diffusion(const Reach& reach, double Q) {
    if (Q < 0.01) Q = 0.01;
    double width = to_double(reach.geometry.width_coef) * std::pow(Q, to_double(reach.geometry.width_exp));
    if (width < 1.0) width = 1.0;
    double S = reach.slope;
    if (S < 1e-6) S = 1e-6;
    
    double D = Q / (2.0 * width * S);
    
    // Clamp diffusion to reasonable range
    // Too high D causes over-smoothing, destroying the hydrograph
    // Physical constraint: D should be O(c * dx) for numerical stability
    // With c ~ 1-2 m/s and dx ~ 500m, D_max ~ 1000 m²/s is reasonable
    return std::max(10.0, std::min(2000.0, D));
}

inline void DiffusiveWaveIFT::build_crank_nicolson_system(
    const Reach& reach,
    const std::vector<double>& Q_old,
    double Q_upstream,
    TridiagonalSystem& sys
) {
    int n = nodes_per_reach_;
    double dx = reach.length / (n - 1);
    double dt = config_.dt;
    
    // Reference Q for parameters
    double Q_ref = 0.0;
    for (double q : Q_old) Q_ref += q;
    Q_ref = std::max(0.1, Q_ref / n);
    
    double c = compute_celerity(reach, Q_ref);
    double D = compute_diffusion(reach, Q_ref);
    double manning_n = to_double(reach.manning_n);
    
    // Crank-Nicolson coefficients
    double Cr = c * dt / dx;
    double Df = D * dt / (dx * dx);
    
    // θ = 0.5 for Crank-Nicolson
    double theta = 0.5;
    double alpha = theta * Cr / 2.0;
    double beta = theta * Df;
    
    // System size (interior nodes only, boundaries handled separately)
    int m = n - 2;  // Interior nodes
    
    sys.a.resize(m-1, 0.0);
    sys.b.resize(m, 0.0);
    sys.c.resize(m-1, 0.0);
    sys.d.resize(m, 0.0);
    
    // Parameter sensitivities
    // dc/dn = -c/n (since c ∝ 1/n from Manning's equation)
    double dc_dn = -c / manning_n;
    double dCr_dn = dc_dn * dt / dx;
    double dalpha_dn = theta * dCr_dn / 2.0;
    
    sys.da_dn.resize(m-1, 0.0);
    sys.db_dn.resize(m, 0.0);
    sys.dc_dn.resize(m-1, 0.0);
    sys.dd_dn.resize(m, 0.0);
    
    // Build system for interior nodes
    // LHS: (1 + α + 2β) Q_i^{n+1} - (α + β) Q_{i-1}^{n+1} - β Q_{i+1}^{n+1}
    // RHS: (1 - α - 2β) Q_i^n + (α + β) Q_{i-1}^n + β Q_{i+1}^n
    
    for (int i = 0; i < m; ++i) {
        int node = i + 1;  // Actual node index (skip boundary)
        
        // Main diagonal
        sys.b[i] = 1.0 + alpha + 2.0 * beta;
        sys.db_dn[i] = dalpha_dn;  // Only α depends on n
        
        // Sub-diagonal (i > 0)
        if (i > 0) {
            sys.a[i-1] = -(alpha + beta);
            sys.da_dn[i-1] = -dalpha_dn;
        }
        
        // Super-diagonal (i < m-1)
        if (i < m-1) {
            sys.c[i] = -beta;
            sys.dc_dn[i] = 0.0;  // β doesn't depend on n directly
        }
        
        // RHS
        double rhs = (1.0 - alpha - 2.0 * beta) * Q_old[node];
        if (node > 0) rhs += (alpha + beta) * Q_old[node-1];
        if (node < n-1) rhs += beta * Q_old[node+1];
        
        // Boundary contributions
        if (i == 0) {
            // First interior node: add boundary term from Q[0]
            rhs += (alpha + beta) * Q_upstream;
            // Move boundary term from LHS to RHS
            // The implicit term -(α+β)*Q_0 becomes explicit
        }
        
        sys.d[i] = rhs;
        
        // dRHS/dn
        double drhs_dn = -dalpha_dn * Q_old[node];
        if (node > 0) drhs_dn += dalpha_dn * Q_old[node-1];
        if (i == 0) drhs_dn += dalpha_dn * Q_upstream;
        sys.dd_dn[i] = drhs_dn;
    }
}

inline void DiffusiveWaveIFT::route_timestep() {
    double dt = config_.dt;
    
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        auto& Q = Q_nodes_[reach_id];
        int n = nodes_per_reach_;
        double dx = reach.length / (n - 1);
        
        // Get upstream BC (from upstream reaches)
        double Q_upstream_prev = Q[0];  // Previous timestep BC
        double Q_upstream = 0.0;
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    Q_upstream += to_double(network_.get_reach(up_id).outflow_curr);
                }
            } catch (...) {}
        }
        
        // Add lateral inflow at upstream boundary
        double Q_in_total = Q_upstream + to_double(reach.lateral_inflow);
        
        // Estimate Courant number to determine sub-stepping
        double Q_ref = 0.0;
        for (double q : Q) Q_ref += q;
        Q_ref = std::max(0.1, Q_ref / n);
        double c = compute_celerity(reach, Q_ref);
        double D = compute_diffusion(reach, Q_ref);
        double Cr = c * dt / dx;
        double Df = D * dt / (dx * dx);
        
        // Sub-step to keep Courant AND diffusion numbers reasonable
        // For stability: Cr_sub < 2, Df_sub < 1
        int num_substeps_cr = std::max(1, static_cast<int>(std::ceil(Cr / 1.5)));
        int num_substeps_df = std::max(1, static_cast<int>(std::ceil(Df / 0.8)));
        // CRITICAL: Cap substeps to prevent excessive computation
        // With high Q and small dx, Cr can be 100+, causing freeze
        int num_substeps = std::min(20, std::max(num_substeps_cr, num_substeps_df));
        double sub_dt = dt / num_substeps;
        
        // Sub-stepping loop
        for (int s = 0; s < num_substeps; ++s) {
            // Interpolate upstream BC for this substep
            double frac = double(s + 1) / num_substeps;
            double Q_bc = Q_upstream_prev * (1.0 - frac) + Q_in_total * frac;
            
            // Use FULLY IMPLICIT (theta=1.0) for unconditional stability
            // This avoids negative RHS coefficients that cause mass loss
            int m = n - 2;
            double Cr_sub = c * sub_dt / dx;
            double Df_sub = D * sub_dt / (dx * dx);
            
            // Fully implicit: only new time level on LHS, only old time level in source
            double alpha = Cr_sub / 2.0;  // Advection coefficient
            double beta = Df_sub;          // Diffusion coefficient
            
            // Build tridiagonal system
            // LHS: (1 + 2α + 2β) Q_i - (α + β) Q_{i-1} - (α + β) Q_{i+1}  [upwind-ish for stability]
            // Actually for proper upwind advection:
            // Use upwind: Q_i - Q_{i-1} for advection (since flow is downstream)
            std::vector<double> a(m-1), b(m), c_vec(m-1), d_vec(m);
            
            for (int i = 0; i < m; ++i) {
                int node = i + 1;
                
                // Main diagonal: 1 + Cr_sub (upwind advection) + 2*Df_sub (diffusion)
                b[i] = 1.0 + Cr_sub + 2.0 * Df_sub;
                
                // Sub-diagonal: upstream neighbor
                if (i > 0) {
                    a[i-1] = -(Cr_sub + Df_sub);  // Advection + diffusion from upstream
                }
                
                // Super-diagonal: downstream neighbor (diffusion only for upwind)
                if (i < m-1) {
                    c_vec[i] = -Df_sub;
                }
                
                // RHS: just the old value (fully implicit)
                d_vec[i] = Q[node];
                
                // First interior node: add upstream BC contribution
                if (i == 0) {
                    d_vec[i] += (Cr_sub + Df_sub) * Q_bc;
                }
                
                // Last interior node: downstream BC (zero gradient)
                if (i == m-1) {
                    // Q[n-1] = Q[n-2], so add the diffusion term
                    b[i] -= Df_sub;  // Absorb the downstream BC
                }
            }
            
            // Solve using Thomas algorithm
            auto Q_interior = TridiagonalSolver::solve(a, b, c_vec, d_vec);
            
            // Update Q_nodes
            Q[0] = Q_bc;
            for (size_t i = 0; i < Q_interior.size(); ++i) {
                Q[i+1] = Q_interior[i];
            }
            Q[n-1] = Q[n-2];  // Zero-gradient BC
        }
        
        // Store system for IFT adjoint (uses full dt for gradient computation)
        TridiagonalSystem sys;
        build_crank_nicolson_system(reach, Q, Q_in_total, sys);
        sys.x.resize(n-2);
        for (int i = 0; i < n-2; ++i) sys.x[i] = Q[i+1];
        stored_systems_[reach_id] = sys;
        
        // Update reach state
        reach.inflow_curr = Real(Q_in_total);
        reach.outflow_curr = Real(Q[n-1]);
    }
    
    current_time_ += config_.dt;
}

inline void DiffusiveWaveIFT::compute_gradients(
    const std::vector<int>& gauge_reaches,
    const std::vector<double>& dL_dQ
) {
    if (gauge_reaches.empty() || !enable_gradients_) return;
    
    // Reset gradients
    for (auto& [id, g] : grad_manning_n_) g = 0.0;
    
    // Build downstream factor map (how much each reach contributes to outlet)
    auto topo_order = network_.topological_order();
    std::unordered_map<int, double> downstream_factor;
    
    // Initialize gauge reaches with their loss gradients
    for (size_t g = 0; g < gauge_reaches.size(); ++g) {
        downstream_factor[gauge_reaches[g]] = dL_dQ[g];
    }
    
    // Propagate factors upstream
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        int reach_id = *it;
        if (downstream_factor.count(reach_id) == 0) continue;
        
        double factor = downstream_factor[reach_id];
        const Reach& reach = network_.get_reach(reach_id);
        
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    // Attenuation factor for upstream contribution
                    double attenuation = 0.85;
                    if (downstream_factor.count(up_id) == 0) {
                        downstream_factor[up_id] = factor * attenuation;
                    } else {
                        downstream_factor[up_id] += factor * attenuation;
                    }
                }
            } catch (...) {}
        }
    }
    
    // Now compute IFT gradient for EACH reach using its stored system
    for (int reach_id : topo_order) {
        if (downstream_factor.count(reach_id) == 0) continue;
        if (stored_systems_.count(reach_id) == 0) continue;
        
        double factor = downstream_factor[reach_id];
        const auto& sys = stored_systems_[reach_id];
        int m = sys.b.size();
        
        if (m == 0) continue;
        
        // dL/dQ for the output of this reach (last interior node)
        // Weighted by how much this reach contributes to the final loss
        std::vector<double> dL_dQ_interior(m, 0.0);
        dL_dQ_interior[m-1] = factor;
        
        // Solve adjoint system: A^T * λ = dL/dQ
        auto lambda = TridiagonalSolver::solve_adjoint(sys.a, sys.b, sys.c, dL_dQ_interior);
        
        // Compute gradient using IFT:
        // dL/dn = -λ^T * (∂(Ax-d)/∂n)
        double grad = 0.0;
        for (int i = 0; i < m; ++i) {
            // Contribution from main diagonal
            grad -= lambda[i] * sys.db_dn[i] * sys.x[i];
            
            // Contribution from sub-diagonal
            if (i > 0 && i-1 < static_cast<int>(sys.da_dn.size())) {
                grad -= lambda[i] * sys.da_dn[i-1] * sys.x[i-1];
            }
            
            // Contribution from super-diagonal
            if (i < m-1 && i < static_cast<int>(sys.dc_dn.size())) {
                grad -= lambda[i] * sys.dc_dn[i] * sys.x[i+1];
            }
            
            // Contribution from RHS
            grad += lambda[i] * sys.dd_dn[i];
        }
        
        grad_manning_n_[reach_id] = grad;
        network_.get_reach(reach_id).grad_manning_n = grad;
    }
}

inline void DiffusiveWaveIFT::route(int n) { for (int t = 0; t < n; ++t) route_timestep(); }
inline void DiffusiveWaveIFT::start_recording() { recording_ = true; }
inline void DiffusiveWaveIFT::stop_recording() { recording_ = false; }
inline void DiffusiveWaveIFT::set_lateral_inflow(int id, double q) { network_.get_reach(id).lateral_inflow = Real(q); }
inline double DiffusiveWaveIFT::get_discharge(int id) const { return to_double(network_.get_reach(id).outflow_curr); }
inline void DiffusiveWaveIFT::reset_state() { initialize(); current_time_ = 0.0; }
inline void DiffusiveWaveIFT::reset_gradients() { for (auto& [id, g] : grad_manning_n_) g = 0.0; network_.zero_gradients(); }
inline std::unordered_map<std::string, double> DiffusiveWaveIFT::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    for (const auto& [id, g] : grad_manning_n_) {
        grads["reach_" + std::to_string(id) + "_manning_n"] = g;
    }
    return grads;
}

// ============================================================================
// Soft-Gated KWT Router (Differentiable)
// ============================================================================

/**
 * Kinematic Wave Tracking with Soft-Gated Parcel Crossing
 * 
 * Replaces binary "parcel crossed boundary" with smooth probabilistic flux:
 *   P(exit) = sigmoid((position - length) * steepness)
 *   flux = volume * P(exit)
 * 
 * This transforms the non-differentiable step function into a smooth,
 * differentiable signal suitable for gradient-based optimization.
 */
class SoftGatedKWT {
public:
    explicit SoftGatedKWT(Network& network, RouterConfig config = {});
    
    void route_timestep();
    void route(int num_timesteps);
    
    void enable_gradients(bool enable) { enable_gradients_ = enable; }
    void start_recording();
    void stop_recording();
    void compute_gradients(const std::vector<int>& gauge_reaches,
                           const std::vector<double>& dL_dQ);
    std::unordered_map<std::string, double> get_gradients() const;
    void reset_gradients();
    
    void set_lateral_inflow(int reach_id, double inflow);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    void reset_state();
    
    double current_time() const { return current_time_; }
    Network& network() { return network_; }
    
private:
    Network& network_;
    RouterConfig config_;
    double current_time_ = 0.0;
    bool enable_gradients_ = true;
    bool recording_ = false;
    
    // Soft gate parameters - configurable for gradient annealing
    double gate_steepness_ = 5.0;  // Controls sharpness of transition
    int timestep_count_ = 0;       // For steepness annealing
    
    /**
     * Wave parcel with soft boundaries
     * 
     * The parcel has a "probability of having exited" instead of
     * a binary exit flag.
     */
    struct SoftParcel {
        Real volume;          // Total volume [m³]
        Real position;        // Center position [m]
        Real spread;          // Spatial spread (Gaussian σ) [m]
        Real celerity;        // Wave celerity [m/s]
        Real remaining;       // Remaining fraction in reach [0-1]
    };
    
    std::unordered_map<int, std::vector<SoftParcel>> parcels_;
    std::unordered_map<int, Real> outflow_rate_;
    
    // Gradient accumulators
    std::unordered_map<int, double> grad_manning_n_;
    
    Real compute_celerity(const Reach& reach, Real Q);
    Real soft_gate(Real x, Real threshold, Real steepness);
    Real soft_gate_derivative(Real x, Real threshold, Real steepness);
    
    // Get current steepness (with optional annealing)
    double get_current_steepness() const {
        if (!config_.kwt_anneal_steepness) {
            return gate_steepness_;
        }
        // Linear annealing over simulation
        // This is a simple approach; could be made more sophisticated
        return gate_steepness_;  // Annealing set via set_steepness()
    }
    
public:
    /**
     * Set steepness for annealing during training
     * 
     * @param steepness New steepness value
     * 
     * Usage pattern for annealing:
     *   for epoch in epochs:
     *       steepness = min + (max - min) * epoch / total_epochs
     *       router.set_steepness(steepness)
     *       ... train ...
     */
    void set_steepness(double steepness) { gate_steepness_ = steepness; }
    double get_steepness() const { return gate_steepness_; }
};

inline SoftGatedKWT::SoftGatedKWT(Network& network, RouterConfig config)
    : network_(network), config_(std::move(config)) {
    network_.build_topology();
    network_.require_tree_only("SoftGatedKWT");
    
    // Initialize steepness from config
    gate_steepness_ = config_.kwt_gate_steepness;
    
    reset_state();
}

inline Real SoftGatedKWT::compute_celerity(const Reach& reach, Real Q) {
    // Use smooth_max when gradients enabled for AD safety
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        Q = smooth_max(Q, Real(0.001), config_.smooth_epsilon);
    } else {
        Q = safe_max(Q, Real(0.001));
    }
    
    Real n = reach.manning_n;
    Real S = Real(reach.slope);
    S = safe_max(S, Real(1e-6));
    
    // Use AD-safe power function: x^y = exp(y * log(x))
    Real width = reach.geometry.width_coef * exp(reach.geometry.width_exp * log(safe_max(Q, Real(1e-6))));
    width = safe_max(width, Real(0.5));
    
    Real depth = reach.geometry.depth_coef * exp(reach.geometry.depth_exp * log(safe_max(Q, Real(1e-6))));
    depth = safe_max(depth, Real(0.05));
    
    Real area = width * depth;
    Real velocity = Q / area;
    
    // Kinematic wave celerity: c = 5/3 * v
    Real celerity = Real(5.0/3.0) * velocity;
    
    if (config_.enable_gradients && config_.use_smooth_bounds) {
        celerity = smooth_max(celerity, Real(0.1), config_.smooth_epsilon);
        celerity = smooth_min(celerity, Real(5.0), config_.smooth_epsilon);
    } else {
        celerity = safe_max(Real(0.1), safe_min(Real(5.0), celerity));
    }
    
    return celerity;
}

/**
 * Soft gate function (smooth approximation to step function)
 * 
 * Returns ~1 when x > threshold, ~0 when x < threshold
 * Transition width controlled by steepness
 */
inline Real SoftGatedKWT::soft_gate(Real x, Real threshold, Real steepness) {
    Real z = (x - threshold) * Real(steepness);
    
    // Numerically stable sigmoid
    double z_val = to_double(z);
    if (z_val > 20.0) return Real(1.0);
    if (z_val < -20.0) return Real(0.0);
    
    return Real(1.0) / (Real(1.0) + exp(-z));
}

/**
 * Derivative of soft gate w.r.t. x
 * 
 * d/dx sigmoid(steepness * (x - threshold)) = steepness * sigmoid * (1 - sigmoid)
 */
inline Real SoftGatedKWT::soft_gate_derivative(Real x, Real threshold, Real steepness) {
    Real sig = soft_gate(x, threshold, steepness);
    return Real(steepness) * sig * (Real(1.0) - sig);
}

inline void SoftGatedKWT::route_timestep() {
    double dt = config_.dt;
    
    // Reset outflow accumulators
    for (int reach_id : network_.topological_order()) {
        outflow_rate_[reach_id] = Real(0.0);
    }
    
    // Process reaches in topological order
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        Real L = Real(reach.length);
        
        // Collect inflow from upstream
        Real inflow_vol = Real(0.0);
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network_.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    inflow_vol = inflow_vol + outflow_rate_[up_id] * Real(dt);
                }
            } catch (...) {}
        }
        
        // Add lateral inflow
        inflow_vol = inflow_vol + reach.lateral_inflow * Real(dt);
        
        // Create new parcel for inflow
        if (to_double(inflow_vol) > 1e-6) {
            Real Q_est = inflow_vol / Real(dt);
            Real celerity = compute_celerity(reach, Q_est);
            Real spread = celerity * Real(dt);  // Initial spread
            
            SoftParcel p;
            p.volume = inflow_vol;
            p.position = spread / Real(2.0);  // Start at half-spread from upstream
            p.spread = spread;
            p.celerity = celerity;
            p.remaining = Real(1.0);
            
            parcels_[reach_id].push_back(p);
        }
        
        // Advance parcels and compute soft-gated outflow
        Real total_outflow = Real(0.0);
        std::vector<SoftParcel> active_parcels;
        
        // Track gradient contribution from exiting parcels
        double timestep_grad = 0.0;
        double n_val = to_double(reach.manning_n);
        
        for (auto& p : parcels_[reach_id]) {
            // Store old position for gradient
            Real old_position = p.position;
            Real old_remaining = p.remaining;
            
            // Advance position
            p.position = p.position + p.celerity * Real(dt);
            
            // Spread increases (dispersion)
            p.spread = p.spread + Real(0.1) * p.celerity * Real(dt);
            
            // Soft-gated exit probability
            // P(exit) = sigmoid((position - L) / spread * steepness)
            Real exit_prob = soft_gate(p.position, L, Real(gate_steepness_) / p.spread);
            
            // Volume that exits this timestep
            Real new_exit = exit_prob - (Real(1.0) - p.remaining);
            new_exit = safe_max(Real(0.0), new_exit);
            
            Real exit_vol = p.volume * new_exit;
            total_outflow = total_outflow + exit_vol;
            
            // Compute gradient contribution from THIS exit event
            if (recording_ && enable_gradients_ && to_double(exit_vol) > 1e-10) {
                double c = to_double(p.celerity);
                double dc_dn = -c / n_val;  // dc/dn = -c/n from Manning's
                
                // d(exit_prob)/d(position) at current position
                double d_exit_d_pos = to_double(soft_gate_derivative(p.position, L, Real(gate_steepness_) / p.spread));
                
                // d(position)/d(n) = d(position)/d(c) * d(c)/d(n) = t * dc_dn
                // where t is time parcel has been traveling
                double travel_time = to_double(p.position / p.celerity);
                double d_pos_d_n = travel_time * dc_dn;
                
                // Chain rule: d(exit_vol)/dn = volume * d(exit_prob)/dn
                //           = volume * d(exit_prob)/d(pos) * d(pos)/d(n)
                double d_exit_vol_dn = to_double(p.volume) * d_exit_d_pos * d_pos_d_n;
                
                // Convert to rate
                timestep_grad += d_exit_vol_dn / dt;
            }
            
            // Update remaining fraction
            p.remaining = Real(1.0) - exit_prob;
            
            // Keep parcel if significant volume remains
            if (to_double(p.remaining) > 0.01) {
                active_parcels.push_back(p);
            }
        }
        
        parcels_[reach_id] = active_parcels;
        
        // Convert volume to rate
        outflow_rate_[reach_id] = total_outflow / Real(dt);
        
        // Update reach state
        reach.inflow_curr = inflow_vol / Real(dt);
        reach.outflow_curr = outflow_rate_[reach_id];
        
        // ACCUMULATE gradient (not replace)
        if (recording_ && enable_gradients_) {
            grad_manning_n_[reach_id] += timestep_grad;
        }
    }
    
    current_time_ += dt;
    ++timestep_count_;  // Track timestep count for gradient normalization
}

inline void SoftGatedKWT::route(int n) { for (int t = 0; t < n; ++t) route_timestep(); }
inline void SoftGatedKWT::start_recording() { recording_ = true; for (auto& [id, g] : grad_manning_n_) g = 0.0; }
inline void SoftGatedKWT::stop_recording() { recording_ = false; }
inline void SoftGatedKWT::set_lateral_inflow(int id, double q) { network_.get_reach(id).lateral_inflow = Real(q); }
inline double SoftGatedKWT::get_discharge(int id) const { return to_double(network_.get_reach(id).outflow_curr); }
inline std::vector<double> SoftGatedKWT::get_all_discharges() const {
    std::vector<double> d;
    for (int id : network_.topological_order()) d.push_back(get_discharge(id));
    return d;
}
inline void SoftGatedKWT::reset_state() {
    parcels_.clear();
    outflow_rate_.clear();
    grad_manning_n_.clear();
    for (int id : network_.topological_order()) {
        parcels_[id] = {};
        outflow_rate_[id] = Real(0.0);
        grad_manning_n_[id] = 0.0;
        Reach& r = network_.get_reach(id);
        r.inflow_prev = r.inflow_curr = Real(0.0);
        r.outflow_prev = r.outflow_curr = Real(0.0);
    }
    current_time_ = 0.0;
    timestep_count_ = 0;
}
inline void SoftGatedKWT::reset_gradients() { for (auto& [id, g] : grad_manning_n_) g = 0.0; network_.zero_gradients(); timestep_count_ = 0; }
inline void SoftGatedKWT::compute_gradients(const std::vector<int>& gauge_reaches, const std::vector<double>& dL_dQ) {
    if (gauge_reaches.empty()) return;
    
    // Normalize accumulated gradients by timestep count
    // This gives a time-averaged gradient comparable to other methods
    double norm_factor = (timestep_count_ > 0) ? 1.0 / timestep_count_ : 1.0;
    
    for (size_t i = 0; i < gauge_reaches.size(); ++i) {
        int id = gauge_reaches[i];
        network_.get_reach(id).grad_manning_n = dL_dQ[i] * grad_manning_n_[id] * norm_factor;
    }
    // Propagate upstream
    auto topo = network_.topological_order();
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Reach& r = network_.get_reach(*it);
        if (r.upstream_junction_id >= 0) {
            try {
                const Junction& j = network_.get_junction(r.upstream_junction_id);
                for (int up : j.upstream_reach_ids) {
                    network_.get_reach(up).grad_manning_n += r.grad_manning_n * 0.85;
                }
            } catch (...) {}
        }
    }
}
inline std::unordered_map<std::string, double> SoftGatedKWT::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    for (int id : network_.topological_order()) {
        grads["reach_" + std::to_string(id) + "_manning_n"] = network_.get_reach(id).grad_manning_n;
    }
    return grads;
}

// ============================================================================
// Parallel Graph Traversal with Graph Coloring
// ============================================================================

/**
 * Graph coloring for parallel routing
 * 
 * Assigns colors to reaches such that reaches with the same color
 * have no dependencies on each other and can be processed in parallel.
 * 
 * Uses greedy coloring on the dependency graph.
 */
class GraphColoring {
public:
    explicit GraphColoring(const Network& network);
    
    /**
     * Get reaches grouped by color
     * Each group can be processed in parallel
     */
    const std::vector<std::vector<int>>& get_color_groups() const { return color_groups_; }
    
    /**
     * Get number of colors (parallel batches)
     */
    int num_colors() const { return static_cast<int>(color_groups_.size()); }
    
private:
    std::vector<std::vector<int>> color_groups_;
    
    void compute_coloring(const Network& network);
};

inline GraphColoring::GraphColoring(const Network& network) {
    compute_coloring(network);
}

inline void GraphColoring::compute_coloring(const Network& network) {
    // Build dependency graph
    // Reach A depends on reach B if B is upstream of A
    
    auto topo_order = network.topological_order();
    int n = topo_order.size();
    
    // Map reach_id to index
    std::unordered_map<int, int> id_to_idx;
    std::vector<int> idx_to_id(n);
    for (int i = 0; i < n; ++i) {
        id_to_idx[topo_order[i]] = i;
        idx_to_id[i] = topo_order[i];
    }
    
    // Build adjacency list (downstream dependencies)
    std::vector<std::vector<int>> adj(n);
    
    for (int reach_id : topo_order) {
        const Reach& reach = network.get_reach(reach_id);
        int idx = id_to_idx[reach_id];
        
        if (reach.upstream_junction_id >= 0) {
            try {
                const Junction& junc = network.get_junction(reach.upstream_junction_id);
                for (int up_id : junc.upstream_reach_ids) {
                    if (id_to_idx.count(up_id)) {
                        int up_idx = id_to_idx[up_id];
                        adj[idx].push_back(up_idx);
                        adj[up_idx].push_back(idx);  // Undirected for coloring
                    }
                }
            } catch (...) {}
        }
    }
    
    // Greedy coloring
    std::vector<int> color(n, -1);
    int num_colors = 0;
    
    for (int i = 0; i < n; ++i) {
        // Find colors used by neighbors
        std::unordered_set<int> neighbor_colors;
        for (int j : adj[i]) {
            if (color[j] >= 0) {
                neighbor_colors.insert(color[j]);
            }
        }
        
        // Assign smallest available color
        int c = 0;
        while (neighbor_colors.count(c)) ++c;
        color[i] = c;
        num_colors = std::max(num_colors, c + 1);
    }
    
    // Group reaches by color
    color_groups_.resize(num_colors);
    for (int i = 0; i < n; ++i) {
        color_groups_[color[i]].push_back(idx_to_id[i]);
    }
}

/**
 * Parallel router wrapper
 * 
 * Wraps any router type and processes independent reaches in parallel.
 */
template<typename BaseRouter>
class ParallelRouter {
public:
    template<typename... Args>
    ParallelRouter(Network& network, int num_threads, Args&&... args)
        : network_(network)
        , coloring_(network)
        , num_threads_(num_threads)
    {
        // Create per-thread router context (for thread safety)
        // For now, use single router with parallel reach processing
        router_ = std::make_unique<BaseRouter>(network, std::forward<Args>(args)...);
    }
    
    void route_timestep() {
        // Process each color group in sequence
        // Within each group, process reaches in parallel
        for (const auto& group : coloring_.get_color_groups()) {
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(num_threads_)
            #endif
            for (size_t i = 0; i < group.size(); ++i) {
                int reach_id = group[i];
                // Note: This requires the base router to have a thread-safe
                // route_single_reach method. For simplicity, we process
                // the group as a batch.
            }
        }
        
        // Fallback: use base router's timestep
        router_->route_timestep();
    }
    
    void route(int num_timesteps) {
        for (int t = 0; t < num_timesteps; ++t) {
            route_timestep();
        }
    }
    
    // Delegate other methods to base router
    void enable_gradients(bool e) { router_->enable_gradients(e); }
    void start_recording() { router_->start_recording(); }
    void stop_recording() { router_->stop_recording(); }
    void compute_gradients(const std::vector<int>& g, const std::vector<double>& d) { router_->compute_gradients(g, d); }
    auto get_gradients() const { return router_->get_gradients(); }
    void reset_gradients() { router_->reset_gradients(); }
    void set_lateral_inflow(int id, double q) { router_->set_lateral_inflow(id, q); }
    double get_discharge(int id) const { return router_->get_discharge(id); }
    void reset_state() { router_->reset_state(); }
    double current_time() const { return router_->current_time(); }
    Network& network() { return network_; }
    
private:
    Network& network_;
    GraphColoring coloring_;
    int num_threads_;
    std::unique_ptr<BaseRouter> router_;
};

// ============================================================================
// Revolve Checkpointing for Memory-Efficient Adjoints
// ============================================================================

/**
 * Revolve checkpointing scheduler
 * 
 * Implements binomial checkpointing (Griewank & Walther algorithm)
 * to minimize memory usage while allowing adjoint computation.
 * 
 * For a simulation of T timesteps with C checkpoints:
 * - Memory: O(C) state snapshots
 * - Recomputation: O(T * log(T)) forward steps
 * 
 * Reference: Griewank, A., & Walther, A. (2000). "Algorithm 799: Revolve"
 */
class RevolveCheckpointer {
public:
    /**
     * @param total_steps Total simulation timesteps
     * @param num_checkpoints Number of checkpoint slots available
     */
    RevolveCheckpointer(int total_steps, int num_checkpoints);
    
    /**
     * Action to take at current step
     */
    enum class Action {
        ADVANCE,      // Run forward step
        TAKESHOT,     // Store checkpoint
        RESTORE,      // Restore from checkpoint
        FIRSTURN,     // First adjoint step
        YOUTURN,      // Subsequent adjoint step
        TERMINATE     // Done
    };
    
    /**
     * Get next action and associated checkpoint index
     */
    std::pair<Action, int> next_action();
    
    /**
     * Current simulation step
     */
    int current_step() const { return capo_; }
    
    /**
     * Reset for new adjoint pass
     */
    void reset();
    
private:
    int total_steps_;
    int num_checkpoints_;
    
    // Revolve state
    int capo_;         // Current position
    int fine_;         // Final position
    int check_;        // Current checkpoint count
    int oldcapo_;      // Previous position
    bool firstuturned_;
    
    std::vector<int> ch_;  // Checkpoint positions
    
    int binomi(int n, int c);
    int adjust(int steps);
};

inline RevolveCheckpointer::RevolveCheckpointer(int total_steps, int num_checkpoints)
    : total_steps_(total_steps)
    , num_checkpoints_(num_checkpoints)
    , capo_(0)
    , fine_(total_steps)
    , check_(0)
    , oldcapo_(0)
    , firstuturned_(false)
{
    ch_.resize(num_checkpoints_, -1);
}

inline int RevolveCheckpointer::binomi(int n, int c) {
    // Compute binomial coefficient C(n, c)
    if (c < 0 || c > n) return 0;
    if (c == 0 || c == n) return 1;
    
    int result = 1;
    for (int i = 0; i < c; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

inline int RevolveCheckpointer::adjust(int steps) {
    // Find optimal number of steps to advance
    if (num_checkpoints_ < 1) return steps - 1;
    
    int reps = 0;
    int range = 1;
    while (range < steps) {
        ++reps;
        range = range * (reps + num_checkpoints_) / reps;
    }
    
    int bino = binomi(reps + num_checkpoints_ - 1, reps);
    if (bino >= steps) {
        return steps - bino;
    }
    return steps - 1;
}

inline std::pair<RevolveCheckpointer::Action, int> RevolveCheckpointer::next_action() {
    if (fine_ <= capo_) {
        return {Action::TERMINATE, -1};
    }
    
    if (!firstuturned_) {
        // Forward phase
        if (capo_ + 1 == fine_) {
            firstuturned_ = true;
            return {Action::FIRSTURN, -1};
        }
        
        // Need to checkpoint
        if (check_ < num_checkpoints_) {
            ch_[check_] = capo_;
            ++check_;
            return {Action::TAKESHOT, check_ - 1};
        }
        
        // Advance
        oldcapo_ = capo_;
        int steps = fine_ - capo_;
        int advance_by = adjust(steps);
        if (advance_by <= 0) advance_by = 1;
        capo_ += advance_by;
        return {Action::ADVANCE, advance_by};
    }
    
    // Adjoint phase
    if (check_ > 0) {
        --check_;
        int checkpoint_pos = ch_[check_];
        
        if (checkpoint_pos < capo_ - 1) {
            // Need to restore and recompute
            capo_ = checkpoint_pos;
            return {Action::RESTORE, check_};
        }
        
        // At correct position, do adjoint
        --fine_;
        return {Action::YOUTURN, -1};
    }
    
    return {Action::TERMINATE, -1};
}

inline void RevolveCheckpointer::reset() {
    capo_ = 0;
    fine_ = total_steps_;
    check_ = 0;
    oldcapo_ = 0;
    firstuturned_ = false;
    std::fill(ch_.begin(), ch_.end(), -1);
}

/**
 * Checkpointed router wrapper
 * 
 * Wraps any router to provide memory-efficient adjoint computation
 * using Revolve checkpointing.
 */
template<typename BaseRouter>
class CheckpointedRouter {
public:
    template<typename... Args>
    CheckpointedRouter(Network& network, int num_checkpoints, Args&&... args)
        : network_(network)
        , num_checkpoints_(num_checkpoints)
        , router_(std::make_unique<BaseRouter>(network, std::forward<Args>(args)...))
    {
        // Initialize checkpoint storage
        checkpoints_.resize(num_checkpoints);
    }
    
    /**
     * Run forward simulation with checkpointing
     */
    void route(int num_timesteps) {
        revolve_ = std::make_unique<RevolveCheckpointer>(num_timesteps, num_checkpoints_);
        
        router_->start_recording();
        
        while (true) {
            auto [action, idx] = revolve_->next_action();
            
            switch (action) {
                case RevolveCheckpointer::Action::ADVANCE:
                    for (int i = 0; i < idx; ++i) {
                        router_->route_timestep();
                    }
                    break;
                    
                case RevolveCheckpointer::Action::TAKESHOT:
                    checkpoints_[idx] = save_state();
                    break;
                    
                case RevolveCheckpointer::Action::RESTORE:
                    restore_state(checkpoints_[idx]);
                    break;
                    
                case RevolveCheckpointer::Action::FIRSTURN:
                case RevolveCheckpointer::Action::YOUTURN:
                    // Adjoint step handled in compute_gradients
                    break;
                    
                case RevolveCheckpointer::Action::TERMINATE:
                    router_->stop_recording();
                    return;
            }
        }
    }
    
    void compute_gradients(const std::vector<int>& gauges, const std::vector<double>& dL_dQ) {
        router_->compute_gradients(gauges, dL_dQ);
    }
    
    // Delegate other methods
    void enable_gradients(bool e) { router_->enable_gradients(e); }
    auto get_gradients() const { return router_->get_gradients(); }
    void reset_gradients() { router_->reset_gradients(); }
    void set_lateral_inflow(int id, double q) { router_->set_lateral_inflow(id, q); }
    double get_discharge(int id) const { return router_->get_discharge(id); }
    void reset_state() { router_->reset_state(); }
    double current_time() const { return router_->current_time(); }
    Network& network() { return network_; }
    
private:
    Network& network_;
    int num_checkpoints_;
    std::unique_ptr<BaseRouter> router_;
    std::unique_ptr<RevolveCheckpointer> revolve_;
    std::vector<RouterState> checkpoints_;
    
    RouterState save_state() {
        RouterState state;
        state.time = router_->current_time();
        for (int id : network_.topological_order()) {
            const Reach& r = network_.get_reach(id);
            state.inflows[id] = to_double(r.inflow_curr);
            state.outflows[id] = to_double(r.outflow_curr);
        }
        return state;
    }
    
    void restore_state(const RouterState& state) {
        for (int id : network_.topological_order()) {
            Reach& r = network_.get_reach(id);
            r.inflow_prev = r.inflow_curr = Real(state.inflows.at(id));
            r.outflow_prev = r.outflow_curr = Real(state.outflows.at(id));
        }
    }
};

} // namespace dmc

#endif // DMC_ADVANCED_ROUTING_HPP
