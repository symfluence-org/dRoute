#ifndef DMC_SAINT_VENANT_ROUTER_HPP
#define DMC_SAINT_VENANT_ROUTER_HPP

/**
 * @file saint_venant_router.hpp
 * @brief Full dynamic Saint-Venant Equations solver using SUNDIALS CVODES
 *
 * Solves the 1D shallow water equations (Saint-Venant):
 *
 *   Continuity:  ∂A/∂t + ∂Q/∂x = q_lat
 *   Momentum:    ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
 *
 * where:
 *   A = cross-sectional area [m²]
 *   Q = discharge [m³/s]
 *   h = water depth [m]
 *   q_lat = lateral inflow per unit length [m²/s]
 *   S₀ = bed slope [-]
 *   Sf = friction slope (Manning) [-]
 *   g = gravitational acceleration [m/s²]
 *
 * Spatial discretization: Finite Volume with Rusanov flux
 * Time integration: CVODES (implicit BDF with Newton iteration)
 * Gradients: Finite difference approximation (for proper CVODES adjoint
 *            sensitivity with Enzyme AD, use SaintVenantEnzyme instead)
 */

#include "network.hpp"
#include "types.hpp"

#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <memory>
#include <iostream>

// SUNDIALS headers
#ifdef DMC_ENABLE_SUNDIALS
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_context.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL 0
#endif
#endif

namespace dmc {

// ============================================================================
// Configuration
// ============================================================================

struct SaintVenantConfig {
    double dt = 3600.0;              // Output timestep [s]
    int n_nodes = 10;                // Spatial nodes per reach
    double g = 9.81;                 // Gravitational acceleration [m/s²]
    
    // Numerical parameters
    double rel_tol = 1e-4;           // Relative tolerance for CVODES
    double abs_tol = 1e-6;           // Absolute tolerance
    int max_steps = 5000;            // Max internal steps per output step
    
    // Initial conditions
    double initial_depth = 0.5;      // Initial water depth [m]
    double initial_velocity = 0.1;   // Initial velocity [m/s]
    
    // Stability
    double min_depth = 0.01;         // Minimum depth to prevent dry bed [m]
    double min_area = 0.1;           // Minimum area [m²]
    
    // Gradient computation
    bool enable_adjoint = true;      // Use adjoint for gradients
    int checkpoint_stride = 100;     // Checkpoint interval for adjoint
};

// ============================================================================
// SVE Channel Geometry Helper (separate from network's ChannelGeometry)
// ============================================================================

struct SVEGeometry {
    double width_coef = 10.0;    // Width = a * Q^b
    double width_exp = 0.3;
    double bed_slope = 0.001;
    double manning_n = 0.035;
    
    // Compute width from area (assuming trapezoidal with 1:1 side slopes)
    inline double width_from_area(double A) const {
        // For rectangular: W = A / h, h = A / W
        // Simple approximation: W ≈ sqrt(A) * factor
        return std::max(1.0, width_coef * std::pow(std::max(A, 0.1), 0.4));
    }
    
    // Compute depth from area
    inline double depth_from_area(double A, double W) const {
        return std::max(0.01, A / W);
    }
    
    // Compute hydraulic radius
    inline double hydraulic_radius(double A, double W, double h) const {
        double P = W + 2.0 * h;  // Wetted perimeter (rectangular)
        return A / std::max(P, 0.1);
    }
    
    // Compute friction slope (Manning)
    inline double friction_slope(double Q, double A, double R) const {
        double n = manning_n;
        double v = Q / std::max(A, 0.1);
        // Sf = (n * |v|)² / R^(4/3)
        return n * n * v * std::abs(v) / std::pow(std::max(R, 0.01), 4.0/3.0);
    }
};

// ============================================================================
// Saint-Venant State
// ============================================================================

/**
 * State for a single reach: A and Q at each node
 */
struct ReachState {
    std::vector<double> A;  // Area at each node [m²]
    std::vector<double> Q;  // Discharge at each node [m³/s]
    
    void resize(int n_nodes) {
        A.resize(n_nodes, 0.0);
        Q.resize(n_nodes, 0.0);
    }
};

// ============================================================================
// SaintVenantRouter Class
// ============================================================================

class SaintVenantRouter {
public:
    SaintVenantRouter(Network& network, SaintVenantConfig config = {});
    ~SaintVenantRouter();
    
    // Prevent copying (SUNDIALS handles are not copyable)
    SaintVenantRouter(const SaintVenantRouter&) = delete;
    SaintVenantRouter& operator=(const SaintVenantRouter&) = delete;
    
    // =========== Core Routing ===========
    
    /**
     * Route entire network for one timestep.
     * Uses CVODES to integrate SVE from t to t + dt.
     */
    void route_timestep();
    
    /**
     * Route for multiple timesteps.
     */
    void route(int num_timesteps);
    
    // =========== State Management ===========
    
    /**
     * Set lateral inflow for a reach [m³/s].
     * Distributed uniformly along reach length.
     */
    void set_lateral_inflow(int reach_id, double inflow);
    
    /**
     * Get current discharge at reach outlet [m³/s].
     */
    double get_discharge(int reach_id) const;
    
    /**
     * Get discharge at all reaches.
     */
    std::vector<double> get_all_discharges() const;
    
    /**
     * Get water depth at reach outlet [m].
     */
    double get_depth(int reach_id) const;
    
    /**
     * Reset state to initial conditions.
     */
    void reset_state();
    
    // =========== Gradient Computation ===========
    
    /**
     * Enable gradient recording for adjoint computation.
     */
    void start_recording();
    
    /**
     * Stop recording.
     */
    void stop_recording();
    
    /**
     * Record current output for gradient computation.
     */
    void record_output(int reach_id);
    
    /**
     * Compute gradients via CVODES adjoint.
     */
    void compute_gradients_timeseries(int reach_id, 
                                      const std::vector<double>& dL_dQ);
    
    /**
     * Get gradients for all parameters.
     */
    std::unordered_map<std::string, double> get_gradients() const;
    
    // =========== Access ===========
    
    const SaintVenantConfig& config() const { return config_; }
    Network& network() { return network_; }
    double current_time() const { return current_time_; }
    
private:
    Network& network_;
    SaintVenantConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    // State: one ReachState per reach
    std::vector<ReachState> reach_states_;
    std::vector<SVEGeometry> reach_geometry_;
    std::vector<double> lateral_inflows_;  // [m³/s] per reach
    // Per-timestep snapshot of lateral_inflows_ during recording, so the finite-
    // difference gradient can replay the exact time-varying forcing.
    std::vector<std::vector<double>> lateral_history_;
    
    // Output history for gradient computation
    std::unordered_map<int, std::vector<double>> output_history_;
    
    // Gradients (accumulated)
    std::vector<double> grad_manning_n_;
    
    // Mapping from reach index to state vector index
    std::vector<int> reach_state_offset_;
    int total_state_size_ = 0;
    
    // Internal methods
    void initialize_state();
    void pack_state(double* y) const;
    void unpack_state(const double* y);
    
    // RHS function: dy/dt = f(t, y)
    void compute_rhs(double t, const double* y, double* ydot);
    
    // Jacobian: df/dy (for implicit solver)
    void compute_jacobian(double t, const double* y, double* J);
    
    // Flux computation (Rusanov/Local Lax-Friedrichs)
    void compute_flux(double A_L, double Q_L, double A_R, double Q_R,
                     const SVEGeometry& geom, double dx,
                     double& F_A, double& F_Q);
    
#ifdef DMC_ENABLE_SUNDIALS
    // SUNDIALS objects
    void* cvode_mem_ = nullptr;
    N_Vector y_ = nullptr;
    SUNMatrix J_ = nullptr;
    SUNLinearSolver LS_ = nullptr;
    SUNContext sunctx_ = nullptr;
    
    // CVODES callbacks (static, use user_data to access this)
    static int rhs_callback(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    static int jac_callback(sunrealtype t, N_Vector y, N_Vector fy, 
                           SUNMatrix J, void* user_data,
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
#endif
};

// ============================================================================
// Implementation
// ============================================================================

inline SaintVenantRouter::SaintVenantRouter(Network& network, SaintVenantConfig config)
    : network_(network), config_(std::move(config)) {
    
    network_.build_topology();
    
    int n_reaches = network_.topological_order().size();
    int n_nodes = config_.n_nodes;
    
    // Allocate state for each reach
    reach_states_.resize(n_reaches);
    reach_geometry_.resize(n_reaches);
    lateral_inflows_.resize(n_reaches, 0.0);
    grad_manning_n_.resize(n_reaches, 0.0);
    reach_state_offset_.resize(n_reaches);
    
    // Build state vector mapping
    int offset = 0;
    for (int i = 0; i < n_reaches; ++i) {
        reach_state_offset_[i] = offset;
        reach_states_[i].resize(n_nodes);
        offset += 2 * n_nodes;  // A and Q for each node
    }
    total_state_size_ = offset;
    
    // Initialize geometry from network reaches
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches; ++i) {
        const Reach& reach = network_.get_reach(topo_order[i]);
        reach_geometry_[i].bed_slope = reach.slope;
        reach_geometry_[i].manning_n = to_double(reach.manning_n);
        reach_geometry_[i].width_coef = to_double(reach.geometry.width_coef);
        reach_geometry_[i].width_exp = to_double(reach.geometry.width_exp);
    }
    
    // Initialize state
    initialize_state();
    
#ifdef DMC_ENABLE_SUNDIALS
    // Create SUNDIALS context
    int flag = SUNContext_Create(SUN_COMM_NULL, &sunctx_);
    if (flag != 0) {
        std::cerr << "Error creating SUNDIALS context" << std::endl;
        return;
    }
    
    // Create state vector
    y_ = N_VNew_Serial(total_state_size_, sunctx_);
    pack_state(N_VGetArrayPointer(y_));
    
    // Create CVODES solver (BDF for stiff problems)
    cvode_mem_ = CVodeCreate(CV_BDF, sunctx_);
    
    // Initialize with RHS function
    flag = CVodeInit(cvode_mem_, rhs_callback, 0.0, y_);
    
    // Set tolerances
    flag = CVodeSStolerances(cvode_mem_, config_.rel_tol, config_.abs_tol);
    
    // Set user data (this pointer)
    flag = CVodeSetUserData(cvode_mem_, this);
    
    // Create dense matrix and linear solver
    J_ = SUNDenseMatrix(total_state_size_, total_state_size_, sunctx_);
    LS_ = SUNLinSol_Dense(y_, J_, sunctx_);
    
    // Attach linear solver
    flag = CVodeSetLinearSolver(cvode_mem_, LS_, J_);
    
    // Set Jacobian function
    flag = CVodeSetJacFn(cvode_mem_, jac_callback);
    
    // Set max steps
    flag = CVodeSetMaxNumSteps(cvode_mem_, config_.max_steps);
#else
    std::cerr << "Warning: SUNDIALS not enabled, using fallback RK4 solver (slower)" << std::endl;
#endif
}

inline SaintVenantRouter::~SaintVenantRouter() {
#ifdef DMC_ENABLE_SUNDIALS
    if (LS_) SUNLinSolFree(LS_);
    if (J_) SUNMatDestroy(J_);
    if (y_) N_VDestroy(y_);
    if (cvode_mem_) CVodeFree(&cvode_mem_);
    if (sunctx_) SUNContext_Free(&sunctx_);
#endif
}

inline void SaintVenantRouter::initialize_state() {
    int n_nodes = config_.n_nodes;
    double h0 = config_.initial_depth;
    double v0 = config_.initial_velocity;
    
    for (size_t i = 0; i < reach_states_.size(); ++i) {
        const SVEGeometry& geom = reach_geometry_[i];
        double W = geom.width_coef * std::pow(h0, 0.5);  // Approximate width
        double A0 = W * h0;
        double Q0 = A0 * v0;
        
        for (int j = 0; j < n_nodes; ++j) {
            reach_states_[i].A[j] = A0;
            reach_states_[i].Q[j] = Q0;
        }
    }
}

inline void SaintVenantRouter::pack_state(double* y) const {
    int n_nodes = config_.n_nodes;
    for (size_t i = 0; i < reach_states_.size(); ++i) {
        int off = reach_state_offset_[i];
        for (int j = 0; j < n_nodes; ++j) {
            y[off + 2*j] = reach_states_[i].A[j];
            y[off + 2*j + 1] = reach_states_[i].Q[j];
        }
    }
}

inline void SaintVenantRouter::unpack_state(const double* y) {
    int n_nodes = config_.n_nodes;
    for (size_t i = 0; i < reach_states_.size(); ++i) {
        int off = reach_state_offset_[i];
        for (int j = 0; j < n_nodes; ++j) {
            reach_states_[i].A[j] = std::max(y[off + 2*j], config_.min_area);
            reach_states_[i].Q[j] = y[off + 2*j + 1];
        }
    }
}

inline void SaintVenantRouter::compute_flux(
    double A_L, double Q_L, double A_R, double Q_R,
    const SVEGeometry& geom, double dx,
    double& F_A, double& F_Q) {
    
    double g = config_.g;
    
    // Ensure positive area
    A_L = std::max(A_L, config_.min_area);
    A_R = std::max(A_R, config_.min_area);
    
    // Compute velocities
    double u_L = Q_L / A_L;
    double u_R = Q_R / A_R;
    
    // Compute widths and depths
    double W_L = geom.width_from_area(A_L);
    double W_R = geom.width_from_area(A_R);
    double h_L = geom.depth_from_area(A_L, W_L);
    double h_R = geom.depth_from_area(A_R, W_R);
    
    // Wave speeds (characteristic speeds)
    double c_L = std::sqrt(g * h_L);
    double c_R = std::sqrt(g * h_R);
    
    // Maximum wave speed (for Rusanov flux)
    double a_max = std::max(std::abs(u_L) + c_L, std::abs(u_R) + c_R);
    
    // Physical fluxes
    // F(U) = [Q, Q²/A + g*A*h/2]
    double F_A_L = Q_L;
    double F_A_R = Q_R;
    double F_Q_L = Q_L * u_L + 0.5 * g * A_L * h_L;
    double F_Q_R = Q_R * u_R + 0.5 * g * A_R * h_R;
    
    // Rusanov flux: F = 0.5*(F_L + F_R) - 0.5*a_max*(U_R - U_L)
    F_A = 0.5 * (F_A_L + F_A_R) - 0.5 * a_max * (A_R - A_L);
    F_Q = 0.5 * (F_Q_L + F_Q_R) - 0.5 * a_max * (Q_R - Q_L);
}

inline void SaintVenantRouter::compute_rhs(double t, const double* y, double* ydot) {
    // Unpack state temporarily
    int n_nodes = config_.n_nodes;
    int n_reaches = reach_states_.size();
    double g = config_.g;
    
    const auto& topo_order = network_.topological_order();
    
    // Process each reach
    for (int r = 0; r < n_reaches; ++r) {
        const Reach& reach = network_.get_reach(topo_order[r]);
        const SVEGeometry& geom = reach_geometry_[r];
        double dx = reach.length / (n_nodes - 1);
        int off = reach_state_offset_[r];
        
        // Lateral inflow as a per-cell source. The continuity source is applied at all
        // n_nodes cells of width dx, so dividing by reach.length over-integrates the
        // source by n_nodes/(n_nodes-1) (the n_nodes cells span n_nodes*dx, not length).
        // Normalize by the discretized length n_nodes*dx so the source integrates to
        // exactly lateral_inflows_[r] and mass is conserved.
        double q_lat = lateral_inflows_[r] / (n_nodes * dx);
        
        // Boundary condition: upstream inflow
        double Q_upstream = 0.0;
        
        // Sum upstream reach outflows
        Junction& up_junc = network_.get_junction(reach.upstream_junction_id);
        for (int up_reach_id : up_junc.upstream_reach_ids) {
            // Find upstream reach in our arrays
            for (int ur = 0; ur < n_reaches; ++ur) {
                if (topo_order[ur] == up_reach_id) {
                    // Get outlet Q from upstream reach (last node)
                    int up_off = reach_state_offset_[ur];
                    Q_upstream += y[up_off + 2*(n_nodes-1) + 1];
                    break;
                }
            }
        }
        
        // NOTE: lateral inflow enters ONLY through the distributed continuity source
        // q_lat below (which integrates to the full lateral_inflows_[r] over the reach).
        // The inlet boundary carries upstream reach outflow only. Previously a spurious
        // "0.5 * lateral at the inlet" term was added here on top of the distributed
        // source, double-counting lateral inflow (~1.5x per reach, compounding downstream)
        // and breaking mass conservation.

        // Interior nodes: finite volume update
        for (int j = 0; j < n_nodes; ++j) {
            double A_j = std::max(y[off + 2*j], config_.min_area);
            double Q_j = y[off + 2*j + 1];
            
            // Left and right states for flux
            double A_L, Q_L, A_R, Q_R;
            
            if (j == 0) {
                // Upstream boundary
                double W_bc = geom.width_from_area(A_j);
                double h_bc = geom.depth_from_area(A_j, W_bc);
                A_L = W_bc * h_bc;  // Assume same depth as first cell
                Q_L = Q_upstream;
            } else {
                A_L = std::max(y[off + 2*(j-1)], config_.min_area);
                Q_L = y[off + 2*(j-1) + 1];
            }
            
            if (j == n_nodes - 1) {
                // Downstream boundary: zero-gradient (transmissive)
                A_R = A_j;
                Q_R = Q_j;
            } else {
                A_R = std::max(y[off + 2*(j+1)], config_.min_area);
                Q_R = y[off + 2*(j+1) + 1];
            }
            
            // Compute fluxes at cell interfaces
            double F_A_left, F_Q_left, F_A_right, F_Q_right;
            
            // Left interface flux
            if (j == 0) {
                // Upstream boundary flux
                F_A_left = Q_upstream;
                double u_up = Q_upstream / std::max(A_L, config_.min_area);
                double W_up = geom.width_from_area(A_L);
                double h_up = geom.depth_from_area(A_L, W_up);
                F_Q_left = Q_upstream * u_up + 0.5 * g * A_L * h_up;
            } else {
                compute_flux(A_L, Q_L, A_j, Q_j, geom, dx, F_A_left, F_Q_left);
            }
            
            // Right interface flux
            compute_flux(A_j, Q_j, A_R, Q_R, geom, dx, F_A_right, F_Q_right);
            
            // Source terms
            double W_j = geom.width_from_area(A_j);
            double h_j = geom.depth_from_area(A_j, W_j);
            double R_j = geom.hydraulic_radius(A_j, W_j, h_j);
            double Sf = geom.friction_slope(Q_j, A_j, R_j);
            
            // dA/dt = -(F_A_right - F_A_left)/dx + q_lat
            ydot[off + 2*j] = -(F_A_right - F_A_left) / dx + q_lat;
            
            // dQ/dt = -(F_Q_right - F_Q_left)/dx + gA(S0 - Sf)
            ydot[off + 2*j + 1] = -(F_Q_right - F_Q_left) / dx 
                                  + g * A_j * (geom.bed_slope - Sf);
        }
    }
}

inline void SaintVenantRouter::compute_jacobian(double t, const double* y, double* J) {
    // Numerical Jacobian via finite differences
    // For large systems, this should be replaced with analytical or AD-generated Jacobian
    
    int N = total_state_size_;
    double eps = 1e-6;
    
    std::vector<double> ydot_base(N);
    std::vector<double> ydot_pert(N);
    std::vector<double> y_pert(y, y + N);
    
    compute_rhs(t, y, ydot_base.data());
    
    for (int j = 0; j < N; ++j) {
        double y_j = y_pert[j];
        double h = eps * std::max(1.0, std::abs(y_j));
        
        y_pert[j] = y_j + h;
        compute_rhs(t, y_pert.data(), ydot_pert.data());
        y_pert[j] = y_j;
        
        for (int i = 0; i < N; ++i) {
            // Column-major for SUNDIALS dense matrix
            J[j * N + i] = (ydot_pert[i] - ydot_base[i]) / h;
        }
    }
}

#ifdef DMC_ENABLE_SUNDIALS
inline int SaintVenantRouter::rhs_callback(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SaintVenantRouter* self = static_cast<SaintVenantRouter*>(user_data);
    self->compute_rhs(t, N_VGetArrayPointer(y), N_VGetArrayPointer(ydot));
    return 0;
}

inline int SaintVenantRouter::jac_callback(sunrealtype t, N_Vector y, N_Vector fy,
                                           SUNMatrix J, void* user_data,
                                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SaintVenantRouter* self = static_cast<SaintVenantRouter*>(user_data);
    self->compute_jacobian(t, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
    return 0;
}
#endif

inline void SaintVenantRouter::route_timestep() {
    // Snapshot the lateral forcing used for this step so the FD gradient can replay it.
    if (recording_) lateral_history_.push_back(lateral_inflows_);
#ifdef DMC_ENABLE_SUNDIALS
    // Advance CVODES from current_time to current_time + dt
    sunrealtype tout = current_time_ + config_.dt;
    sunrealtype tret;
    
    int flag = CVode(cvode_mem_, tout, y_, &tret, CV_NORMAL);
    
    if (flag < 0) {
        std::cerr << "CVODES error: " << flag << " at t = " << current_time_ << std::endl;
    }
    
    // Unpack state
    unpack_state(N_VGetArrayPointer(y_));
    
    current_time_ = tret;
#else
    // Fallback: explicit RK4 with CFL-based substepping
    // Note: This is slower and less accurate than SUNDIALS, but works without dependencies
    
    std::vector<double> y(total_state_size_);
    std::vector<double> k1(total_state_size_);
    std::vector<double> k2(total_state_size_);
    std::vector<double> k3(total_state_size_);
    std::vector<double> k4(total_state_size_);
    std::vector<double> y_temp(total_state_size_);
    
    pack_state(y.data());
    
    // Estimate CFL-stable timestep
    // For SVE: dt <= CFL * dx / (|u| + sqrt(g*h))
    // Use conservative CFL = 0.5
    double min_dx = 1e10;
    double max_wave_speed = 0.1;
    
    const auto& topo_order = network_.topological_order();
    for (size_t r = 0; r < reach_geometry_.size(); ++r) {
        const Reach& reach = network_.get_reach(topo_order[r]);
        double dx = reach.length / (config_.n_nodes - 1);
        min_dx = std::min(min_dx, dx);
        
        // Estimate wave speed from current state
        int off = reach_state_offset_[r];
        for (int j = 0; j < config_.n_nodes; ++j) {
            double A = std::max(y[off + 2*j], config_.min_area);
            double Q = y[off + 2*j + 1];
            double W = reach_geometry_[r].width_from_area(A);
            double h = reach_geometry_[r].depth_from_area(A, W);
            double u = Q / A;
            double c = std::sqrt(config_.g * h);
            max_wave_speed = std::max(max_wave_speed, std::abs(u) + c);
        }
    }
    
    double dt_cfl = 0.5 * min_dx / max_wave_speed;
    int n_sub = std::max(10, static_cast<int>(std::ceil(config_.dt / dt_cfl)));
    n_sub = std::min(n_sub, 10000);  // Cap to prevent runaway
    double dt_sub = config_.dt / n_sub;
    
    // RK4 integration
    for (int s = 0; s < n_sub; ++s) {
        double t = current_time_ + s * dt_sub;
        
        // k1 = f(t, y)
        compute_rhs(t, y.data(), k1.data());
        
        // k2 = f(t + dt/2, y + dt/2 * k1)
        for (int i = 0; i < total_state_size_; ++i) {
            y_temp[i] = y[i] + 0.5 * dt_sub * k1[i];
            if (i % 2 == 0) y_temp[i] = std::max(y_temp[i], config_.min_area);
        }
        compute_rhs(t + 0.5 * dt_sub, y_temp.data(), k2.data());
        
        // k3 = f(t + dt/2, y + dt/2 * k2)
        for (int i = 0; i < total_state_size_; ++i) {
            y_temp[i] = y[i] + 0.5 * dt_sub * k2[i];
            if (i % 2 == 0) y_temp[i] = std::max(y_temp[i], config_.min_area);
        }
        compute_rhs(t + 0.5 * dt_sub, y_temp.data(), k3.data());
        
        // k4 = f(t + dt, y + dt * k3)
        for (int i = 0; i < total_state_size_; ++i) {
            y_temp[i] = y[i] + dt_sub * k3[i];
            if (i % 2 == 0) y_temp[i] = std::max(y_temp[i], config_.min_area);
        }
        compute_rhs(t + dt_sub, y_temp.data(), k4.data());
        
        // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (int i = 0; i < total_state_size_; ++i) {
            y[i] += dt_sub / 6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
            if (i % 2 == 0) y[i] = std::max(y[i], config_.min_area);
        }
    }
    
    unpack_state(y.data());
    current_time_ += config_.dt;
#endif
}

inline void SaintVenantRouter::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) {
        route_timestep();
    }
}

inline void SaintVenantRouter::set_lateral_inflow(int reach_id, double inflow) {
    // Find reach index in topological order
    const auto& topo_order = network_.topological_order();
    for (size_t i = 0; i < topo_order.size(); ++i) {
        if (topo_order[i] == reach_id) {
            lateral_inflows_[i] = inflow;
            return;
        }
    }
}

inline double SaintVenantRouter::get_discharge(int reach_id) const {
    const auto& topo_order = network_.topological_order();
    for (size_t i = 0; i < topo_order.size(); ++i) {
        if (topo_order[i] == reach_id) {
            // Return outlet discharge (last node)
            return reach_states_[i].Q.back();
        }
    }
    return 0.0;
}

inline std::vector<double> SaintVenantRouter::get_all_discharges() const {
    std::vector<double> result;
    for (const auto& state : reach_states_) {
        result.push_back(state.Q.back());
    }
    return result;
}

inline double SaintVenantRouter::get_depth(int reach_id) const {
    const auto& topo_order = network_.topological_order();
    for (size_t i = 0; i < topo_order.size(); ++i) {
        if (topo_order[i] == reach_id) {
            double A = reach_states_[i].A.back();
            double W = reach_geometry_[i].width_from_area(A);
            return reach_geometry_[i].depth_from_area(A, W);
        }
    }
    return 0.0;
}

inline void SaintVenantRouter::reset_state() {
    initialize_state();
    current_time_ = 0.0;
    
#ifdef DMC_ENABLE_SUNDIALS
    pack_state(N_VGetArrayPointer(y_));
    CVodeReInit(cvode_mem_, 0.0, y_);
#endif
}

inline void SaintVenantRouter::start_recording() {
    recording_ = true;
    output_history_.clear();
    lateral_history_.clear();
    std::fill(grad_manning_n_.begin(), grad_manning_n_.end(), 0.0);
}

inline void SaintVenantRouter::stop_recording() {
    recording_ = false;
}

inline void SaintVenantRouter::record_output(int reach_id) {
    if (!recording_) return;
    output_history_[reach_id].push_back(get_discharge(reach_id));
}

inline void SaintVenantRouter::compute_gradients_timeseries(
    int reach_id,
    const std::vector<double>& dL_dQ) {

    // This implementation uses finite difference approximation for gradients.
    // For efficient, exact gradients via CVODES adjoint sensitivity with Enzyme AD,
    // use the SaintVenantEnzyme class from saint_venant_enzyme.hpp instead.

    auto it = output_history_.find(reach_id);
    if (it == output_history_.end()) {
        std::cerr << "No recorded outputs for reach " << reach_id << std::endl;
        return;
    }
    
    const auto& recorded = it->second;
    const size_t nT = recorded.size();
    if (dL_dQ.size() != nT) {
        std::cerr << "dL_dQ size mismatch" << std::endl;
        return;
    }
    if (lateral_history_.size() < nT) {
        std::cerr << "lateral history (" << lateral_history_.size()
                  << ") shorter than recorded outputs (" << nT
                  << "); cannot compute gradients" << std::endl;
        return;
    }

    // Central finite-difference gradient. The perturbed runs must replay the SAME
    // time-varying lateral forcing as the recorded forward pass (previously the
    // forcing was not re-applied, so the perturbed run used a constant inflow and the
    // gradient came out with the wrong sign and magnitude). Disable recording during
    // the replays so they don't append to the history.
    const double eps = 1e-5;
    const int n_reaches = static_cast<int>(reach_geometry_.size());
    const auto& topo_order = network_.topological_order();
    const bool was_recording = recording_;
    recording_ = false;

    auto replay = [&]() {
        reset_state();
        std::vector<double> sim;
        sim.reserve(nT);
        for (size_t t = 0; t < nT; ++t) {
            lateral_inflows_ = lateral_history_[t];   // replay exact forcing
            route_timestep();
            sim.push_back(get_discharge(reach_id));
        }
        return sim;
    };

    for (int r = 0; r < n_reaches; ++r) {
        const double n_orig = reach_geometry_[r].manning_n;

        reach_geometry_[r].manning_n = n_orig + eps;
        network_.get_reach(topo_order[r]).manning_n = Real(n_orig + eps);
        std::vector<double> sim_plus = replay();

        reach_geometry_[r].manning_n = n_orig - eps;
        network_.get_reach(topo_order[r]).manning_n = Real(n_orig - eps);
        std::vector<double> sim_minus = replay();

        reach_geometry_[r].manning_n = n_orig;
        network_.get_reach(topo_order[r]).manning_n = Real(n_orig);

        double grad = 0.0;
        for (size_t t = 0; t < nT; ++t) {
            const double dQ_dn = (sim_plus[t] - sim_minus[t]) / (2.0 * eps);
            grad += dL_dQ[t] * dQ_dn;
        }
        grad_manning_n_[r] = grad;
    }

    recording_ = was_recording;
    reset_state();
}

inline std::unordered_map<std::string, double> SaintVenantRouter::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    const auto& topo_order = network_.topological_order();
    
    for (size_t i = 0; i < grad_manning_n_.size(); ++i) {
        std::string key = "reach_" + std::to_string(topo_order[i]) + "_manning_n";
        grads[key] = grad_manning_n_[i];
    }
    
    return grads;
}

} // namespace dmc

#endif // DMC_SAINT_VENANT_ROUTER_HPP
