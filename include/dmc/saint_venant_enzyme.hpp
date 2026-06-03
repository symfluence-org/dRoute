#ifndef DMC_SAINT_VENANT_ENZYME_HPP
#define DMC_SAINT_VENANT_ENZYME_HPP

/**
 * @file saint_venant_enzyme.hpp
 * @brief Full dynamic Saint-Venant Equations solver with Enzyme AD gradients
 * 
 * This implementation combines:
 * - SUNDIALS CVODES for implicit time integration (forward pass)
 * - CVODES Adjoint Sensitivity (CVSAS) for backward pass
 * - Enzyme AD for Jacobian and adjoint RHS computation
 * 
 * Gradient computation strategy:
 * 1. Forward pass: CVODES with Enzyme-computed Jacobian (faster Newton)
 * 2. Backward pass: CVODES adjoint with Enzyme-computed Jᵀλ products
 * 3. Parameter sensitivity: Accumulate ∫ λᵀ (∂f/∂p) dt via Enzyme
 * 
 * This provides exact gradients with O(1) memory overhead relative to
 * finite difference, and significantly faster computation.
 */

#include "network.hpp"
#include "types.hpp"
#include "ad_backend.hpp"  // For Enzyme declarations
#include "saint_venant_router.hpp"

#include <vector>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <iostream>

// SUNDIALS headers for adjoint sensitivity
#ifdef DMC_ENABLE_SUNDIALS
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_ls.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_context.h>

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL 0
#endif
#endif

// Additional Enzyme declaration for forward-mode AD
#ifdef DMC_USE_ENZYME
extern "C" {
    void __enzyme_fwddiff(void*, ...);
}
#endif

namespace dmc {

// ============================================================================
// Extended Configuration for Enzyme-based solver
// ============================================================================

struct SaintVenantEnzymeConfig : public SaintVenantConfig {
    // Adjoint-specific settings
    int adjoint_checkpoint_steps = 100;   // Steps between checkpoints
    bool use_hermite_interpolation = true; // Hermite vs polynomial interpolation

    // Enzyme options
    bool use_enzyme_jacobian = true;      // Use Enzyme for Jacobian (vs FD)
    bool use_enzyme_adjoint = true;       // Use Enzyme for adjoint RHS

    // Debug options
    bool verbose = false;                 // Print debug information during initialization
};

// ============================================================================
// SaintVenantEnzyme Class - Enzyme-enabled SVE solver
// ============================================================================

class SaintVenantEnzyme {
public:
    SaintVenantEnzyme(Network& network, SaintVenantEnzymeConfig config = {});
    ~SaintVenantEnzyme();
    
    // Prevent copying
    SaintVenantEnzyme(const SaintVenantEnzyme&) = delete;
    SaintVenantEnzyme& operator=(const SaintVenantEnzyme&) = delete;
    
    // =========== Core Routing ===========
    void route_timestep();
    void route(int num_timesteps);
    
    // =========== State Management ===========
    void set_lateral_inflow(int reach_id, double inflow);
    double get_discharge(int reach_id) const;
    std::vector<double> get_all_discharges() const;
    double get_depth(int reach_id) const;
    void reset_state();
    
    // =========== Gradient Computation ===========
    
    /**
     * Enable gradient recording for adjoint computation.
     * This activates CVODES checkpointing.
     */
    void start_recording();
    
    /**
     * Stop recording and finalize forward pass.
     */
    void stop_recording();
    
    /**
     * Compute gradients w.r.t. Manning's n for all reaches.
     * 
     * Uses CVODES adjoint sensitivity with Enzyme-computed adjoints:
     * 1. Backward integration of adjoint ODE: λ' = -Jᵀλ
     * 2. Accumulation of parameter sensitivity: dL/dn = ∫ λᵀ (∂f/∂n) dt
     * 
     * @param gauge_reach_id  Reach ID where loss is computed
     * @param dL_dQ           Gradient of loss w.r.t. Q at each recorded time [n_times]
     */
    void compute_gradients(int gauge_reach_id, const std::vector<double>& dL_dQ);
    
    /**
     * Get gradients for all parameters.
     * @return Map from "reach_X_manning_n" to gradient value
     */
    std::unordered_map<std::string, double> get_gradients() const;
    
    /**
     * Reset accumulated gradients.
     */
    void reset_gradients();
    
    // =========== Access ===========
    const SaintVenantEnzymeConfig& config() const { return config_; }
    Network& network() { return network_; }
    double current_time() const { return current_time_; }
    
private:
    Network& network_;
    SaintVenantEnzymeConfig config_;
    double current_time_ = 0.0;
    bool recording_ = false;
    
    // State: one ReachState per reach
    std::vector<ReachState> reach_states_;
    std::vector<SVEGeometry> reach_geometry_;
    std::vector<double> lateral_inflows_;  // [m³/s] per reach
    
    // State vector mapping
    std::vector<int> reach_state_offset_;
    int total_state_size_ = 0;
    int n_reaches_ = 0;
    int n_nodes_ = 0;
    
    // Recording for gradients
    std::vector<double> recorded_times_;
    std::vector<std::vector<double>> recorded_states_;  // [n_times][state_size]
    int gauge_state_idx_ = -1;  // Index of gauge reach outlet Q in state vector

    // Lateral-inflow forcing history, recorded per outer step during the forward pass.
    // The CVODES adjoint reconstructs the forward solution by re-integrating the RHS from
    // checkpoints, so the forcing MUST be a reproducible function of time t (not the mutable
    // lateral_inflows_ member, which the caller overwrites each step). Without this, the
    // reconstruction diverges and overflows CVODES' data store -> NULL deref in CVAdataStore.
    std::vector<std::vector<double>> lateral_history_;  // [step][reach]
    double record_t0_ = 0.0;                            // time at start of recording
    std::vector<double> record_y0_;                     // packed state at start of recording
    bool adjoint_initialized_ = false;                  // CVodeAdjInit called exactly once
    std::vector<double> dL_dQ_;                          // per-observation dL/dQ(t_k) (running-cost source)

    // Gradients (accumulated during backward pass)
    std::vector<double> grad_manning_n_;
    
    // =========== Core computation methods ===========
    
    void initialize_state();
    void pack_state(double* y) const;
    void unpack_state(const double* y);
    
    /**
     * RHS function: dy/dt = f(t, y, p)
     * 
     * This is the core physics - needs to be differentiable by Enzyme.
     */
    void compute_rhs(double t, const double* y, double* ydot);
    
    /**
     * RHS with explicit parameter dependency for Enzyme sensitivity.
     * 
     * @param t       Current time
     * @param y       State vector [A_0, Q_0, A_1, Q_1, ...]
     * @param params  Parameter vector [n_0, n_1, ..., n_{n_reaches-1}]
     * @param ydot    Output: time derivatives
     */
    void compute_rhs_with_params(double t, const double* y, 
                                  const double* params, double* ydot);
    
    /**
     * Flux computation (Rusanov/Local Lax-Friedrichs)
     */
    void compute_flux(double A_L, double Q_L, double A_R, double Q_R,
                     const SVEGeometry& geom, double dx,
                     double& F_A, double& F_Q);
    
#ifdef DMC_USE_ENZYME
    // =========== Enzyme-based Jacobian and Adjoints ===========
    
    /**
     * Compute Jacobian using Enzyme forward-mode AD.
     * 
     * J[i,j] = ∂ydot[i]/∂y[j]
     * 
     * Uses __enzyme_fwddiff to compute columns of J efficiently.
     */
    void compute_jacobian_enzyme(double t, const double* y, double* J);
    
    /**
     * Compute parameter sensitivity: (∂f/∂p)ᵀλ
     * 
     * Used to accumulate: dL/dp = ∫ λᵀ (∂f/∂p) dt
     * 
     * @param t       Current time
     * @param y       State vector
     * @param lambda  Adjoint vector
     * @param grad_p  Output: gradient w.r.t. each parameter (Manning's n)
     */
    void compute_param_sensitivity(double t, const double* y,
                                   const double* lambda, double* grad_p);
    
    // Static wrappers for Enzyme (need free function pointers)
    static void rhs_wrapper(double t, const double* y, const double* params,
                           double* ydot, SaintVenantEnzyme* self);
    
#endif // DMC_USE_ENZYME
    
    // Finite difference Jacobian (always available as fallback)
    void compute_jacobian_fd(double t, const double* y, double* J);
    
#ifdef DMC_ENABLE_SUNDIALS
    // SUNDIALS objects for forward pass
    void* cvode_mem_ = nullptr;
    N_Vector y_ = nullptr;
    SUNMatrix J_ = nullptr;
    SUNLinearSolver LS_ = nullptr;
    SUNContext sunctx_ = nullptr;
    
    // SUNDIALS objects for adjoint (backward) pass
    void* cvode_mem_B_ = nullptr;  // Backward problem
    N_Vector yB_ = nullptr;        // Adjoint state
    int indexB_ = -1;              // Backward problem index
    int ncheck_ = 0;               // Number of checkpoints
    
    // CVODES callbacks
    static int rhs_callback(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    static int jac_callback(sunrealtype t, N_Vector y, N_Vector fy,
                           SUNMatrix J, void* user_data,
                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    
    // Adjoint RHS callback: λ' = -Jᵀλ + g_y
    static int adjoint_rhs_callback(sunrealtype t, N_Vector y, N_Vector yB,
                                    N_Vector yBdot, void* user_data);
    
    // Quadrature RHS for parameter sensitivity: (∂f/∂p)ᵀλ
    static int quad_rhs_callback(sunrealtype t, N_Vector y, N_Vector yB,
                                 N_Vector qBdot, void* user_data);
    
    void setup_cvodes();
    void setup_adjoint();
    void cleanup_cvodes();
#endif
};

// ============================================================================
// Implementation
// ============================================================================

inline SaintVenantEnzyme::SaintVenantEnzyme(Network& network, SaintVenantEnzymeConfig config)
    : network_(network), config_(std::move(config)) {
    
    network_.build_topology();
    
    n_reaches_ = network_.topological_order().size();
    n_nodes_ = config_.n_nodes;
    
    // Allocate state for each reach
    reach_states_.resize(n_reaches_);
    reach_geometry_.resize(n_reaches_);
    lateral_inflows_.resize(n_reaches_, 0.0);
    grad_manning_n_.resize(n_reaches_, 0.0);
    reach_state_offset_.resize(n_reaches_);
    
    // Build state vector mapping: [A_0, Q_0, A_1, Q_1, ...] for each reach
    int offset = 0;
    for (int i = 0; i < n_reaches_; ++i) {
        reach_state_offset_[i] = offset;
        reach_states_[i].resize(n_nodes_);
        offset += 2 * n_nodes_;  // A and Q for each node
    }
    total_state_size_ = offset;
    
    // Initialize geometry from network reaches
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches_; ++i) {
        const Reach& reach = network_.get_reach(topo_order[i]);
        reach_geometry_[i].bed_slope = reach.slope;
        reach_geometry_[i].manning_n = to_double(reach.manning_n);
        reach_geometry_[i].width_coef = to_double(reach.geometry.width_coef);
        reach_geometry_[i].width_exp = to_double(reach.geometry.width_exp);
    }
    
    // Initialize state
    initialize_state();
    
#ifdef DMC_ENABLE_SUNDIALS
    setup_cvodes();
#else
    std::cerr << "Warning: SUNDIALS not enabled. Enzyme-SVE requires SUNDIALS." << std::endl;
#endif
}

inline SaintVenantEnzyme::~SaintVenantEnzyme() {
#ifdef DMC_ENABLE_SUNDIALS
    cleanup_cvodes();
#endif
}

#ifdef DMC_ENABLE_SUNDIALS

inline void SaintVenantEnzyme::setup_cvodes() {
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
    
    // Set Jacobian function (Enzyme-computed)
    flag = CVodeSetJacFn(cvode_mem_, jac_callback);
    
    // Set max steps
    flag = CVodeSetMaxNumSteps(cvode_mem_, config_.max_steps);
    
    if (config_.verbose) {
        std::cout << "SaintVenantEnzyme: CVODES initialized with "
                  << total_state_size_ << " state variables" << std::endl;
#ifdef DMC_USE_ENZYME
        if (config_.use_enzyme_jacobian) {
            std::cout << "  Using Enzyme AD for Jacobian computation" << std::endl;
        }
        if (config_.use_enzyme_adjoint) {
            std::cout << "  Using Enzyme AD for adjoint sensitivity" << std::endl;
        }
#endif
    }
}

inline void SaintVenantEnzyme::setup_adjoint() {
    if (!config_.enable_adjoint) return;
    
    // Initialize adjoint module with checkpointing. CVodeAdjInit allocates the adjoint
    // checkpoint memory and MUST be called exactly once per cvode_mem; on subsequent
    // recordings (e.g. each calibration iteration) we reset it with CVodeAdjReInit instead
    // -- calling CVodeAdjInit twice leaks and corrupts the checkpoint structure.
    int interp = config_.use_hermite_interpolation ? CV_HERMITE : CV_POLYNOMIAL;
    int flag;
    if (!adjoint_initialized_) {
        flag = CVodeAdjInit(cvode_mem_, config_.adjoint_checkpoint_steps, interp);
        adjoint_initialized_ = (flag == CV_SUCCESS);
    } else {
        flag = CVodeAdjReInit(cvode_mem_);
    }

    if (flag != CV_SUCCESS) {
        std::cerr << "Error initializing CVODES adjoint" << std::endl;
        return;
    }

    if (config_.verbose) {
        std::cout << "  CVODES Adjoint initialized with checkpoint stride = "
                  << config_.adjoint_checkpoint_steps << std::endl;
    }
}

inline void SaintVenantEnzyme::cleanup_cvodes() {
    // Clean up backward problem first
    if (yB_) N_VDestroy(yB_);
    
    // Clean up forward problem
    if (LS_) SUNLinSolFree(LS_);
    if (J_) SUNMatDestroy(J_);
    if (y_) N_VDestroy(y_);
    if (cvode_mem_) {
        CVodeFree(&cvode_mem_);
    }
    if (sunctx_) SUNContext_Free(&sunctx_);
}

// CVODES callbacks
inline int SaintVenantEnzyme::rhs_callback(sunrealtype t, N_Vector y, 
                                           N_Vector ydot, void* user_data) {
    SaintVenantEnzyme* self = static_cast<SaintVenantEnzyme*>(user_data);
    self->compute_rhs(t, N_VGetArrayPointer(y), N_VGetArrayPointer(ydot));
    return 0;
}

inline int SaintVenantEnzyme::jac_callback(sunrealtype t, N_Vector y, N_Vector fy,
                                           SUNMatrix J, void* user_data,
                                           N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SaintVenantEnzyme* self = static_cast<SaintVenantEnzyme*>(user_data);
    
#ifdef DMC_USE_ENZYME
    if (self->config_.use_enzyme_jacobian) {
        self->compute_jacobian_enzyme(t, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
    } else {
        self->compute_jacobian_fd(t, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
    }
#else
    self->compute_jacobian_fd(t, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
#endif
    return 0;
}

inline int SaintVenantEnzyme::adjoint_rhs_callback(sunrealtype t, N_Vector y, 
                                                    N_Vector yB, N_Vector yBdot,
                                                    void* user_data) {
    // Adjoint ODE: λ' = -Jᵀλ
    // yB is λ (adjoint), yBdot is λ'
    
    SaintVenantEnzyme* self = static_cast<SaintVenantEnzyme*>(user_data);
    int N = self->total_state_size_;
    
    double* y_ptr = N_VGetArrayPointer(y);
    double* lambda = N_VGetArrayPointer(yB);
    double* lambda_dot = N_VGetArrayPointer(yBdot);
    
    // λ' = -Jᵀλ. We build the full J with the stable FORWARD-mode Enzyme Jacobian and
    // contract, rather than a reverse-mode VJP: reverse-mode __enzyme_autodiff segfaults
    // in this build, so the entire adjoint path is forward-mode only.
    std::vector<double> J(N * N);
#ifdef DMC_USE_ENZYME
    self->compute_jacobian_enzyme(t, y_ptr, J.data());
#else
    self->compute_jacobian_fd(t, y_ptr, J.data());
#endif
    // J stored column-major: J[j*N + i] = J(i,j) = ∂ydot_i/∂y_j.
    // (Jᵀλ)_i = Σ_j J(j,i) λ_j = Σ_j J[i*N + j] λ_j
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        for (int j = 0; j < N; ++j) acc += J[i * N + j] * lambda[j];
        lambda_dot[i] = -acc;
    }
    // Running-cost source: the discrete loss L = sum_k l_k(Q(t_k)) is folded into a
    // continuous adjoint source g_y(t) = dL/dQ(t_k(t))/dt at the gauge, so the WHOLE backward
    // solve is a single CVodeB with NO per-observation CVodeReInitB. The per-step reinit was
    // restarting the BDF integrator at order 1 every step, compounding error and washing out
    // the costate (multi-reach gradients collapsed to near-equal values). λ' = -Jᵀλ - g_y.
    // Observation k applies over (t_{k-1}, t_k]; with t_k = record_t0_+(k+1)*dt that is
    // k(t) = ceil((t-record_t0_)/dt)-1, clamped to [0, N-1].
    if (!self->dL_dQ_.empty()) {
        const double dt = self->config_.dt;
        int k = static_cast<int>(std::ceil((t - self->record_t0_) / dt)) - 1;
        if (k < 0) k = 0;
        if (k >= static_cast<int>(self->dL_dQ_.size())) k = static_cast<int>(self->dL_dQ_.size()) - 1;
        lambda_dot[self->gauge_state_idx_] -= self->dL_dQ_[k] / dt;
    }
    if (self->config_.verbose) {
        static int cnt = 0;
        if ((cnt++ % 60) == 0) {
            // per-reach forward Q (last node) and costate λ_Q (last node), to see whether λ
            // is washed out (uniform across reaches) vs concentrated upstream as it should be.
            std::cerr << "[adjcb] t=" << t << " yQ:";
            for (int r = 0; r < self->n_reaches_; ++r)
                std::cerr << " " << y_ptr[self->reach_state_offset_[r] + 2*(self->n_nodes_-1)+1];
            std::cerr << "  lamQ:";
            for (int r = 0; r < self->n_reaches_; ++r)
                std::cerr << " " << lambda[self->reach_state_offset_[r] + 2*(self->n_nodes_-1)+1];
            std::cerr << std::endl;
        }
    }
    return 0;
}

inline int SaintVenantEnzyme::quad_rhs_callback(sunrealtype t, N_Vector y,
                                                 N_Vector yB, N_Vector qBdot,
                                                 void* user_data) {
    // Quadrature for parameter sensitivity: (∂f/∂p)ᵀλ
    // Accumulates: dL/dp = ∫ λᵀ (∂f/∂p) dt
    
    SaintVenantEnzyme* self = static_cast<SaintVenantEnzyme*>(user_data);

#ifdef DMC_USE_ENZYME
    // Parameter sensitivity (∂f/∂p)ᵀλ via stable forward-mode Enzyme.
    double* y_ptr = N_VGetArrayPointer(y);
    double* lambda = N_VGetArrayPointer(yB);
    double* grad = N_VGetArrayPointer(qBdot);
    self->compute_param_sensitivity(t, y_ptr, lambda, grad);
#else
    N_VConst(0.0, qBdot);
#endif
    return 0;
}

#endif // DMC_ENABLE_SUNDIALS

// ============================================================================
// Enzyme-based Jacobian and Adjoint computation
// ============================================================================

#ifdef DMC_USE_ENZYME

// Static wrapper function that Enzyme can differentiate
inline void SaintVenantEnzyme::rhs_wrapper(double t, const double* y, 
                                            const double* params,
                                            double* ydot, 
                                            SaintVenantEnzyme* self) {
    self->compute_rhs_with_params(t, y, params, ydot);
}

inline void SaintVenantEnzyme::compute_jacobian_enzyme(double t, const double* y, double* J) {
    // Use Enzyme forward-mode to compute Jacobian columns
    // J[i,j] = ∂ydot[i]/∂y[j]
    
    int N = total_state_size_;
    
    // Collect current parameters
    std::vector<double> params(n_reaches_);
    for (int r = 0; r < n_reaches_; ++r) {
        params[r] = reach_geometry_[r].manning_n;
    }
    
    std::vector<double> dy(N, 0.0);
    std::vector<double> dydot(N, 0.0);
    std::vector<double> ydot_base(N);
    
    // Compute base RHS (for reference)
    compute_rhs_with_params(t, y, params.data(), ydot_base.data());
    
    // Compute each column of Jacobian via forward-mode AD
    for (int j = 0; j < N; ++j) {
        // Seed: dy[j] = 1, all others = 0
        std::fill(dy.begin(), dy.end(), 0.0);
        dy[j] = 1.0;
        std::fill(dydot.begin(), dydot.end(), 0.0);
        
        // Forward-mode AD: compute directional derivative d(ydot)/d(y) * dy
        __enzyme_fwddiff((void*)rhs_wrapper,
            enzyme_const, t,
            enzyme_dup, y, dy.data(),
            enzyme_const, params.data(),
            enzyme_dupnoneed, nullptr, dydot.data(),
            enzyme_const, this);
        
        // dydot now contains column j of the Jacobian
        for (int i = 0; i < N; ++i) {
            // Column-major for SUNDIALS: J[j*N + i] = J(i,j)
            J[j * N + i] = dydot[i];
        }
    }
}

inline void SaintVenantEnzyme::compute_param_sensitivity(double t, const double* y,
                                                          const double* lambda,
                                                          double* grad_p) {
    // Compute (∂f/∂p)ᵀλ for parameter gradients
    // p = [manning_n_0, manning_n_1, ...]
    
    int N = total_state_size_;
    int P = n_reaches_;
    
    // Collect current parameters
    std::vector<double> params(P);
    for (int r = 0; r < P; ++r) {
        params[r] = reach_geometry_[r].manning_n;
    }
    
    // Initialize output gradients
    std::fill(grad_p, grad_p + P, 0.0);

    // Parameter sensitivity via FORWARD-mode AD (reverse-mode __enzyme_autodiff
    // segfaults in this build; forward-mode __enzyme_fwddiff is stable). For each
    // parameter p_r we seed dp_r = 1 and read dydot = ∂f/∂p_r, then contract with
    // the adjoint state: (∂f/∂p_r)ᵀλ = Σ_i dydot_i · λ_i.
    std::vector<double> dparams(P, 0.0);
    std::vector<double> dydot(N, 0.0);
    for (int r = 0; r < P; ++r) {
        std::fill(dparams.begin(), dparams.end(), 0.0);
        dparams[r] = 1.0;
        std::fill(dydot.begin(), dydot.end(), 0.0);
        __enzyme_fwddiff((void*)rhs_wrapper,
            enzyme_const, t,
            enzyme_const, y,                          // y held constant
            enzyme_dup, params.data(), dparams.data(),// seed parameter r
            enzyme_dupnoneed, nullptr, dydot.data(),  // tangent output ∂f/∂p_r
            enzyme_const, this);
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += dydot[i] * lambda[i];
        grad_p[r] = s;
    }
}

#endif // DMC_USE_ENZYME

// Finite difference Jacobian (always available as fallback)
inline void SaintVenantEnzyme::compute_jacobian_fd(double t, const double* y, double* J) {
    // Numerical Jacobian via finite differences
    int N = total_state_size_;
    double eps = 1e-7;
    
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
            J[j * N + i] = (ydot_pert[i] - ydot_base[i]) / h;
        }
    }
}

// ============================================================================
// Core physics (must be Enzyme-compatible)
// ============================================================================

inline void SaintVenantEnzyme::initialize_state() {
    double h0 = config_.initial_depth;
    double v0 = config_.initial_velocity;
    
    for (int i = 0; i < n_reaches_; ++i) {
        const SVEGeometry& geom = reach_geometry_[i];
        double W = geom.width_coef * std::pow(h0, 0.5);
        double A0 = W * h0;
        double Q0 = A0 * v0;
        
        for (int j = 0; j < n_nodes_; ++j) {
            reach_states_[i].A[j] = A0;
            reach_states_[i].Q[j] = Q0;
        }
    }
}

inline void SaintVenantEnzyme::pack_state(double* y) const {
    for (int i = 0; i < n_reaches_; ++i) {
        int off = reach_state_offset_[i];
        for (int j = 0; j < n_nodes_; ++j) {
            y[off + 2*j] = reach_states_[i].A[j];
            y[off + 2*j + 1] = reach_states_[i].Q[j];
        }
    }
}

inline void SaintVenantEnzyme::unpack_state(const double* y) {
    for (int i = 0; i < n_reaches_; ++i) {
        int off = reach_state_offset_[i];
        for (int j = 0; j < n_nodes_; ++j) {
            reach_states_[i].A[j] = std::max(y[off + 2*j], config_.min_area);
            reach_states_[i].Q[j] = y[off + 2*j + 1];
        }
    }
}

inline void SaintVenantEnzyme::compute_rhs(double t, const double* y, double* ydot) {
    // Use current Manning's n from geometry
    std::vector<double> params(n_reaches_);
    for (int r = 0; r < n_reaches_; ++r) {
        params[r] = reach_geometry_[r].manning_n;
    }
    compute_rhs_with_params(t, y, params.data(), ydot);
}

inline void SaintVenantEnzyme::compute_rhs_with_params(
    double t, const double* y, const double* params, double* ydot) {
    
    // This function must be Enzyme-compatible!
    // Avoid: std containers in hot path, branches that depend on values being differentiated
    
    double g = config_.g;
    double min_area = config_.min_area;
    const auto& topo_order = network_.topological_order();

    // Resolve the lateral forcing as a function of time t so the adjoint reconstruction
    // reproduces the forward exactly. During/after recording use the per-step history
    // (piecewise-constant in [t_k, t_{k+1})); otherwise fall back to the live member.
    const bool use_hist = !lateral_history_.empty();
    int lstep = 0;
    if (use_hist) {
        lstep = static_cast<int>(std::floor((t - record_t0_) / config_.dt));
        if (lstep < 0) lstep = 0;
        if (lstep >= static_cast<int>(lateral_history_.size()))
            lstep = static_cast<int>(lateral_history_.size()) - 1;
    }

    // Process each reach
    for (int r = 0; r < n_reaches_; ++r) {
        const Reach& reach = network_.get_reach(topo_order[r]);
        SVEGeometry geom = reach_geometry_[r];
        // NOTE: do NOT route the active parameter through geom.manning_n -- Enzyme drops the
        // tangent through this struct member (geom is copied from an enzyme_const member).
        // Instead pass params[r] directly into friction_slope below. geom.manning_n is left
        // as-is and only used by the non-differentiated friction_slope(Q,A,R) overload, which
        // we no longer call on this path.
        const double manning_r = params[r];  // active parameter, flows straight into Sf

        double dx = reach.length / (n_nodes_ - 1);
        int off = reach_state_offset_[r];

        double lat_r = use_hist ? lateral_history_[lstep][r] : lateral_inflows_[r];
        // Lateral inflow per unit length
        double q_lat = lat_r / reach.length;
        
        // Upstream boundary: sum of upstream reach outflows
        double Q_upstream = 0.0;
        Junction& up_junc = network_.get_junction(reach.upstream_junction_id);
        for (int up_reach_id : up_junc.upstream_reach_ids) {
            for (int ur = 0; ur < n_reaches_; ++ur) {
                if (topo_order[ur] == up_reach_id) {
                    int up_off = reach_state_offset_[ur];
                    Q_upstream += y[up_off + 2*(n_nodes_-1) + 1];
                    break;
                }
            }
        }
        Q_upstream += lat_r * 0.5;  // Half at inlet
        
        // Interior nodes
        for (int j = 0; j < n_nodes_; ++j) {
            double A_j = y[off + 2*j];
            if (A_j < min_area) A_j = min_area;
            double Q_j = y[off + 2*j + 1];
            
            // Left and right states
            double A_L, Q_L, A_R, Q_R;
            
            if (j == 0) {
                double W_bc = geom.width_from_area(A_j);
                double h_bc = geom.depth_from_area(A_j, W_bc);
                A_L = W_bc * h_bc;
                Q_L = Q_upstream;
            } else {
                A_L = y[off + 2*(j-1)];
                if (A_L < min_area) A_L = min_area;
                Q_L = y[off + 2*(j-1) + 1];
            }
            
            if (j == n_nodes_ - 1) {
                A_R = A_j;
                Q_R = Q_j;
            } else {
                A_R = y[off + 2*(j+1)];
                if (A_R < min_area) A_R = min_area;
                Q_R = y[off + 2*(j+1) + 1];
            }
            
            // Fluxes
            double F_A_left, F_Q_left, F_A_right, F_Q_right;
            
            if (j == 0) {
                F_A_left = Q_upstream;
                double u_up = Q_upstream / A_L;
                double W_up = geom.width_from_area(A_L);
                double h_up = geom.depth_from_area(A_L, W_up);
                F_Q_left = Q_upstream * u_up + 0.5 * g * A_L * h_up;
            } else {
                compute_flux(A_L, Q_L, A_j, Q_j, geom, dx, F_A_left, F_Q_left);
            }
            
            compute_flux(A_j, Q_j, A_R, Q_R, geom, dx, F_A_right, F_Q_right);
            
            // Source terms
            double W_j = geom.width_from_area(A_j);
            double h_j = geom.depth_from_area(A_j, W_j);
            double R_j = geom.hydraulic_radius(A_j, W_j, h_j);
            double Sf = geom.friction_slope(Q_j, A_j, R_j, manning_r);
            
            // dA/dt
            ydot[off + 2*j] = -(F_A_right - F_A_left) / dx + q_lat;
            
            // dQ/dt
            ydot[off + 2*j + 1] = -(F_Q_right - F_Q_left) / dx 
                                  + g * A_j * (geom.bed_slope - Sf);
        }
    }
}

inline void SaintVenantEnzyme::compute_flux(
    double A_L, double Q_L, double A_R, double Q_R,
    const SVEGeometry& geom, double dx,
    double& F_A, double& F_Q) {
    
    double g = config_.g;
    double min_area = config_.min_area;
    
    if (A_L < min_area) A_L = min_area;
    if (A_R < min_area) A_R = min_area;
    
    double u_L = Q_L / A_L;
    double u_R = Q_R / A_R;
    
    double W_L = geom.width_from_area(A_L);
    double W_R = geom.width_from_area(A_R);
    double h_L = geom.depth_from_area(A_L, W_L);
    double h_R = geom.depth_from_area(A_R, W_R);

    // Floor depth inside the wave-celerity sqrt. d(sqrt(g h))/dh ~ 1/sqrt(h) blows up
    // as h->0, so at near-dry states the forward-mode Jacobian becomes ill-conditioned
    // and crashes the CVODES dense backward solve. Clamping to min_depth bounds the
    // celerity derivative while leaving the (well-behaved) advective/pressure terms intact.
    double hmin = config_.min_depth;
    double hc_L = h_L > hmin ? h_L : hmin;
    double hc_R = h_R > hmin ? h_R : hmin;
    double c_L = std::sqrt(g * hc_L);
    double c_R = std::sqrt(g * hc_R);
    
    double a_max = std::max(std::abs(u_L) + c_L, std::abs(u_R) + c_R);
    
    double F_A_L = Q_L;
    double F_A_R = Q_R;
    double F_Q_L = Q_L * u_L + 0.5 * g * A_L * h_L;
    double F_Q_R = Q_R * u_R + 0.5 * g * A_R * h_R;
    
    // Rusanov flux
    F_A = 0.5 * (F_A_L + F_A_R) - 0.5 * a_max * (A_R - A_L);
    F_Q = 0.5 * (F_Q_L + F_Q_R) - 0.5 * a_max * (Q_R - Q_L);
}

// ============================================================================
// Public interface methods
// ============================================================================

inline void SaintVenantEnzyme::route_timestep() {
#ifdef DMC_ENABLE_SUNDIALS
    sunrealtype tout = current_time_ + config_.dt;
    sunrealtype tret;
    
    int flag;
    if (recording_) {
        // Record the lateral forcing for THIS step so the adjoint can reconstruct the
        // forward RHS by time (forcing applies over [current_time_, current_time_+dt]).
        lateral_history_.push_back(lateral_inflows_);
        // Forward with checkpointing for adjoint
        flag = CVodeF(cvode_mem_, tout, y_, &tret, CV_NORMAL, &ncheck_);
    } else {
        flag = CVode(cvode_mem_, tout, y_, &tret, CV_NORMAL);
    }
    
    if (flag < 0) {
        std::cerr << "CVODES error: " << flag << " at t = " << current_time_ << std::endl;
    }
    
    unpack_state(N_VGetArrayPointer(y_));
    current_time_ = tret;
    
    // Record state if needed
    if (recording_) {
        recorded_times_.push_back(current_time_);
        std::vector<double> state(total_state_size_);
        std::copy(N_VGetArrayPointer(y_), 
                  N_VGetArrayPointer(y_) + total_state_size_,
                  state.begin());
        recorded_states_.push_back(std::move(state));
    }
#else
    std::cerr << "SUNDIALS required for SaintVenantEnzyme" << std::endl;
#endif
}

inline void SaintVenantEnzyme::route(int num_timesteps) {
    for (int t = 0; t < num_timesteps; ++t) {
        route_timestep();
    }
}

inline void SaintVenantEnzyme::set_lateral_inflow(int reach_id, double inflow) {
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches_; ++i) {
        if (topo_order[i] == reach_id) {
            lateral_inflows_[i] = inflow;
            return;
        }
    }
}

inline double SaintVenantEnzyme::get_discharge(int reach_id) const {
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches_; ++i) {
        if (topo_order[i] == reach_id) {
            return reach_states_[i].Q.back();
        }
    }
    return 0.0;
}

inline std::vector<double> SaintVenantEnzyme::get_all_discharges() const {
    std::vector<double> result;
    for (const auto& state : reach_states_) {
        result.push_back(state.Q.back());
    }
    return result;
}

inline double SaintVenantEnzyme::get_depth(int reach_id) const {
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches_; ++i) {
        if (topo_order[i] == reach_id) {
            double A = reach_states_[i].A.back();
            double W = reach_geometry_[i].width_from_area(A);
            return reach_geometry_[i].depth_from_area(A, W);
        }
    }
    return 0.0;
}

inline void SaintVenantEnzyme::reset_state() {
    initialize_state();
    current_time_ = 0.0;
    
#ifdef DMC_ENABLE_SUNDIALS
    pack_state(N_VGetArrayPointer(y_));
    CVodeReInit(cvode_mem_, 0.0, y_);
#endif
    
    recorded_times_.clear();
    recorded_states_.clear();
}

inline void SaintVenantEnzyme::start_recording() {
    recording_ = true;
    recorded_times_.clear();
    recorded_states_.clear();
    lateral_history_.clear();
    record_t0_ = current_time_;

#ifdef DMC_ENABLE_SUNDIALS
    ncheck_ = 0;
    // Snapshot the state at the start of recording so the forward checkpointed trajectory
    // can be regenerated (with the complete forcing history) before the backward solve.
    record_y0_.assign(N_VGetArrayPointer(y_),
                      N_VGetArrayPointer(y_) + total_state_size_);
    setup_adjoint();
#endif
}

inline void SaintVenantEnzyme::stop_recording() {
    recording_ = false;
}

inline void SaintVenantEnzyme::compute_gradients(int gauge_reach_id, 
                                                  const std::vector<double>& dL_dQ) {
#ifdef DMC_ENABLE_SUNDIALS
    if (recorded_times_.empty()) {
        std::cerr << "No recorded outputs for gradient computation" << std::endl;
        return;
    }
    
    if (dL_dQ.size() != recorded_times_.size()) {
        std::cerr << "dL_dQ size (" << dL_dQ.size() << ") doesn't match recorded times ("
                  << recorded_times_.size() << ")" << std::endl;
        return;
    }
    
    // Find gauge reach index
    int gauge_idx = -1;
    const auto& topo_order = network_.topological_order();
    for (int i = 0; i < n_reaches_; ++i) {
        if (topo_order[i] == gauge_reach_id) {
            gauge_idx = i;
            // Index of outlet Q in state vector
            gauge_state_idx_ = reach_state_offset_[i] + 2*(n_nodes_-1) + 1;
            break;
        }
    }
    
    if (gauge_idx < 0) {
        std::cerr << "Gauge reach " << gauge_reach_id << " not found" << std::endl;
        return;
    }
    
    // Initialize adjoint state: λ(T) = ∂L/∂y(T)
    // Only the gauge Q component is non-zero
    yB_ = N_VNew_Serial(total_state_size_, sunctx_);
    N_VConst(0.0, yB_);
    
    // Reset gradient accumulators
    std::fill(grad_manning_n_.begin(), grad_manning_n_.end(), 0.0);
    
    // --- Regenerate the checkpointed forward trajectory with the COMPLETE forcing history.
    // During the live forward pass lateral_history_ is built up one step at a time, so at
    // each window endpoint the time->step index clamps down to the last-pushed entry. The
    // adjoint reconstruction (CVAdataStore), however, runs after recording finishes and sees
    // the COMPLETE history, so the same endpoint maps to the next step's forcing. With
    // time-varying forcing those differ, the reconstruction's adaptive stepping diverges from
    // the forward pass, overruns the pre-allocated checkpoint buffer, and dereferences NULL.
    // Re-running the forward here -- history now complete and fixed -- makes the checkpoints
    // bit-consistent with reconstruction. (Confirmed root cause: constant-in-time forcing
    // never crashes; time-varying forcing crashes at exit 139, fixed by this regeneration.)
    {
        N_Vector y0 = N_VNew_Serial(total_state_size_, sunctx_);
        std::copy(record_y0_.begin(), record_y0_.end(), N_VGetArrayPointer(y0));
        CVodeReInit(cvode_mem_, record_t0_, y0);
        CVodeAdjReInit(cvode_mem_);
        sunrealtype tret; int nck = 0;
        for (size_t k = 0; k < recorded_times_.size(); ++k) {
            int fflag = CVodeF(cvode_mem_, recorded_times_[k], y_, &tret, CV_NORMAL, &nck);
            if (fflag < 0) {
                std::cerr << "Forward checkpoint regeneration error " << fflag
                          << " at t=" << recorded_times_[k] << std::endl;
                break;
            }
        }
        ncheck_ = nck;
        N_VDestroy(y0);
    }

    // Backward integration through recorded times
    // Start from final time
    double t_final = recorded_times_.back();

    // Create backward problem
    int flag = CVodeCreateB(cvode_mem_, CV_BDF, &indexB_);
    if (flag != CV_SUCCESS) {
        std::cerr << "Error creating backward problem" << std::endl;
        return;
    }
    
    // Store the per-observation loss gradient; the adjoint RHS consumes it as a continuous
    // running-cost source (see adjoint_rhs_callback). Terminal condition is therefore
    // λ(T) = 0 -- ALL observations (including the one at t_final) enter via the source term.
    dL_dQ_ = dL_dQ;
    N_VConst(0.0, yB_);

    flag = CVodeInitB(cvode_mem_, indexB_, adjoint_rhs_callback, t_final, yB_);
    flag = CVodeSStolerancesB(cvode_mem_, indexB_, config_.rel_tol, config_.abs_tol);
    flag = CVodeSetUserDataB(cvode_mem_, indexB_, this);
    // The running-cost source g_y(t) is piecewise-constant with a jump at every observation
    // time (stride dt), which forces the BDF integrator to take many tiny steps on the
    // backward solve. CVodeB defaults to mxstep=500 (it does NOT inherit the forward
    // CVodeSetMaxNumSteps), so without this the backward integration dies with a -1 mxstep
    // error a few percent of the way home -- leaving the quadrature with a tiny fraction of
    // the true gradient (observed: AD ~1000x smaller than FD). Raise it generously.
    flag = CVodeSetMaxNumStepsB(cvode_mem_, indexB_, 1000000);

    // Create linear solver for backward problem
    SUNMatrix JB = SUNDenseMatrix(total_state_size_, total_state_size_, sunctx_);
    SUNLinearSolver LSB = SUNLinSol_Dense(yB_, JB, sunctx_);
    flag = CVodeSetLinearSolverB(cvode_mem_, indexB_, LSB, JB);
    
    // Initialize quadrature for parameter sensitivity
    N_Vector qB = N_VNew_Serial(n_reaches_, sunctx_);
    N_VConst(0.0, qB);
    flag = CVodeQuadInitB(cvode_mem_, indexB_, quad_rhs_callback, qB);
    flag = CVodeQuadSStolerancesB(cvode_mem_, indexB_, config_.rel_tol, config_.abs_tol);
    flag = CVodeSetQuadErrConB(cvode_mem_, indexB_, SUNTRUE);
    
    // Backward integration of the discrete-observation adjoint.
    // The loss L = sum_k l_k(Q(t_k)) contributes an impulse dL/dQ(t_k) to the adjoint at
    // every observation time; between observations the adjoint obeys lambda' = -J^T lambda.
    // Rather than the textbook approach of stepping CVodeB to each t_k and applying the
    // impulse via CVodeGetB/CVodeReInitB (which restarts BDF at order 1 every step ->
    // compounding error that washes out the multi-reach costate), we fold the impulses into
    // a single continuous running-cost source g_y(t) in adjoint_rhs_callback and do ONE
    // CVodeB from t_final to record_t0_. Terminal condition lambda(t_final) = 0.
    const bool dbg = config_.verbose;
    if (dbg) {
        std::cerr << "[adj] gauge_state_idx=" << gauge_state_idx_
                  << " t_final=" << t_final << " record_t0=" << record_t0_
                  << " N_obs=" << recorded_times_.size() << std::endl;
    }
    sunrealtype tBret;
    // SINGLE continuous backward solve from t_final to record_t0_, no per-observation reinit.
    // The discrete observations enter as the running-cost source in adjoint_rhs_callback.
    flag = CVodeB(cvode_mem_, record_t0_, CV_NORMAL);
    if (flag < 0) {
        std::cerr << "Backward integration error " << flag
                  << " integrating to t=" << record_t0_ << std::endl;
    }

    // Final accumulated quadrature (parameter gradients).
    // CVODES integrates the backward problem (and its quadrature) from t_final down to t0,
    // so CVodeGetQuadB returns the integral over the BACKWARD path; dL/dp = +integral over
    // forward time, hence the sign flip below (validated: single-step adjoint matches FD).
    flag = CVodeGetQuadB(cvode_mem_, indexB_, &tBret, qB);

    double* grad = N_VGetArrayPointer(qB);
    for (int r = 0; r < n_reaches_; ++r) {
        grad_manning_n_[r] = -grad[r];
    }
    if (dbg) {
        std::cerr << "[adj] final grad:";
        for (int r = 0; r < n_reaches_; ++r) std::cerr << " " << grad_manning_n_[r];
        std::cerr << std::endl;
    }
    
    // Cleanup backward problem
    SUNLinSolFree(LSB);
    SUNMatDestroy(JB);
    N_VDestroy(qB);

    if (config_.verbose) {
        std::cout << "  Gradients computed via CVODES adjoint + Enzyme" << std::endl;
    }

#else
    std::cerr << "SUNDIALS required for adjoint gradient computation" << std::endl;
#endif
}

inline std::unordered_map<std::string, double> SaintVenantEnzyme::get_gradients() const {
    std::unordered_map<std::string, double> grads;
    const auto& topo_order = network_.topological_order();
    
    for (int i = 0; i < n_reaches_; ++i) {
        std::string key = "reach_" + std::to_string(topo_order[i]) + "_manning_n";
        grads[key] = grad_manning_n_[i];
    }
    
    return grads;
}

inline void SaintVenantEnzyme::reset_gradients() {
    std::fill(grad_manning_n_.begin(), grad_manning_n_.end(), 0.0);
}

} // namespace dmc

#endif // DMC_SAINT_VENANT_ENZYME_HPP
