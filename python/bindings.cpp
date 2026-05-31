/**

 * @file bindings.cpp
 * @brief Python bindings for dMC-Route using pybind11
 * 
 * Exposes the C++ routing library to Python for efficient integration
 * with PyTorch and other ML frameworks. Eliminates subprocess/CSV overhead.
 * 
 * Usage:
 *   import droute
 *   
 *   net = droute.Network()
 *   net.load_from_geojson("network.geojson")
 *   
 *   config = droute.RouterConfig()
 *   config.dt = 3600.0
 *   config.enable_gradients = True
 *   
 *   router = droute.MuskingumCungeRouter(net, config)
 *   router.start_recording()
 *   
 *   for t in range(num_timesteps):
 *       router.set_lateral_inflows(runoff[t, :])
 *       router.route_timestep()
 *   
 *   router.stop_recording()
 *   router.compute_gradients([outlet_id], [1.0])
 *   grads = router.get_gradients()
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 #include <pybind11/numpy.h>
 #include <pybind11/functional.h>
 #include <dmc/ad_backend.hpp>
 #include <dmc/router.hpp>
 #include <dmc/advanced_routing.hpp>
 #include <dmc/network.hpp>
 #include <dmc/network_io.hpp>
 #include <dmc/unified_router.hpp>
 #include <dmc/saint_venant_router.hpp>
 #include <dmc/saint_venant_enzyme.hpp>
 #include <dmc/parallel_routing.hpp>
 
 namespace py = pybind11;
 using namespace dmc;
 
 // Helper to convert std::vector to numpy array
 template<typename T>
 py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
     return py::array_t<T>(vec.size(), vec.data());
 }
 
 // Helper to get Real value as double
 inline double get_real_value(const Real& r) {
     return to_double(r);
 }
 
 // Helper to set Real value from double
 inline void set_real_value(Real& r, double v) {
     r = Real(v);
 }
 
 // Wrapper function for loading network from GeoJSON
 inline Network load_network_geojson(const std::string& filepath) {
     NetworkIO io;
     return io.load_geojson(filepath);
 }
 
 // Wrapper function for loading network from CSV
 inline Network load_network_csv(const std::string& reaches_csv, 
                                  const std::string& params_csv = "") {
     NetworkIO io;
     return io.load_csv(reaches_csv, params_csv);
 }
 
// =============================================================================
// ENZYME AD DECLARATIONS AND OBJECTIVE FUNCTION (at file scope)
// =============================================================================

#ifdef DMC_USE_ENZYME

// Simple objective function for Enzyme to differentiate
// Computes: sum_t( grad_output[t] * Q_outlet[t] )
static double routing_objective(
    int n_reaches,
    int n_timesteps,
    int outlet_reach,
    double dt,
    const double* manning_n,        // [n_reaches] - DIFFERENTIATE W.R.T. THIS
    const double* lateral_inflows,  // [n_timesteps * n_reaches] 
    const double* grad_output,      // [n_timesteps]
    const double* lengths,          // [n_reaches]
    const double* slopes,           // [n_reaches]
    const double* width_coefs,      // [n_reaches]
    const double* width_exps,       // [n_reaches]
    const double* depth_coefs,      // [n_reaches]
    const double* depth_exps,       // [n_reaches]
    const int* topo_order,          // [n_reaches]
    const int* upstream_counts,     // [n_reaches]
    const int* upstream_offsets,    // [n_reaches + 1]
    const int* upstream_indices     // [total_upstream]
) {
    // Allocate state on stack for small networks, heap for large
    constexpr int STACK_THRESHOLD = 256;
    double stack_Q[STACK_THRESHOLD];
    double stack_Q_prev[STACK_THRESHOLD];
    
    double* Q = (n_reaches <= STACK_THRESHOLD) ? stack_Q : new double[n_reaches];
    double* Q_prev = (n_reaches <= STACK_THRESHOLD) ? stack_Q_prev : new double[n_reaches];
    
    // Initialize
    for (int i = 0; i < n_reaches; ++i) {
        Q[i] = 0.0;
        Q_prev[i] = 0.0;
    }
    
    double weighted_sum = 0.0;
    
    for (int t = 0; t < n_timesteps; ++t) {
        // Process reaches in topological order
        for (int idx = 0; idx < n_reaches; ++idx) {
            int reach = topo_order[idx];
            
            // Sum upstream inflows
            double Q_upstream = 0.0;
            int n_up = upstream_counts[reach];
            int offset = upstream_offsets[reach];
            for (int u = 0; u < n_up; ++u) {
                int up_reach = upstream_indices[offset + u];
                Q_upstream += Q[up_reach];
            }
            
            // Add lateral inflow
            double Q_in = Q_upstream + lateral_inflows[t * n_reaches + reach];
            
            // Muskingum-Cunge routing
            double length = lengths[reach];
            double slope = std::max(slopes[reach], 1e-6);
            double n = std::max(manning_n[reach], 0.001);
            double w_coef = width_coefs[reach];
            double w_exp = width_exps[reach];
            double d_coef = depth_coefs[reach];
            double d_exp = depth_exps[reach];
            
            // Reference discharge
            double Q_ref = std::max(std::max(Q_prev[reach], Q_in), 0.01);
            
            // Hydraulic geometry (power law)
            double width = w_coef * std::pow(Q_ref, w_exp);
            double depth = d_coef * std::pow(Q_ref, d_exp);
            width = std::max(width, 0.5);
            depth = std::max(depth, 0.05);
            
            // Hydraulic radius
            double area = width * depth;
            double perimeter = width + 2.0 * depth;
            double R_h = area / perimeter;
            
            // Velocity and celerity (Manning's equation)
            double velocity = (1.0 / n) * std::pow(R_h, 2.0/3.0) * std::sqrt(slope);
            double celerity = (5.0 / 3.0) * velocity;
            celerity = std::max(celerity, 0.1);
            
            // Muskingum parameters
            double K = length / celerity;
            double X = 0.2;
            
            // Muskingum coefficients
            double denom = 2.0 * K * (1.0 - X) + dt;
            double C0 = (dt - 2.0 * K * X) / denom;
            double C1 = (dt + 2.0 * K * X) / denom;
            double C2 = (2.0 * K * (1.0 - X) - dt) / denom;
            
            // Route
            double Q_out = C0 * Q_in + C1 * Q_in + C2 * Q_prev[reach];
            Q[reach] = std::max(Q_out, 0.0);
        }
        
        // Accumulate weighted objective
        weighted_sum += grad_output[t] * Q[outlet_reach];
        
        // Update previous state
        for (int i = 0; i < n_reaches; ++i) {
            Q_prev[i] = Q[i];
        }
    }
    
    // Cleanup heap allocation if used
    if (n_reaches > STACK_THRESHOLD) {
        delete[] Q;
        delete[] Q_prev;
    }
    
    return weighted_sum;
}

#endif // DMC_USE_ENZYME

 PYBIND11_MODULE(_droute_core, m) {
     m.doc() = R"pbdoc(
         dRoute: Differentiable River Routing Library
         ------------------------------------------------
         
         A high-performance, differentiable routing library for
         hydrological modeling and machine learning applications.
         
         Features:
         - Multiple routing methods (Muskingum-Cunge, IRF, Diffusive Wave, KWT)
         - Automatic differentiation via CoDiPack
         - Direct memory access (no CSV/subprocess overhead)
         - NumPy array support for efficient data transfer
         
         Example:
            >>> import droute
             >>> net = droute.Network()
             >>> config = droute.RouterConfig()
             >>> router = droute.MuskingumCungeRouter(net, config)
     )pbdoc";
 
     // =========================================================================
     // RouterConfig
     // =========================================================================
     py::class_<RouterConfig>(m, "RouterConfig", "Configuration for routing methods")
         .def(py::init<>())
         
         // Basic settings
         .def_readwrite("dt", &RouterConfig::dt,
             "Timestep in seconds (default: 3600.0)")
         .def_readwrite("enable_gradients", &RouterConfig::enable_gradients,
             "Enable gradient computation via AD (default: True)")
         .def_readwrite("min_flow", &RouterConfig::min_flow,
             "Minimum flow for numerical stability (default: 1e-6)")
         .def_readwrite("x_lower_bound", &RouterConfig::x_lower_bound,
             "Lower bound for Muskingum X parameter (default: 0.0)")
         .def_readwrite("x_upper_bound", &RouterConfig::x_upper_bound,
             "Upper bound for Muskingum X parameter (default: 0.5)")
         
         // AD smoothing
         .def_readwrite("use_smooth_bounds", &RouterConfig::use_smooth_bounds,
             "Use smooth min/max/clamp for AD (default: True)")
         .def_readwrite("smooth_epsilon", &RouterConfig::smooth_epsilon,
             "Epsilon for smooth functions (default: 1e-6)")
         
         // Sub-stepping
         .def_readwrite("fixed_substepping", &RouterConfig::fixed_substepping,
             "Use fixed substep count for AD safety (default: True)")
         .def_readwrite("num_substeps", &RouterConfig::num_substeps,
             "Number of substeps per reach (default: 4)")
         .def_readwrite("adaptive_substepping", &RouterConfig::adaptive_substepping,
             "Use adaptive substeps - breaks AD (default: False)")
         .def_readwrite("max_substeps", &RouterConfig::max_substeps,
             "Maximum substeps for adaptive mode (default: 10)")
         
         // IRF options
         .def_readwrite("irf_max_kernel_size", &RouterConfig::irf_max_kernel_size,
             "Maximum IRF kernel length in timesteps (default: 200)")
         .def_readwrite("irf_shape_param", &RouterConfig::irf_shape_param,
             "Gamma shape parameter for IRF (default: 2.5)")
         .def_readwrite("irf_soft_mask", &RouterConfig::irf_soft_mask,
             "Use sigmoid masking for differentiable IRF (default: True)")
         .def_readwrite("irf_mask_steepness", &RouterConfig::irf_mask_steepness,
             "Steepness of IRF sigmoid mask (default: 10.0)")
         
         // Diffusive wave options
         .def_readwrite("dw_num_nodes", &RouterConfig::dw_num_nodes,
             "Spatial nodes per reach for diffusive wave (default: 10)")
         .def_readwrite("dw_use_ift_adjoint", &RouterConfig::dw_use_ift_adjoint,
             "Use IFT for diffusive wave adjoints (default: True)")
         
         // KWT-Soft options
         .def_readwrite("kwt_gate_steepness", &RouterConfig::kwt_gate_steepness,
             "Soft gate steepness for KWT (default: 5.0)")
         .def_readwrite("kwt_anneal_steepness", &RouterConfig::kwt_anneal_steepness,
             "Enable steepness annealing during training (default: False)")
         .def_readwrite("kwt_steepness_min", &RouterConfig::kwt_steepness_min,
             "Minimum steepness for annealing (default: 1.0)")
         .def_readwrite("kwt_steepness_max", &RouterConfig::kwt_steepness_max,
             "Maximum steepness for annealing (default: 20.0)")
         
         // Memory management
         .def_readwrite("enable_checkpointing", &RouterConfig::enable_checkpointing,
             "Enable Revolve checkpointing (default: False)")
         .def_readwrite("checkpoint_interval", &RouterConfig::checkpoint_interval,
             "Timesteps between checkpoints (default: 1000)")
         
         // Parallelization
         .def_readwrite("parallel_routing", &RouterConfig::parallel_routing,
             "Enable OpenMP parallel routing (default: False)")
         .def_readwrite("num_threads", &RouterConfig::num_threads,
             "Thread count for parallel mode (default: 4)")
         
         .def("__repr__", [](const RouterConfig& c) {
             return "<RouterConfig dt=" + std::to_string(c.dt) + 
                    " enable_gradients=" + (c.enable_gradients ? "True" : "False") + ">";
         });
 
     // =========================================================================
     // ChannelGeometry
     // =========================================================================
     py::class_<ChannelGeometry>(m, "ChannelGeometry", "Channel geometry parameters")
         .def(py::init<>())
         .def_property("width_coef",
             [](const ChannelGeometry& g) { return to_double(g.width_coef); },
             [](ChannelGeometry& g, double v) { g.width_coef = Real(v); },
             "Width coefficient in W = a * Q^b")
         .def_property("width_exp",
             [](const ChannelGeometry& g) { return to_double(g.width_exp); },
             [](ChannelGeometry& g, double v) { g.width_exp = Real(v); },
             "Width exponent in W = a * Q^b")
         .def_property("depth_coef",
             [](const ChannelGeometry& g) { return to_double(g.depth_coef); },
             [](ChannelGeometry& g, double v) { g.depth_coef = Real(v); },
             "Depth coefficient in D = c * Q^d")
         .def_property("depth_exp",
             [](const ChannelGeometry& g) { return to_double(g.depth_exp); },
             [](ChannelGeometry& g, double v) { g.depth_exp = Real(v); },
             "Depth exponent in D = c * Q^d")
         .def("width", [](const ChannelGeometry& g, double Q) {
             return to_double(g.width(Real(Q)));
         }, "Compute channel width for given discharge")
         .def("depth", [](const ChannelGeometry& g, double Q) {
             return to_double(g.depth(Real(Q)));
         }, "Compute channel depth for given discharge");
 
     // =========================================================================
     // Reach
     // =========================================================================
     py::class_<Reach>(m, "Reach", "River reach segment")
         .def(py::init<>())
         .def_readwrite("id", &Reach::id)
         .def_readwrite("name", &Reach::name)
         .def_readwrite("length", &Reach::length, "Reach length in meters")
         .def_readwrite("slope", &Reach::slope, "Channel slope (m/m)")
         .def_readwrite("geometry", &Reach::geometry, "Channel geometry")
         .def_readwrite("upstream_junction_id", &Reach::upstream_junction_id)
         .def_readwrite("downstream_junction_id", &Reach::downstream_junction_id)
         
         // Manning's n as property (Real -> double conversion)
         .def_property("manning_n",
             [](const Reach& r) { return to_double(r.manning_n); },
             [](Reach& r, double v) { r.manning_n = Real(v); },
             "Manning's roughness coefficient")
         
         // Lateral inflow
         .def_property("lateral_inflow",
             [](const Reach& r) { return to_double(r.lateral_inflow); },
             [](Reach& r, double v) { r.lateral_inflow = Real(v); },
             "Lateral inflow rate (m³/s)")
         
         // Current state (read-only)
         .def_property_readonly("inflow",
             [](const Reach& r) { return to_double(r.inflow_curr); },
             "Current inflow (m³/s)")
         .def_property_readonly("outflow",
             [](const Reach& r) { return to_double(r.outflow_curr); },
             "Current outflow (m³/s)")
         
         // Gradients (read-only) - use lambdas for member access
         .def_property_readonly("grad_manning_n", 
             [](const Reach& r) { return r.grad_manning_n; },
             "Gradient w.r.t. Manning's n")
         .def_property_readonly("grad_width_coef",
             [](const Reach& r) { return r.grad_width_coef; },
             "Gradient w.r.t. width coefficient")
         .def_property_readonly("grad_width_exp",
             [](const Reach& r) { return r.grad_width_exp; },
             "Gradient w.r.t. width exponent")
         .def_property_readonly("grad_depth_coef",
             [](const Reach& r) { return r.grad_depth_coef; },
             "Gradient w.r.t. depth coefficient")
         .def_property_readonly("grad_depth_exp",
             [](const Reach& r) { return r.grad_depth_exp; },
             "Gradient w.r.t. depth exponent");
 
     // =========================================================================
     // Junction
     // =========================================================================
     py::class_<Junction>(m, "Junction", "River junction (confluence/bifurcation)")
         .def(py::init<>())
         .def_readwrite("id", &Junction::id)
         .def_readwrite("name", &Junction::name)
         .def_readwrite("upstream_reach_ids", &Junction::upstream_reach_ids)
         .def_readwrite("downstream_reach_ids", &Junction::downstream_reach_ids)
         .def_property("external_inflow",
             [](const Junction& j) { return to_double(j.external_inflow); },
             [](Junction& j, double v) { j.external_inflow = Real(v); },
             "External inflow at junction (m³/s)");
 
     // =========================================================================
     // Network
     // =========================================================================
     py::class_<Network>(m, "Network", "River network topology")
         .def(py::init<>())
         
         // Building network
         .def("add_reach", &Network::add_reach, "Add a reach to the network")
         .def("add_junction", &Network::add_junction, "Add a junction to the network")
         .def("build_topology", &Network::build_topology, "Build network topology")
         
         // Accessors
         .def("get_reach", py::overload_cast<int>(&Network::get_reach),
             py::return_value_policy::reference, "Get reach by ID")
         .def("get_junction", py::overload_cast<int>(&Network::get_junction),
             py::return_value_policy::reference, "Get junction by ID")
         .def("topological_order", &Network::topological_order,
             "Get reach IDs in topological (upstream-to-downstream) order")
         .def("num_reaches", &Network::num_reaches, "Number of reaches")
         .def("num_junctions", &Network::num_junctions, "Number of junctions")
         
         // Gradients
         .def("zero_gradients", &Network::zero_gradients, "Reset all gradients to zero")
         
         // Bulk operations for efficiency
         .def("set_manning_n_all", [](Network& net, py::array_t<double> values) {
             auto buf = values.request();
             double* ptr = static_cast<double*>(buf.ptr);
             const auto& order = net.topological_order();
             for (size_t i = 0; i < order.size() && i < static_cast<size_t>(buf.size); ++i) {
                 net.get_reach(order[i]).manning_n = Real(ptr[i]);
             }
         }, "Set Manning's n for all reaches (in topological order)")
         
         .def("get_manning_n_all", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> values(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 values[i] = to_double(net.get_reach(order[i]).manning_n);
             }
             return vector_to_numpy(values);
         }, "Get Manning's n for all reaches (in topological order)")
         
         .def("get_grad_manning_n_all", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> grads(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 grads[i] = net.get_reach(order[i]).grad_manning_n;
             }
             return vector_to_numpy(grads);
         }, "Get gradients w.r.t. Manning's n for all reaches")
         
         .def("set_lateral_inflows", [](Network& net, py::array_t<double> values) {
             auto buf = values.request();
             double* ptr = static_cast<double*>(buf.ptr);
             const auto& order = net.topological_order();
             for (size_t i = 0; i < order.size() && i < static_cast<size_t>(buf.size); ++i) {
                 net.get_reach(order[i]).lateral_inflow = Real(ptr[i]);
             }
         }, "Set lateral inflows for all reaches (in topological order)")
         
         .def("get_outflows", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> outflows(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 outflows[i] = to_double(net.get_reach(order[i]).outflow_curr);
             }
             return vector_to_numpy(outflows);
         }, "Get current outflows for all reaches (in topological order)");
 
     // =========================================================================
     // Network I/O
     // =========================================================================
     m.def("load_network_geojson", &load_network_geojson,
         "Load network from GeoJSON file",
         py::arg("filepath"));
     
     m.def("load_network_csv", &load_network_csv,
         "Load network from CSV file",
         py::arg("reaches_csv"),
         py::arg("params_csv") = "");
     
     py::class_<NetworkLoadConfig>(m, "NetworkLoadConfig",
         "Configuration for network loading")
         .def(py::init<>())
         .def_readwrite("reach_id_prop", &NetworkLoadConfig::reach_id_prop)
         .def_readwrite("reach_length_prop", &NetworkLoadConfig::reach_length_prop)
         .def_readwrite("reach_slope_prop", &NetworkLoadConfig::reach_slope_prop)
         .def_readwrite("from_node_prop", &NetworkLoadConfig::from_node_prop)
         .def_readwrite("to_node_prop", &NetworkLoadConfig::to_node_prop)
         .def_readwrite("manning_n_prop", &NetworkLoadConfig::manning_n_prop)
         .def_readwrite("length_to_meters", &NetworkLoadConfig::length_to_meters)
         .def_readwrite("slope_factor", &NetworkLoadConfig::slope_factor)
         .def_readwrite("default_slope", &NetworkLoadConfig::default_slope)
         .def_readwrite("default_manning_n", &NetworkLoadConfig::default_manning_n)
         .def_readwrite("default_length", &NetworkLoadConfig::default_length)
         .def_readwrite("min_slope", &NetworkLoadConfig::min_slope);
     
     py::class_<NetworkIO>(m, "NetworkIO",
         "Network I/O helper class")
         .def(py::init<NetworkLoadConfig>(), py::arg("config") = NetworkLoadConfig())
         .def("load_geojson", &NetworkIO::load_geojson, py::arg("filepath"))
         .def("load_csv", &NetworkIO::load_csv, 
             py::arg("reaches_csv"), py::arg("params_csv") = "")
         .def("save_geojson", &NetworkIO::save_geojson,
             py::arg("network"), py::arg("filepath"))
         .def("save_csv", &NetworkIO::save_csv,
             py::arg("network"), py::arg("filepath"));
 
     // =========================================================================
     // MuskingumCungeRouter
     // =========================================================================
     py::class_<MuskingumCungeRouter>(m, "MuskingumCungeRouter",
         "Muskingum-Cunge routing with full AD support")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         
         // Core routing
         .def("route_timestep", &MuskingumCungeRouter::route_timestep,
             "Route one timestep")
         .def("route", &MuskingumCungeRouter::route, "Route multiple timesteps",
             py::arg("num_timesteps"))
         
         // Gradient computation
         .def("enable_gradients", &MuskingumCungeRouter::enable_gradients,
             "Enable/disable gradient computation")
         .def("start_recording", &MuskingumCungeRouter::start_recording,
             "Start recording for AD")
         .def("stop_recording", &MuskingumCungeRouter::stop_recording,
             "Stop recording for AD")
         .def("compute_gradients", &MuskingumCungeRouter::compute_gradients,
             "Compute gradients w.r.t. loss at gauge reaches",
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &MuskingumCungeRouter::get_gradients,
             "Get gradients as dict")
         .def("reset_gradients", &MuskingumCungeRouter::reset_gradients,
             "Reset gradients to zero")
         
         // State access
         .def("set_lateral_inflow", &MuskingumCungeRouter::set_lateral_inflow,
             "Set lateral inflow for a reach",
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &MuskingumCungeRouter::set_lateral_inflows,
             "Set lateral inflows for all reaches",
             py::arg("inflows"))
         .def("get_discharge", &MuskingumCungeRouter::get_discharge,
             "Get discharge at a reach",
             py::arg("reach_id"))
         .def("get_all_discharges", &MuskingumCungeRouter::get_all_discharges,
             "Get discharges at all reaches")
         .def("reset_state", &MuskingumCungeRouter::reset_state,
             "Reset router state")
         
         // Properties
         .def("current_time", &MuskingumCungeRouter::current_time,
             "Current simulation time")
         .def_property_readonly("network", 
             py::overload_cast<>(&MuskingumCungeRouter::network),
             py::return_value_policy::reference,
             "Access the network")
         .def_property_readonly("config", &MuskingumCungeRouter::config,
             "Access the configuration")
         
         // Timeseries gradient methods
         .def("record_output", &MuskingumCungeRouter::record_output,
             "Record current discharge at a reach for gradient computation",
             py::arg("reach_id"))
         .def("record_outputs", &MuskingumCungeRouter::record_outputs,
             "Record current discharge at multiple reaches",
             py::arg("reach_ids"))
         .def("clear_output_history", &MuskingumCungeRouter::clear_output_history,
             "Clear recorded output history")
         .def("get_output_history_size", &MuskingumCungeRouter::get_output_history_size,
             "Get number of recorded timesteps for a reach",
             py::arg("reach_id"))
         .def("get_output_history", &MuskingumCungeRouter::get_output_history,
             "Get recorded output values for a reach",
             py::arg("reach_id"))
         .def("compute_gradients_timeseries", 
             py::overload_cast<int, const std::vector<double>&>(
                 &MuskingumCungeRouter::compute_gradients_timeseries),
             R"doc(
             Compute gradients for full timeseries at a single reach.
             
             This properly accumulates gradients across all timesteps.
             
             Args:
                 reach_id: Reach ID where we have observations
                 dL_dQ: Gradient of loss w.r.t. discharge at each timestep.
                        For MSE: dL_dQ[t] = 2 * (sim[t] - obs[t]) / n_timesteps
                        Must have same length as recorded outputs.
             )doc",
             py::arg("reach_id"), py::arg("dL_dQ"))
         .def("compute_gradients_timeseries",
             py::overload_cast<const std::vector<int>&,
                               const std::vector<std::vector<double>>&>(
                 &MuskingumCungeRouter::compute_gradients_timeseries),
             R"doc(
             Compute gradients for full timeseries at multiple gauge reaches.

             Seeds the adjoint at every gauge and performs a single reverse pass,
             so the returned gradients are w.r.t. a loss summed over all gauges.

             Args:
                 reach_ids: list of reach IDs with observations.
                 dL_dQ: list (one per reach) of per-timestep loss gradients; each
                        inner list must match the recorded-output length.
             )doc",
             py::arg("reach_ids"), py::arg("dL_dQ"))

         // State serialization for checkpointing
         .def("save_state", &MuskingumCungeRouter::save_state,
             "Save current router state for checkpointing")
         .def("load_state", &MuskingumCungeRouter::load_state,
             "Load router state from checkpoint",
             py::arg("state"))
         .def("save_state_to_file", &MuskingumCungeRouter::save_state_to_file,
             "Save state to file",
             py::arg("filepath"))
         .def("load_state_from_file", &MuskingumCungeRouter::load_state_from_file,
             "Load state from file",
             py::arg("filepath"));
    
     // RouterState for checkpointing
     py::class_<RouterState>(m, "RouterState",
         "Router state for checkpointing and serialization")
         .def(py::init<>())
         .def_readwrite("time", &RouterState::time)
         .def_readwrite("inflows", &RouterState::inflows)
         .def_readwrite("outflows", &RouterState::outflows)
         .def("save", &RouterState::save,
             "Save state to file",
             py::arg("filepath"))
         .def_static("load", &RouterState::load,
             "Load state from file",
             py::arg("filepath"));
 
     // =========================================================================
     // IRFRouter
     // =========================================================================
     py::class_<IRFRouter>(m, "IRFRouter",
         "Impulse Response Function routing with soft-masked kernel")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &IRFRouter::route_timestep)
         .def("route", &IRFRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &IRFRouter::enable_gradients)
         .def("start_recording", &IRFRouter::start_recording)
         .def("stop_recording", &IRFRouter::stop_recording)
         .def("compute_gradients", &IRFRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &IRFRouter::get_gradients)
         .def("reset_gradients", &IRFRouter::reset_gradients)
         .def("set_lateral_inflow", &IRFRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &IRFRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &IRFRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &IRFRouter::get_all_discharges)
         .def("reset_state", &IRFRouter::reset_state)
         .def("current_time", &IRFRouter::current_time);
 
     // =========================================================================
     // DiffusiveWaveRouter
     // =========================================================================
     py::class_<DiffusiveWaveRouter>(m, "DiffusiveWaveRouter",
         "Diffusive wave routing with analytical gradients")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &DiffusiveWaveRouter::route_timestep)
         .def("route", &DiffusiveWaveRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &DiffusiveWaveRouter::enable_gradients)
         .def("start_recording", &DiffusiveWaveRouter::start_recording)
         .def("stop_recording", &DiffusiveWaveRouter::stop_recording)
         .def("compute_gradients", &DiffusiveWaveRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &DiffusiveWaveRouter::get_gradients)
         .def("reset_gradients", &DiffusiveWaveRouter::reset_gradients)
         .def("set_lateral_inflow", &DiffusiveWaveRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &DiffusiveWaveRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &DiffusiveWaveRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &DiffusiveWaveRouter::get_all_discharges)
         .def("reset_state", &DiffusiveWaveRouter::reset_state)
         .def("current_time", &DiffusiveWaveRouter::current_time);
 
     // =========================================================================
     // LagRouter
     // =========================================================================
     py::class_<LagRouter>(m, "LagRouter",
         "Simple lag routing (forward-only, not recommended for calibration)")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &LagRouter::route_timestep)
         .def("route", &LagRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &LagRouter::enable_gradients)
         .def("start_recording", &LagRouter::start_recording)
         .def("stop_recording", &LagRouter::stop_recording)
         .def("compute_gradients", &LagRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &LagRouter::get_gradients)
         .def("reset_gradients", &LagRouter::reset_gradients)
         .def("set_lateral_inflow", &LagRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &LagRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &LagRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &LagRouter::get_all_discharges)
         .def("reset_state", &LagRouter::reset_state)
         .def("current_time", &LagRouter::current_time);
 
     // =========================================================================
     // KWTRouter
     // =========================================================================
     py::class_<KWTRouter>(m, "KWTRouter",
         "Kinematic Wave Tracking (forward-only, mizuRoute compatible)")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &KWTRouter::route_timestep)
         .def("route", &KWTRouter::route, py::arg("num_timesteps"))
         .def("set_lateral_inflow", &KWTRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &KWTRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &KWTRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &KWTRouter::get_all_discharges)
         .def("reset_state", &KWTRouter::reset_state)
         .def("current_time", &KWTRouter::current_time);
 
     // =========================================================================
     // Advanced Routers (from advanced_routing.hpp)
     // =========================================================================
     py::class_<DiffusiveWaveIFT>(m, "DiffusiveWaveIFT",
         "Diffusive wave with IFT adjoint for exact gradients")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &DiffusiveWaveIFT::route_timestep)
         .def("route", &DiffusiveWaveIFT::route, py::arg("num_timesteps"))
         .def("enable_gradients", &DiffusiveWaveIFT::enable_gradients)
         .def("start_recording", &DiffusiveWaveIFT::start_recording)
         .def("stop_recording", &DiffusiveWaveIFT::stop_recording)
         .def("compute_gradients", &DiffusiveWaveIFT::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &DiffusiveWaveIFT::get_gradients)
         .def("reset_gradients", &DiffusiveWaveIFT::reset_gradients)
         .def("set_lateral_inflow", &DiffusiveWaveIFT::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", [](DiffusiveWaveIFT& router, const std::vector<std::pair<int, double>>& inflows) {
             for (const auto& [id, q] : inflows) {
                 router.set_lateral_inflow(id, q);
             }
         }, py::arg("inflows"), "Set lateral inflows as list of (reach_id, inflow) pairs")
         .def("get_discharge", &DiffusiveWaveIFT::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", [](DiffusiveWaveIFT& router) {
             std::vector<double> discharges;
             for (int id : router.network().topological_order()) {
                 discharges.push_back(router.get_discharge(id));
             }
             return discharges;
         }, "Get discharges for all reaches in topological order")
         .def("reset_state", &DiffusiveWaveIFT::reset_state)
         .def("current_time", &DiffusiveWaveIFT::current_time);
 
     py::class_<SoftGatedKWT>(m, "SoftGatedKWT",
         "Soft-gated KWT with differentiable parcel tracking")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &SoftGatedKWT::route_timestep)
         .def("route", &SoftGatedKWT::route, py::arg("num_timesteps"))
         .def("enable_gradients", &SoftGatedKWT::enable_gradients)
         .def("start_recording", &SoftGatedKWT::start_recording)
         .def("stop_recording", &SoftGatedKWT::stop_recording)
         .def("compute_gradients", &SoftGatedKWT::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &SoftGatedKWT::get_gradients)
         .def("reset_gradients", &SoftGatedKWT::reset_gradients)
         .def("set_lateral_inflow", &SoftGatedKWT::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", [](SoftGatedKWT& router, const std::vector<std::pair<int, double>>& inflows) {
             for (const auto& [id, q] : inflows) {
                 router.set_lateral_inflow(id, q);
             }
         }, py::arg("inflows"), "Set lateral inflows as list of (reach_id, inflow) pairs")
         .def("get_discharge", &SoftGatedKWT::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &SoftGatedKWT::get_all_discharges)
         .def("reset_state", &SoftGatedKWT::reset_state)
         .def("current_time", &SoftGatedKWT::current_time)
         .def("set_steepness", &SoftGatedKWT::set_steepness,
             "Set soft gate steepness (for annealing)")
         .def("get_steepness", &SoftGatedKWT::get_steepness,
             "Get current soft gate steepness");
 
     // =========================================================================
     // Enzyme Kernels (Fast, AD-compatible)
     // =========================================================================
     
     py::module_ enzyme = m.def_submodule("enzyme", "Fast Enzyme-compatible routing kernels");
     
     // EnzymeRouter: High-performance router using flat-array kernels
     py::class_<EnzymeRouter>(enzyme, "EnzymeRouter",
         R"pbdoc(
         High-performance router using Enzyme-compatible flat-array kernels.
         
         This router is optimized for speed and can be used with numerical
         differentiation or (when compiled with Enzyme) automatic differentiation.
         
         Supports multiple routing methods:
         - 0: Muskingum-Cunge (default)
         - 1: Lag (simple delay)
         - 2: IRF (Impulse Response Function)
         - 3: KWT (Kinematic Wave Tracking)
         - 4: Diffusive Wave
         
         Example:
             >>> router = droute.enzyme.EnzymeRouter(network, method=0)
             >>> router.set_lateral_inflows(runoff)
             >>> router.route_timestep()
             >>> Q = router.get_discharges()
         )pbdoc")
         .def(py::init([](Network& network, double dt, int num_substeps, int method) {
             UnifiedRouterConfig config;
             config.dt = dt;
             config.num_substeps = num_substeps;
             config.routing_method = static_cast<EnzymeRoutingMethod>(method);
             return new EnzymeRouter(network, config);
         }),
             py::arg("network"),
             py::arg("dt") = 3600.0,
             py::arg("num_substeps") = 4,
             py::arg("method") = 0,
             "Create EnzymeRouter from network with specified routing method")
         
         .def("set_routing_method", [](EnzymeRouter& router, int method) {
             router.set_routing_method(static_cast<EnzymeRoutingMethod>(method));
         }, py::arg("method"), 
            "Set routing method: 0=MC, 1=Lag, 2=IRF, 3=KWT, 4=Diffusive")
         
         .def("get_routing_method", [](EnzymeRouter& router) {
             return static_cast<int>(router.get_routing_method());
         }, "Get current routing method")
         
         .def("route_timestep", &EnzymeRouter::route_timestep,
             "Route one timestep using selected routing method")
         
         .def("route", &EnzymeRouter::route, py::arg("num_timesteps"),
             "Route multiple timesteps")
         
         .def("set_lateral_inflow", &EnzymeRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"),
             "Set lateral inflow for a single reach")
         
         .def("set_lateral_inflows", [](EnzymeRouter& router, py::array_t<double> inflows) {
             auto buf = inflows.request();
             if (buf.ndim != 1) throw std::runtime_error("Expected 1D array");
             double* ptr = static_cast<double*>(buf.ptr);
             for (size_t i = 0; i < static_cast<size_t>(buf.size); ++i) {
                 router.set_lateral_inflow(static_cast<int>(i), ptr[i]);
             }
         }, py::arg("inflows"), "Set lateral inflows from numpy array")
         
         .def("get_discharge", &EnzymeRouter::get_discharge, py::arg("reach_id"),
             "Get discharge at a reach")
         
         .def("get_discharges", [](EnzymeRouter& router) {
             auto discharges = router.get_all_discharges();
             return py::array_t<double>(discharges.size(), discharges.data());
         }, "Get all discharges as numpy array")
         
         .def("reset_state", &EnzymeRouter::reset_state,
             "Reset router state to initial conditions")
         
         .def("set_manning_n", &EnzymeRouter::set_manning_n,
             py::arg("reach_id"), py::arg("manning_n"),
             "Set Manning's n for a reach")
         
         .def("set_manning_n_all", [](EnzymeRouter& router, py::array_t<double> values) {
             auto buf = values.request();
             if (buf.ndim != 1) throw std::runtime_error("Expected 1D array");
             double* ptr = static_cast<double*>(buf.ptr);
             for (size_t i = 0; i < static_cast<size_t>(buf.size); ++i) {
                 router.set_manning_n(static_cast<int>(i), ptr[i]);
             }
         }, py::arg("manning_n"), "Set Manning's n for all reaches from numpy array")
         
         .def("get_manning_n_all", [](EnzymeRouter& router) {
             auto values = router.get_manning_n_all();
             return py::array_t<double>(values.size(), values.data());
         }, "Get Manning's n for all reaches as numpy array")
         
         .def_property_readonly("num_reaches", &EnzymeRouter::num_reaches,
             "Number of reaches in network")
         
         .def_property_readonly("dt", &EnzymeRouter::dt,
             "Timestep in seconds")
         
         .def("get_topology_debug", &EnzymeRouter::get_topology_debug,
             "Get debug string showing network topology structure");
     
     // Compute numerical gradients
     enzyme.def("compute_gradients_numerical", [](
         EnzymeRouter& router,
         py::array_t<double> runoff,
         py::array_t<double> observed,
         int outlet_reach,
         double eps
     ) {
         auto runoff_buf = runoff.request();
         auto obs_buf = observed.request();
         
         if (runoff_buf.ndim != 2) throw std::runtime_error("runoff must be 2D (timesteps x reaches)");
         if (obs_buf.ndim != 1) throw std::runtime_error("observed must be 1D");
         
         int n_timesteps = runoff_buf.shape[0];
         int n_reaches = runoff_buf.shape[1];
         double* runoff_ptr = static_cast<double*>(runoff_buf.ptr);
         double* obs_ptr = static_cast<double*>(obs_buf.ptr);
         
         // Lambda to run simulation and compute MSE loss
         auto run_and_loss = [&](EnzymeRouter& r) -> double {
             r.reset_state();
             double mse = 0.0;
             for (int t = 0; t < n_timesteps; ++t) {
                 for (int i = 0; i < n_reaches; ++i) {
                     r.set_lateral_inflow(i, runoff_ptr[t * n_reaches + i]);
                 }
                 r.route_timestep();
                 double sim = r.get_discharge(outlet_reach);
                 double diff = sim - obs_ptr[t];
                 mse += diff * diff;
             }
             return mse / n_timesteps;
         };
         
         // Get current Manning's n values
         auto manning_n = router.get_manning_n_all();
         
         // Compute base loss
         double base_loss = run_and_loss(router);
         
         // Compute gradients via central differences
         std::vector<double> gradients(n_reaches);
         for (int i = 0; i < n_reaches; ++i) {
             double orig = manning_n[i];
             double h = eps * std::max(std::abs(orig), 0.001);
             
             // Forward perturbation
             router.set_manning_n(i, orig + h);
             double loss_plus = run_and_loss(router);
             
             // Backward perturbation  
             router.set_manning_n(i, orig - h);
             double loss_minus = run_and_loss(router);
             
             // Central difference
             gradients[i] = (loss_plus - loss_minus) / (2 * h);
             
             // Restore
             router.set_manning_n(i, orig);
         }
         
         // Return gradients and base loss - use py::cast for proper memory handling
         py::dict result;
         result["gradients"] = py::cast(gradients);
         result["loss"] = base_loss;
         return result;
     },
     py::arg("router"),
     py::arg("runoff"),
     py::arg("observed"),
     py::arg("outlet_reach"),
     py::arg("eps") = 0.001,
     R"pbdoc(
         Compute gradients of MSE loss w.r.t. Manning's n using numerical differentiation.
         
         This is fast because it uses the Enzyme kernels (no AD tape overhead).
         
         Args:
             router: EnzymeRouter instance
             runoff: Lateral inflows array (n_timesteps, n_reaches)
             observed: Observed discharge at outlet (n_timesteps,)
             outlet_reach: Index of outlet reach
             eps: Perturbation size for finite differences
             
         Returns:
             dict with 'gradients' (numpy array) and 'loss' (float)
     )pbdoc");
     
     // Run simulation and return timeseries
     enzyme.def("simulate", [](
         EnzymeRouter& router,
         py::array_t<double> runoff,
         int outlet_reach
     ) {
         auto runoff_buf = runoff.request();
         if (runoff_buf.ndim != 2) throw std::runtime_error("runoff must be 2D (timesteps x reaches)");
         
         int n_timesteps = runoff_buf.shape[0];
         int n_reaches = runoff_buf.shape[1];
         double* runoff_ptr = static_cast<double*>(runoff_buf.ptr);
         
         router.reset_state();
         
         // Simulate into vector
         std::vector<double> sim(n_timesteps);
         for (int t = 0; t < n_timesteps; ++t) {
             for (int i = 0; i < n_reaches; ++i) {
                 router.set_lateral_inflow(i, runoff_ptr[t * n_reaches + i]);
             }
             router.route_timestep();
             sim[t] = router.get_discharge(outlet_reach);
         }
         
         // Use py::cast for proper memory handling
         return py::cast(sim);
     },
     py::arg("router"),
     py::arg("runoff"),
     py::arg("outlet_reach"),
     "Run simulation and return outlet discharge timeseries");
     
     // Optimize using gradient descent
     enzyme.def("optimize", [](
         EnzymeRouter& router,
         py::array_t<double> runoff,
         py::array_t<double> observed,
         int outlet_reach,
         int n_epochs,
         double lr,
         double eps,
         bool verbose
     ) {
         auto runoff_buf = runoff.request();
         auto obs_buf = observed.request();
         
         if (runoff_buf.ndim != 2) throw std::runtime_error("runoff must be 2D");
         if (obs_buf.ndim != 1) throw std::runtime_error("observed must be 1D");
         
         int n_timesteps = runoff_buf.shape[0];
         int n_reaches = runoff_buf.shape[1];
         double* runoff_ptr = static_cast<double*>(runoff_buf.ptr);
         double* obs_ptr = static_cast<double*>(obs_buf.ptr);
         
         // Get initial parameters in log space
         auto manning_n = router.get_manning_n_all();
         std::vector<double> log_n(n_reaches);
         for (int i = 0; i < n_reaches; ++i) {
             log_n[i] = std::log(manning_n[i]);
         }
         
         // Lambda to run simulation and compute MSE loss
         auto run_and_loss = [&]() -> std::pair<double, std::vector<double>> {
             // Set current parameters
             for (int i = 0; i < n_reaches; ++i) {
                 router.set_manning_n(i, std::exp(log_n[i]));
             }
             
             router.reset_state();
             std::vector<double> sim(n_timesteps);
             for (int t = 0; t < n_timesteps; ++t) {
                 for (int i = 0; i < n_reaches; ++i) {
                     router.set_lateral_inflow(i, runoff_ptr[t * n_reaches + i]);
                 }
                 router.route_timestep();
                 sim[t] = router.get_discharge(outlet_reach);
             }
             
             double mse = 0.0;
             for (int t = 0; t < n_timesteps; ++t) {
                 double diff = sim[t] - obs_ptr[t];
                 mse += diff * diff;
             }
             return {mse / n_timesteps, sim};
         };
         
         // Optimization loop
         std::vector<double> losses;
         
         for (int epoch = 0; epoch < n_epochs; ++epoch) {
             // Compute gradients via finite differences in log space
             std::vector<double> grad(n_reaches, 0.0);
             auto [base_loss, _] = run_and_loss();
             losses.push_back(base_loss);
             
             for (int i = 0; i < n_reaches; ++i) {
                 double orig = log_n[i];
                 double h = eps;
                 
                 log_n[i] = orig + h;
                 auto [loss_plus, __] = run_and_loss();
                 
                 log_n[i] = orig - h;
                 auto [loss_minus, ___] = run_and_loss();
                 
                 grad[i] = (loss_plus - loss_minus) / (2 * h);
                 log_n[i] = orig;
             }
             
             // Gradient descent update
             for (int i = 0; i < n_reaches; ++i) {
                 log_n[i] -= lr * grad[i];
                 // Clip to reasonable range
                 log_n[i] = std::max(std::log(0.01), std::min(std::log(0.2), log_n[i]));
             }
             
             if (verbose && ((epoch + 1) % 5 == 0 || epoch == 0)) {
                 // Compute NSE for reporting
                 auto [loss, sim] = run_and_loss();
                 double obs_mean = 0.0;
                 for (int t = 0; t < n_timesteps; ++t) obs_mean += obs_ptr[t];
                 obs_mean /= n_timesteps;
                 
                 double ss_res = 0.0, ss_tot = 0.0;
                 for (int t = 0; t < n_timesteps; ++t) {
                     ss_res += (sim[t] - obs_ptr[t]) * (sim[t] - obs_ptr[t]);
                     ss_tot += (obs_ptr[t] - obs_mean) * (obs_ptr[t] - obs_mean);
                 }
                 double nse = 1.0 - ss_res / ss_tot;
                 
                 std::cerr << "  Epoch " << (epoch + 1) << "/" << n_epochs 
                          << ": MSE = " << loss << ", NSE = " << nse << std::endl;
             }
         }
         
         // Final simulation - avoid structured bindings for reliability
         for (int i = 0; i < n_reaches; ++i) {
             router.set_manning_n(i, std::exp(log_n[i]));
         }
         
         // Run final simulation explicitly (not through lambda)
         router.reset_state();
         std::vector<double> final_sim(n_timesteps);
         for (int t = 0; t < n_timesteps; ++t) {
             for (int i = 0; i < n_reaches; ++i) {
                 router.set_lateral_inflow(i, runoff_ptr[t * n_reaches + i]);
             }
             router.route_timestep();
             final_sim[t] = router.get_discharge(outlet_reach);
         }
         
         double final_loss = 0.0;
         for (int t = 0; t < n_timesteps; ++t) {
             double diff = final_sim[t] - obs_ptr[t];
             final_loss += diff * diff;
         }
         final_loss /= n_timesteps;
         
         // Compute final metrics
         double obs_mean = 0.0;
         for (int t = 0; t < n_timesteps; ++t) obs_mean += obs_ptr[t];
         obs_mean /= n_timesteps;
         
         double ss_res = 0.0, ss_tot = 0.0;
         for (int t = 0; t < n_timesteps; ++t) {
             ss_res += (final_sim[t] - obs_ptr[t]) * (final_sim[t] - obs_ptr[t]);
             ss_tot += (obs_ptr[t] - obs_mean) * (obs_ptr[t] - obs_mean);
         }
         double nse = 1.0 - ss_res / ss_tot;
         
         // Return results - use py::cast for automatic vector-to-numpy conversion
         py::dict result;
         
         // Use py::cast to properly convert vector to numpy array
         // This handles memory ownership correctly
         result["simulated"] = py::cast(final_sim);
         result["losses"] = py::cast(losses);
         result["nse"] = nse;
         result["final_loss"] = final_loss;
         
         // Convert manning_n to vector then cast
         std::vector<double> manning_vec(n_reaches);
         for (int i = 0; i < n_reaches; ++i) {
             manning_vec[i] = std::exp(log_n[i]);
         }
         result["optimized_manning_n"] = py::cast(manning_vec);
         
         return result;
     },
     py::arg("router"),
     py::arg("runoff"),
     py::arg("observed"),
     py::arg("outlet_reach"),
     py::arg("n_epochs") = 30,
     py::arg("lr") = 0.1,
     py::arg("eps") = 0.01,
     py::arg("verbose") = true,
     R"pbdoc(
         Optimize Manning's n using gradient descent with numerical gradients.
         
         This is much faster than CoDiPack-based optimization because it uses
         the fast Enzyme kernels without AD tape overhead.
         
         Args:
             router: EnzymeRouter instance
             runoff: Lateral inflows (n_timesteps, n_reaches)
             observed: Observed discharge at outlet (n_timesteps,)
             outlet_reach: Index of outlet reach
             n_epochs: Number of optimization epochs
             lr: Learning rate
             eps: Perturbation size for finite differences
             verbose: Print progress
             
         Returns:
             dict with 'simulated', 'losses', 'nse', 'final_loss', 'optimized_manning_n'
     )pbdoc");

    // Python binding for Enzyme AD gradients
    enzyme.def("compute_manning_gradients", [](
            py::array_t<double> manning_n,
            py::array_t<double> lateral_inflows,
            py::array_t<double> grad_output,
            py::array_t<double> lengths,
            py::array_t<double> slopes,
            py::array_t<double> width_coefs,
            py::array_t<double> width_exps,
            py::array_t<double> depth_coefs,
            py::array_t<double> depth_exps,
            py::array_t<int> topo_order,
            py::array_t<int> upstream_counts,
            py::array_t<int> upstream_offsets,
            py::array_t<int> upstream_indices,
            int outlet_reach,
            double dt
        ) -> py::array_t<double> {
        #ifdef DMC_USE_ENZYME
            // Get buffer info
            auto mann_buf = manning_n.request();
            auto inflows_buf = lateral_inflows.request();
            auto grad_buf = grad_output.request();
            
            int n_reaches = mann_buf.shape[0];
            int n_timesteps = inflows_buf.shape[0];
            
            // Get raw pointers
            double* mann_ptr = static_cast<double*>(mann_buf.ptr);
            double* inflows_ptr = static_cast<double*>(inflows_buf.ptr);
            double* grad_ptr = static_cast<double*>(grad_buf.ptr);
            double* lengths_ptr = static_cast<double*>(lengths.request().ptr);
            double* slopes_ptr = static_cast<double*>(slopes.request().ptr);
            double* w_coef_ptr = static_cast<double*>(width_coefs.request().ptr);
            double* w_exp_ptr = static_cast<double*>(width_exps.request().ptr);
            double* d_coef_ptr = static_cast<double*>(depth_coefs.request().ptr);
            double* d_exp_ptr = static_cast<double*>(depth_exps.request().ptr);
            int* topo_ptr = static_cast<int*>(topo_order.request().ptr);
            int* counts_ptr = static_cast<int*>(upstream_counts.request().ptr);
            int* offsets_ptr = static_cast<int*>(upstream_offsets.request().ptr);
            int* indices_ptr = static_cast<int*>(upstream_indices.request().ptr);
            
            // Allocate output gradient
            std::vector<double> d_manning(n_reaches, 0.0);
            
            // Call Enzyme reverse-mode AD
            __enzyme_autodiff(
                (void*)routing_objective,
                enzyme_const, n_reaches,
                enzyme_const, n_timesteps,
                enzyme_const, outlet_reach,
                enzyme_const, dt,
                enzyme_dup, mann_ptr, d_manning.data(),  // Active variable
                enzyme_const, inflows_ptr,
                enzyme_const, grad_ptr,
                enzyme_const, lengths_ptr,
                enzyme_const, slopes_ptr,
                enzyme_const, w_coef_ptr,
                enzyme_const, w_exp_ptr,
                enzyme_const, d_coef_ptr,
                enzyme_const, d_exp_ptr,
                enzyme_const, topo_ptr,
                enzyme_const, counts_ptr,
                enzyme_const, offsets_ptr,
                enzyme_const, indices_ptr
            );
            
            // Return as numpy array
            return py::array_t<double>(n_reaches, d_manning.data());
        #else
            throw std::runtime_error("Enzyme AD not available. Rebuild with -DDMC_USE_ENZYME=ON");
        #endif
        },
        py::arg("manning_n"),
        py::arg("lateral_inflows"), 
        py::arg("grad_output"),
        py::arg("lengths"),
        py::arg("slopes"),
        py::arg("width_coefs"),
        py::arg("width_exps"),
        py::arg("depth_coefs"),
        py::arg("depth_exps"),
        py::arg("topo_order"),
        py::arg("upstream_counts"),
        py::arg("upstream_offsets"),
        py::arg("upstream_indices"),
        py::arg("outlet_reach"),
        py::arg("dt"),
        R"doc(
        Compute gradients of routing loss w.r.t. Manning's n using Enzyme AD.

        TRUE reverse-mode automatic differentiation - not numerical!

        Args:
            manning_n: [n_reaches] Current Manning's n values
            lateral_inflows: [n_timesteps, n_reaches] Lateral inflows
            grad_output: [n_timesteps] Upstream gradient dL/dQ_outlet  
            lengths: [n_reaches] Reach lengths in meters
            slopes: [n_reaches] Bed slopes
            width_coefs, width_exps: Power law width parameters
            depth_coefs, depth_exps: Power law depth parameters
            topo_order: [n_reaches] Topological order of reaches
            upstream_counts: [n_reaches] Number of upstream reaches
            upstream_offsets: [n_reaches+1] Offsets into upstream_indices
            upstream_indices: Flattened upstream reach indices
            outlet_reach: Index of outlet reach
            dt: Timestep in seconds

        Returns:
            [n_reaches] Gradients dL/d(manning_n)
        )doc");
        
     // =========================================================================
     // SaintVenantRouter - Full dynamic SVE solver
     // =========================================================================
     py::class_<SaintVenantConfig>(m, "SaintVenantConfig",
         "Configuration for Saint-Venant Equations solver")
         .def(py::init<>())
         .def_readwrite("dt", &SaintVenantConfig::dt, "Output timestep [s]")
         .def_readwrite("n_nodes", &SaintVenantConfig::n_nodes, "Spatial nodes per reach")
         .def_readwrite("g", &SaintVenantConfig::g, "Gravitational acceleration [m/s²]")
         .def_readwrite("rel_tol", &SaintVenantConfig::rel_tol, "Relative tolerance")
         .def_readwrite("abs_tol", &SaintVenantConfig::abs_tol, "Absolute tolerance")
         .def_readwrite("max_steps", &SaintVenantConfig::max_steps, "Max CVODES steps")
         .def_readwrite("initial_depth", &SaintVenantConfig::initial_depth, "Initial water depth [m]")
         .def_readwrite("initial_velocity", &SaintVenantConfig::initial_velocity, "Initial velocity [m/s]")
         .def_readwrite("min_depth", &SaintVenantConfig::min_depth, "Minimum depth [m]")
         .def_readwrite("min_area", &SaintVenantConfig::min_area, "Minimum area [m²]")
         .def_readwrite("enable_adjoint", &SaintVenantConfig::enable_adjoint, "Use adjoint for gradients")
         .def_readwrite("checkpoint_stride", &SaintVenantConfig::checkpoint_stride, "Checkpoint interval");
     
     py::class_<SaintVenantRouter>(m, "SaintVenantRouter",
         R"doc(
         Full dynamic Saint-Venant Equations (SVE) solver.
         
         Solves the 1D shallow water equations using SUNDIALS CVODES:
         - Continuity: ∂A/∂t + ∂Q/∂x = q_lat
         - Momentum: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
         
         Uses finite volume spatial discretization with Rusanov flux and
         implicit BDF time integration for stability.
         
         This is the highest-fidelity routing option, suitable for benchmarking
         simpler routing methods.
         )doc")
         .def(py::init<Network&, SaintVenantConfig>(),
             py::arg("network"), py::arg("config") = SaintVenantConfig(),
             "Create SVE router with network and configuration")
         
         // Core routing
         .def("route_timestep", &SaintVenantRouter::route_timestep,
             "Advance one timestep using CVODES")
         .def("route", &SaintVenantRouter::route,
             "Route multiple timesteps", py::arg("num_timesteps"))
         
         // State access
         .def("set_lateral_inflow", &SaintVenantRouter::set_lateral_inflow,
             "Set lateral inflow for a reach [m³/s]",
             py::arg("reach_id"), py::arg("inflow"))
         .def("get_discharge", &SaintVenantRouter::get_discharge,
             "Get discharge at reach outlet [m³/s]",
             py::arg("reach_id"))
         .def("get_all_discharges", &SaintVenantRouter::get_all_discharges,
             "Get discharge at all reach outlets")
         .def("get_depth", &SaintVenantRouter::get_depth,
             "Get water depth at reach outlet [m]",
             py::arg("reach_id"))
         .def("reset_state", &SaintVenantRouter::reset_state,
             "Reset to initial conditions")
         
         // Gradient computation
         .def("start_recording", &SaintVenantRouter::start_recording,
             "Start recording for gradient computation")
         .def("stop_recording", &SaintVenantRouter::stop_recording,
             "Stop recording")
         .def("record_output", &SaintVenantRouter::record_output,
             "Record current output for gradient computation",
             py::arg("reach_id"))
         .def("compute_gradients_timeseries", &SaintVenantRouter::compute_gradients_timeseries,
             "Compute gradients for timeseries (numerical for now)",
             py::arg("reach_id"), py::arg("dL_dQ"))
         .def("get_gradients", &SaintVenantRouter::get_gradients,
             "Get accumulated gradients")
         
         // Properties
         .def("current_time", &SaintVenantRouter::current_time,
             "Current simulation time [s]")
         .def_property_readonly("config", &SaintVenantRouter::config,
             "Access configuration");
 
     // =========================================================================
     // SaintVenantEnzyme - Enzyme AD + CVODES Adjoint
     // =========================================================================
     py::class_<SaintVenantEnzymeConfig, SaintVenantConfig>(m, "SaintVenantEnzymeConfig",
         "Extended configuration for Enzyme-enabled Saint-Venant solver")
         .def(py::init<>())
         .def_readwrite("adjoint_checkpoint_steps", &SaintVenantEnzymeConfig::adjoint_checkpoint_steps,
             "Steps between checkpoints for adjoint")
         .def_readwrite("use_hermite_interpolation", &SaintVenantEnzymeConfig::use_hermite_interpolation,
             "Use Hermite vs polynomial interpolation")
         .def_readwrite("use_enzyme_jacobian", &SaintVenantEnzymeConfig::use_enzyme_jacobian,
             "Use Enzyme AD for Jacobian computation")
         .def_readwrite("use_enzyme_adjoint", &SaintVenantEnzymeConfig::use_enzyme_adjoint,
             "Use Enzyme AD for adjoint RHS computation");
 
     py::class_<SaintVenantEnzyme>(m, "SaintVenantEnzyme",
         R"doc(
         Full dynamic Saint-Venant solver with Enzyme AD gradients.
         
         This implementation combines:
         - SUNDIALS CVODES for implicit time integration (forward pass)
         - CVODES Adjoint Sensitivity for backward pass  
         - Enzyme AD for exact Jacobian and adjoint RHS computation
         
         The gradient computation strategy:
         1. Forward pass: CVODES with Enzyme-computed Jacobian (faster Newton)
         2. Backward pass: CVODES adjoint with Enzyme-computed J^T λ products
         3. Parameter sensitivity: ∫ λ^T (∂f/∂p) dt via Enzyme
         
         This provides exact gradients with O(n_params) cost, significantly 
         faster than finite difference which has O(n_params * n_timesteps) cost.
         
         Example:
             config = droute.SaintVenantEnzymeConfig()
             config.dt = 3600.0  # 1 hour
             config.use_enzyme_jacobian = True
             config.use_enzyme_adjoint = True
             
             router = droute.SaintVenantEnzyme(network, config)
             router.start_recording()
             
             for t in range(n_timesteps):
                 router.set_lateral_inflow(reach_id, inflow[t])
                 router.route_timestep()
             
             router.stop_recording()
             router.compute_gradients(outlet_reach_id, dL_dQ)
             grads = router.get_gradients()
         )doc")
         
         .def(py::init<Network&, SaintVenantEnzymeConfig>(),
             py::arg("network"), py::arg("config") = SaintVenantEnzymeConfig(),
             "Create Enzyme-enabled Saint-Venant router")
         
         // Core routing
         .def("route_timestep", &SaintVenantEnzyme::route_timestep,
             "Route one timestep using CVODES implicit solver")
         .def("route", &SaintVenantEnzyme::route,
             "Route multiple timesteps",
             py::arg("num_timesteps"))
         
         // State management  
         .def("set_lateral_inflow", &SaintVenantEnzyme::set_lateral_inflow,
             "Set lateral inflow for a reach [m³/s]",
             py::arg("reach_id"), py::arg("inflow"))
         .def("get_discharge", &SaintVenantEnzyme::get_discharge,
             "Get discharge at reach outlet [m³/s]",
             py::arg("reach_id"))
         .def("get_all_discharges", &SaintVenantEnzyme::get_all_discharges,
             "Get discharges for all reaches in topological order")
         .def("get_depth", &SaintVenantEnzyme::get_depth,
             "Get water depth at reach outlet [m]",
             py::arg("reach_id"))
         .def("reset_state", &SaintVenantEnzyme::reset_state,
             "Reset to initial conditions")
         
         // Gradient computation (Enzyme + CVODES Adjoint)
         .def("start_recording", &SaintVenantEnzyme::start_recording,
             "Enable gradient recording (activates CVODES checkpointing)")
         .def("stop_recording", &SaintVenantEnzyme::stop_recording,
             "Stop recording")
         .def("compute_gradients", &SaintVenantEnzyme::compute_gradients,
             R"doc(
             Compute gradients via CVODES adjoint with Enzyme AD.
             
             Uses the adjoint method:
             1. Backward integration: λ' = -J^T λ (Enzyme provides J^T λ)
             2. Accumulate: dL/dp = ∫ λ^T (∂f/∂p) dt (Enzyme provides ∂f/∂p)
             
             Args:
                 gauge_reach_id: Reach where loss is computed
                 dL_dQ: Gradient of loss w.r.t. Q at each recorded time
             )doc",
             py::arg("gauge_reach_id"), py::arg("dL_dQ"))
         .def("get_gradients", &SaintVenantEnzyme::get_gradients,
             "Get accumulated gradients as dict")
         .def("reset_gradients", &SaintVenantEnzyme::reset_gradients,
             "Reset gradient accumulators")
         
         // Properties
         .def("current_time", &SaintVenantEnzyme::current_time,
             "Current simulation time [s]")
         .def_property_readonly("config", &SaintVenantEnzyme::config,
             "Access configuration");
 
     // =========================================================================
     // ParallelEnzymeRouter - OpenMP parallelized routing
     // =========================================================================
     py::class_<ParallelEnzymeRouter>(m, "ParallelEnzymeRouter",
         R"doc(
         OpenMP-parallelized Muskingum-Cunge router.
         
         Uses graph coloring to identify independent reaches that can be
         processed in parallel. Provides significant speedup for large basins.
         
         Example:
             router = droute.ParallelEnzymeRouter(network, num_threads=4)
             for t in range(n_timesteps):
                 router.set_lateral_inflow(reach_id, inflow)
                 router.route_timestep()
         )doc")
         .def(py::init<Network&, int, double, int>(),
             py::arg("network"),
             py::arg("num_threads") = 0,
             py::arg("dt") = 3600.0,
             py::arg("num_substeps") = 4,
             "Create parallel router (num_threads=0 uses all available)")
         
         .def("route_timestep", &ParallelEnzymeRouter::route_timestep,
             "Route one timestep using parallel processing")
         .def("set_lateral_inflow", &ParallelEnzymeRouter::set_lateral_inflow,
             "Set lateral inflow for a reach",
             py::arg("reach_id"), py::arg("inflow"))
         .def("get_discharge", &ParallelEnzymeRouter::get_discharge,
             "Get discharge at reach outlet",
             py::arg("reach_id"))
         .def("reset_state", &ParallelEnzymeRouter::reset_state,
             "Reset to initial conditions")
         .def("num_colors", &ParallelEnzymeRouter::num_colors,
             "Number of color groups (parallel batches)")
         .def("num_threads", &ParallelEnzymeRouter::num_threads,
             "Number of OpenMP threads");
 
     // =========================================================================
     // Version info
     // =========================================================================
#ifdef DMC_VERSION
    m.attr("__version__") = DMC_VERSION;
#else
    m.attr("__version__") = "0.5.1";  // Fallback
#endif
     m.attr("__author__") = "Darri Eythorsson";
 }
