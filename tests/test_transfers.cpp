/**
 * Transfers, level-pool lakes, cycle rejection, mass conservation and
 * CoDiPack gradients of transfer/lake parameters (C++ side; the Python suite
 * in tests/python/test_transfers_lakes.py covers the same ground in more
 * detail).
 */

#include <dmc/network.hpp>
#include <dmc/router.hpp>
#include <dmc/transfer_routing.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace dmc;

static int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cout << "FAIL: " << msg << " (" #cond ")\n";                 \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

// downstream[i] = -1 for terminal reaches; junction i feeds reach i
static Network make_net(const std::vector<int>& downstream) {
    Network net;
    int n = static_cast<int>(downstream.size());
    std::vector<std::vector<int>> ups(n + 1);
    for (int i = 0; i < n; ++i) ups[downstream[i] >= 0 ? downstream[i] : n].push_back(i);
    for (int j = 0; j <= n; ++j) {
        Junction junc;
        junc.id = j;
        junc.upstream_reach_ids = ups[j];
        net.add_junction(junc);
    }
    for (int i = 0; i < n; ++i) {
        Reach r;
        r.id = i;
        r.length = 2000.0;
        r.slope = 0.002;
        r.manning_n = Real(0.035);
        r.upstream_junction_id = i;
        r.downstream_junction_id = downstream[i] >= 0 ? downstream[i] : n;
        net.add_reach(r);
    }
    net.build_topology();
    return net;
}

static Transfer edge(const std::string& id, TransferType type, int from, int to,
                     double fraction, double capacity = -1.0) {
    Transfer t;
    t.id = id;
    t.type = type;
    t.operated = (type == TransferType::Headrace || type == TransferType::Diversion);
    t.from_reach = from;
    t.to_reach = to;
    t.fraction = Real(fraction);
    if (capacity >= 0.0) {
        t.has_capacity = true;
        t.capacity = Real(capacity);
    }
    return t;
}

// 0 -> 1 -> 2 (reservoir) -> 3 -> out ; 4 -> 5 -> 6 -> out
static Network full_network(double fraction_bf = 0.2) {
    Network net = make_net({1, 2, 3, -1, 5, 6, -1});
    Reach& lake = net.get_reach(2);
    lake.is_lake = true;
    lake.lake_model = LAKE_LEVEL_POOL;
    lake.lake_type = 1;
    lake.lake_area = 3e6;
    lake.lake_weir_coef = Real(25.0);
    lake.lake_min_storage = -9e6;
    Transfer hr = edge("HR", TransferType::Headrace, 2, 5, 1.0, 3.0);
    hr.from_storage = true;
    net.add_transfer(hr);
    net.add_transfer(edge("BF", TransferType::NaturalBifurcation, 1, 6, fraction_bf));
    Transfer gw = edge("GW", TransferType::Groundwater, 3, 6, 0.1);
    gw.gw_k = Real(86400.0);
    net.add_transfer(gw);
    Transfer loss = edge("LOSS", TransferType::Groundwater, 0, -1, 0.05);
    loss.gw_k = Real(86400.0);
    net.add_transfer(loss);
    net.transfer_options().enable_groundwater = true;
    net.build_topology();
    return net;
}

static const double LAT[7] = {5.0, 2.0, 1.0, 1.5, 3.0, 0.5, 0.25};

template <typename R>
static void drive(R& router, int steps) {
    for (int k = 0; k < steps; ++k) {
        double f = 1.0 + 0.8 * std::sin(2.0 * M_PI * k / 48.0);
        for (int i = 0; i < 7; ++i) router.set_lateral_inflow(i, LAT[i] * f);
        router.route_timestep();
    }
}

void test_cycle_rejected() {
    std::cout << "cycle rejection... ";
    Network net = make_net({1, 2, -1});
    net.add_transfer(edge("BACK", TransferType::Diversion, 2, 0, 0.5));
    bool threw = false;
    try {
        net.build_topology();
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("BACK") != std::string::npos;
    }
    CHECK(threw, "cycle through transfer BACK must be rejected");
    std::cout << "done\n";
}

void test_conservation() {
    std::cout << "mass conservation (IRF exact, MC transient)... ";
    {
        Network net = full_network();
        RouterConfig cfg;
        cfg.dt = 3600.0;
        cfg.enable_gradients = false;
        cfg.irf_max_kernel_size = 200;
        IRFRouter router(net, cfg);
        drive(router, 240);
        MassBalance mb = router.mass_balance();
        CHECK(mb.declared_losses > 0.0, "groundwater loss edge must declare losses");
        CHECK(std::abs(mb.relative_residual) < 1e-10, "IRF residual " << mb.relative_residual);
    }
    {
        Network net = full_network();
        RouterConfig cfg;
        cfg.dt = 3600.0;
        cfg.enable_gradients = false;
        MuskingumCungeRouter router(net, cfg);
        drive(router, 240);
        MassBalance mb = router.mass_balance();
        CHECK(std::abs(mb.relative_residual) < 2e-3, "MC residual " << mb.relative_residual);
    }
    std::cout << "done\n";
}

static double outlet_sum(double fraction_bf, int steps) {
    Network net = full_network(fraction_bf);
    RouterConfig cfg;
    cfg.dt = 3600.0;
    cfg.enable_gradients = false;
    MuskingumCungeRouter router(net, cfg);
    double total = 0.0;
    for (int k = 0; k < steps; ++k) {
        drive(router, 1);
        total += router.get_discharge(6);
    }
    return total;
}

void test_gradient() {
    if (!AD_ENABLED) {
        std::cout << "gradient: skipped (no AD)\n";
        return;
    }
    std::cout << "gradient of bifurcation fraction (AD vs FD)... ";
    const int steps = 48;
    Network net = full_network(0.2);
    RouterConfig cfg;
    cfg.dt = 3600.0;
    cfg.enable_gradients = true;
    MuskingumCungeRouter router(net, cfg);
    router.start_recording();
    for (int k = 0; k < steps; ++k) {
        drive(router, 1);
        router.record_output(6);
    }
    router.stop_recording();
    router.compute_gradients_timeseries(6, std::vector<double>(steps, 1.0));
    double ad = router.get_gradients()["transfer_BF_fraction"];
    double h = 1e-4;
    double fd = (outlet_sum(0.2 + h, steps) - outlet_sum(0.2 - h, steps)) / (2 * h);
    CHECK(fd > 0.0, "fd " << fd);
    CHECK(std::abs(ad - fd) < 2e-3 * std::abs(fd), "AD " << ad << " vs FD " << fd);
    std::cout << "done (AD " << ad << ", FD " << fd << ")\n";
}

int main() {
    test_cycle_rejected();
    test_conservation();
    test_gradient();
    if (failures) {
        std::cout << failures << " failure(s)\n";
        return EXIT_FAILURE;
    }
    std::cout << "All transfer tests passed\n";
    return EXIT_SUCCESS;
}
