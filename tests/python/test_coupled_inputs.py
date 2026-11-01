"""Tests for the coupled (multi-gauge path) input loader of the dRoute calibration worker.

A synthetic two-GRU domain whose GRUs are split into elevation-band HRUs (ids differ from the GRU ids)
and which has no dRoute multi-gauge layout, only SYMFLUENCE's preprocessed streamflow.
"""

import numpy as np
import pytest

pytest.importorskip("symfluence")
gpd = pytest.importorskip("geopandas")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")
shapely = pytest.importorskip("shapely")

DOMAIN = "TEST"
# GRU 10 = HRUs 1+2 (3 + 1 km2), GRU 20 = HRUs 3+4+5 (2 + 2 + 4 km2)
HRU_ID = [1, 2, 3, 4, 5]
HRU_GRU = [10, 10, 20, 20, 20]
HRU_AREA = [3e6, 1e6, 2e6, 2e6, 4e6]
RUNOFF = {10: 1e-6, 20: 2e-6}   # m/s, constant


@pytest.fixture
def domain(tmp_path):
    root = tmp_path / f"domain_{DOMAIN}"
    rn = root / "shapefiles" / "river_network"
    rn.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"LINKNO": [10, 20], "DSLINKNO": [20, -1], "Length": [1000.0, 1000.0], "Slope": [0.01, 0.01]},
        geometry=[shapely.LineString([(0, 0), (1, 0)]), shapely.LineString([(1, 0), (2, 0)])],
        crs="EPSG:4326").to_file(rn / "network.shp")

    summa = root / "settings" / "SUMMA"
    summa.mkdir(parents=True)
    xr.Dataset({"hruId": ("hru", HRU_ID), "hru2gruId": ("hru", HRU_GRU),
                "HRUarea": ("hru", HRU_AREA)}).to_netcdf(summa / "attributes.nc")

    time = pd.date_range("2000-01-01", "2000-03-31 21:00", freq="3h")
    out = root / "simulations" / "run" / "SUMMA"
    out.mkdir(parents=True)
    xr.Dataset({"averageRoutedRunoff": (("time", "gru"), np.tile([RUNOFF[10], RUNOFF[20]], (len(time), 1))),
                "gruId": ("gru", [10, 20])}, coords={"time": time}).to_netcdf(out / "run_timestep.nc")

    obs = root / "data" / "observations" / "streamflow" / "preprocessed"
    obs.mkdir(parents=True)
    days = pd.date_range("2000-01-01", "2000-03-31")
    pd.DataFrame({"datetime": days, "discharge_cms": np.arange(len(days), dtype=float)}).to_csv(
        obs / f"{DOMAIN}_streamflow_processed.csv", index=False)
    (root / "settings" / "dRoute").mkdir(parents=True)
    return tmp_path


def _worker(data_dir, **extra):
    from droute.calibration.worker import DRouteWorker
    cfg = {"SYMFLUENCE_DATA_DIR": str(data_dir), "DOMAIN_NAME": DOMAIN,
           "CALIBRATION_PERIOD": "2000-02-01, 2000-03-31", "EXPERIMENT_TIME_START": "2000-01-01 00:00",
           "DROUTE_RUNOFF_CAP": 1e9, **extra}
    return DRouteWorker(config=cfg)


def test_runoff_scaled_by_gru_area(domain):
    inp = _worker(domain)._load_inputs(domain / f"domain_{DOMAIN}" / "settings" / "dRoute")
    i10, i20 = inp["id_to_idx"][10], inp["id_to_idx"][20]
    np.testing.assert_allclose(inp["daily"][:, i10], RUNOFF[10] * 4e6)   # 4 m3/s
    np.testing.assert_allclose(inp["daily"][:, i20], RUNOFF[20] * 8e6)   # 16 m3/s


def test_outlet_gauge_from_preprocessed_streamflow(domain):
    inp = _worker(domain)._load_inputs(domain / f"domain_{DOMAIN}" / "settings" / "dRoute")
    assert len(inp["gauges"]) == 1
    g = inp["gauges"][0]
    assert g["ridx"] == inp["id_to_idx"][20]          # the reach with no downstream reach
    assert inp["dates"][inp["i_eval0"]] == pd.Timestamp("2000-02-01")
    assert len(g["obs"]) == 60 and g["obs"][0] == 31.0   # obs value = day index since 1 Jan


def test_sim_reach_id_selects_gauge_reach(domain):
    inp = _worker(domain, SIM_REACH_ID=10)._load_inputs(domain / f"domain_{DOMAIN}" / "settings" / "dRoute")
    assert inp["gauges"][0]["ridx"] == inp["id_to_idx"][10]
