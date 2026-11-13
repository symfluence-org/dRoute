# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SYMFLUENCE Team <dev@symfluence.org>

"""
dRoute Network Adapter.

Handles conversion between mizuRoute-compatible topology format and dRoute's
internal network representation. Enables seamless switching between routing
models using the same preprocessed network topology.

The adapter supports:
- Loading mizuRoute NetCDF topology files
- Converting to dRoute's Python API format
- Writing dRoute-specific configuration files
- Validating network topology consistency
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import netCDF4 as nc4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class DRouteNetworkAdapter:
    """
    Adapter for converting mizuRoute topology to dRoute format.

    Supports loading topology from:
    - NetCDF files (mizuRoute format)
    - GeoJSON files
    - CSV files

    The dRoute network representation consists of:
    - Segment IDs and downstream connectivity
    - Segment properties (length, slope, width)
    - HRU-to-segment drainage mapping
    - Optional channel geometry parameters

    Example:
        >>> adapter = DRouteNetworkAdapter(logger)
        >>> network = adapter.load_topology('topology.nc')
        >>> droute_config = adapter.to_droute_format(network)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the network adapter.

        Args:
            logger: Logger instance for status messages
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_topology(
        self,
        topology_path: Path,
        format: str = 'netcdf'
    ) -> Dict[str, Any]:
        """
        Load network topology from file.

        Args:
            topology_path: Path to topology file
            format: File format ('netcdf', 'geojson', 'csv')

        Returns:
            Dictionary containing network topology data:
            - seg_ids: Segment IDs
            - down_seg_ids: Downstream segment IDs
            - slopes: Segment slopes
            - lengths: Segment lengths (m)
            - hru_ids: HRU IDs
            - hru_to_seg: HRU-to-segment mapping
            - hru_areas: HRU areas (m^2)

        Raises:
            FileNotFoundError: If topology file doesn't exist
            ValueError: If format is unsupported
        """
        topology_path = Path(topology_path)

        if not topology_path.exists():
            raise FileNotFoundError(f"Topology file not found: {topology_path}")

        if format == 'netcdf':
            return self._load_netcdf_topology(topology_path)
        elif format == 'geojson':
            return self._load_geojson_topology(topology_path)
        elif format == 'csv':
            return self._load_csv_topology(topology_path)
        else:
            raise ValueError(f"Unsupported topology format: {format}")

    def _load_netcdf_topology(self, path: Path) -> Dict[str, Any]:
        """Load mizuRoute-format NetCDF topology."""
        self.logger.debug(f"Loading NetCDF topology from {path}")

        if not HAS_XARRAY:
            if not HAS_NETCDF4:
                raise ImportError("Either xarray or netCDF4 required to load NetCDF topology")
            return self._load_netcdf_topology_nc4(path)

        with xr.open_dataset(path) as ds:
            topology = {
                'seg_ids': ds['segId'].values.astype(int),
                'down_seg_ids': ds['downSegId'].values.astype(int),
                'slopes': ds['slope'].values.astype(float),
                'lengths': ds['length'].values.astype(float),
                'hru_ids': ds['hruId'].values.astype(int),
                'hru_to_seg': ds['hruToSegId'].values.astype(int),
                'hru_areas': ds['area'].values.astype(float),
            }

            # Optional: load width if available
            if 'width' in ds:
                topology['widths'] = ds['width'].values.astype(float)
            # Optional: per-segment Manning's n prior
            if 'mann_n' in ds:
                topology['mann_n'] = ds['mann_n'].values.astype(float)
            # Optional (Landslikan >= v0.4): 0 drains on, 1 sea, 2 closed-basin sink
            if 'terminal_type' in ds:
                topology['terminal_type'] = ds['terminal_type'].values.astype(int)

        self.logger.info(
            f"Loaded topology: {len(topology['seg_ids'])} segments, "
            f"{len(topology['hru_ids'])} HRUs"
        )

        return topology

    def _load_netcdf_topology_nc4(self, path: Path) -> Dict[str, Any]:
        """Fallback: Load NetCDF using netCDF4 directly."""
        with nc4.Dataset(path, 'r') as ds:
            topology = {
                'seg_ids': ds.variables['segId'][:].astype(int),
                'down_seg_ids': ds.variables['downSegId'][:].astype(int),
                'slopes': ds.variables['slope'][:].astype(float),
                'lengths': ds.variables['length'][:].astype(float),
                'hru_ids': ds.variables['hruId'][:].astype(int),
                'hru_to_seg': ds.variables['hruToSegId'][:].astype(int),
                'hru_areas': ds.variables['area'][:].astype(float),
            }

            if 'width' in ds.variables:
                topology['widths'] = ds.variables['width'][:].astype(float)
            if 'mann_n' in ds.variables:
                topology['mann_n'] = ds.variables['mann_n'][:].astype(float)
            if 'terminal_type' in ds.variables:
                topology['terminal_type'] = ds.variables['terminal_type'][:].astype(int)

        return topology

    def _load_geojson_topology(self, path: Path) -> Dict[str, Any]:
        """Load GeoJSON topology (segments as LineStrings)."""
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas required to load GeoJSON topology") from None

        self.logger.debug(f"Loading GeoJSON topology from {path}")

        gdf = gpd.read_file(path)

        # Map column names (flexible naming)
        seg_id_col = self._find_column(gdf, ['segId', 'seg_id', 'SEGID', 'id'])
        down_seg_col = self._find_column(gdf, ['downSegId', 'down_seg_id', 'DOWNSEGID', 'downstream'])
        slope_col = self._find_column(gdf, ['slope', 'SLOPE'])
        length_col = self._find_column(gdf, ['length', 'LENGTH', 'len'])

        topology = {
            'seg_ids': gdf[seg_id_col].values.astype(int),
            'down_seg_ids': gdf[down_seg_col].values.astype(int),
            'slopes': gdf[slope_col].values.astype(float) if slope_col else np.full(len(gdf), 0.001),
            'lengths': gdf[length_col].values.astype(float) if length_col else gdf.geometry.length.values,
            'hru_ids': gdf[seg_id_col].values.astype(int),  # Default: same as seg_ids
            'hru_to_seg': gdf[seg_id_col].values.astype(int),  # Default: 1:1 mapping
            'hru_areas': np.full(len(gdf), 1e6),  # Default: 1 km^2
        }

        return topology

    def _load_csv_topology(self, path: Path) -> Dict[str, Any]:
        """Load CSV topology."""
        import pandas as pd

        self.logger.debug(f"Loading CSV topology from {path}")

        df = pd.read_csv(path)

        seg_id_col = self._find_column(df, ['segId', 'seg_id', 'SEGID', 'id'])
        down_seg_col = self._find_column(df, ['downSegId', 'down_seg_id', 'DOWNSEGID', 'downstream'])

        topology = {
            'seg_ids': df[seg_id_col].values.astype(int),
            'down_seg_ids': df[down_seg_col].values.astype(int),
            'slopes': df.get('slope', pd.Series(np.full(len(df), 0.001))).values.astype(float),
            'lengths': df.get('length', pd.Series(np.full(len(df), 1000.0))).values.astype(float),
            'hru_ids': df[seg_id_col].values.astype(int),
            'hru_to_seg': df[seg_id_col].values.astype(int),
            'hru_areas': df.get('area', pd.Series(np.full(len(df), 1e6))).values.astype(float),
        }

        return topology

    def _find_column(self, df, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def to_droute_format(
        self,
        topology: Dict[str, Any],
        routing_method: str = 'muskingum_cunge',
        routing_dt: int = 3600
    ) -> Dict[str, Any]:
        """
        Convert topology to dRoute's expected format.

        Args:
            topology: Network topology dictionary from load_topology()
            routing_method: Routing method to use
            routing_dt: Routing timestep in seconds

        Returns:
            Dictionary formatted for dRoute Python API
        """
        # Build adjacency list for downstream connectivity
        seg_ids = topology['seg_ids']
        down_seg_ids = topology['down_seg_ids']

        # Create ID-to-index mapping
        id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

        # Build downstream index array (-1 for outlets)
        downstream_idx = np.full(len(seg_ids), -1, dtype=int)
        for i, down_id in enumerate(down_seg_ids):
            if down_id in id_to_idx:
                downstream_idx[i] = id_to_idx[down_id]

        # Find outlet segments
        outlet_mask = downstream_idx == -1
        outlet_indices = np.where(outlet_mask)[0].tolist()

        # Compute routing order (topological sort)
        routing_order = self._compute_routing_order(downstream_idx)

        droute_config = {
            'n_segments': len(seg_ids),
            'segment_ids': seg_ids.tolist(),
            'downstream_idx': downstream_idx.tolist(),
            'outlet_indices': outlet_indices,
            'routing_order': routing_order,
            'slopes': topology['slopes'].tolist(),
            'lengths': topology['lengths'].tolist(),
            'routing_method': routing_method,
            'routing_dt': routing_dt,
            'hru_ids': topology['hru_ids'].tolist(),
            'hru_to_seg_idx': [id_to_idx.get(sid, -1) for sid in topology['hru_to_seg']],
            'hru_areas': topology['hru_areas'].tolist(),
        }

        if 'mann_n' in topology:
            droute_config['mann_n'] = np.asarray(topology['mann_n']).tolist()

        if 'terminal_type' in topology:
            # Informational: every reach without a downstream segment is a
            # terminal in dRoute (its outflow is reported as terminal_outflow),
            # whether it reaches the sea (1) or a closed-basin sink (2).
            tt = np.asarray(topology['terminal_type']).astype(int)
            droute_config['terminal_type'] = tt.tolist()
            bad = int(np.sum((tt != 0) != outlet_mask))
            if bad:
                self.logger.warning(
                    f"terminal_type disagrees with downSegId for {bad} segments "
                    "(terminal_type 0 must have a downstream segment, 1/2 must not)")

        # Add optional width if available
        if 'widths' in topology:
            droute_config['widths'] = topology['widths'].tolist()
        else:
            # Estimate width from contributing area (Leopold & Maddock, 1953)
            # W = a * Q^b, approximate with sqrt(upstream_area)
            droute_config['widths'] = self._estimate_channel_widths(topology)

        return droute_config

    def _compute_routing_order(self, downstream_idx: np.ndarray) -> List[int]:
        """
        Compute topological routing order (upstream to downstream).

        Args:
            downstream_idx: Array of downstream segment indices (-1 for outlets)

        Returns:
            List of segment indices in routing order
        """
        n_segments = len(downstream_idx)

        # Count incoming edges (upstream connections)
        in_degree = np.zeros(n_segments, dtype=int)
        for down_idx in downstream_idx:
            if down_idx >= 0:
                in_degree[down_idx] += 1

        # Start with headwater segments (no upstream)
        queue = [i for i in range(n_segments) if in_degree[i] == 0]
        routing_order = []

        while queue:
            current = queue.pop(0)
            routing_order.append(current)

            down_idx = downstream_idx[current]
            if down_idx >= 0:
                in_degree[down_idx] -= 1
                if in_degree[down_idx] == 0:
                    queue.append(down_idx)

        return routing_order

    def _estimate_channel_widths(self, topology: Dict[str, Any]) -> List[float]:
        """
        Estimate channel widths from contributing areas.

        Uses empirical relationship: W ~ sqrt(A)
        """
        # Simple estimate: width proportional to sqrt of contributing area
        hru_to_seg = topology['hru_to_seg']
        hru_areas = topology['hru_areas']
        seg_ids = topology['seg_ids']

        # Sum contributing area for each segment
        seg_areas = {}
        for hru_id, seg_id, area in zip(topology['hru_ids'], hru_to_seg, hru_areas):
            if seg_id not in seg_areas:
                seg_areas[seg_id] = 0.0
            seg_areas[seg_id] += area

        # Estimate width: W = 4.8 * Q^0.5 where Q ~ sqrt(A)
        # Simplified: W ~ 2 * A^0.25 (in meters, A in m^2)
        widths = []
        for seg_id in seg_ids:
            area = seg_areas.get(seg_id, 1e6)  # Default 1 km^2
            width = max(1.0, 2.0 * (area ** 0.25))  # Min 1m width
            widths.append(width)

        return widths

    def write_droute_config(
        self,
        droute_config: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Write dRoute configuration to YAML file.

        Args:
            droute_config: Configuration from to_droute_format()
            output_path: Path to write YAML file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required to write dRoute config") from None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Separate network data from routing config
        network_config = {
            'network': {
                'n_segments': droute_config['n_segments'],
                'segment_ids': droute_config['segment_ids'],
                'downstream_idx': droute_config['downstream_idx'],
                'outlet_indices': droute_config['outlet_indices'],
                'routing_order': droute_config['routing_order'],
            },
            'geometry': {
                'slopes': droute_config['slopes'],
                'lengths': droute_config['lengths'],
                'widths': droute_config['widths'],
            },
            'hru_mapping': {
                'hru_ids': droute_config['hru_ids'],
                'hru_to_seg_idx': droute_config['hru_to_seg_idx'],
                'hru_areas': droute_config['hru_areas'],
            },
            'routing': {
                'method': droute_config['routing_method'],
                'dt': droute_config['routing_dt'],
            },
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(network_config, f, default_flow_style=False)

        self.logger.info(f"Wrote dRoute config to {output_path}")

    def validate_topology(self, topology: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate and fix network topology issues.

        Fixes cycles using the elevation-based rule from BaseTopologyGenerator,
        repairs invalid downstream references, and enforces minimum values.

        Args:
            topology: Network topology dictionary (modified in place)

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        import numpy as np

        warnings = []
        is_valid = True

        seg_ids_arr = np.array(topology['seg_ids'])
        down_seg_ids_arr = np.array(topology['down_seg_ids'])
        seg_ids_set = set(topology['seg_ids'])

        # Fix disconnected segments
        for i, down_id in enumerate(down_seg_ids_arr):
            if down_id != 0 and down_id not in seg_ids_set:
                warnings.append(
                    f"Segment {seg_ids_arr[i]} had invalid downstream {down_id} -> fixed to outlet"
                )
                down_seg_ids_arr[i] = 0

        # Fix cycles using shared algorithm
        elevations = np.array(topology.get('elevations', np.zeros(len(seg_ids_arr))))
        from ._symfluence_compat import BaseTopologyGenerator
        fixed_down_ids = BaseTopologyGenerator.fix_routing_cycles(
            seg_ids_arr, down_seg_ids_arr, elevations
        )
        n_fixed = np.sum(fixed_down_ids != down_seg_ids_arr)
        if n_fixed > 0:
            warnings.append(f"Fixed {n_fixed} cycles using elevation-based rule")
            topology['down_seg_ids'] = fixed_down_ids.tolist()
        else:
            topology['down_seg_ids'] = down_seg_ids_arr.tolist()

        # Fix zero/negative lengths
        for i, length in enumerate(topology['lengths']):
            if length <= 0:
                warnings.append(f"Segment {topology['seg_ids'][i]} had invalid length {length} -> set to 1.0")
                topology['lengths'][i] = 1.0

        # Fix zero/negative slopes
        for i, slope in enumerate(topology['slopes']):
            if slope <= 0:
                warnings.append(f"Segment {topology['seg_ids'][i]} had invalid slope {slope} -> set to 0.001")
                topology['slopes'][i] = 0.001

        return is_valid, warnings


    # =========================================================================
    # Hydrofabric extensions: transfers.csv and lakes.csv (non-tree layers)
    # =========================================================================

    def build_network(self, droute_config: Dict[str, Any], mannings_n: Optional[float] = None):
        """Build a dRoute C++ Network from :meth:`to_droute_format` output.

        Reach ids are segment indices (0..n-1); junction i collects the reaches
        draining into reach i, and terminal reaches drain into junction n.
        Uses the per-segment ``mann_n`` prior when present unless ``mannings_n``
        is given.
        """
        import droute

        n = droute_config['n_segments']
        downstream_idx = droute_config['downstream_idx']
        lengths = droute_config['lengths']
        slopes = droute_config['slopes']
        widths = droute_config.get('widths')
        mann = droute_config.get('mann_n')

        net = droute.Network()
        upstream: Dict[int, List[int]] = {i: [] for i in range(n + 1)}
        for i in range(n):
            ds_idx = downstream_idx[i]
            upstream[ds_idx if ds_idx >= 0 else n].append(i)
        for jid in range(n + 1):
            junc = droute.Junction()
            junc.id = jid
            junc.upstream_reach_ids = upstream[jid]
            net.add_junction(junc)
        for i in range(n):
            reach = droute.Reach()
            reach.id = i
            reach.length = float(lengths[i])
            reach.slope = max(float(slopes[i]), 1e-4)
            if mannings_n is not None:
                reach.manning_n = float(mannings_n)
            elif mann is not None:
                reach.manning_n = float(mann[i])
            reach.upstream_junction_id = i
            ds_idx = downstream_idx[i]
            reach.downstream_junction_id = ds_idx if ds_idx >= 0 else n
            if widths is not None:
                geom = reach.geometry
                geom.width_coef = float(widths[i])
                geom.width_exp = 0.0
                reach.geometry = geom
            net.add_reach(reach)
        net.build_topology()
        return net

    @staticmethod
    def upstream_areas_km2(droute_config: Dict[str, Any]) -> np.ndarray:
        """Tree-only upstream drainage area of each segment [km^2]."""
        n = droute_config['n_segments']
        local = np.zeros(n)
        for hru_idx, seg_idx in enumerate(droute_config['hru_to_seg_idx']):
            if seg_idx >= 0:
                local[seg_idx] += droute_config['hru_areas'][hru_idx] / 1e6
        acc = local.copy()
        for i in droute_config['routing_order']:
            d = droute_config['downstream_idx'][i]
            if d >= 0:
                acc[d] += acc[i]
        return acc

    def load_transfers(self, path: Path) -> List[Dict[str, Any]]:
        """Read a transfers table (Landslikan ``droute/transfers.csv``).

        Required columns: transfer_id, type, operated, from_unit, to_unit,
        capacity_m3s, fraction, rule, commissioned. Optional: decommissioned,
        status, topology_effect, confidence. Rows whose ``status`` is not
        ``resolved`` or that lack endpoints are skipped with a warning.
        """
        import csv

        required = {'transfer_id', 'type', 'operated', 'from_unit', 'to_unit',
                    'capacity_m3s', 'fraction', 'rule', 'commissioned'}
        records = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing transfer columns {sorted(missing)}")
            if 'decommissioned' not in (reader.fieldnames or []):
                self.logger.info(f"{path}: no 'decommissioned' column; edges stay active once commissioned")
            for row in reader:
                tid = row['transfer_id']
                status = (row.get('status') or 'resolved').strip()
                if status != 'resolved':
                    self.logger.warning(f"transfer {tid}: status '{status}', skipped ({row.get('reason', '')})")
                    continue
                from_unit = _num(row.get('from_unit'))
                to_unit = _num(row.get('to_unit'))
                if from_unit is None or to_unit is None:
                    self.logger.warning(f"transfer {tid}: unresolved endpoint, skipped")
                    continue
                records.append({
                    'transfer_id': tid,
                    'type': row['type'].strip(),
                    'operated': str(row.get('operated', '')).strip().lower() in ('true', '1', 'yes'),
                    'from_unit': int(from_unit),
                    'to_unit': int(to_unit),
                    'capacity_m3s': _num(row.get('capacity_m3s')),
                    'fraction': _num(row.get('fraction')),
                    'rule': (row.get('rule') or '').strip(),
                    'commissioned': _num(row.get('commissioned')),
                    'decommissioned': _num(row.get('decommissioned')),
                    'topology_effect': (row.get('topology_effect') or '').strip(),
                    'confidence': (row.get('confidence') or '').strip(),
                })
        return records

    def apply_transfers(
        self,
        net,
        transfers: List[Dict[str, Any]],
        droute_config: Dict[str, Any],
        storage_withdrawals: Optional[Dict[str, int]] = None,
        defaults: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Add transfer edges to a built Network (reach id = segment index).

        Empty ``fraction`` / ``capacity_m3s`` get the defaults in
        :data:`TRANSFER_DEFAULTS` (calibration parameters). Operated edges listed
        in a lake's ``operated_withdrawals`` (``storage_withdrawals``) withdraw
        from that lake's storage. Rebuilds the topology, which raises if the
        combined graph has a cycle.
        """
        import droute

        d = dict(TRANSFER_DEFAULTS)
        d.update(defaults or {})
        storage_withdrawals = storage_withdrawals or {}
        id_to_idx = {int(s): i for i, s in enumerate(droute_config['segment_ids'])}
        added = []
        needs_capacity = []
        for rec in transfers:
            tid = rec['transfer_id']
            if rec['from_unit'] not in id_to_idx:
                self.logger.warning(f"transfer {tid}: from_unit {rec['from_unit']} not in topology, skipped")
                continue
            to_unit = rec['to_unit']
            if to_unit != 0 and to_unit not in id_to_idx:
                self.logger.warning(f"transfer {tid}: to_unit {to_unit} not in topology, skipped")
                continue
            t = droute.Transfer()
            t.id = tid
            t.type = droute.parse_transfer_type(rec['type'])
            t.operated = bool(rec['operated'])
            t.from_reach = id_to_idx[rec['from_unit']]
            t.to_reach = id_to_idx[to_unit] if to_unit != 0 else -1
            kind = rec['type']
            if rec['fraction'] is not None:
                t.fraction = float(rec['fraction'])
            elif kind in ('headrace', 'diversion'):
                t.fraction = d['operated_fraction']
            elif kind == 'groundwater':
                t.fraction = d['groundwater_fraction']
            else:
                t.fraction = d['bifurcation_fraction']
            if kind in ('headrace', 'diversion'):
                if rec['capacity_m3s'] is not None:
                    t.capacity = float(rec['capacity_m3s'])
                elif d['capacity_factor'] is not None and d['capacity_factor'] > 0:
                    needs_capacity.append(tid)
            if kind == 'groundwater':
                t.gw_k = d['groundwater_k_days'] * 86400.0
            if rec['commissioned'] is not None:
                t.commissioned = float(rec['commissioned'])
            if rec.get('decommissioned') is not None:
                t.decommissioned = float(rec['decommissioned'])
            if t.operated and tid in storage_withdrawals:
                if storage_withdrawals[tid] == t.from_reach:
                    t.from_storage = True
                else:
                    self.logger.warning(
                        f"transfer {tid}: listed as withdrawal of a lake at reach "
                        f"{storage_withdrawals[tid]} but from_unit is reach {t.from_reach}")
            if rec['rule']:
                self.logger.warning(
                    f"transfer {tid}: rule '{rec['rule']}' is not interpreted; register a callable "
                    "with net.get_transfer(id).rule = f(year, time_s, q_available)")
            net.add_transfer(t)
            added.append(tid)
        net.build_topology()   # raises RuntimeError on a cycle
        if needs_capacity:
            # Capacity prior: factor * mean-flow prior at the source, using the
            # effective drainage area of the combined graph (upstream diversions
            # feed the source, e.g. Thorisvatn receives Kvislaveita).
            area = self.effective_areas_km2(net, droute_config)
            for tid in needs_capacity:
                t = net.get_transfer(tid)
                q_prior = area[t.from_reach] * d['specific_runoff_m3s_km2']
                t.capacity = float(d['capacity_factor'] * q_prior)
        self.logger.info(f"Applied {len(added)} transfer edges")
        return added

    def effective_areas_km2(self, net, droute_config: Dict[str, Any]) -> np.ndarray:
        """Drainage area of each segment on the combined tree + transfer graph.

        Transfers move ``fraction`` of the area-weighted flow (capacities and
        commissioning dates are ignored; groundwater edges are skipped).
        """
        import droute

        n = droute_config['n_segments']
        acc = np.zeros(n)
        for hru_idx, seg_idx in enumerate(droute_config['hru_to_seg_idx']):
            if seg_idx >= 0:
                acc[seg_idx] += droute_config['hru_areas'][hru_idx] / 1e6
        by_from: Dict[int, List[Any]] = {}
        for tid in net.transfer_ids():
            t = net.get_transfer(tid)
            if t.enabled and t.type != droute.TransferType.groundwater:
                by_from.setdefault(t.from_reach, []).append(t)
        for i in net.topological_order():
            out = acc[i]
            rem = out
            for t in by_from.get(i, []):
                moved = min(max(t.fraction, 0.0) * out, rem)
                rem -= moved
                if t.to_reach >= 0:
                    acc[t.to_reach] += moved
            dsi = droute_config['downstream_idx'][i]
            if dsi >= 0:
                acc[dsi] += rem
        return acc

    def apply_hydrofabric(
        self,
        net,
        droute_config: Dict[str, Any],
        transfers_path: Optional[Path] = None,
        lakes_path: Optional[Path] = None,
        enable_groundwater: bool = False,
        sim_start_year: Optional[float] = None,
        transfer_defaults: Optional[Dict[str, float]] = None,
        lake_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Apply lakes.csv (level-pool nodes) and transfers.csv to a built Network."""
        from .lake_preprocessor import apply_level_pool_lakes, load_lakes_table

        summary: Dict[str, Any] = {'lakes': 0, 'transfers': []}
        storage_withdrawals: Dict[str, int] = {}
        id_to_idx = {int(s): i for i, s in enumerate(droute_config['segment_ids'])}
        if lakes_path is not None and Path(lakes_path).exists():
            lakes = load_lakes_table(Path(lakes_path), logger=self.logger)
            storage_withdrawals = apply_level_pool_lakes(
                net, lakes, id_to_idx, widths=droute_config.get('widths'),
                params=lake_params, logger=self.logger)
            summary['lakes'] = sum(1 for u in lakes if int(u) in id_to_idx)
        if transfers_path is not None and Path(transfers_path).exists():
            records = self.load_transfers(Path(transfers_path))
            summary['transfers'] = self.apply_transfers(
                net, records, droute_config, storage_withdrawals, transfer_defaults)
        opts = net.transfer_options
        opts.enable_groundwater = bool(enable_groundwater)
        if sim_start_year is not None:
            opts.sim_start_year = float(sim_start_year)
        return summary

    def load_hydrofabric(
        self,
        directory: Path,
        mannings_n: Optional[float] = None,
        **kwargs: Any,
    ):
        """Load topology.nc, transfers.csv and lakes.csv from one directory.

        Returns ``(network, droute_config, summary)``. Extra keyword arguments go
        to :meth:`apply_hydrofabric` (enable_groundwater, sim_start_year, ...).
        """
        directory = Path(directory)
        topology = self.load_topology(directory / 'topology.nc')
        droute_config = self.to_droute_format(topology)
        net = self.build_network(droute_config, mannings_n=mannings_n)
        summary = self.apply_hydrofabric(
            net, droute_config,
            transfers_path=directory / 'transfers.csv',
            lakes_path=directory / 'lakes.csv',
            **kwargs)
        return net, droute_config, summary


#: Starting values for transfer parameters left empty in transfers.csv. They are
#: calibration parameters (fraction, capacity, gw_k are on the AD tape).
TRANSFER_DEFAULTS: Dict[str, float] = {
    'operated_fraction': 1.0,          # headrace/diversion: take up to capacity
    'capacity_factor': 1.0,            # capacity = factor * prior mean flow at from_unit
    'specific_runoff_m3s_km2': 0.05,   # prior mean flow = upstream area * 50 L/s/km2
    'bifurcation_fraction': 0.25,      # natural_bifurcation share
    'groundwater_fraction': 0.05,      # groundwater leak share
    'groundwater_k_days': 30.0,        # groundwater store residence time
}


def _num(v: Any) -> Optional[float]:
    """Parse a CSV number; empty / NaN -> None."""
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f != f else f


__all__ = ['DRouteNetworkAdapter', 'TRANSFER_DEFAULTS']
