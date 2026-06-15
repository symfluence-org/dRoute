# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 Darri Eythorsson

"""
dRoute topology generator.

Subclass of BaseTopologyGenerator that writes dRoute's YAML network config
format and optionally a mizuRoute-compatible NetCDF topology for interop.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from symfluence.models.utilities.base_topology_generator import BaseTopologyGenerator, TopologyData

if TYPE_CHECKING:
    from droute.preprocessor import DRoutePreProcessor


class DRouteTopologyGenerator(BaseTopologyGenerator):
    """
    Generates network topology for dRoute from shapefiles.

    Produces dRoute's YAML network config format with segments, connectivity,
    geometry, and HRU mapping. Also applies cycle FIXING (not just detection).
    """

    def __init__(self, preprocessor: 'DRoutePreProcessor'):
        super().__init__(preprocessor)

    def get_topology_output_path(self) -> Path:
        topology_name = self.pp._get_config_value(
            lambda: self.pp.config.model.droute.topology_file, default='droute_topology.nc'
        )
        return self.pp.setup_dir / topology_name

    def write_topology_file(self, topology_data: TopologyData, output_path: Path) -> None:
        """Write topology in dRoute's YAML network config format."""
        import yaml

        # Build routing order via topological sort
        routing_order = self.topological_sort(topology_data.seg_ids, topology_data.down_seg_ids)

        # Build downstream index mapping
        id_to_idx = {sid: i for i, sid in enumerate(topology_data.seg_ids)}
        downstream_idx = []
        for down_sid in topology_data.down_seg_ids:
            if down_sid in id_to_idx:
                downstream_idx.append(id_to_idx[down_sid])
            else:
                downstream_idx.append(-1)

        # Identify outlets
        outlet_indices = [i for i, down in enumerate(downstream_idx) if down == -1]

        # HRU-to-segment index mapping
        hru_to_seg_idx = []
        for hru_seg_id in topology_data.hru_to_seg_ids:
            if hru_seg_id in id_to_idx:
                hru_to_seg_idx.append(id_to_idx[hru_seg_id])
            else:
                hru_to_seg_idx.append(0)

        # Estimate channel widths from HRU areas (Leopold & Maddock approx)
        hru_areas_km2 = topology_data.hru_areas / 1e6
        widths = np.maximum(2.71 * np.maximum(hru_areas_km2, 0.01) ** 0.557, 1.0)

        network_config: Dict[str, Any] = {
            'n_segments': int(topology_data.num_seg),
            'segment_ids': topology_data.seg_ids.tolist(),
            'downstream_idx': downstream_idx,
            'routing_order': routing_order,
            'outlet_indices': outlet_indices,
            'slopes': np.maximum(topology_data.slopes, 0.001).tolist(),
            'lengths': topology_data.lengths.tolist(),
            'widths': widths[:topology_data.num_seg].tolist(),
            'routing_method': 'muskingum_cunge',
            'routing_dt': 3600,
            'hru_ids': topology_data.hru_ids.tolist(),
            'hru_to_seg_idx': hru_to_seg_idx,
            'hru_areas': topology_data.hru_areas.tolist(),
        }

        # Write as YAML
        yaml_output = output_path.with_suffix('.yaml') if output_path.suffix == '.nc' else output_path
        with open(yaml_output, 'w', encoding='utf-8') as f:
            yaml.dump(network_config, f, default_flow_style=False)

        self.pp.logger.info(f"dRoute topology written to {yaml_output}")

    def build_and_write(self) -> TopologyData:
        """
        Build topology from shapefiles and write dRoute config.

        Also applies cycle fixing using the elevation-based rule.
        Returns TopologyData for use by the preprocessor.
        """
        topology_data = self.build_topology()
        output_path = self.get_topology_output_path()
        self.write_topology_file(topology_data, output_path)
        return topology_data
