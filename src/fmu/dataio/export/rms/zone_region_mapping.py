from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa

import fmu.dataio as dio
from fmu.dataio._export_service import ExportService
from fmu.dataio._logging import null_logger
from fmu.dataio.export._export_result import ExportResult, ExportResultItem
from fmu.dataio.export.rms._base import SimpleExportRMSBase
from fmu.dataio.export.rms._conditional_rms_imports import import_rms_package
from fmu.dataio.export.rms._utils import check_rmsapi_version
from fmu.datamodels.common.enums import Classification
from fmu.datamodels.fmu_results.enums import Content
from fmu.datamodels.standard_results import enums

rmsapi, rmsjobs = import_rms_package()

_logger: Final = null_logger(__name__)

_TableIndexColumns = enums.ZoneRegionIndex.TableIndexColumns


def create_zone_region_mapping(
    project: Any,
    grid_name: str,
    volume_job_name: str,
) -> pd.DataFrame:
    """Create a mapping between zone/region combinations and unique integers.

    This function reads the zone and region configuration from a volumetrics job
    and creates a mapping table where each zone/region combination is assigned
    a unique integer ID. This ID (similar to Eclipse FIPNUM) can be used to
    create a grid property vector that enables comparison between RMS volumes
    and simulator volumes.

    The mapping is created by looping through zones (outer loop) and regions
    (inner loop), assigning incrementing integers starting from 1.

    Args:
        project: The 'magic' project variable in RMS.
        grid_name: Name of 3D grid model in RMS.
        volume_job_name: Name of the volume job (same job used for volume export).

    Returns:
        DataFrame with columns ZONE, REGION, and FIPGRP (the unique integer ID).

    Examples:
        Example usage in an RMS script::

            from fmu.dataio.export.rms import create_zone_region_mapping

            mapping = create_zone_region_mapping(project, "Geogrid", "geogrid_volumes")
            print(mapping)
            # Output:
            #       ZONE      REGION  FIPGRP
            # 0  Valysar  WestLowland       1
            # 1  Valysar  CentralSouth      2
            # ...

    """
    check_rmsapi_version(minimum_version="1.10")

    if grid_name not in project.grid_models:
        raise ValueError(f"No grid model with name '{grid_name}' exists.")

    available_volume_jobs = rmsapi.jobs.Job.get_job_names(owner=["Grid models", grid_name, "Grid"], type="Volumetrics")

    if volume_job_name not in available_volume_jobs:
        raise ValueError(
            f"No volume job with name '{volume_job_name}' exists "
            f"for grid model named '{grid_name}'.\n"
            f"Available volume jobs:\n{available_volume_jobs}"
        )

    volume_job = rmsjobs.Job.get_job(
        owner=["Grid models", grid_name, "Grid"],
        type="Volumetrics",
        name=volume_job_name,
    ).get_arguments()

    input_params = volume_job.get("Input", [{}])[0]
    zone_names = input_params.get("SelectedZoneNames", [])
    region_names = input_params.get("SelectedRegionNames", [])

    if not zone_names:
        raise ValueError(
            f"No zones selected in volume job '{volume_job_name}'. " "Please configure zones in the volumetric job."
        )

    if not region_names:
        raise ValueError(
            f"No regions selected in volume job '{volume_job_name}'. " "Please configure regions in the volumetric job."
        )

    # Create mapping: outer loop zones, inner loop regions
    mapping = []
    fipgrp_id = 1
    for zone in zone_names:
        for region in region_names:
            mapping.append(
                {
                    _TableIndexColumns.ZONE.value: zone,
                    _TableIndexColumns.REGION.value: region,
                    _TableIndexColumns.FIPGRP.value: fipgrp_id,
                }
            )
            fipgrp_id += 1

    return pd.DataFrame(mapping)


class _ExportZoneRegionMapping(SimpleExportRMSBase):
    """Export zone/region mapping table from RMS."""

    def __init__(
        self,
        project: Any,
        grid_name: str,
        volume_job_name: str,
    ) -> None:
        super().__init__()

        self.project = project
        self.grid_name = grid_name
        self.volume_job_name = volume_job_name

        _logger.debug("Creating zone/region mapping...")
        self._dataframe = create_zone_region_mapping(project, grid_name, volume_job_name)
        _logger.debug("Zone/region mapping created with %d rows", len(self._dataframe))

    @property
    def _standard_result(self) -> None:  # type: ignore
        """No standard result for zone/region mapping."""
        return None

    # @property
    # def _subfolder(self) -> str:
    #     """Use 'mappings' as subfolder for zone/region mapping."""
    #     return "mappings"

    @property
    def _content(self) -> Content:
        """Get content for the exported data."""
        return Content.mappings

    @property
    def _classification(self) -> Classification:
        """Get default classification."""
        return Classification.internal

    @property
    def _rep_include(self) -> bool:
        """rep_include status"""
        return False

    def _export_data_as_standard_result(self) -> ExportResult:
        """Export the zone/region mapping table."""

        edata = dio.ExportData(
            config=self._config,
            content=self._content,
            subfolder=self._subfolder,
            classification=self._classification,
            name=f"{self.grid_name}_zone_region_mapping",
            tagname="zone_region_mapping",
            rep_include=self._rep_include,
            table_index=[_TableIndexColumns.ZONE.value, _TableIndexColumns.REGION.value],
        )

        mapping_table = pa.Table.from_pandas(self._dataframe)

        export_service = ExportService(export_config=edata._export_config)
        absolute_export_path = export_service.export_with_metadata(mapping_table)

        _logger.debug("Zone/region mapping exported to: %s", absolute_export_path)
        return ExportResult(
            items=[
                ExportResultItem(
                    absolute_path=Path(absolute_export_path),
                )
            ],
        )

    def _validate_data_pre_export(self) -> None:
        """Validate the mapping table."""
        if self._dataframe.empty:
            raise RuntimeError("Zone/region mapping table is empty.")

        # Check for required columns
        required_cols = [
            _TableIndexColumns.ZONE.value,
            _TableIndexColumns.REGION.value,
            _TableIndexColumns.FIPGRP.value,
        ]
        for col in required_cols:
            if col not in self._dataframe.columns:
                raise RuntimeError(f"Required column '{col}' missing from mapping table.")

        # Check for duplicate zone/region combinations
        duplicates = self._dataframe.duplicated(subset=[_TableIndexColumns.ZONE.value, _TableIndexColumns.REGION.value])
        if duplicates.any():
            raise RuntimeError("Duplicate zone/region combinations found in mapping.")

        # Check FIPGRP values are unique and sequential
        fipgrp_values = sorted(self._dataframe[_TableIndexColumns.FIPGRP.value].values)
        expected_values = list(range(1, len(self._dataframe) + 1))
        if fipgrp_values != expected_values:
            raise RuntimeError(f"{_TableIndexColumns.FIPGRP.value} values are not sequential integers starting from 1.")


def export_zone_region_mapping(
    project: Any,
    grid_name: str,
    volume_job_name: str,
) -> ExportResult:
    """Export zone/region to FIPGRP mapping table for RMS-simulator comparison.

    This function creates and exports a mapping table between zone/region combinations
    and unique integer IDs (similar to Eclipse FIPNUM). The mapping uses the exact
    zones and regions configured in the specified volumetrics job, ensuring consistency
    with exported RMS volumes.

    This mapping table is intended to be used together with inplace volumes to enable
    comparison between RMS volumetrics and simulator (Eclipse) volumes.

    Args:
        project: The 'magic' project variable in RMS.
        grid_name: Name of 3D grid model in RMS.
        volume_job_name: Name of the volume job (should be the same job used for
            exporting volumes via export_inplace_volumes).

    Returns:
        ExportResult containing the path to the exported mapping table.

    Examples:
        Example usage in an RMS script::

            from fmu.dataio.export.rms import (
                export_inplace_volumes,
                export_zone_region_mapping,
            )

            # Export volumes
            export_inplace_volumes(project, "Geogrid", "geogrid_volumes")

            # Export mapping for comparison with simulator
            export_zone_region_mapping(project, "Geogrid", "geogrid_volumes")

    """
    check_rmsapi_version(minimum_version="1.10")

    return _ExportZoneRegionMapping(
        project,
        grid_name,
        volume_job_name,
    ).export()
