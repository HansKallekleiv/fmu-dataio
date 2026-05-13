from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
import xtgeo

from fmu.dataio._export import ExportConfig, export_with_metadata
from fmu.dataio._logging import null_logger
from fmu.dataio.exceptions import ValidationError
from fmu.dataio.export._base import SimpleExportBase
from fmu.dataio.export._export_result import ExportResult, ExportResultItem
from fmu.dataio.export.rms._utils import get_rms_project_units
from fmu.datamodels.common.enums import Classification
from fmu.datamodels.fmu_results.enums import (
    Content,
    DomainReference,
    VerticalDomain,
)
from fmu.datamodels.standard_results.enums import (
    StandardResultName,
    WellboreLogs,
    WellboreTrajectory,
)

_logger: Final = null_logger(__name__)

_MD_COLUMN_CANDIDATES: Final = ("MD", "M_MDEPTH", "Q_MDEPTH")
_SURVEY_COLUMN_CANDIDATES: Final = (
    "M_INCL",
    "Q_INCL",
    "M_AZI",
    "Q_AZI",
)


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _well_parent_name(
    wellbore_name: str, well_parent_names: dict[str, str] | None
) -> str:
    if well_parent_names is None:
        return wellbore_name
    return well_parent_names.get(wellbore_name, wellbore_name)


def _find_md_column(well: xtgeo.Well, dataframe: pd.DataFrame) -> str:
    if well.mdlogname in dataframe.columns:
        return str(well.mdlogname)

    for column in _MD_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValidationError(
        "The well dataframe must contain a measured depth column. "
        f"Expected one of: {list(_MD_COLUMN_CANDIDATES)}."
    )


def _load_wells_from_rms(
    project: Any,
    well_names: str | Sequence[str],
    trajectory: str | None,
    logrun: str | None,
    lognames: str | Sequence[str] | None,
) -> list[xtgeo.Well]:
    return [
        xtgeo.well_from_roxar(
            project,
            well_name,
            trajectory=trajectory,
            logrun=logrun,
            lognames=lognames,
            inclmd=True,
        )
        for well_name in _as_list(well_names)
    ]


def _trajectory_table(
    wells: Sequence[xtgeo.Well], well_parent_names: dict[str, str] | None
) -> pd.DataFrame:
    tables = []

    for well in wells:
        dataframe = well.get_dataframe()
        md_column = _find_md_column(well, dataframe)
        table = dataframe[[well.xname, well.yname, well.zname, md_column]].copy()
        table = table.rename(
            columns={
                well.xname: "X_UTME",
                well.yname: "Y_UTMN",
                well.zname: "Z_TVDSS",
                md_column: "MD",
            }
        )
        table.insert(0, "WELLBORE", well.wellname)
        table.insert(0, "WELL", _well_parent_name(well.wellname, well_parent_names))
        tables.append(
            table[["WELL", "WELLBORE", "MD", "X_UTME", "Y_UTMN", "Z_TVDSS"]]
        )

    return pd.concat(tables, ignore_index=True)


def _logs_table(
    wells: Sequence[xtgeo.Well], well_parent_names: dict[str, str] | None
) -> pd.DataFrame:
    tables = []

    for well in wells:
        dataframe = well.get_dataframe()
        md_column = _find_md_column(well, dataframe)
        excluded_columns = {
            well.xname,
            well.yname,
            well.zname,
            md_column,
            *_MD_COLUMN_CANDIDATES,
            *_SURVEY_COLUMN_CANDIDATES,
        }
        log_columns = [
            column for column in dataframe.columns if column not in excluded_columns
        ]
        table = dataframe[[md_column, *log_columns]].copy()
        table = table.rename(columns={md_column: "MD"})
        table.insert(0, "WELLBORE", well.wellname)
        table.insert(0, "WELL", _well_parent_name(well.wellname, well_parent_names))
        tables.append(table)

    return pd.concat(tables, ignore_index=True, sort=False)


class _ExportWellboreTrajectory(SimpleExportBase):
    def __init__(
        self,
        project: Any,
        well_names: str | Sequence[str],
        trajectory: str | None,
        logrun: str | None,
        well_parent_names: dict[str, str] | None,
    ) -> None:
        super().__init__()

        _logger.debug("Process wellbore trajectory data from RMS.")
        wells = _load_wells_from_rms(
            project, well_names, trajectory=trajectory, logrun=logrun, lognames=[]
        )
        self._table = _trajectory_table(wells, well_parent_names)
        self._unit = "m" if get_rms_project_units(project) == "metric" else "ft"

    def _get_export_config(self) -> ExportConfig:
        return (
            ExportConfig.builder()
            .content(Content.wellbore_trajectory)
            .domain(VerticalDomain.depth, DomainReference.msl)
            .unit(self._unit)
            .table_config(
                table_index=WellboreTrajectory.TableIndexColumns.index_columns()
            )
            .file_config(
                name=StandardResultName.wellbore_trajectory.value,
                subfolder=StandardResultName.wellbore_trajectory.value,
            )
            .access(Classification.internal, rep_include=True)
            .global_config(self._config)
            .standard_result(StandardResultName.wellbore_trajectory)
            .build()
        )

    def _export_data_as_standard_result(self) -> ExportResult:
        absolute_export_path = export_with_metadata(
            self._get_export_config(), self._table
        )
        _logger.debug("Wellbore trajectory exported to: %s", absolute_export_path)

        return ExportResult(
            items=[ExportResultItem(absolute_path=Path(absolute_export_path))],
        )

    def _validate_data_pre_export(self) -> None:
        if self._table.empty:
            raise ValidationError("The wellbore trajectory table is empty.")


class _ExportWellboreLogs(SimpleExportBase):
    def __init__(
        self,
        project: Any,
        well_names: str | Sequence[str],
        trajectory: str | None,
        logrun: str | None,
        lognames: str | Sequence[str] | None,
        well_parent_names: dict[str, str] | None,
    ) -> None:
        super().__init__()

        _logger.debug("Process wellbore log data from RMS.")
        wells = _load_wells_from_rms(
            project,
            well_names,
            trajectory=trajectory,
            logrun=logrun,
            lognames=lognames,
        )
        self._table = _logs_table(wells, well_parent_names)
        self._unit = "m" if get_rms_project_units(project) == "metric" else "ft"

    def _get_export_config(self) -> ExportConfig:
        return (
            ExportConfig.builder()
            .content(Content.wellbore_logs)
            .domain(VerticalDomain.depth, DomainReference.msl)
            .unit(self._unit)
            .table_config(table_index=WellboreLogs.TableIndexColumns.index_columns())
            .file_config(
                name=StandardResultName.wellbore_logs.value,
                subfolder=StandardResultName.wellbore_logs.value,
            )
            .access(Classification.internal, rep_include=True)
            .global_config(self._config)
            .standard_result(StandardResultName.wellbore_logs)
            .build()
        )

    def _export_data_as_standard_result(self) -> ExportResult:
        absolute_export_path = export_with_metadata(
            self._get_export_config(), self._table
        )
        _logger.debug("Wellbore logs exported to: %s", absolute_export_path)

        return ExportResult(
            items=[ExportResultItem(absolute_path=Path(absolute_export_path))],
        )

    def _validate_data_pre_export(self) -> None:
        if self._table.empty:
            raise ValidationError("The wellbore logs table is empty.")

        if len(self._table.columns) <= len(
            WellboreLogs.TableIndexColumns.index_columns()
        ):
            raise ValidationError(
                "The wellbore logs table does not contain any log columns."
            )


def export_wellbore_trajectories(
    project: Any,
    well_names: str | Sequence[str],
    trajectory: str | None = "Drilled trajectory",
    logrun: str | None = "log",
    well_parent_names: dict[str, str] | None = None,
) -> ExportResult:
    """Simplified interface when exporting wellbore trajectories from RMS.

    Args:
        project: The 'magic' project variable in RMS.
        well_names: One or more RMS well names to export.
        trajectory: Name of trajectory in RMS.
        logrun: Name of logrun in RMS. Used to include measured depth.
        well_parent_names: Optional mapping from wellbore name to local well name.
    """

    return _ExportWellboreTrajectory(
        project, well_names, trajectory, logrun, well_parent_names
    ).export()


def export_wellbore_logs(
    project: Any,
    well_names: str | Sequence[str],
    trajectory: str | None = "Drilled trajectory",
    logrun: str | None = "log",
    lognames: str | Sequence[str] | None = "all",
    well_parent_names: dict[str, str] | None = None,
) -> ExportResult:
    """Simplified interface when exporting wellbore logs from RMS.

    Args:
        project: The 'magic' project variable in RMS.
        well_names: One or more RMS well names to export.
        trajectory: Name of trajectory in RMS.
        logrun: Name of logrun in RMS.
        lognames: Log names to export, or 'all' for all logs in the logrun.
        well_parent_names: Optional mapping from wellbore name to local well name.
    """

    return _ExportWellboreLogs(
        project, well_names, trajectory, logrun, lognames, well_parent_names
    ).export()