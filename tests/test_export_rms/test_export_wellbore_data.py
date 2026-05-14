"""Test RMS wellbore trajectory and log export helpers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xtgeo
from fmu.datamodels.standard_results.enums import StandardResultName
from pytest import MonkeyPatch

from fmu import dataio

if TYPE_CHECKING:
    from fmu.dataio.export.rms.wellbore_data import (
        _ExportWellboreLogs,
        _ExportWellboreTrajectory,
    )


@pytest.fixture
def xtgeo_wells() -> list[xtgeo.Well]:
    return [
        xtgeo.Well(
            wname="31/2-A-1",
            mdlogname="M_MDEPTH",
            df=pd.DataFrame(
                {
                    "X_UTME": [461000.0, 461010.0],
                    "Y_UTMN": [5930000.0, 5930010.0],
                    "Z_TVDSS": [1500.0, 1510.0],
                    "M_MDEPTH": [1500.0, 1514.0],
                    "GR": [80.0, 82.0],
                    "FACIES": [1.0, 2.0],
                }
            ),
        ),
        xtgeo.Well(
            wname="31/2-A-1 AH",
            mdlogname="M_MDEPTH",
            df=pd.DataFrame(
                {
                    "X_UTME": [461020.0],
                    "Y_UTMN": [5930020.0],
                    "Z_TVDSS": [1520.0],
                    "M_MDEPTH": [1528.0],
                    "GR": [84.0],
                    "FACIES": [2.0],
                }
            ),
        ),
    ]


@pytest.fixture
def mock_well_from_roxar(
    xtgeo_wells: list[xtgeo.Well],
) -> Generator[MagicMock, None, None]:
    wells_by_name = {well.wellname: well for well in xtgeo_wells}

    with mock.patch(
        "fmu.dataio.export.rms.wellbore_data.xtgeo.well_from_roxar",
        side_effect=lambda _project, well_name, **_kwargs: wells_by_name[well_name],
    ) as patched:
        yield patched


@pytest.fixture
def trajectory_exporter(
    mock_project_variable: MagicMock,
    monkeypatch: MonkeyPatch,
    rmssetup_with_fmuconfig: Path,
    mock_well_from_roxar: MagicMock,
) -> _ExportWellboreTrajectory:
    monkeypatch.chdir(rmssetup_with_fmuconfig)

    from fmu.dataio.export.rms.wellbore_data import _ExportWellboreTrajectory

    return _ExportWellboreTrajectory(
        mock_project_variable,
        ["31/2-A-1", "31/2-A-1 AH"],
        trajectory="Drilled trajectory",
        logrun="log",
        well_parent_names={"31/2-A-1 AH": "31/2-A-1"},
    )


@pytest.fixture
def logs_exporter(
    mock_project_variable: MagicMock,
    monkeypatch: MonkeyPatch,
    rmssetup_with_fmuconfig: Path,
    mock_well_from_roxar: MagicMock,
) -> _ExportWellboreLogs:
    monkeypatch.chdir(rmssetup_with_fmuconfig)

    from fmu.dataio.export.rms.wellbore_data import _ExportWellboreLogs

    return _ExportWellboreLogs(
        mock_project_variable,
        ["31/2-A-1", "31/2-A-1 AH"],
        trajectory="Drilled trajectory",
        logrun="log",
        lognames=["GR", "FACIES"],
        well_parent_names={"31/2-A-1 AH": "31/2-A-1"},
    )


@pytest.mark.usefixtures("inside_rms_interactive")
def test_export_wellbore_trajectory_metadata(
    trajectory_exporter: _ExportWellboreTrajectory,
    rmssetup_with_fmuconfig: Path,
) -> None:
    out = trajectory_exporter.export()

    export_folder = (
        rmssetup_with_fmuconfig / "../../share/results/tables/wellbore_trajectory"
    ).resolve()
    assert out.items[0].absolute_path == export_folder / "wellbore_trajectory.parquet"
    assert out.items[0].absolute_path.exists()
    assert out.items[0].absolute_path.read_bytes()[:4] == b"PAR1"
    assert (export_folder / ".wellbore_trajectory.parquet.yml").exists()

    metadata = dataio.read_metadata(out.items[0].absolute_path)

    assert metadata["class"] == "table"
    assert metadata["data"]["content"] == "wellbore_trajectory"
    assert (
        metadata["data"]["standard_result"]["name"]
        == StandardResultName.wellbore_trajectory
    )
    assert metadata["data"]["spec"]["columns"] == [
        "WELL",
        "WELLBORE",
        "MD",
        "X_UTME",
        "Y_UTMN",
        "Z_TVDSS",
    ]
    assert metadata["data"]["table_index"] == ["WELL", "WELLBORE", "MD"]


@pytest.mark.usefixtures("inside_rms_interactive")
def test_export_wellbore_logs_metadata(
    logs_exporter: _ExportWellboreLogs,
) -> None:
    out = logs_exporter.export()

    assert out.items[0].absolute_path.read_bytes()[:4] == b"PAR1"

    metadata = dataio.read_metadata(out.items[0].absolute_path)

    assert metadata["class"] == "table"
    assert metadata["data"]["content"] == "wellbore_logs"
    assert (
        metadata["data"]["standard_result"]["name"] == StandardResultName.wellbore_logs
    )
    assert metadata["data"]["spec"]["columns"] == [
        "WELL",
        "WELLBORE",
        "MD",
        "GR",
        "FACIES",
    ]
    assert metadata["data"]["table_index"] == ["WELL", "WELLBORE", "MD"]


@pytest.mark.usefixtures("inside_rms_interactive")
def test_public_export_functions(
    mock_project_variable: MagicMock,
    mock_well_from_roxar: MagicMock,
    rmssetup_with_fmuconfig: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(rmssetup_with_fmuconfig)

    from fmu.dataio.export.rms import export_wellbore_logs, export_wellbore_trajectories

    trajectory_out = export_wellbore_trajectories(
        mock_project_variable, ["31/2-A-1", "31/2-A-1 AH"]
    )
    logs_out = export_wellbore_logs(
        mock_project_variable,
        ["31/2-A-1", "31/2-A-1 AH"],
        lognames=["GR", "FACIES"],
    )

    assert len(trajectory_out.items) == 1
    assert len(logs_out.items) == 1
    assert mock_well_from_roxar.call_count == 4

    assert (
        dataio.read_metadata(trajectory_out.items[0].absolute_path)["data"][
            "standard_result"
        ]["name"]
        == StandardResultName.wellbore_trajectory
    )
    assert (
        dataio.read_metadata(logs_out.items[0].absolute_path)["data"][
            "standard_result"
        ]["name"]
        == StandardResultName.wellbore_logs
    )


@pytest.mark.usefixtures("inside_rms_interactive")
def test_logs_without_log_columns_raise(
    mock_project_variable: MagicMock,
    rmssetup_with_fmuconfig: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(rmssetup_with_fmuconfig)

    well = xtgeo.Well(
        wname="31/2-A-1",
        mdlogname="M_MDEPTH",
        df=pd.DataFrame(
            {
                "X_UTME": [461000.0],
                "Y_UTMN": [5930000.0],
                "Z_TVDSS": [1500.0],
                "M_MDEPTH": [1500.0],
            }
        ),
    )

    with mock.patch(
        "fmu.dataio.export.rms.wellbore_data.xtgeo.well_from_roxar",
        return_value=well,
    ):
        from fmu.dataio.export.rms.wellbore_data import _ExportWellboreLogs

        exporter = _ExportWellboreLogs(
            mock_project_variable,
            "31/2-A-1",
            trajectory="Drilled trajectory",
            logrun="log",
            lognames="all",
            well_parent_names=None,
        )

    with pytest.raises(ValueError, match="does not contain any log columns"):
        exporter.export()
