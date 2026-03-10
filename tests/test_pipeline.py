"""Tests for schema, validator, exporter, and pipeline discovery."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from bds.schema import (
    CellRecord,
    CycleRecord,
    calculate_cycles_stat,
    dict_to_cell_record,
)
from bds.validator import Validator
from bds.exporter import export_cell_record
from bds.pipeline import discover_files


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# --- Schema tests ---

class TestCycleRecord:
    def test_to_dict_basic(self):
        c = CycleRecord(
            cycle_number=1,
            voltage_v=[3.5, 3.6, 3.7],
            current_a=[0.5, 0.5, 0.5],
            time_s=[0.0, 10.0, 20.0],
        )
        d = c.to_dict()
        assert d["cycle_number"] == 1
        assert d["voltage_v"] == [3.5, 3.6, 3.7]
        assert "temperature_c" not in d  # None fields excluded

    def test_additional_data(self):
        c = CycleRecord(cycle_number=1, additional_data={"custom_field": 42})
        d = c.to_dict()
        assert d["custom_field"] == 42


class TestCellRecord:
    def test_dump_and_load(self, tmp_dir):
        cell = CellRecord(
            cell_id="test_cell",
            cycles=[CycleRecord(cycle_number=1, voltage_v=[3.5])],
        )
        path = tmp_dir / "test.pkl"
        cell.dump(path)
        loaded = CellRecord.load(path)
        assert loaded.cell_id == "test_cell"
        assert len(loaded.cycles) == 1

    def test_to_dict(self):
        cell = CellRecord(
            cell_id="B0005",
            nominal_capacity_ah=2.0,
            cycles=[CycleRecord(cycle_number=1)],
        )
        d = cell.to_dict()
        assert d["cell_id"] == "B0005"
        assert d["nominal_capacity_ah"] == 2.0


class TestCalculateCyclesStat:
    def test_basic_stats(self):
        cycles = [
            CycleRecord(
                cycle_number=1,
                voltage_v=[3.0, 3.5, 4.0],
                current_a=[0.5, 0.5, 0.5],
            ),
            CycleRecord(
                cycle_number=2,
                voltage_v=[3.1, 3.6, 4.1],
                current_a=[-0.5, -0.5, -0.5],
            ),
        ]
        stat = calculate_cycles_stat(cycles)
        assert stat["max_voltages"] == [4.0, 4.1]
        assert stat["min_voltages"] == [3.0, 3.1]
        assert stat["max_currents"] == [0.5, -0.5]

    def test_missing_fields(self):
        cycles = [CycleRecord(cycle_number=1)]
        stat = calculate_cycles_stat(cycles)
        assert stat["max_voltages"] == [None]
        assert stat["min_temperatures"] == [None]


class TestDictToCellRecord:
    def test_basic_conversion(self):
        data = {
            "cell_id": "test",
            "cycles": [
                {"cycle_number": 1, "voltage_v": [3.5, 3.6]},
                {"cycle_number": 2, "voltage_v": [3.4, 3.5]},
            ],
        }
        cell = dict_to_cell_record(data)
        assert cell.cell_id == "test"
        assert len(cell.cycles) == 2
        assert cell.cycles[0].voltage_v == [3.5, 3.6]


# --- Validator tests ---

class TestValidator:
    def setup_method(self):
        self.v = Validator()

    def test_valid_data(self):
        data = {
            "cell_id": "test",
            "cycles": [
                {
                    "cycle_number": 1,
                    "voltage_v": [3.0, 3.5, 4.0],
                    "current_a": [0.5, 0.5, 0.5],
                    "time_s": [0.0, 10.0, 20.0],
                }
            ],
        }
        result = self.v.validate(data)
        assert result.is_valid

    def test_missing_cell_id(self):
        data = {"cycles": [{"cycle_number": 1}]}
        result = self.v.validate(data)
        assert not result.is_valid
        assert any("cell_id" in i for i in result.issues)

    def test_empty_cycles(self):
        data = {"cell_id": "test", "cycles": []}
        result = self.v.validate(data)
        assert not result.is_valid

    def test_voltage_out_of_range(self):
        data = {
            "cell_id": "test",
            "cycles": [
                {"cycle_number": 1, "voltage_v": [3000.0, 3500.0]}  # mV not converted
            ],
        }
        result = self.v.validate(data)
        assert not result.is_valid
        assert any("voltage_v" in i for i in result.issues)

    def test_non_monotonic_time(self):
        data = {
            "cell_id": "test",
            "cycles": [
                {"cycle_number": 1, "time_s": [0.0, 10.0, 5.0]}  # not monotonic
            ],
        }
        result = self.v.validate(data)
        assert not result.is_valid
        assert any("monotonic" in i for i in result.issues)

    def test_inconsistent_array_lengths(self):
        data = {
            "cell_id": "test",
            "cycles": [
                {
                    "cycle_number": 1,
                    "voltage_v": [3.0, 3.5],
                    "current_a": [0.5, 0.5, 0.5],  # different length
                    "time_s": [0.0, 10.0],
                }
            ],
        }
        result = self.v.validate(data)
        assert not result.is_valid
        assert any("inconsistent" in i for i in result.issues)


# --- Exporter tests ---

class TestExporter:
    def test_export_cell_record(self, tmp_dir):
        data = {
            "cell_id": "export_test",
            "cycles": [
                {
                    "cycle_number": 1,
                    "voltage_v": [3.0, 3.5, 4.0],
                    "current_a": [0.5, 0.5, 0.5],
                },
            ],
        }
        out_path = tmp_dir / "export_test.pkl"
        cell = export_cell_record(data, out_path)

        assert out_path.exists()
        assert cell.cell_id == "export_test"
        assert cell.cycles_stat is not None
        assert len(cell.cycles_stat["max_voltages"]) == 1

        # Verify it's loadable
        loaded = CellRecord.load(out_path)
        assert loaded.cell_id == "export_test"


# --- Pipeline discovery tests ---

class TestDiscoverFiles:
    def test_discover_csv(self, tmp_dir):
        (tmp_dir / "data.csv").write_text("a,b\n1,2\n")
        (tmp_dir / "readme.md").write_text("# readme")
        files = discover_files(tmp_dir)
        assert len(files) == 1
        assert files[0].name == "data.csv"

    def test_discover_nested(self, tmp_dir):
        sub = tmp_dir / "sub"
        sub.mkdir()
        (sub / "data.mat").write_bytes(b"fake")
        (tmp_dir / "top.csv").write_text("a\n1\n")
        files = discover_files(tmp_dir)
        assert len(files) == 2

    def test_single_file(self, tmp_dir):
        f = tmp_dir / "test.json"
        f.write_text("{}")
        files = discover_files(f)
        assert len(files) == 1
