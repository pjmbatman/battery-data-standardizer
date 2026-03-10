"""Target schema definitions — CellRecord / CycleRecord compatible with BFF."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import pickle
from pathlib import Path


@dataclass
class CyclingProtocol:
    c_rate: Optional[float] = None
    current_a: Optional[float] = None
    voltage_v: Optional[float] = None
    power_w: Optional[float] = None
    start_voltage_v: Optional[float] = None
    start_soc: Optional[float] = None
    end_voltage_v: Optional[float] = None
    end_soc: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CycleRecord:
    cycle_number: int
    voltage_v: Optional[List[float]] = None
    current_a: Optional[List[float]] = None
    charge_capacity_ah: Optional[List[float]] = None
    discharge_capacity_ah: Optional[List[float]] = None
    time_s: Optional[List[float]] = None
    temperature_c: Optional[List[float]] = None
    internal_resistance_ohm: Optional[float] = None
    soh: Optional[float] = None
    soc: Optional[List[float]] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if k == "additional_data":
                d.update(v)
            elif v is not None:
                d[k] = v
        return d


@dataclass
class CellRecord:
    cell_id: str
    cycles: List[CycleRecord] = field(default_factory=list)
    form_factor: Optional[str] = None
    anode_material: Optional[str] = None
    cathode_material: Optional[str] = None
    electrolyte_material: Optional[str] = None
    nominal_capacity_ah: Optional[float] = None
    depth_of_charge: float = 1.0
    depth_of_discharge: float = 1.0
    initial_cycles: int = 0
    charge_protocol: Optional[List[CyclingProtocol]] = None
    discharge_protocol: Optional[List[CyclingProtocol]] = None
    max_voltage_limit_v: Optional[float] = None
    min_voltage_limit_v: Optional[float] = None
    max_current_limit_a: Optional[float] = None
    min_current_limit_a: Optional[float] = None
    reference: Optional[str] = None
    description: Optional[str] = None
    cycles_stat: Optional[Dict[str, List[Optional[float]]]] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k in [
            "cell_id", "form_factor", "anode_material", "cathode_material",
            "electrolyte_material", "nominal_capacity_ah", "depth_of_charge",
            "depth_of_discharge", "initial_cycles", "max_voltage_limit_v",
            "min_voltage_limit_v", "max_current_limit_a", "min_current_limit_a",
            "reference", "description", "cycles_stat",
        ]:
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.charge_protocol:
            d["charge_protocol"] = [p.to_dict() for p in self.charge_protocol]
        if self.discharge_protocol:
            d["discharge_protocol"] = [p.to_dict() for p in self.discharge_protocol]
        d["cycles"] = [c.to_dict() for c in self.cycles]
        d.update(self.additional_data)
        return d

    def dump(self, path: str | Path) -> None:
        """Save as dict pickle — compatible with BFF CellRecord.load()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str | Path) -> "CellRecord":
        """Load from pickle — handles both dict and CellRecord objects."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            # Dict format (BFF-compatible)
            if obj.get("charge_protocol"):
                obj["charge_protocol"] = [
                    CyclingProtocol(**p) if isinstance(p, dict) else p
                    for p in obj["charge_protocol"]
                ]
            if obj.get("discharge_protocol"):
                obj["discharge_protocol"] = [
                    CyclingProtocol(**p) if isinstance(p, dict) else p
                    for p in obj["discharge_protocol"]
                ]
            if obj.get("cycles"):
                obj["cycles"] = [
                    CycleRecord(**c) if isinstance(c, dict) else c
                    for c in obj["cycles"]
                ]
            return CellRecord(**obj)
        return obj  # Already a CellRecord instance


def calculate_cycles_stat(cycles: List[CycleRecord]) -> Dict[str, List[Optional[float]]]:
    """Calculate per-cycle statistics matching BFF base_standardizer output."""
    stat = {
        "max_voltages": [], "min_voltages": [], "mean_voltages": [],
        "max_currents": [], "min_currents": [], "mean_currents": [],
        "max_temperatures": [], "min_temperatures": [], "mean_temperatures": [],
        "max_discharge_capacities": [], "min_discharge_capacities": [], "mean_discharge_capacities": [],
        "max_charge_capacities": [], "min_charge_capacities": [], "mean_charge_capacities": [],
    }

    for cycle in cycles:
        _append_stats(stat, "voltages", cycle.voltage_v)
        _append_stats(stat, "currents", cycle.current_a)
        _append_stats(stat, "temperatures", cycle.temperature_c)
        _append_stats(stat, "discharge_capacities", cycle.discharge_capacity_ah)
        _append_stats(stat, "charge_capacities", cycle.charge_capacity_ah)

    return stat


def _append_stats(
    stat: dict, name: str, values: Optional[List[float]]
) -> None:
    if values is None:
        stat[f"max_{name}"].append(None)
        stat[f"min_{name}"].append(None)
        stat[f"mean_{name}"].append(None)
        return
    # Handle scalar values (e.g., single capacity float from LLM)
    if isinstance(values, (int, float)):
        stat[f"max_{name}"].append(float(values))
        stat[f"min_{name}"].append(float(values))
        stat[f"mean_{name}"].append(float(values))
        return
    if len(values) > 0:
        stat[f"max_{name}"].append(max(values))
        stat[f"min_{name}"].append(min(values))
        stat[f"mean_{name}"].append(sum(values) / len(values))
    else:
        stat[f"max_{name}"].append(None)
        stat[f"min_{name}"].append(None)
        stat[f"mean_{name}"].append(None)


def dict_to_cell_record(data: dict) -> CellRecord:
    """Convert a raw dict (from LLM output) into a CellRecord instance."""
    cycles_raw = data.pop("cycles", [])
    cycles = []
    for c in cycles_raw:
        additional = {}
        known_keys = {
            "cycle_number", "voltage_v", "current_a", "charge_capacity_ah",
            "discharge_capacity_ah", "time_s", "temperature_c",
            "internal_resistance_ohm", "soh", "soc",
        }
        for k, v in c.items():
            if k not in known_keys:
                additional[k] = v
        cycle = CycleRecord(
            cycle_number=c.get("cycle_number", 0),
            voltage_v=c.get("voltage_v"),
            current_a=c.get("current_a"),
            charge_capacity_ah=c.get("charge_capacity_ah"),
            discharge_capacity_ah=c.get("discharge_capacity_ah"),
            time_s=c.get("time_s"),
            temperature_c=c.get("temperature_c"),
            internal_resistance_ohm=c.get("internal_resistance_ohm"),
            soh=c.get("soh"),
            soc=c.get("soc"),
            additional_data=additional,
        )
        cycles.append(cycle)

    charge_proto = None
    if "charge_protocol" in data:
        raw = data.pop("charge_protocol")
        if raw:
            charge_proto = [CyclingProtocol(**p) if isinstance(p, dict) else p for p in raw]

    discharge_proto = None
    if "discharge_protocol" in data:
        raw = data.pop("discharge_protocol")
        if raw:
            discharge_proto = [CyclingProtocol(**p) if isinstance(p, dict) else p for p in raw]

    known_cell_keys = {
        "cell_id", "form_factor", "anode_material", "cathode_material",
        "electrolyte_material", "nominal_capacity_ah", "depth_of_charge",
        "depth_of_discharge", "initial_cycles", "max_voltage_limit_v",
        "min_voltage_limit_v", "max_current_limit_a", "min_current_limit_a",
        "reference", "description", "cycles_stat",
    }
    additional = {k: v for k, v in data.items() if k not in known_cell_keys}

    return CellRecord(
        cell_id=data.get("cell_id", "unknown"),
        cycles=cycles,
        form_factor=data.get("form_factor"),
        anode_material=data.get("anode_material"),
        cathode_material=data.get("cathode_material"),
        electrolyte_material=data.get("electrolyte_material"),
        nominal_capacity_ah=data.get("nominal_capacity_ah"),
        depth_of_charge=data.get("depth_of_charge", 1.0),
        depth_of_discharge=data.get("depth_of_discharge", 1.0),
        initial_cycles=data.get("initial_cycles", 0),
        charge_protocol=charge_proto,
        discharge_protocol=discharge_proto,
        max_voltage_limit_v=data.get("max_voltage_limit_v"),
        min_voltage_limit_v=data.get("min_voltage_limit_v"),
        max_current_limit_a=data.get("max_current_limit_a"),
        min_current_limit_a=data.get("min_current_limit_a"),
        reference=data.get("reference"),
        description=data.get("description"),
        cycles_stat=data.get("cycles_stat"),
        additional_data=additional,
    )
