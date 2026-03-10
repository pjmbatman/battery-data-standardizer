"""Output data validation — ensures LLM-extracted data conforms to CellRecord schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "Validation passed"
        return "Validation failed:\n" + "\n".join(f"  - {i}" for i in self.issues)


class Validator:
    """Validate LLM-extracted battery data against the target schema."""

    def validate(self, data: dict) -> ValidationResult:
        issues: List[str] = []

        # Top-level checks
        if "cell_id" not in data:
            issues.append("missing 'cell_id'")

        cycles = data.get("cycles")
        if not cycles or not isinstance(cycles, list):
            issues.append("missing or empty 'cycles' list")
            return ValidationResult(is_valid=False, issues=issues)

        if len(cycles) == 0:
            issues.append("no cycles found")
            return ValidationResult(is_valid=False, issues=issues)

        for i, cycle in enumerate(cycles):
            self._validate_cycle(i, cycle, issues)

        return ValidationResult(is_valid=len(issues) == 0, issues=issues)

    def _validate_cycle(self, idx: int, cycle: dict, issues: List[str]) -> None:
        prefix = f"cycle {idx}"

        # cycle_number should exist
        if "cycle_number" not in cycle:
            issues.append(f"{prefix}: missing 'cycle_number'")

        # Voltage range check (0 ~ 6V typical for Li-ion)
        self._check_range(cycle, "voltage_v", 0.0, 6.0, prefix, issues)

        # Current range check (sanity: -1000A to 1000A)
        self._check_range(cycle, "current_a", -1000.0, 1000.0, prefix, issues)

        # Temperature range check (-40 ~ 100°C)
        self._check_range(cycle, "temperature_c", -40.0, 100.0, prefix, issues)

        # Time monotonicity
        time_s = cycle.get("time_s")
        if time_s and isinstance(time_s, list) and len(time_s) > 1:
            if not all(a <= b for a, b in zip(time_s, time_s[1:])):
                issues.append(f"{prefix}: time_s not monotonically increasing")

        # Array length consistency
        lengths = {}
        for field_name in ("voltage_v", "current_a", "time_s", "temperature_c",
                           "charge_capacity_ah", "discharge_capacity_ah", "soc"):
            val = cycle.get(field_name)
            if val and isinstance(val, list):
                lengths[field_name] = len(val)
        if lengths:
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                issues.append(f"{prefix}: inconsistent array lengths: {lengths}")

    @staticmethod
    def _check_range(
        cycle: dict,
        field_name: str,
        lo: float,
        hi: float,
        prefix: str,
        issues: List[str],
    ) -> None:
        values = cycle.get(field_name)
        if not values or not isinstance(values, list) or len(values) == 0:
            return
        # Filter to numeric values only
        numeric = [v for v in values if isinstance(v, (int, float))]
        if not numeric:
            return
        v_min = min(numeric)
        v_max = max(numeric)
        if v_max > hi or v_min < lo:
            issues.append(
                f"{prefix}: {field_name} out of range [{v_min:.4f}, {v_max:.4f}] "
                f"(expected [{lo}, {hi}])"
            )
