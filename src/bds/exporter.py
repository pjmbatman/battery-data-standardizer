"""Export standardized data as CellRecord-compatible pickle files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from bds.schema import CellRecord, calculate_cycles_stat, dict_to_cell_record


def export_cell_record(data: dict, output_path: Path) -> CellRecord:
    """Convert LLM-extracted dict to CellRecord and save as pickle.

    Returns the CellRecord instance.
    """
    cell = dict_to_cell_record(data)

    # Compute cycles_stat if not already present
    if cell.cycles_stat is None and cell.cycles:
        cell.cycles_stat = calculate_cycles_stat(cell.cycles)

    output_path = Path(output_path)
    cell.dump(output_path)
    return cell


def export_multiple(
    data_list: List[dict],
    output_dir: Path,
) -> List[CellRecord]:
    """Export multiple cell records to a directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for data in data_list:
        cell_id = data.get("cell_id", "unknown")
        out_path = output_dir / f"{cell_id}.pkl"
        cell = export_cell_record(data, out_path)
        cells.append(cell)
    return cells
