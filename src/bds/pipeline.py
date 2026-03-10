"""Full orchestration pipeline: file discovery → inspect → agent → validate → export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from bds.agent.orchestrator import AgentOrchestrator
from bds.config import BDSConfig, load_config
from bds.exporter import export_cell_record
from bds.inspector.archive import extract_archive, is_archive
from bds.inspector.preview import inspect_file
from bds.schema import CellRecord
from bds.validator import ValidationResult, Validator

logger = logging.getLogger(__name__)

# File extensions we attempt to process
SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".mat",
    ".json",
    ".pkl", ".pickle",
    ".h5", ".hdf5",
}


def discover_files(input_path: Path) -> List[Path]:
    """Recursively find all processable data files under input_path."""
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    files: List[Path] = []
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return files


class StandardizationPipeline:
    """End-to-end pipeline: discover → extract archives → standardize → validate → export."""

    def __init__(
        self,
        config: Optional[BDSConfig] = None,
        config_path: Optional[Path] = None,
    ):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self.agent = AgentOrchestrator(config)
        self.validator = Validator()

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Tuple[Path, Optional[CellRecord], Optional[str]]]:
        """Run the full pipeline.

        Args:
            input_path: File or directory to process.
            output_dir: Directory for output pickle files.
            progress_callback: Optional callback(file_name, current_index, total).

        Returns:
            List of (input_file, CellRecord_or_None, error_or_None) tuples.
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract archives
        files = self._prepare_files(input_path)
        total = len(files)
        logger.info("Found %d files to process", total)

        results: List[Tuple[Path, Optional[CellRecord], Optional[str]]] = []

        for idx, file_path in enumerate(files):
            name = file_path.name
            if progress_callback:
                progress_callback(name, idx + 1, total)
            logger.info("[%d/%d] Processing: %s", idx + 1, total, name)

            try:
                data = self.agent.standardize(file_path)

                if data is None:
                    results.append((file_path, None, "Agent returned no result"))
                    continue

                # Validate
                validation = self.validator.validate(data)
                if not validation.is_valid:
                    logger.warning(
                        "Validation issues for %s: %s", name, validation.issues[:3]
                    )
                    # Still export with warnings
                    data["_validation_issues"] = validation.issues

                # Export — use input filename for output to avoid collisions
                out_path = output_dir / f"{file_path.stem}.pkl"
                cell = export_cell_record(data, out_path)
                logger.info("Exported: %s", out_path)
                results.append((file_path, cell, None))

            except Exception as exc:
                logger.error("Error processing %s: %s", name, exc, exc_info=True)
                results.append((file_path, None, str(exc)))

        return results

    def _prepare_files(self, input_path: Path) -> List[Path]:
        """Discover files, extracting archives as needed."""
        if input_path.is_file() and is_archive(input_path):
            extracted = extract_archive(input_path)
            files = []
            for p in extracted:
                if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(p)
            return files

        return discover_files(input_path)

    def inspect_only(self, input_path: Path) -> str:
        """Preview file structure without LLM calls."""
        return inspect_file(Path(input_path))
