"""CLI entry point for Battery Data Standardizer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from bds.config import BDSConfig, find_config, load_config

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.version_option(package_name="battery-data-standardizer")
def main():
    """Battery Data Standardizer — AI-powered automatic standardization."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", default="./standardized", help="Output directory")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Config YAML path")
@click.option("--api-base", default=None, help="vLLM server base URL")
@click.option("--model", default=None, help="Model name override")
@click.option("--batch", is_flag=True, help="Batch mode (no interactive confirmation)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def standardize(
    input_path: str,
    output_dir: str,
    config_path: Optional[str],
    api_base: Optional[str],
    model: Optional[str],
    batch: bool,
    verbose: bool,
):
    """Standardize battery data files into CellRecord format."""
    _setup_logging(verbose)

    # Build config
    overrides = {}
    if api_base:
        overrides.setdefault("llm", {})["api_base"] = api_base
    if model:
        overrides.setdefault("llm", {})["model"] = model

    cfg_path = config_path or find_config()
    config = load_config(cfg_path, overrides if overrides else None)

    from bds.pipeline import StandardizationPipeline, discover_files
    from bds.agent.llm_client import LLMClient

    # Check vLLM server connection
    llm = LLMClient(config.llm)
    if not llm.check_connection():
        console.print(
            "[red]Cannot connect to vLLM server.[/red]\n"
            f"  api_base: {config.llm.api_base}\n"
            f"  model: {config.llm.model}\n\n"
            "Start the server with: bash scripts/serve_model.sh"
        )
        sys.exit(1)
    # Propagate auto-selected model back to config
    config.llm.model = llm.model

    pipeline = StandardizationPipeline(config=config)

    input_p = Path(input_path)
    output_p = Path(output_dir)

    # Preview files
    files = discover_files(input_p)
    if not files:
        console.print("[red]No supported files found.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Found {len(files)} file(s) to standardize:[/bold]")
    for f in files[:20]:
        console.print(f"  {f.name}")
    if len(files) > 20:
        console.print(f"  ... and {len(files) - 20} more")

    if not batch:
        if not click.confirm("\nProceed?", default=True):
            sys.exit(0)

    # Run pipeline with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Standardizing...", total=len(files))

        def on_progress(name: str, current: int, total: int):
            progress.update(task, completed=current, description=f"Processing {name}")

        results = pipeline.run(input_p, output_p, progress_callback=on_progress)

    # Summary
    console.print()
    table = Table(title="Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    success = 0
    for file_path, cell, error in results:
        if cell is not None:
            n_cycles = len(cell.cycles)
            table.add_row(file_path.name, "[green]OK[/green]", f"{n_cycles} cycles")
            success += 1
        else:
            table.add_row(file_path.name, "[red]FAIL[/red]", error or "unknown error")

    console.print(table)
    console.print(f"\n[bold]{success}/{len(results)} files standardized successfully.[/bold]")
    console.print(f"Output: {output_p.resolve()}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def inspect(file_path: str, verbose: bool):
    """Preview file structure without calling LLM."""
    _setup_logging(verbose)

    from bds.inspector.preview import inspect_file

    preview = inspect_file(Path(file_path))
    console.print(preview)


@main.group()
def cache():
    """Manage extraction code cache."""
    pass


@cache.command("list")
@click.option("--config", "config_path", type=click.Path(), default=None)
def cache_list(config_path: Optional[str]):
    """List cached extraction entries."""
    cfg_path = config_path or find_config()
    config = load_config(cfg_path)

    from bds.cache import MappingCache

    mc = MappingCache(config.cache)
    entries = mc.list_entries()

    if not entries:
        console.print("Cache is empty.")
        return

    table = Table(title="Cache Entries")
    table.add_column("Signature", style="dim")
    table.add_column("Source File", style="cyan")
    table.add_column("Created", style="green")

    for e in entries:
        table.add_row(e["signature"], e["source_file"], e["created_at"])

    console.print(table)


@cache.command("clear")
@click.option("--config", "config_path", type=click.Path(), default=None)
def cache_clear(config_path: Optional[str]):
    """Clear all cached entries."""
    cfg_path = config_path or find_config()
    config = load_config(cfg_path)

    from bds.cache import MappingCache

    mc = MappingCache(config.cache)
    count = mc.clear()
    console.print(f"Cleared {count} cache entries.")


if __name__ == "__main__":
    main()
