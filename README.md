# Battery Data Standardizer (BDS)

An AI-powered tool for automatically standardizing heterogeneous battery data files.

An LLM (EXAONE) autonomously analyzes any file format, understands field semantics, maps them to a target schema, and generates extraction code on the fly — no hardcoded parsers required. Add a new dataset without writing a single line of code.

## Architecture

```
Input (any file format)
    │
    ▼
[File Inspector] ─── Converts file structure into a text preview
    │
    ▼
[LLM Agent] ─── EXAONE analyzes the structure and decides how to extract
    │   ├─ Primary: Code Generation & Execution
    │   └─ Fallback: Iterative Tool Use (inspect, read, extract)
    │
    ▼
[Sandbox Executor] ─── Safely executes generated code in subprocess
    │
    ▼
[Validator & Exporter] ─── Validates output → CellRecord pickle
```

## Supported Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| CSV/TSV | `.csv`, `.tsv`, `.txt` | Auto delimiter detection |
| Excel | `.xlsx`, `.xls` | Multi-sheet support |
| MATLAB v5 | `.mat` | scipy.io.loadmat |
| MATLAB v7.3 | `.mat` | HDF5 via h5py |
| HDF5 | `.h5`, `.hdf5` | Hierarchical structure traversal |
| JSON | `.json` | Nested structure support |
| Pickle | `.pkl`, `.pickle` | Python objects |
| Archives | `.zip`, `.tar.gz`, etc. | Auto-extraction before processing |

## Output

Outputs [BatteryFoundationFramework](https://github.com/pjmbatman/BatteryFoundationFramework)-compatible `CellRecord` pickle files.

```python
from pipeline.standardizer.cell_record import CellRecord
cell = CellRecord.load("output/B0025.pkl")
```

## Quick Start

### 1. Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install dependencies
git clone https://github.com/pjmbatman/battery-data-standardizer.git
cd battery-data-standardizer
uv venv
uv pip install -e ".[dev]"
```

### 2. Start the LLM Server

Serve the EXAONE model via vLLM. Run this on a GPU machine.

```bash
# Default model (EXAONE-4.0-32B-FP8, ~34GB VRAM)
bash scripts/serve_model.sh

# Or specify a different model
MODEL=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct bash scripts/serve_model.sh

# Verify the server is running
curl http://localhost:8000/v1/models
```

> The model will be downloaded automatically on first run. This may take some time.

### 3. Standardize Data

```bash
# Standardize a single file
uv run bds standardize ./data/sample.csv -o ./output/

# Batch mode (entire directory, no interactive prompts)
uv run bds standardize ./raw_data/ -o ./standardized/ --batch

# Use a remote vLLM server
uv run bds standardize ./data/ -o ./output/ --api-base http://gpu-server:8000/v1
```

### 4. Inspect File Structure (no LLM calls)

```bash
uv run bds inspect ./data/B0025.mat
```

Example output:
```
[MAT v5, 2,753,821 bytes]
B0025: structured array, shape=(1, 1), fields=['cycle']
  .cycle: shape=(1, 80), dtype=[('type', 'O'), ('ambient_temperature', 'O'), ...]

## B0025.cycle: 80 records
Types: {'impedance': 21, 'charge': 31, 'discharge': 28}
...
```

### 5. Cache Management

Files with the same structure reuse cached extraction code without calling the LLM.

```bash
# List cached entries
uv run bds cache list

# Clear all cache
uv run bds cache clear
```

## Configuration

Edit `configs/default.yaml`:

```yaml
llm:
  api_base: "http://localhost:8000/v1"
  model: "LGAI-EXAONE/EXAONE-4.0-32B-FP8"
  temperature: 0.1
  max_retries: 3

sandbox:
  timeout: 120

agent:
  max_tool_steps: 20
  fallback_to_tool_use: true

cache:
  enabled: true
  db_path: ".bds_cache/cache.db"
```

## How It Works

1. **File Inspector**: Converts any file into a text preview (structure, field names, sample data)
2. **Code Generation Agent**: LLM reads the preview and generates a complete Python extraction script
3. **Sandbox Execution**: The generated script runs in a subprocess; JSON output is parsed from stdout
4. **Validation**: Checks voltage range (0–6V), time monotonicity, array length consistency, etc.
5. **Auto-retry**: On execution errors or validation failures, the error is fed back to the LLM for code correction (up to 3 retries)
6. **Tool Use Agent (fallback)**: If code generation fails, the LLM iteratively uses tools (inspect, read_sample, extract, execute_code, profile) to explore and extract data step by step
7. **Caching**: Successful extraction code is cached by file structure signature (e.g., header hash) and reused for files with the same structure

## Testing

```bash
uv run python -m pytest tests/
```

## Project Structure

```
battery-data-standardizer/
├── configs/default.yaml
├── scripts/
│   ├── serve_model.sh          # vLLM serving script
│   └── download_model.py       # Model downloader
├── src/bds/
│   ├── cli.py                  # CLI (standardize, inspect, cache)
│   ├── config.py               # Config loader
│   ├── pipeline.py             # End-to-end pipeline
│   ├── inspector/
│   │   ├── preview.py          # File structure preview
│   │   └── archive.py          # Archive extraction
│   ├── agent/
│   │   ├── orchestrator.py     # Agent orchestrator
│   │   ├── code_generator.py   # Code generation agent
│   │   ├── tool_use.py         # Tool use agent (fallback)
│   │   ├── tools.py            # Tool definitions
│   │   ├── prompts.py          # Prompt templates
│   │   └── llm_client.py       # vLLM OpenAI-compatible client
│   ├── sandbox/
│   │   └── executor.py         # Sandboxed code execution
│   ├── schema.py               # CellRecord / CycleRecord schema
│   ├── validator.py            # Output data validation
│   ├── exporter.py             # Pickle exporter
│   └── cache.py                # SQLite cache
└── tests/                      # Unit tests (38 tests)
```

## Validation Results

| Dataset | Format | Cycles | Attempt |
|---------|--------|--------|---------|
| SNL | CSV | 388 | 1st |
| CALCE | XLSX | 7 | 1st |
| UL-PUR | CSV | 205 | 1st |
| HNEI | CSV (45MB) | 1,101 | Cached |
| NASA | MAT v5 | 59 | 2nd |

## Requirements

- Python >= 3.10
- GPU server for vLLM (~34GB VRAM for EXAONE-4.0-32B-FP8)
- [uv](https://github.com/astral-sh/uv) for package management
