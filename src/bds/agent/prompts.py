"""Prompt templates for the LLM agent."""

from __future__ import annotations

TARGET_SCHEMA_DESCRIPTION = """\
Each cell produces a JSON dict with:
{
    "cell_id": str,           // unique identifier for the cell
    "cycles": [
        {
            "cycle_number": int,
            "voltage_v": [float, ...],             // Volts
            "current_a": [float, ...],             // Amperes (positive=charge, negative=discharge)
            "temperature_c": [float, ...],         // Celsius (optional, include if available)
            "time_s": [float, ...],                // seconds from start of cycle
            "charge_capacity_ah": [float, ...],    // Ah (optional, compute if missing)
            "discharge_capacity_ah": [float, ...], // Ah (optional, compute if missing)
        },
        ...
    ],
    // Optional metadata (include if discoverable):
    "nominal_capacity_ah": float,   // nominal capacity in Ah
    "max_voltage_limit_v": float,
    "min_voltage_limit_v": float,
    "form_factor": str,             // e.g., "18650", "pouch"
    "cathode_material": str,        // e.g., "NMC", "LFP"
    "anode_material": str,          // e.g., "graphite"
}"""


CODE_GENERATION_SYSTEM = """\
You are a battery data extraction expert. Given a file structure preview, \
write a complete Python script that reads the file and outputs battery cycling \
data in the target schema as JSON to stdout.

## Target Schema
{schema}

## Critical Instructions
1. Write a COMPLETE, self-contained Python script.
2. The script MUST print a single JSON object to stdout (use json.dumps with default=str for safety).
3. Handle unit conversions based on field names and value ranges:
   - mV → V (divide by 1000)
   - mA → A (divide by 1000)
   - mAh → Ah (divide by 1000)
   - kΩ → Ω (multiply by 1000)
   - minutes → seconds (multiply by 60)
   - hours → seconds (multiply by 3600)
4. If capacity columns are missing, calculate from current integration: Q = ∫|I|·dt / 3600
5. Detect cycle boundaries from:
   - Explicit cycle index/number column (preferred)
   - Current sign transitions (charge→discharge or vice versa)
6. Handle multilingual field names (e.g., German: Spannung=voltage, Strom=current, Temperatur=temperature)
7. Available libraries: pandas, numpy, scipy, h5py, openpyxl, json, pickle, pathlib, sys
8. IMPORTANT: Use numpy.trapezoid() (NOT scipy.integrate.trapz or cumtrapz — they are removed in recent scipy)
9. IMPORTANT: cell_id MUST be set. Derive from filename if not in data (e.g., path stem).
10. IMPORTANT: The output MUST have "cell_id" (str) and "cycles" (list of dicts) at the top level.
11. Group rows by Cycle_Index or similar column. Each group = one cycle dict in the output.
12. For .mat files: If the preview says "MAT v5", use scipy.io.loadmat(path, squeeze_me=True). Do NOT use h5py for MAT v5 files.
    If the preview says "MAT v7.3 (HDF5)", use h5py.
13. For nested structured arrays (e.g., NASA battery data):
    - numpy.void objects do NOT have .get(). Use bracket indexing: record['field_name']
    - To check field existence: 'field_name' in record.dtype.names
    - Use [()] to unwrap 0-d arrays: data['key']['cycle'][()] → array of records
    - Access nested data: cycle['data'][()]['Voltage_measured'].tolist()

Write ONLY the Python code. No explanations, no markdown fences."""


CODE_GENERATION_USER = """\
## File Info
Path: {file_path}

## File Structure Preview
{file_preview}

IMPORTANT: Hardcode the file path directly in the script as:
FILE_PATH = "{file_path}"

Write the extraction script now."""


CODE_FIX_USER = """\
The previous extraction code failed. Fix it.

## Error
{error}

## Stdout (partial)
{partial_output}

## Original Code
```python
{original_code}
```

## File Info
Path: {file_path}

## File Structure Preview
{file_preview}

Write the FIXED Python code. Only code, no explanation."""


VALIDATION_FIX_USER = """\
The extraction code ran but produced data that failed validation.

## Validation Issues
{issues}

## Output Data Summary
Number of cycles: {num_cycles}
Sample cycle keys: {sample_keys}
{sample_data}

## Original Code
```python
{original_code}
```

## File Info
Path: {file_path}

Fix the code to address these validation issues. Write only Python code."""


# ---------------------------------------------------------------------------
# Tool-use mode prompts
# ---------------------------------------------------------------------------

TOOL_USE_SYSTEM = """\
You are a battery data extraction agent. Your task is to extract battery cycling \
data from a file and produce a JSON object matching the target schema.

## Target Schema
{schema}

## Available Tools
- inspect(path, key?): View the structure of a file or a nested key within it
- read_sample(path, key): Read sample data (first 5 rows/elements) at a key path
- extract(path, key, max_rows?): Extract full data at a key path as a list
- execute_code(code): Run a short Python snippet and get its output
- profile(path, key): Get statistics (min, max, mean, dtype, null_count) for data at a key

## Instructions
1. Start by inspecting the file structure to understand what data is available.
2. Use read_sample to look at actual values and understand field meanings.
3. Use profile to check value ranges and determine if unit conversion is needed.
4. Use extract or execute_code to pull out the actual data.
5. When you have enough information, return the final JSON result as a message (not a tool call).

Handle unit conversions, cycle boundary detection, and capacity calculation as needed.
When done, respond with ONLY the final JSON object (no markdown, no explanation)."""


TOOL_USE_USER = """\
## File to Process
Path: {file_path}

## Initial Structure Preview
{file_preview}

Begin extracting the battery cycling data."""
