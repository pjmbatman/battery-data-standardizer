"""Tool definitions for the Tool-Use agent mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# OpenAI function-calling tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "View the structure of a file or a nested key within it. Returns key names, types, shapes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to inspect"},
                    "key": {"type": "string", "description": "Dot-separated key path for nested access (e.g., 'cycle.0.voltage')"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_sample",
            "description": "Read sample data (first 5 rows/elements) at a specific key path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "key": {"type": "string", "description": "Dot-separated key path"},
                },
                "required": ["path", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract",
            "description": "Extract full data at a key path as a list. Use max_rows to limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "key": {"type": "string", "description": "Dot-separated key path"},
                    "max_rows": {"type": "integer", "description": "Max rows to extract (default 100000)", "default": 100000},
                },
                "required": ["path", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run a short Python code snippet. The code should print its result to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile",
            "description": "Get statistics (min, max, mean, dtype, null_count, length) for data at a key path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "key": {"type": "string", "description": "Dot-separated key path"},
                },
                "required": ["path", "key"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute tool calls from the LLM agent."""

    def __init__(self, sandbox_executor=None):
        self._sandbox = sandbox_executor
        self._file_cache: Dict[str, Any] = {}

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call and return the result as a string."""
        handlers = {
            "inspect": self._inspect,
            "read_sample": self._read_sample,
            "extract": self._extract,
            "execute_code": self._execute_code,
            "profile": self._profile,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error in {tool_name}: {type(e).__name__}: {e}"

    # -- inspect --
    def _inspect(self, path: str, key: str = None) -> str:
        data = self._load_file(path)
        if key:
            data = self._navigate(data, key)
        return self._describe(data)

    # -- read_sample --
    def _read_sample(self, path: str, key: str) -> str:
        data = self._load_file(path)
        data = self._navigate(data, key)
        return self._sample(data, n=5)

    # -- extract --
    def _extract(self, path: str, key: str, max_rows: int = 100000) -> str:
        data = self._load_file(path)
        data = self._navigate(data, key)
        return self._to_list_str(data, max_rows)

    # -- execute_code --
    def _execute_code(self, code: str) -> str:
        if self._sandbox is None:
            return "Sandbox not available"
        result = self._sandbox.execute(code, timeout=30)
        if result.success:
            return json.dumps(result.output, default=str)[:10000]
        return f"Error: {result.error}\nPartial: {result.partial_output or ''}"

    # -- profile --
    def _profile(self, path: str, key: str) -> str:
        data = self._load_file(path)
        data = self._navigate(data, key)
        return self._compute_profile(data)

    # -------------------------------------------------------------------
    # File loading (cached)
    # -------------------------------------------------------------------

    def _load_file(self, path: str) -> Any:
        if path in self._file_cache:
            return self._file_cache[path]

        p = Path(path)
        ext = p.suffix.lower()

        if ext in (".csv", ".tsv", ".txt"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            data = pd.read_csv(p, sep=sep)
        elif ext in (".xlsx", ".xls"):
            import pandas as pd
            data = pd.read_excel(p, sheet_name=None)  # dict of DataFrames
        elif ext == ".mat":
            data = self._load_mat(p)
        elif ext == ".json":
            data = json.loads(p.read_text())
        elif ext in (".pkl", ".pickle"):
            import pickle
            with open(p, "rb") as f:
                data = pickle.load(f)
        elif ext in (".h5", ".hdf5"):
            data = self._load_hdf5(p)
        else:
            data = p.read_text(errors="replace")

        self._file_cache[path] = data
        return data

    @staticmethod
    def _load_mat(path: Path) -> Any:
        try:
            import h5py
            f = h5py.File(path, "r")
            return f  # keep open for navigation
        except Exception:
            import scipy.io
            return scipy.io.loadmat(path, squeeze_me=False)

    @staticmethod
    def _load_hdf5(path: Path) -> Any:
        import h5py
        return h5py.File(path, "r")

    # -------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------

    def _navigate(self, data: Any, key: str) -> Any:
        """Navigate nested data using dot-separated key path."""
        parts = key.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, (list, tuple)):
                current = current[int(part)]
            elif isinstance(current, np.ndarray):
                if current.dtype.names and part in current.dtype.names:
                    current = current[part]
                else:
                    current = current[int(part)]
            elif hasattr(current, "__getitem__"):
                # h5py Group/Dataset, pandas DataFrame
                try:
                    current = current[part]
                except (KeyError, TypeError):
                    current = current[int(part)]
            else:
                raise KeyError(f"Cannot navigate to '{part}' in {type(current).__name__}")
        return current

    # -------------------------------------------------------------------
    # Description / sampling
    # -------------------------------------------------------------------

    def _describe(self, data: Any, max_keys: int = 50) -> str:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return (
                f"DataFrame: shape={data.shape}, columns={list(data.columns)}\n"
                f"dtypes:\n{data.dtypes.to_string()}"
            )
        if isinstance(data, dict):
            keys = list(data.keys())[:max_keys]
            lines = [f"dict ({len(data)} keys):"]
            for k in keys:
                v = data[k]
                lines.append(f"  {k}: {self._type_info(v)}")
            return "\n".join(lines)
        if isinstance(data, np.ndarray):
            info = f"ndarray: shape={data.shape}, dtype={data.dtype}"
            if data.dtype.names:
                info += f", fields={list(data.dtype.names)}"
            return info
        if isinstance(data, (list, tuple)):
            return f"{type(data).__name__}: len={len(data)}, sample_type={type(data[0]).__name__ if data else 'empty'}"

        # h5py
        try:
            import h5py
            if isinstance(data, h5py.Group):
                keys = list(data.keys())[:max_keys]
                lines = [f"HDF5 Group ({len(data)} items):"]
                for k in keys:
                    item = data[k]
                    if isinstance(item, h5py.Dataset):
                        lines.append(f"  {k}: Dataset shape={item.shape}, dtype={item.dtype}")
                    elif isinstance(item, h5py.Group):
                        lines.append(f"  {k}: Group ({len(item)} items)")
                return "\n".join(lines)
            if isinstance(data, h5py.Dataset):
                return f"HDF5 Dataset: shape={data.shape}, dtype={data.dtype}"
        except ImportError:
            pass

        return f"{type(data).__name__}: {str(data)[:500]}"

    def _sample(self, data: Any, n: int = 5) -> str:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.head(n).to_string()
        if isinstance(data, pd.Series):
            return data.head(n).to_string()
        if isinstance(data, np.ndarray):
            flat = data.flatten()[:n]
            return f"shape={data.shape}, first {n}: {flat.tolist()}"
        if isinstance(data, (list, tuple)):
            return json.dumps(data[:n], default=str)
        if isinstance(data, dict):
            items = list(data.items())[:n]
            return json.dumps(dict(items), default=str, indent=2)

        try:
            import h5py
            if isinstance(data, h5py.Dataset):
                arr = data[:n] if len(data.shape) == 1 else data[:n]
                return f"shape={data.shape}, first {n}: {np.array(arr).tolist()}"
        except ImportError:
            pass

        return str(data)[:1000]

    def _to_list_str(self, data: Any, max_rows: int) -> str:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.head(max_rows).to_json(orient="records")
        if isinstance(data, pd.Series):
            return json.dumps(data.head(max_rows).tolist(), default=str)
        if isinstance(data, np.ndarray):
            flat = data.flatten()[:max_rows]
            return json.dumps(flat.tolist(), default=str)
        if isinstance(data, (list, tuple)):
            return json.dumps(data[:max_rows], default=str)

        try:
            import h5py
            if isinstance(data, h5py.Dataset):
                arr = data[:]
                flat = np.array(arr).flatten()[:max_rows]
                return json.dumps(flat.tolist(), default=str)
        except ImportError:
            pass

        return str(data)[:10000]

    def _compute_profile(self, data: Any) -> str:
        import pandas as pd

        arr = None
        if isinstance(data, pd.Series):
            arr = data.dropna().values
        elif isinstance(data, pd.DataFrame):
            return data.describe().to_string()
        elif isinstance(data, np.ndarray):
            arr = data.flatten()
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=float)

        try:
            import h5py
            if isinstance(data, h5py.Dataset):
                arr = np.array(data[:]).flatten()
        except ImportError:
            pass

        if arr is not None and len(arr) > 0:
            try:
                numeric = arr.astype(float)
                return json.dumps({
                    "dtype": str(arr.dtype),
                    "length": len(arr),
                    "min": float(np.nanmin(numeric)),
                    "max": float(np.nanmax(numeric)),
                    "mean": float(np.nanmean(numeric)),
                    "null_count": int(np.isnan(numeric).sum()),
                })
            except (ValueError, TypeError):
                return json.dumps({
                    "dtype": str(arr.dtype),
                    "length": len(arr),
                    "note": "non-numeric data",
                })

        return f"Cannot profile: {type(data).__name__}"

    @staticmethod
    def _type_info(obj: Any) -> str:
        if isinstance(obj, np.ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        if isinstance(obj, dict):
            return f"dict({len(obj)} keys)"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}(len={len(obj)})"
        return type(obj).__name__
