# Crypto History Pipeline

## Overview

This document describes the execution procedures for the crypto history pipeline, which supports two mutually exclusive modes of operation:

1. **Backfill mode** — downloads historical market data from Binance and persists detector state to a Supabase database.
2. **Dataset mode** — constructs a breakout-level dataset from locally stored data without performing any database writes.

Both modes share a common configuration and a single entry point (`main.py`), differing only in their data source and output destination.

---

## 1. Prerequisites

### 1.1 Environment

The pipeline requires Python 3.10 or later. Dependencies are managed via a virtual environment:

```bash
python -m venv trading_env
trading_env\Scripts\activate       # Windows
source trading_env/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

### 1.2 Environment Variables

Create a `.env` file in the project root containing the Supabase credentials:

```
SUPABASE_URL=<your-project-url>
SUPABASE_SECRET_KEY=<your-service-role-key>
```

These are loaded at runtime via `python-dotenv`.

### 1.3 Data Layout

The pipeline expects two distinct data roots, configured independently in `configs/config.py`:

| Constant | Purpose | Typical Value |
|---|---|---|
| `BASE_PATH` | Source for backfill (daily archives) | `upload/data/spot/daily/klines` |
| `DATASET_PATH` | Source for dataset construction (monthly archives) | `upload/data/spot/monthly/klines` |
| `OUTPUT_PATH` | Destination for generated parquet files | `output/` |

Each root follows the structure:

```
<root>/<TICKER>/1m/<YYYY-MM-DD>_<YYYY-MM-DD>/<TICKER>-1m-<YYYY-MM-DD>.zip
```

The `BTCUSDT` pair must be present in the relevant root, as it is used as a reference series for all other instruments.

---

## 2. Execution

The pipeline is invoked through `main.py` with a mandatory `--mode` argument.

### 2.1 Backfill Mode

```bash
python main.py --mode backfill
```

**Behaviour.** The pipeline downloads historical kline data for all USDT pairs from Binance, runs the breakout detector against each instrument, and persists:

- the processing log (first and last processed timestamps, detector parameters),
- the serialized detector state.

Both records are written to Supabase, enabling incremental resumption on subsequent runs. Downloaded archives are removed after successful processing, with the exception of `BTCUSDT`, which is retained.

**Data source.** `BASE_PATH`.

### 2.2 Dataset Mode

```bash
python main.py --mode dataset
```

**Behaviour.** The pipeline enumerates locally available ticker directories, executes the detector on each instrument starting from a clean state, and writes the resulting breakout records to parquet files. No database interaction occurs; the detector state stored in Supabase is neither read nor modified.

**Data source.** `DATASET_PATH`.

**Output.** A per-ticker file `<TICKER>_breakout_levels.parquet` and a combined file `_all_breakouts.parquet` under `OUTPUT_PATH/breakout_dataset/`.

An optional override is available:

```bash
python main.py --mode dataset --output-dir /custom/path
```

---

## 3. Design Notes

### 3.1 Separation of Concerns

The `HistoryBackfill` class exposes an internal method `_run_detection(ticker, load_state)` that encapsulates data loading, BTC alignment, and detector execution, but performs no persistence. Two public entry points build upon it:

- `_level_detection(ticker)` — invokes `_run_detection` with `load_state=True` and subsequently saves the processing log and detector state.
- `create_dataset(...)` — invokes `_run_detection` with `load_state=False` and serialises the results to parquet.

This structure ensures that dataset construction cannot inadvertently mutate the database.

### 3.2 Data Source Isolation

Two loader callables are injected into `HistoryBackfill` during construction:

- `loader_func` — bound to `BASE_PATH`, used by backfill mode;
- `dataset_loader_func` — bound to `DATASET_PATH`, used by dataset mode.

The binding is performed in `main.py` via `functools.partial`, preserving a single source of truth for each path.

### 3.3 Logging

Logging behaviour is controlled by the `LOG_MODE` constant at the top of `main.py`. Four presets are provided — `NORMAL`, `DOWNLOAD`, `DETECTOR`, and `STORAGE` — each defining per-module log levels. Output is duplicated to `app.log` and to the standard output stream.

---

## 4. Troubleshooting

| Symptom | Probable Cause | Remedy |
|---|---|---|
| `Base path does not exist` | `BASE_PATH` or `DATASET_PATH` points to a non-existent directory | Verify the path in `configs/config.py` |
| `No zip files found for <TICKER>` | The relevant data root is empty for that ticker | Ensure archives have been downloaded into the correct root |
| `AttributeError: 'HistoryBackfill' object has no attribute 'create_dataset'` | The method is absent from the installed source | Update `src/services/HistoryBackfill.py` |
| `NameError: name 'Iterable' is not defined` | Missing typing import | Add `from typing import Iterable` to `src/utils/utils.py` |

---

## 5. Summary

| Mode | Command | Reads From | Writes To | Database Access |
|---|---|---|---|---|
| Backfill | `python main.py --mode backfill` | `BASE_PATH` | Supabase | Read / Write |
| Dataset | `python main.py --mode dataset` | `DATASET_PATH` | `OUTPUT_PATH/breakout_dataset/` | None |

The two modes are independent and may be executed in either order. Neither mode requires the other to have been run previously, provided that the corresponding data root is populated.