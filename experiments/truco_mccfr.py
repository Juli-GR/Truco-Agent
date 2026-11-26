"""Stage 2b: tabular MCCFR (external sampling) on one hand of truco, full deck by default.

Runs `truco.algorithms.mccfr` on the numba simulator, with the information abstraction by default
(`--key abstract`; `--key exact` only fits small decks). Iterations run in blocks of `--block`;
after each block the script prints a progress line and logs to MLflow (experiment "truco-mccfr"):
- block_seconds and iterations_per_second (time per block, as a chart);
- entries and load of the table (information sets seen, fraction of the slots used) and rescales;
- rss_gb and peak_rss_gb: the RAM the process really uses, now and at its peak.

Checkpoints: every `--checkpoint-hours` (and at the end, or if the run stops on an error or
Ctrl+C) the tables are written to `<run dir>/checkpoint_<iteration>/`, ignored by git. They are big
(about 32 GB for the full deck), so only the last one is kept: the new one is written to a temporary
folder, renamed when complete, and only then the previous one is deleted. A crash while writing
leaves the previous checkpoint intact; the disk needs room for two checkpoints for a moment.

Resuming: `--resume <run dir>` loads the last checkpoint and continues the same MLflow run, with
the configuration saved in the run dir (the command line only sets how long to keep going and how
many threads). Every block is seeded from `--seed` and its index; with `--threads 1` a resumed run
repeats exactly what was lost, with more threads it is statistically the same but not identical.

Threads: `--threads` iterations run in parallel on the same tables (see mccfr.train). The speed-up
depends on the machine (memory, core types): try a short run with a few values first; the
iterations_per_second metric shows it.

Usage, from the repo root:
    python experiments/truco_mccfr.py --hours 48
    python experiments/truco_mccfr.py --niveles 5 --capacity-log2 21 --iterations 10000000
    python experiments/truco_mccfr.py --resume results/mccfr/<run dir> --hours 24
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import psutil

from truco.algorithms import mccfr
from truco.games import truco_numba as tn
from truco.games.cards import NUM_TRUCO_LEVELS
from truco.tracking import REPO_ROOT, setup_experiment

CONFIG_FILE = "config.json"
CHECKPOINT_PREFIX = "checkpoint_"
TMP_CHECKPOINT = "checkpoint.tmp"
STATE_FILE = "progress.json"  # inside each checkpoint: iteration and block reached
GB = 1e9


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / GB


def peak_rss_gb() -> float:
    """Highest RAM the process has used, as the operating system counts it."""
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / GB
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB  # Linux: KB


def block_seed(seed: int, block: int) -> int:
    return (seed * 1_000_003 + block) % (2**32)


def last_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob(f"{CHECKPOINT_PREFIX}*"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(tables: mccfr.Tables, run_dir: Path, iteration: int, block: int) -> float:
    """Writes the checkpoint safely and deletes the previous one; returns the seconds it took."""
    start = time.perf_counter()
    tmp = run_dir / TMP_CHECKPOINT
    if tmp.exists():
        shutil.rmtree(tmp)
    tables.save(tmp)
    (tmp / STATE_FILE).write_text(json.dumps({"iteration": iteration, "block": block}))
    previous = last_checkpoint(run_dir)
    tmp.rename(run_dir / f"{CHECKPOINT_PREFIX}{iteration:015d}")
    if previous is not None:
        shutil.rmtree(previous)
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--niveles", type=int, default=NUM_TRUCO_LEVELS)
    parser.add_argument("--key", choices=mccfr.KEY_MODES, default="abstract")
    parser.add_argument(
        "--capacity-log2",
        type=int,
        default=30,
        help="table slots = 2**this; 30 (~32 GB) fits the full deck with the abstraction",
    )
    parser.add_argument("--block", type=int, default=1_000_000, help="iterations per block")
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="parallel threads; 1 makes the run exactly reproducible (see mccfr.train)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, help="stop after this many iterations in total")
    parser.add_argument("--hours", type=float, help="stop after this many hours (this session)")
    parser.add_argument("--checkpoint-hours", type=float, default=2.0)
    parser.add_argument("--resume", type=Path, help="run dir of a previous run to continue")
    args = parser.parse_args()
    if args.iterations is None and args.hours is None:
        parser.error("give --iterations and/or --hours")
    return args


def check_resources(table_bytes: int, run_dir: Path) -> None:
    available = psutil.virtual_memory().available
    if table_bytes > available:
        raise SystemExit(
            f"The tables need {table_bytes / GB:.1f} GB of RAM and only "
            f"{available / GB:.1f} GB are available: lower --capacity-log2."
        )
    free = shutil.disk_usage(run_dir).free
    if free < 2 * table_bytes:
        print(
            f"WARNING: {free / GB:.1f} GB free on disk; writing a checkpoint needs up to "
            f"{2 * table_bytes / GB:.1f} GB while the previous one still exists.",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    if args.resume:
        run_dir = args.resume
        config = json.loads((run_dir / CONFIG_FILE).read_text())
        checkpoint = last_checkpoint(run_dir)
        if checkpoint is None:
            raise SystemExit(f"No checkpoint in {run_dir}")
        print(f"Resuming from {checkpoint}", flush=True)
        tables = mccfr.Tables.load(checkpoint)
        progress = json.loads((checkpoint / STATE_FILE).read_text())
        iteration, block = progress["iteration"], progress["block"]
    else:
        config = {
            "niveles": args.niveles,
            "key": args.key,
            "capacity_log2": args.capacity_log2,
            "block": args.block,
            "seed": args.seed,
        }
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = REPO_ROOT / "results" / "mccfr" / f"{stamp}_niveles{args.niveles}_{args.key}"
        run_dir.mkdir(parents=True)
        table_bytes = (1 << args.capacity_log2) * mccfr.Tables.bytes_per_slot()
        check_resources(table_bytes, run_dir)
        tables = mccfr.Tables.create(args.capacity_log2)
        iteration = block = 0

    deck = tn.make_deck(config["niveles"])
    key_mode = mccfr.KEY_MODES[config["key"]]

    setup_experiment("truco-mccfr")
    run_id = config.get("mlflow_run_id")
    with mlflow.start_run(run_id=run_id, run_name=run_dir.name):
        if run_id is None:
            config["mlflow_run_id"] = mlflow.active_run().info.run_id
            (run_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))
            mlflow.log_params({**config, "table_gb": round(tables.nbytes() / GB, 2)})
            mlflow.set_tag("run_dir", str(run_dir))
        print(
            f"{run_dir.name}: table {tables.nbytes() / GB:.1f} GB, "
            f"{tables.num_entries:,} entries, iteration {iteration:,}",
            flush=True,
        )

        session_start = last_checkpoint_time = time.time()
        checkpointed_at = iteration if args.resume else None
        stop_reason = "done"
        try:
            while True:
                if args.iterations is not None and iteration >= args.iterations:
                    break
                if args.hours is not None and time.time() - session_start >= args.hours * 3600:
                    stop_reason = "time limit"
                    break
                size = config["block"]
                if args.iterations is not None:
                    size = min(size, args.iterations - iteration)

                start = time.perf_counter()
                seed = block_seed(config["seed"], block)
                ran = mccfr.train(tables, deck, iteration, size, seed, key_mode, args.threads)
                seconds = time.perf_counter() - start
                iteration += ran
                block += 1

                speed = ran / seconds
                metrics = {
                    "block_seconds": seconds,
                    "iterations_per_second": speed,
                    "threads": args.threads,
                    "entries": tables.num_entries,
                    "load": tables.load_factor,
                    "rescales": int(tables.stats[mccfr.RESCALES]),
                    "rss_gb": rss_gb(),
                    "peak_rss_gb": peak_rss_gb(),
                }
                mlflow.log_metrics(metrics, step=iteration)
                eta = ""
                if args.iterations is not None:
                    eta = f"  ETA {(args.iterations - iteration) / speed / 3600:.1f} h"
                print(
                    f"iter {iteration:>14,}  block {seconds:6.1f} s  {speed:>9,.0f} it/s  "
                    f"entries {tables.num_entries:>13,} (load {tables.load_factor:.2f})  "
                    f"RAM {metrics['rss_gb']:.1f} GB (peak {metrics['peak_rss_gb']:.1f}){eta}",
                    flush=True,
                )

                if tables.is_full:
                    stop_reason = "table full"
                    break
                if time.time() - last_checkpoint_time >= args.checkpoint_hours * 3600:
                    seconds = save_checkpoint(tables, run_dir, iteration, block)
                    mlflow.log_metric("checkpoint_seconds", seconds, step=iteration)
                    print(f"checkpoint written in {seconds:.0f} s", flush=True)
                    last_checkpoint_time = time.time()
                    checkpointed_at = iteration
        except BaseException:  # error or Ctrl+C: keep what was learned (blocks are atomic)
            stop_reason = "error"
            raise
        finally:
            if iteration != checkpointed_at:
                seconds = save_checkpoint(tables, run_dir, iteration, block)
                mlflow.log_metric("checkpoint_seconds", seconds, step=iteration)
            mlflow.set_tag("stop_reason", stop_reason)
            print(f"{stop_reason}: checkpoint at iteration {iteration:,} in {run_dir}", flush=True)

        if stop_reason == "table full":
            raise SystemExit(
                "The table is full: some updates were skipped. Resume is not useful; "
                "start again with a larger --capacity-log2."
            )


if __name__ == "__main__":
    main()
