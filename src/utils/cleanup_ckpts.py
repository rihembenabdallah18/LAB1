"""Prune checkpoints under outputs/checkpoints/{run_name}/.

Usage:
    python -m src.utils.cleanup_ckpts --run-name student_direct_ft_large --keep best
    python -m src.utils.cleanup_ckpts --all --keep none
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ckpt_root() -> Path:
    import yaml
    cfg_path = REPO_ROOT / "config" / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    return REPO_ROOT / cfg["paths"]["ckpt_root"]


def _best_ckpt_path(run_dir: Path) -> Path | None:
    """Return the best checkpoint recorded in trainer_state.json, or None."""
    state_file = run_dir / "trainer_state.json"
    if state_file.exists():
        with state_file.open() as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best:
            p = Path(best)
            if p.exists():
                return p
            # trainer_state paths are sometimes absolute; try relative to run_dir
            rel = run_dir / p.name
            if rel.exists():
                return rel
    return None


def _last_ckpt_path(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def prune_run(run_dir: Path, keep: str, dry_run: bool = False) -> None:
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        print(f"  [{run_dir.name}] no checkpoints found, skipping")
        return

    if keep == "none":
        keep_path = None
    elif keep == "best":
        keep_path = _best_ckpt_path(run_dir) or _last_ckpt_path(run_dir)
    elif keep == "last":
        keep_path = _last_ckpt_path(run_dir)
    else:
        raise ValueError(f"Unknown --keep value: {keep!r}")

    for ckpt in ckpts:
        if keep_path and ckpt.resolve() == keep_path.resolve():
            size_gb = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()) / 1e9
            print(f"  [{run_dir.name}] keeping {ckpt.name} ({size_gb:.1f} GB)")
        else:
            size_gb = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()) / 1e9
            print(f"  [{run_dir.name}] removing {ckpt.name} ({size_gb:.1f} GB)")
            if not dry_run:
                shutil.rmtree(ckpt)


def main() -> None:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-name", help="Prune a single run by name")
    group.add_argument("--all", action="store_true", help="Prune every run under ckpt_root")
    p.add_argument("--keep", choices=["best", "last", "none"], default="best",
                   help="Which checkpoint to keep (default: best)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without deleting")
    args = p.parse_args()

    ckpt_root = _ckpt_root()

    if args.all:
        run_dirs = [d for d in ckpt_root.iterdir() if d.is_dir()]
        if not run_dirs:
            print(f"No runs found under {ckpt_root}")
            return
        for run_dir in sorted(run_dirs):
            prune_run(run_dir, args.keep, dry_run=args.dry_run)
    else:
        run_dir = ckpt_root / args.run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        prune_run(run_dir, args.keep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
