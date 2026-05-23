"""Stage 1: download GSM8K + SVAMP and Ho et al. teacher CoTs.

GSM8K:  HuggingFace `gsm8k`/`main` config.
SVAMP:  HuggingFace `ChilleD/SVAMP` (700 train / 300 test, 1 000 total).
Teacher CoTs: itsnamgyu/reasoning-teacher release. The shared Dropbox/Drive folder
contains `teacher_completion_data.tar.gz`, which holds Zero-shot-CoT outputs from
text-davinci-002 across many datasets (including SVAMP). We extract both files.

Resumable: each step skips work if its output exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
GSM8K_DIR = RAW_DIR / "gsm8k"
SVAMP_DIR = RAW_DIR / "svamp"
HO_DIR = RAW_DIR / "ho_et_al_cots"

DROPBOX_ZIP_URL = (
    "https://www.dropbox.com/sh/hwcncpyomx87h20/"
    "AACqgVdd-ZzBQ3ncJcKqw0cVa?dl=1"
)
# Within the unzipped folder, the per-archive path. Confirmed by inspecting the
# Dropbox listing in the project README.
TEACHER_TARBALL_NAME = "teacher_completion_data.tar.gz"
# Within `teacher_completion_data.tar.gz`, the GSM8K file lives at this path.
# The schema is documented in the reasoning-teacher README under
# data.completion.CompletionDataset.
GSM8K_COMPLETION_PATH_IN_TAR = (
    "completion_data/B_text-davinci-002__C_zs_cot/D_gsm8k.json"
)


def _save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def download_gsm8k() -> None:
    train_path = GSM8K_DIR / "train.jsonl"
    test_path = GSM8K_DIR / "test.jsonl"
    if train_path.exists() and test_path.exists():
        print(f"[gsm8k] already downloaded at {GSM8K_DIR}")
        return
    print("[gsm8k] downloading via datasets.load_dataset('gsm8k', 'main')")
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")
    GSM8K_DIR.mkdir(parents=True, exist_ok=True)
    _save_jsonl(ds["train"], train_path)
    _save_jsonl(ds["test"], test_path)
    print(f"[gsm8k] saved {len(ds['train'])} train, {len(ds['test'])} test -> {GSM8K_DIR}")


def _curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -L follow redirects, -f fail on http errors, --retry for transient failures
    cmd = ["curl", "-fSL", "--retry", "3", "-o", str(dest), url]
    print("[curl]", " ".join(cmd))
    subprocess.check_call(cmd)


def download_svamp() -> None:
    train_path = SVAMP_DIR / "train.jsonl"
    test_path = SVAMP_DIR / "test.jsonl"
    if train_path.exists() and test_path.exists():
        print(f"[svamp] already downloaded at {SVAMP_DIR}")
        return
    print("[svamp] downloading via datasets.load_dataset('ChilleD/SVAMP')")
    from datasets import load_dataset

    ds = load_dataset("ChilleD/SVAMP")
    SVAMP_DIR.mkdir(parents=True, exist_ok=True)
    # Normalise field names to match GSM8K convention used by filter.py:
    #   question  -> Body + " " + Question (the full problem text)
    #   answer    -> Answer (bare number string)
    #   equation  -> Equation (kept as metadata)
    #   type      -> Type (kept as metadata)
    def _normalise(row: dict) -> dict:
        return {
            "question": (row.get("Body", "") + " " + row.get("Question", "")).strip(),
            "answer": str(row["Answer"]),
            "equation": row.get("Equation", ""),
            "type": row.get("Type", ""),
            "id": row.get("ID", ""),
        }

    train_rows = [_normalise(r) for r in ds["train"]]
    test_rows  = [_normalise(r) for r in ds["test"]]
    _save_jsonl(train_rows, train_path)
    _save_jsonl(test_rows, test_path)
    print(f"[svamp] saved {len(train_rows)} train, {len(test_rows)} test -> {SVAMP_DIR}")


def _extract_from_tarball(tar_path: Path, dataset_key: str, out_json: Path) -> None:
    """Extract the zs_cot / text-davinci-002 JSON for `dataset_key` from the tarball."""
    print(f"[ho] opening tarball {tar_path} for dataset '{dataset_key}'")
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getnames()
        member = next(
            (m for m in members
             if f"D_{dataset_key}.json" in m and "zs_cot" in m and "text-davinci-002" in m),
            None,
        )
        if member is None:
            # Fallback: accept any member whose basename matches D_<key>.json
            member = next(
                (m for m in members if m.endswith(f"D_{dataset_key}.json")),
                None,
            )
        if member is None:
            available = [m for m in members if m.endswith(".json")]
            raise RuntimeError(
                f"No file matching D_{dataset_key}.json found in tarball.\n"
                f"Available JSON members: {available[:30]}"
            )
        print(f"[ho] extracting {member}")
        ti = tf.getmember(member)
        fh = tf.extractfile(ti)
        assert fh is not None
        out_json.write_bytes(fh.read())
    print(f"[ho] saved {out_json} ({out_json.stat().st_size:,} bytes)")


def download_ho_et_al() -> None:
    gsm8k_json = HO_DIR / "gsm8k_zs_cot_text-davinci-002.json"
    svamp_json  = HO_DIR / "svamp_zs_cot_text-davinci-002.json"
    if gsm8k_json.exists() and svamp_json.exists():
        print(f"[ho] already extracted at {HO_DIR}")
        return

    HO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "release.zip"
        print(f"[ho] downloading shared folder zip (~924 MB) to {zip_path}")
        _curl_download(DROPBOX_ZIP_URL, zip_path)

        # Extract only the teacher tarball to save disk
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            tar_member = next((n for n in names if n.endswith(TEACHER_TARBALL_NAME)), None)
            if tar_member is None:
                raise RuntimeError(
                    f"{TEACHER_TARBALL_NAME} not found in zip. Members: {names[:20]}..."
                )
            print(f"[ho] extracting {tar_member} from zip")
            tar_path = tmp_path / TEACHER_TARBALL_NAME
            with zf.open(tar_member) as src, tar_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        if not gsm8k_json.exists():
            _extract_from_tarball(tar_path, "gsm8k", gsm8k_json)
        else:
            print(f"[ho] gsm8k already extracted, skipping")

        if not svamp_json.exists():
            _extract_from_tarball(tar_path, "svamp", svamp_json)
        else:
            print(f"[ho] svamp already extracted, skipping")


def _smoke_dataset(label: str, train_path: Path, test_path: Path, teacher_path: Path) -> None:
    if not train_path.exists() or not teacher_path.exists():
        print(f"[smoke:{label}] data missing; skipping")
        return
    train_rows = [json.loads(l) for l in train_path.open()]
    test_rows  = [json.loads(l) for l in test_path.open()] if test_path.exists() else []
    teacher = json.loads(teacher_path.read_text())
    teacher_data = teacher["data"]

    print("=" * 60)
    print(f"HO ET AL. SCHEMA REPORT — {label.upper()}")
    print("=" * 60)
    print("[meta]", json.dumps(teacher.get("metadata", {}), indent=2))
    print(f"[counts] teacher records: {len(teacher_data)}, "
          f"train={len(train_rows)}, test={len(test_rows)}")
    first_key = next(iter(teacher_data))
    one = teacher_data[first_key][0]
    print(f"[fields] {list(one.keys())}")
    print("\nTHREE EXAMPLES + TEACHER CoT")
    print("-" * 60)
    for i, ex in enumerate(train_rows[:3]):
        t_recs = teacher_data.get(str(i))
        if not t_recs:
            continue
        t = t_recs[0]
        print(f"\n--- train[{i}] ---")
        print("Q:", ex["question"])
        print("Gold:", ex["answer"])
        print("Teacher CoT:", (t.get("reasoning_completion") or "")[:200].strip())
        print("Teacher answer:", t.get("answer"))


def smoke_print() -> None:
    _smoke_dataset(
        "gsm8k",
        GSM8K_DIR / "train.jsonl",
        GSM8K_DIR / "test.jsonl",
        HO_DIR / "gsm8k_zs_cot_text-davinci-002.json",
    )
    _smoke_dataset(
        "svamp",
        SVAMP_DIR / "train.jsonl",
        SVAMP_DIR / "test.jsonl",
        HO_DIR / "svamp_zs_cot_text-davinci-002.json",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ho", action="store_true", help="Skip Ho et al. download (debug)")
    p.add_argument("--skip-svamp", action="store_true", help="Skip SVAMP download")
    p.add_argument("--smoke-only", action="store_true", help="Only print smoke output")
    args = p.parse_args()

    if not args.smoke_only:
        download_gsm8k()
        if not args.skip_svamp:
            download_svamp()
        if not args.skip_ho:
            download_ho_et_al()
    smoke_print()


if __name__ == "__main__":
    main()
