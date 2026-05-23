"""Stage 2: build training sets from GSM8K or SVAMP + Ho et al.'s CoTs.

Run with --dataset gsm8k  (default) or --dataset svamp.

GSM8K outputs (data/processed/):
  set_a_nofilter.jsonl      all teacher CoTs (no filter)
  set_b_magister.jsonl      teacher CoTs whose final answer == gold
  set_c_calculator.jsonl    same membership as Set B, arithmetic-corrected
  direct_ft.jsonl           answer-only baseline

SVAMP outputs (data/processed/svamp/):
  svamp_set_a_nofilter.jsonl
  svamp_set_b_magister.jsonl
  svamp_set_c_calculator.jsonl
  svamp_direct_ft.jsonl

Common record schema:
  {sample_index, question, cot, gold_answer, teacher_predicted_answer,
   calculator_corrected_cot?, n_calc_edits?}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data.calculator import correct_equations
from src.data.parse_answer import parse_answer
from src.utils.runcard import finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "data" / "raw"
OUT_DIR   = REPO_ROOT / "data" / "processed"

# GSM8K paths
GSM8K_TRAIN = RAW_DIR / "gsm8k" / "train.jsonl"
HO_GSM8K    = RAW_DIR / "ho_et_al_cots" / "gsm8k_zs_cot_text-davinci-002.json"
HO_JSON     = HO_GSM8K  # kept for backward-compat imports

# SVAMP paths
SVAMP_TRAIN = RAW_DIR / "svamp" / "train.jsonl"
HO_SVAMP    = RAW_DIR / "ho_et_al_cots" / "svamp_zs_cot_text-davinci-002.json"
SVAMP_OUT   = OUT_DIR / "svamp"

ANS_TOL = 1e-6


def _gold_str(parsed: float) -> str:
    """GSM8K answers are always integer-valued; render compactly."""
    return str(int(parsed)) if float(parsed).is_integer() else repr(parsed)


def _eq(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ANS_TOL)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "dataset": "gsm8k",
        "ans_tol": ANS_TOL,
        "calculator": "src.data.calculator.correct_equations",
        "set_c_membership": "same as Set B (Magister filter)",
    }
    card = start("02", "filter", config_snapshot)

    train = [json.loads(l) for l in GSM8K_TRAIN.open()]
    teacher_blob = json.loads(HO_GSM8K.read_text())
    teacher = teacher_blob["data"]

    # Counters
    n_total = 0
    n_set_a = 0
    n_set_b = 0
    n_set_c = 0
    n_direct = 0
    n_skipped_no_teacher = 0
    n_skipped_unparseable_gold = 0
    n_skipped_unparseable_teacher = 0

    # Set B / Set C contingency
    in_b_only = 0
    in_c_only = 0
    in_both = 0
    in_neither = 0
    n_calc_edited = 0

    samples_a: list[dict] = []
    samples_b: list[dict] = []
    samples_c: list[dict] = []
    samples_direct: list[dict] = []

    with OUT_A.open("w") as fa, OUT_B.open("w") as fb, \
         OUT_C.open("w") as fc, OUT_DIRECT.open("w") as fd:

        for i, ex in enumerate(train):
            n_total += 1

            gold = parse_answer(ex["answer"])
            if gold is None:
                n_skipped_unparseable_gold += 1
                continue

            # Direct FT — one row per train example, no teacher needed.
            direct_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": "",
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": None,
            }
            fd.write(json.dumps(direct_record) + "\n")
            n_direct += 1
            if len(samples_direct) < 2:
                samples_direct.append(direct_record)

            recs = teacher.get(str(i))
            if not recs:
                n_skipped_no_teacher += 1
                continue
            t = recs[0]    # zs_cot has one completion per sample_index

            cot = (t.get("reasoning_completion") or "").strip()
            teacher_pred = parse_answer(t.get("completion"))

            base_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": cot,
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": (t.get("completion") or "").strip(),
            }

            # Set A — always included (provided we have a teacher CoT).
            fa.write(json.dumps(base_record) + "\n")
            n_set_a += 1
            if len(samples_a) < 2:
                samples_a.append(base_record)

            # Set B — Magister's filter on the raw teacher prediction.
            in_b = teacher_pred is not None and _eq(teacher_pred, gold)
            if in_b:
                fb.write(json.dumps(base_record) + "\n")
                n_set_b += 1
                if len(samples_b) < 2:
                    samples_b.append(base_record)
            elif teacher_pred is None:
                n_skipped_unparseable_teacher += 1

            # Set C — same membership as Set B, but with calculator-corrected
            # intermediate arithmetic. We always apply the rewrite; membership
            # is inherited from Set B so the two sets are directly comparable.
            corrected_cot, edits = correct_equations(cot)
            n_calc_edits = len(edits)
            if n_calc_edits > 0:
                n_calc_edited += 1
            in_c = in_b
            if in_c:
                rec_c = {
                    **base_record,
                    "cot": corrected_cot,             # train students on the corrected version
                    "calculator_corrected_cot": n_calc_edits > 0,
                    "n_calc_edits": n_calc_edits,
                }
                fc.write(json.dumps(rec_c) + "\n")
                n_set_c += 1
                if len(samples_c) < 2:
                    samples_c.append(rec_c)

            # Contingency
            if in_b and in_c:
                in_both += 1
            elif in_b and not in_c:
                in_b_only += 1
            elif in_c and not in_b:
                in_c_only += 1
            else:
                in_neither += 1

    # ---- Console report -------------------------------------------------
    print(f"GSM8K train rows: {n_total}")
    print(f"  Direct FT      : {n_direct} -> {OUT_DIRECT}")
    print(f"  Set A (no flt) : {n_set_a} -> {OUT_A}")
    print(f"  Set B (Magist) : {n_set_b} -> {OUT_B}")
    print(f"  Set C (calc.)  : {n_set_c} -> {OUT_C}")
    print(f"  skipped: no_teacher={n_skipped_no_teacher} "
          f"unparseable_gold={n_skipped_unparseable_gold} "
          f"unparseable_teacher_pred={n_skipped_unparseable_teacher}")
    print(f"  Set B keep rate (of A): {n_set_b / max(n_set_a, 1):.1%}")
    print(f"  Set C keep rate (of A): {n_set_c / max(n_set_a, 1):.1%}")
    print(f"  CoTs the calculator edited: {n_calc_edited} / {n_set_a}")

    print("\n--- Set B / Set C contingency ---")
    print(f"  in both    : {in_both}")
    print(f"  Set C only : {in_c_only}   (chains rescued by calculator)")
    print(f"  Set B only : {in_b_only}   (calc broke a previously-correct answer)")
    print(f"  in neither : {in_neither}")

    print("\n--- 2 example records, Set A ---")
    for rec in samples_a:
        print(_short(rec))
    print("\n--- 2 example records, Set B ---")
    for rec in samples_b:
        print(_short(rec))
    print("\n--- 2 example records, Set C ---")
    for rec in samples_c:
        print(_short(rec))

    # ---- Run-card -------------------------------------------------------
    finish(
        card,
        metrics={
            "gsm8k_train_rows": n_total,
            "set_a_size": n_set_a,
            "set_b_size": n_set_b,
            "set_c_size": n_set_c,
            "direct_ft_size": n_direct,
            "set_b_keep_rate": round(n_set_b / max(n_set_a, 1), 4),
            "set_c_keep_rate": round(n_set_c / max(n_set_a, 1), 4),
            "n_calculator_edited": n_calc_edited,
            "skipped_no_teacher": n_skipped_no_teacher,
            "skipped_unparseable_gold": n_skipped_unparseable_gold,
            "skipped_unparseable_teacher_pred": n_skipped_unparseable_teacher,
            "contingency_b_and_c": in_both,
            "contingency_c_only": in_c_only,
            "contingency_b_only": in_b_only,
            "contingency_neither": in_neither,
        },
        inputs=[str(GSM8K_TRAIN.relative_to(REPO_ROOT)),
                str(HO_GSM8K.relative_to(REPO_ROOT))],
        outputs=[str(p.relative_to(REPO_ROOT)) for p in (OUT_A, OUT_B, OUT_C, OUT_DIRECT)],
        samples=[{"set": "A", **samples_a[0]} if samples_a else {},
                 {"set": "B", **samples_b[0]} if samples_b else {},
                 {"set": "C", **samples_c[0]} if samples_c else {},
                 {"set": "direct", **samples_direct[0]} if samples_direct else {}],
        notes="Stage 2 v2: emits Set A, Set B, Set C (calculator), and Direct FT.",
    )


def build_svamp() -> None:
    """Build Set A / B / C / Direct-FT for SVAMP (mirrors build() for GSM8K).

    SVAMP teacher CoTs (Ho et al., text-davinci-002, zero-shot-CoT) are keyed
    by sample_index over the full 1 000-example dataset.  We process only the
    700-example train split downloaded by Stage 1.
    """
    SVAMP_OUT.mkdir(parents=True, exist_ok=True)

    out_a      = SVAMP_OUT / "svamp_set_a_nofilter.jsonl"
    out_b      = SVAMP_OUT / "svamp_set_b_magister.jsonl"
    out_c      = SVAMP_OUT / "svamp_set_c_calculator.jsonl"
    out_direct = SVAMP_OUT / "svamp_direct_ft.jsonl"

    config_snapshot = {
        "dataset": "svamp",
        "ans_tol": ANS_TOL,
        "calculator": "src.data.calculator.correct_equations",
        "set_c_membership": "same as Set B (Magister filter)",
    }
    card = start("02", "filter_svamp", config_snapshot)

    train = [json.loads(l) for l in SVAMP_TRAIN.open()]
    teacher_blob = json.loads(HO_SVAMP.read_text())
    teacher = teacher_blob["data"]

    n_total = n_set_a = n_set_b = n_set_c = n_direct = 0
    n_skipped_no_teacher = n_skipped_unparseable_gold = n_skipped_unparseable_teacher = 0
    in_both = in_b_only = in_c_only = in_neither = n_calc_edited = 0
    samples_a: list[dict] = []
    samples_b: list[dict] = []
    samples_c: list[dict] = []
    samples_direct: list[dict] = []

    with out_a.open("w") as fa, out_b.open("w") as fb, \
         out_c.open("w") as fc, out_direct.open("w") as fd:

        for i, ex in enumerate(train):
            n_total += 1

            # SVAMP gold answers are plain numeric strings (e.g. "145")
            gold = parse_answer(ex["answer"])
            if gold is None:
                n_skipped_unparseable_gold += 1
                continue

            direct_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": "",
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": None,
            }
            fd.write(json.dumps(direct_record) + "\n")
            n_direct += 1
            if len(samples_direct) < 2:
                samples_direct.append(direct_record)

            recs = teacher.get(str(i))
            if not recs:
                n_skipped_no_teacher += 1
                continue
            t = recs[0]

            cot = (t.get("reasoning_completion") or "").strip()
            teacher_pred = parse_answer(t.get("completion"))

            base_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": cot,
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": (t.get("completion") or "").strip(),
            }

            fa.write(json.dumps(base_record) + "\n")
            n_set_a += 1
            if len(samples_a) < 2:
                samples_a.append(base_record)

            in_b = teacher_pred is not None and _eq(teacher_pred, gold)
            if in_b:
                fb.write(json.dumps(base_record) + "\n")
                n_set_b += 1
                if len(samples_b) < 2:
                    samples_b.append(base_record)
            elif teacher_pred is None:
                n_skipped_unparseable_teacher += 1

            corrected_cot, edits = correct_equations(cot)
            n_calc_edits = len(edits)
            if n_calc_edits > 0:
                n_calc_edited += 1
            in_c = in_b
            if in_c:
                rec_c = {
                    **base_record,
                    "cot": corrected_cot,
                    "calculator_corrected_cot": n_calc_edits > 0,
                    "n_calc_edits": n_calc_edits,
                }
                fc.write(json.dumps(rec_c) + "\n")
                n_set_c += 1
                if len(samples_c) < 2:
                    samples_c.append(rec_c)

            if in_b and in_c:
                in_both += 1
            elif in_b and not in_c:
                in_b_only += 1
            elif in_c and not in_b:
                in_c_only += 1
            else:
                in_neither += 1

    print(f"SVAMP train rows: {n_total}")
    print(f"  Direct FT      : {n_direct} -> {out_direct}")
    print(f"  Set A (no flt) : {n_set_a} -> {out_a}")
    print(f"  Set B (Magist) : {n_set_b} -> {out_b}")
    print(f"  Set C (calc.)  : {n_set_c} -> {out_c}")
    print(f"  skipped: no_teacher={n_skipped_no_teacher} "
          f"unparseable_gold={n_skipped_unparseable_gold} "
          f"unparseable_teacher_pred={n_skipped_unparseable_teacher}")
    print(f"  Set B keep rate (of A): {n_set_b / max(n_set_a, 1):.1%}")
    print(f"  Set C keep rate (of A): {n_set_c / max(n_set_a, 1):.1%}")
    print(f"  CoTs the calculator edited: {n_calc_edited} / {n_set_a}")

    finish(
        card,
        metrics={
            "svamp_train_rows": n_total,
            "set_a_size": n_set_a,
            "set_b_size": n_set_b,
            "set_c_size": n_set_c,
            "direct_ft_size": n_direct,
            "set_b_keep_rate": round(n_set_b / max(n_set_a, 1), 4),
            "set_c_keep_rate": round(n_set_c / max(n_set_a, 1), 4),
            "n_calculator_edited": n_calc_edited,
            "skipped_no_teacher": n_skipped_no_teacher,
            "skipped_unparseable_gold": n_skipped_unparseable_gold,
            "skipped_unparseable_teacher_pred": n_skipped_unparseable_teacher,
        },
        inputs=[str(SVAMP_TRAIN.relative_to(REPO_ROOT)),
                str(HO_SVAMP.relative_to(REPO_ROOT))],
        outputs=[str(p.relative_to(REPO_ROOT)) for p in (out_a, out_b, out_c, out_direct)],
        samples=[{"set": "A", **samples_a[0]} if samples_a else {},
                 {"set": "B", **samples_b[0]} if samples_b else {},
                 {"set": "C", **samples_c[0]} if samples_c else {},
                 {"set": "direct", **samples_direct[0]} if samples_direct else {}],
        notes="SVAMP Stage 2: Set A, B, C (calculator), Direct FT.",
    )


def _short(rec: dict) -> str:
    """Pretty-print a record with long strings clipped to 120 chars."""
    return json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                       for k, v in rec.items()}, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "svamp", "all"], default="gsm8k",
                    help="Which dataset to build training sets for (default: gsm8k)")
    args = ap.parse_args()

    if args.dataset in ("gsm8k", "all"):
        build()
    if args.dataset in ("svamp", "all"):
        build_svamp()
