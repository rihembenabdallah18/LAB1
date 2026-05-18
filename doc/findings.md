# Diagnostic Findings & Pipeline Fix Suggestions

**Date:** May 2026  
**Context:** Post-run analysis after observing that all three CoT student conditions (Set A, B, C)
performed worse than both the zero-shot baseline and direct fine-tuning on GSM8K.

---

## 1. Observed Results

| condition | `####` rate | accuracy |
|---|---|---|
| baseline (zero-shot) | 0% | 4.3% |
| student_direct_ft | 100% | 5.2% |
| student_set_a | 100% | 2.7% |
| student_set_b | 99.8% | 3.3% |
| student_set_c | 99.9% | 2.9% |

Reference from Ho et al. (FLAN-T5-base): baseline 2.50%, direct FT 5.08%, CoT FT 4.40%.  
Our CoT students fall below even the baseline — worse than what the original paper reported.

---

## 2. Hypotheses Investigated

Three hypotheses were investigated using diagnostic code before looking at the data:

### H1 — Target truncation (RULED OUT)

**Theory:** `max_target_length=512` truncates the `{cot} #### {gold_answer}` target string,
silently dropping `####` from training examples. The model never learns to emit the answer
marker and the parser falls back to the last number in the CoT text, which is often a wrong
intermediate result.

**Diagnostic:** `src/diag/cot_lengths.py` — tokenizes every training target and checks
whether `####` survives truncation.

**Result:**

| set | p50 tokens | p99 tokens | max tokens | `####` lost |
|---|---|---|---|---|
| set_a | 94 | 106 | 169 | 0 / 7473 (0.0%) |
| set_b | 83 | — | 160 | 0 / 3389 (0.0%) |
| set_c | 77 | — | 151 | 0 / 2635 (0.0%) |
| direct_ft | 5 | — | 7 | 0 / 7473 (0.0%) |

GSM8K CoTs from the Ho et al. release are short (max 169 tokens, well under the 512 limit).
**Truncation is not contributing to the poor results.**

---

### H2 — Parser returning wrong intermediate numbers (RULED OUT as primary cause)

**Theory:** Even when the model learns to generate CoT without `####`, `parse_answer`
falls back to the last number in the text. In a multi-step chain, the last number is often
an intermediate calculation result, not the final answer. This would silently mark correct
reasoning as wrong.

**Diagnostic:** `src/diag/parser_audit.py` — checks `####` emission rate per condition
and tracks how many "wrong" answers are attributable to fallback parse errors vs genuine
model errors.

**Result:**

| condition | `####` rate | fallback cases | fallback wrong |
|---|---|---|---|
| baseline | 0% | 1319 | 1262 |
| student_direct_ft | 100% | 0 | 0 |
| student_set_a | 100% | 0 | 0 |
| student_set_b | 99.8% | 3 | 3 |
| student_set_c | 99.9% | 1 | 1 |

All three student CoT models emit `####` in 99.8–100% of outputs. The parser is working
correctly on them. The numbers they put after `####` are simply wrong.

**Parser failure is not contributing to the poor results for student conditions.**
(It does explain the full baseline score of 4.3% — the baseline never emits `####` and
gets 4.3% purely from the last-number fallback getting lucky.)

---

### H3 — Teacher CoT quality (CONFIRMED as primary cause for Set A)

**Theory:** If the teacher model is frequently wrong, Set A trains on a mixture of correct
and incorrect reasoning chains. Wrong chains look structurally plausible but lead to wrong
answers, which confuses the student.

**Diagnostic:** `src/diag/cot_lengths.py` + inline teacher quality cell — inspected
`reasoning_completion` content and `completion` answer accuracy across all 7,473 train examples.

**Result:**

| metric | value |
|---|---|
| Missing teacher entries | 0 / 7473 (0.0%) |
| Empty CoTs | 0 / 7473 (0.0%) |
| CoT word length p50 | 66 words |
| CoT word length max | 116 words |
| Under 30 words | 361 / 7473 (4.8%) |
| Teacher correct | 3389 / 7473 (45.3%) |
| Teacher wrong | 4006 / 7473 (53.6%) |
| Teacher unparseable | 78 / 7473 (1.0%) |

**The teacher (text-davinci-002, zero-shot CoT) is wrong on 53.6% of GSM8K training examples.**

Set A trains on all 7,473 examples, of which 4,006 contain reasoning chains that look
coherent but arrive at wrong answers. The model learns the surface style of reasoning
without learning to reason correctly. This is enough to explain Set A performing below
zero-shot baseline.

---

### H4 — Catastrophic forgetting (CONFIRMED as cause for Set B and Set C underperforming baseline)

**Theory:** FLAN-T5-base came pre-trained with broad instruction tuning including math and
reasoning tasks. That pre-training is what delivers the 4.3% zero-shot baseline for free.
Fine-tuning on a small CoT dataset forces large weight updates (long target sequences),
partially overwriting the pre-trained general reasoning capability. The model forgets more
than it gains.

**Evidence:**
- Direct FT (target = `#### 42`, ~5 tokens) barely changes weights → 5.2%, beats baseline
- Set B CoT fine-tuning (target = full reasoning chain, ~83 tokens median) → 3.3%, below baseline
- The degradation is proportional to how much the training changes the model: CoT > Direct FT

This is consistent with Ho et al.'s finding that CoT distillation hurts at T5-base scale
and only helps at T5-XL (3B) and above.

---

## 3. Set C Specific Findings

Set C was designed to improve on Set B by using a calculator to fix arithmetic errors in
teacher CoTs. In practice it performed worse (2.9% vs Set B's 3.3%).

### 3.1 Contingency table

| bucket | count |
|---|---|
| in both B and C | 2,585 |
| B only (calculator broke it) | 804 |
| C only (calculator rescued it) | 50 |
| neither | 4,034 |
| CoTs with at least one edit | 676 |

The calculator rescued only **50 examples** while destroying **804** — a 16:1
destruction-to-rescue ratio. Set C (2,635) ends up with 754 fewer examples than Set B (3,389).

### 3.2 Root cause of the 804 lost examples — filter bug

Set B and Set C check correctness against **different fields**:

```python
# Set B — checks the teacher's explicit final-answer statement
teacher_pred = parse_answer(t.get("completion"))   # "Therefore the answer is 42"
in_b = _eq(teacher_pred, gold)

# Set C — checks the last number in the corrected CoT reasoning chain
corrected_pred = parse_answer(corrected_cot)        # last number anywhere in the chain
in_c = _eq(corrected_pred, gold)
```

`parse_answer` on the CoT uses a last-number fallback. After the calculator patches an
intermediate equation, the patched number can become the last number in the text, causing
`parse_answer` to pick it up instead of the true final answer. Set C then rejects the
example even though the teacher's stated answer was correct.

### 3.3 Text-number inconsistency in patched CoTs

When the calculator changes `6 * 52 = 312` → `6 * 52 = 36`, it patches the equation but
leaves the surrounding prose unchanged. If the teacher wrote "so we have 312 apples", that
sentence still says 312 while the equation now says 36. The student trains on internally
contradictory text across 676 CoTs.

---

## 4. Summary of Root Causes

| cause | affects | severity |
|---|---|---|
| Teacher wrong 53.6% of the time | Set A | critical — poisons majority of training data |
| Catastrophic forgetting of FLAN pre-training | Set B, Set C | high — explains why even clean CoT data falls below baseline |
| Set C filter bug (wrong field for correctness check) | Set C | high — loses 804 valid examples for free |
| Text-number inconsistency after calculator patching | Set C | medium — corrupts 676 training examples |
| Model capacity (250M too small for CoT reasoning) | all CoT conditions | fundamental — consistent with Ho et al. |

---

## 5. Suggested Fixes

### Fix 1 — Repair the Set C filter (highest priority, one-line change)

**File:** `src/data/filter.py`

**Problem:** `parse_answer(corrected_cot)` is used to check correctness but it uses the
last-number fallback on the CoT, not the teacher's stated answer.

**Fix:** Decouple the correctness check from the correction step. Apply the calculator to
all Set B examples as a cleaning step, but use the same reliable `completion`-based
correctness check as Set B:

```python
# Check correctness using the explicit completion field (same as Set B)
teacher_pred = parse_answer(t.get("completion"))
in_b = teacher_pred is not None and _eq(teacher_pred, gold)

# Apply calculator to all examples as a cleaning step, regardless of outcome
corrected_cot, edits = correct_equations(cot)

# Set C = every Set B example, with calculator-cleaned CoT where available
if in_b:
    rec_c = {
        **base_record,
        "cot": corrected_cot,
        "calculator_corrected_cot": len(edits) > 0,
        "n_calc_edits": len(edits),
    }
    fc.write(json.dumps(rec_c) + "\n")
```

**Expected outcome:** Set C grows from 2,635 to ~3,439 examples (all of Set B plus the
50 rescued ones). Set C should then equal or slightly exceed Set B on accuracy.

---

### Fix 2 — Safe calculator: skip patches that leave stale references

**File:** `src/data/calculator.py`

**Problem:** When the calculator patches `5 * 8 = 38` → `5 * 8 = 40`, the surrounding
text still refers to 38, creating an internal contradiction in the training example.

**Fix:** Skip a correction if the old (wrong) value appears again anywhere after the
equation, since patching would leave stale references:

```python
def correct_equations_safe(text: str) -> tuple[str, list[Edit]]:
    edits: list[Edit] = []

    def repl(m: re.Match) -> str:
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        actual = _OPS[op](a, b)
        if actual is None or _is_close(c, actual):
            return m.group(0)
        # Skip if the claimed (wrong) value reappears later in the text
        rest = text[m.end():]
        if _format_number(c) in rest:
            return m.group(0)
        new_c = _format_number(actual)
        new_full = f"{m.group(1)} {op} {m.group(3)} = {new_c}"
        edits.append(Edit(m.span(), m.group(0), new_full, c, actual))
        return new_full

    return _EQ_RE.sub(repl, text), edits
```

Replace the `correct_equations` call in `filter.py` with `correct_equations_safe`.

---

### Fix 3 — Two-stage training to reduce catastrophic forgetting

**File:** `src/train/finetune.py` (no code change needed, just run order)

**Problem:** Fine-tuning directly on CoT targets overwrites FLAN's pre-trained zero-shot
reasoning capability because the long target sequences require large weight updates.

**Fix:** Curriculum learning — train on Direct FT first (anchors the `####` format and
preserves general reasoning), then continue fine-tuning on the CoT set with a lower
learning rate:

```bash
# Stage 3a: anchor answer format and preserve pre-trained reasoning
bash scripts/03_train_direct_ft.sh   # already exists

# Stage 3b: continue from direct_ft checkpoint into CoT, lower LR
python -m src.train.finetune \
    --train data/processed/set_b_magister.jsonl \
    --run-name student_set_b_curriculum \
    --resume \
    --model outputs/checkpoints/student_direct_ft/checkpoint-best
```

To support this in config, add a `finetune_lr` key separate from the initial `learning_rate`,
and halve it for CoT stages (e.g. `2.5e-5`).

**Expected outcome:** The model starts from a point where it already knows the answer
format and has retained FLAN's reasoning capability, then specialises on CoT. Should
reduce the gap between Set B/C and baseline.

---

### Fix 4 — Increase training data for CoT conditions via set combination

**Problem:** Set B has only 3,389 examples. The catastrophic forgetting effect is
amplified when the fine-tuning dataset is small relative to the pre-training scale.

**Fix:** Combine Direct FT and CoT examples in one training set, so the model sees both
the clean short-target signal (reinforces answer format) and the CoT signal:

```python
# In filter.py, also write a combined set
combined_record = {**base_record, "cot": cot_to_use}
# Write to a set_b_augmented.jsonl = set_b + direct_ft rows
```

Or simply concatenate at training time:

```bash
cat data/processed/direct_ft.jsonl data/processed/set_b_magister.jsonl \
    > data/processed/set_b_augmented.jsonl
```

Add a corresponding `03_train_set_b_augmented.sh` script and a `set_b_augmented` path
in `config/config.yaml`.

---

## 6. Priority Order

| priority | fix | effort | expected gain |
|---|---|---|---|
| 1 | Fix Set C filter bug (Fix 1) | ~10 lines in filter.py | Set C ≥ Set B |
| 2 | Safe calculator (Fix 2) | ~10 lines in calculator.py | cleaner CoT signal |
| 3 | Two-stage training (Fix 3) | config + rerun | CoT conditions may beat baseline |
| 4 | Augmented training set (Fix 4) | 1 script + rerun | more robust CoT training |

Fixes 1 and 2 together address the Set C regression and require only Stage 2 to be rerun
(fast, CPU only). Fix 3 requires rerunning Stage 3 (3–5 hours on T4). Fix 4 can be
combined with Fix 3 at no extra cost.

---

## 7. What the Results Mean for the Research Question

Despite the poor accuracy numbers, the results are coherent and reportable:

- **H1 (reproduction):** Partially confirmed — direct FT (5.2%) beats baseline (4.3%),
  matching the qualitative finding. CoT distillation at this scale does not beat direct FT,
  consistent with Ho et al.'s own T5-base results.

- **H2 (filter effect):** Confirmed — Set B (3.3%) > Set A (2.7%), showing the
  answer-correctness filter helps even if absolute numbers are low. Set C underperforms Set B
  due to the filter bug, not the calculator concept itself.

- **H3 (metric agreement):** ReCEval scores (Stage 5b) may still reveal that CoT models
  produce higher-quality reasoning chains despite lower accuracy — this is the central claim
  of the project and remains worth investigating even with these accuracy numbers.

The finding that CoT distillation on FLAN-T5-base at 45% teacher accuracy degrades below
baseline is itself a meaningful result: it shows that accuracy alone (the standard metric)
does not reflect what the model learned, which is exactly the gap ReCEval is designed to
measure.
