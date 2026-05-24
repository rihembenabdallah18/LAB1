# Pipeline — teacher, training sets, and stages

A plain-English walkthrough of what the project trains, what each training set contributes, and what each pipeline stage does. For the academic motivation and hypotheses, see [RESEARCH.md](RESEARCH.md). For the literature this builds on, see [Related work.md](Related%20work.md).

---

## Teacher model

| | |
|---|---|
| **Source** | Ho et al. 2022 — pre-released Zero-shot-CoT outputs from **GPT-3 `text-davinci-002`** on GSM8K. Hosted at `itsnamgyu/reasoning-teacher`. |
| **What it gives us** | One reasoning chain (CoT) per GSM8K **train** question, plus the teacher's own final-answer extraction. 8,792 records covering all 7,473 train examples. |
| **Why we don't run it ourselves** | API cost and reproducibility. Using the released file means anyone can rerun the pipeline without an OpenAI key. |
| **Role in the pipeline** | The teacher is the **source of supervision** for chain-of-thought fine-tuning. The student learns to imitate these chains. The teacher is never evaluated as a model in our results — only its chains are used. |

The teacher's chains are noisy: roughly 45% reach the correct final answer. The three "Set X" filters below differ in **how strictly they screen this noise** before showing it to the student.

---

## Student training sets

All four sets are built from the same source (GSM8K train + teacher CoTs) by [`src/data/filter.py`](../src/data/filter.py). All four students are FLAN-T5-base (220M), trained with identical hyperparameters in [`config/config.yaml`](../config/config.yaml). Only the **training data** changes.

The input/target format is the same for every set:

```
input  : "Q: {question}"
target : "{cot} #### {gold_answer}"      (cot is empty for Direct FT)
```

### Direct FT — no chain of thought

| | |
|---|---|
| **Size** | 7,473 (one row per train example) |
| **How it's built** | `cot=""` for every row. The target collapses to `#### {gold_answer}`. |
| **Purpose** | Reference point for "what does fine-tuning give us **without** any CoT?" |
| **What to expect** | Matches Ho et al.'s **5.08%** standard-FT line. Should beat the un-finetuned baseline; should also (interestingly) beat the CoT sets in our small-model regime, because the model isn't asked to generate intermediate reasoning. |
| **Value it adds** | Isolates the contribution of CoT supervision. If a CoT student beats Direct FT, the CoT is helping; if not, the gain (or loss) comes from elsewhere. |
| **Evaluation** | Accuracy only — there is no chain to score with ReCEval. |

### Set A — all teacher CoTs (no filter)

| | |
|---|---|
| **Size** | 7,473 |
| **How it's built** | Every teacher CoT, kept as-is. No quality check. |
| **Purpose** | Baseline for filtered conditions — measures the cost of training on raw, unfiltered teacher noise. |
| **What to expect** | Worst CoT result. The student sees many chains that lead to wrong answers, so it learns to imitate flawed reasoning. |
| **Value it adds** | The denominator for Set B and Set C. Differences between A and B/C isolate the **filter's** contribution from the **CoT's** contribution. |
| **Evaluation** | Accuracy + full ReCEval (intra / inter / info). |

### Set B — answer-correctness filter (Magister / STaR)

| | |
|---|---|
| **Size** | ~3,389 (45% of Set A) |
| **How it's built** | Keep a CoT only if the teacher's **final number** equals gold. The chain itself is not inspected — only the answer extracted from the teacher's completion. |
| **Purpose** | Reproduces the standard distillation filter from Magister 2022 / STaR 2022. |
| **What to expect** | Reproduces Ho et al.'s **4.40%** CoT-FT line. Should beat Set A. May or may not beat Direct FT — that's the open question. |
| **Value it adds** | This is the **outcome-only** filter. It rewards chains that arrive at the right answer, no matter how. |
| **Evaluation** | Accuracy + full ReCEval. Comparing Set B's accuracy uplift to its ReCEval uplift tests **H2**: does the filter help process or only outcome? |

### Set C — calculator-corrected filter (process-aware)

| | |
|---|---|
| **Size** | 3,389 (same membership as Set B) |
| **How it's built** | Membership uses the same answer-correctness check as Set B (teacher's `completion` field parsed against gold). On top of that, the CoT text is walked and each `A op B = C` substring whose claimed result is wrong is rewritten to the true value. The **corrected** text — not the original — is what gets trained on. In the current run 676 of the 3,389 examples received at least one calculator edit. |
| **Purpose** | A **process-aware** cleaning step. Same set of training questions as Set B, but with arithmetic mistakes inside the chain repaired before fine-tuning. |
| **Why membership matches Set B** | An earlier version selected on `parse_answer(corrected_cot)`, which used a last-number fallback and silently dropped ~800 valid examples whenever the calculator patch shifted which number was last in the text. Decoupling the correctness check (use `completion`) from the cleaning step (use `corrected_cot`) brought Set C in line with Set B. |
| **Value it adds** | Differences between Set B and Set C now come from the calculator-edited 676 examples only, isolating "cleaner arithmetic in supervision" from "different training subset". If Set C beats Set B on **ReCEval** but not on accuracy, that supports H2. |
| **Evaluation** | Accuracy + full ReCEval. The Set B vs. Set C comparison is the project's most direct test of "process vs. outcome supervision". |

### At a glance

| Set | Size | Filter type | Compares against | Headline question |
|---|---|---|---|---|
| Direct FT | 7,473 | none (no CoT) | un-finetuned baseline | Does fine-tuning help at all? |
| Set A | 7,473 | none | Direct FT | Does adding CoTs help? |
| Set B | 3,389 | outcome (answer-correct) | Set A | Does filtering by **final answer** help? |
| Set C | 3,389 (676 with calculator edits) | process (calculator-cleaned, same membership as B) | Set B | Does **cleaning arithmetic inside the chain** help reasoning quality? |

---

## Pipeline stages

Every stage is one shell script in [`scripts/`](../scripts/) and one Python module in [`src/`](../src/). Every stage writes a JSON run-card to `outputs/runs/`; `python -m src.status` reads them and prints a one-screen view of the whole project.

### Stage 1 — Download

**Script:** [`scripts/01_download.sh`](../scripts/01_download.sh) · **Module:** [`src/data/download.py`](../src/data/download.py)

Downloads GSM8K (7,473 train / 1,319 test) from HuggingFace and the Ho et al. teacher CoTs (~25 MB) from the released Dropbox archive. Resumable — if the files exist, it is a no-op. Run once per environment.

### Stage 2 — Build training sets

**Script:** [`scripts/02_filter.sh`](../scripts/02_filter.sh) · **Modules:** [`src/data/filter.py`](../src/data/filter.py), [`src/data/calculator.py`](../src/data/calculator.py)

Reads GSM8K train + teacher CoTs, writes four JSONL files to `data/processed/`: `direct_ft.jsonl`, `set_a_nofilter.jsonl`, `set_b_magister.jsonl`, `set_c_calculator.jsonl`. Prints set sizes and a Set B / Set C contingency table. Fast (a few seconds, CPU only).

### Stage 3 — Fine-tune

**Scripts:** `scripts/03_train_{direct_ft,set_a,set_b,set_c}.sh` · **Module:** [`src/train/finetune.py`](../src/train/finetune.py)

Fine-tunes FLAN-T5-base on one training set. Saves a checkpoint per epoch to `outputs/checkpoints/{run_name}/`. Early-stops on validation loss (patience 2). Resumable via `--resume`. Run cheapest first: `direct_ft` → `set_c` → `set_b` → `set_a`. On a T4 each run takes 30 min – 2 hours depending on set size.

### Stage 4 — Inference

**Script:** [`scripts/04_inference.sh`](../scripts/04_inference.sh) · **Module:** [`src/inference/generate.py`](../src/inference/generate.py)

Runs every condition on the 1,319 GSM8K test problems with **beam=4, no_repeat_ngram=4, repetition_penalty=1.15**. Writes one JSONL per condition to `outputs/generations/`. Each line carries the question, generated CoT, parsed answer, and gold answer. Resumable — if a file already has all 1,319 rows, it's skipped.

### Stage 5a — Accuracy

**Script:** [`scripts/05a_accuracy.sh`](../scripts/05a_accuracy.sh) · **Module:** [`src/eval/accuracy.py`](../src/eval/accuracy.py)

For each condition, parses the final number from the generated text (last `#### N` if present, otherwise the last number in the string) and compares to gold within `abs < 1e-6`. Writes `accuracy.csv` and `accuracy_bar.png`. Pure CPU, seconds.

### Stage 5b — ReCEval

**Script:** [`scripts/05b_receval.sh`](../scripts/05b_receval.sh) · **Module:** [`src/eval/receval/`](../src/eval/receval/)

The project's novel evaluation axis. For each generated chain, computes three per-chain scalars, each aggregated as the **minimum over steps**:

| Metric | What it measures | Backed by |
|---|---|---|
| **intra** | Step internal entailment (simplified RCU — premise = hypothesis = step) | DeBERTa-v3 NLI |
| **inter** | `1 − max P(contradiction)` between step *i* and prior context (question + earlier steps) | DeBERTa-v3 NLI |
| **info** | Incremental log-likelihood gain of the gold answer when step *i* is added to the context | Frozen causal LM (distilgpt2 by default) |

Writes one JSONL per condition to `outputs/eval_results/`, plus a summary CSV and a violin plot. Smoke-test first with `bash scripts/05b_receval.sh --smoke 20`. On a T4 a full sweep takes 1–3 hours per condition; if budget is tight, cap with `--max-examples 500`.

---

## How a result is read

The headline comparison is **always two columns side by side**:

- **Accuracy** — does the student get the right number? (Stage 5a)
- **ReCEval** — does the student get there for the right reason? (Stage 5b)

A student that wins on accuracy but loses on ReCEval is the project's central object of study: a model that picks up the answer pattern without picking up the reasoning behind it. The four-set design exists so that this gap can be attributed to a specific cause (no CoT, no filter, outcome filter, process filter) rather than left as a single number.

---

## Extensions on top of the main GSM8K experiment

### Scale ablation — FLAN-T5-large

The same four training sets are also used to fine-tune **FLAN-T5-large** (~780M, ~3× the parameters of base). Each large-model condition writes to a `*_large` run-name (e.g. `student_set_b_large`) and appears as an extra row in `outputs/eval_results/accuracy.csv` and the ReCEval summary. Two purposes:

- Test whether Ho et al.'s observation that CoT distillation only helps at scale starts to appear between base and large.
- Anchor the small-model numbers — if `base` and `large` rank conditions the same way, the small-model story is not a small-model artifact.

These runs are driven from [`notebooks/Kaggle.ipynb`](../notebooks/Kaggle.ipynb) rather than dedicated shell scripts; they share the same `src/train/finetune.py` entry-point with `--model google/flan-t5-large` and a slightly smaller batch size.

### SVAMP transfer

A separate but structurally identical pipeline runs on **SVAMP** (Patel et al. 2021, ~700 train / 300 test, simpler arithmetic). Same A/B/C/Direct-FT design, same teacher source (`itsnamgyu/reasoning-teacher` SVAMP CoTs), same `src/data/filter.py` invoked with `--dataset svamp`. SVAMP-only paths live in `config/config.yaml` under `svamp_*` and outputs in `outputs/{generations,eval_results}/svamp/`.

| Stage | Script |
|---|---|
| Filter | [`scripts/02_filter_svamp.sh`](../scripts/02_filter_svamp.sh) |
| Train  | [`scripts/03_train_svamp_{direct_ft,set_a,set_b,set_c}.sh`](../scripts/) |
| Inference | [`scripts/04_inference_svamp.sh`](../scripts/04_inference_svamp.sh) |
| Accuracy | [`scripts/05a_accuracy_svamp.sh`](../scripts/05a_accuracy_svamp.sh) |

SVAMP serves two roles:

1. **Re-train transfer** — run the full A/B/C/Direct-FT pipeline on SVAMP train, evaluate on SVAMP test. Tests whether the GSM8K cross-condition ranking reproduces on a different (smaller, simpler) word-problem benchmark.
2. **Zero-shot cross-dataset transfer** — take the GSM8K-trained `student_set_b` checkpoint as-is and evaluate it on SVAMP test (`outputs/generations/svamp/student_set_b.jsonl`, no `svamp_` prefix). Quantifies how much the GSM8K-distilled CoT capability survives a distribution shift, with no additional fine-tuning.

The corresponding plots are `outputs/plots/svamp_overview.png` and `outputs/plots/svamp_transfer_gap.png`.

### Variant — `student_set_c_mix` (mixed CoT + answer-only supervision)

An additional GSM8K-only condition that does not fit the clean four-set design. The training set is **Set C concatenated with a same-sized random sample of Direct FT** (3,389 + 3,389 = 6,778 examples, shuffled, seed 42), built in [`notebooks/Kaggle.ipynb`](../notebooks/Kaggle.ipynb) and written to `data/processed/set_c_plus_direct.jsonl`. The run-name is `student_set_c_mix`.

| | |
|---|---|
| **Purpose** | Test whether interleaving CoT supervision with short answer-only targets reduces the catastrophic-forgetting effect that pushes pure-CoT students below the zero-shot baseline at FLAN-T5-base scale. |
| **What it adds beyond Set C** | The student sees both long CoT targets (~80 tokens) and short answer-only targets (~5 tokens) in every batch. The short targets anchor the answer format and require small weight updates; the long ones supply the reasoning signal. |
| **Evaluation** | Accuracy + ReCEval. Appears as `student_set_c_mix` in `accuracy.csv` and `receval_summary.csv`. |
| **Where it lives** | Notebook-only — no shell script under `scripts/`. The notebook cells write the mixed JSONL, then call `src/train/finetune.py` and `src/inference/generate.py` directly with `--run-name student_set_c_mix`. |