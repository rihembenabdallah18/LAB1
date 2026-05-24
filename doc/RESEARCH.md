# Research Plan — Chain-of-Thought Knowledge Distillation from LLMs to SLMs with Reasoning-Quality Evaluation

**Date:** May 2026

For the operational walkthrough (teacher, training sets, per-stage scripts and modules), see [pipeline.md](pipeline.md). For paper notes, see [Related work.md](Related%20work.md).

---

## 1. Summary

This project reproduces, at small scale and with limited compute, the chain-of-thought (CoT) knowledge distillation pipeline of Ho et al. (ACL 2023, *Large Language Models Are Reasoning Teachers*), which itself builds on the small-student CoT distillation framing of Magister et al. (2022, *Teaching Small Language Models to Reason*). The calculator rewrite-trick used in Set C comes from Magister et al.; answer-correctness filtering (Set B) predates Magister (e.g. STaR / Zelikman 2022). Beyond simple reproduction, the project adds a **second evaluation axis**: in addition to final-answer accuracy, the student model's generated reasoning chains are scored using ReCEval (Prasad et al., EMNLP 2023), a reference-free framework for measuring reasoning correctness and informativeness.

The central claim being tested is whether final-answer accuracy alone — the standard metric in CoT distillation work — adequately reflects whether reasoning capability has actually transferred to the student.

The pipeline is extended along two axes beyond a one-shot reproduction:
- **Scale ablation** — every condition is re-run with FLAN-T5-large (~780M) alongside FLAN-T5-base (~250M), to test whether Ho et al.'s "CoT only helps at scale" claim begins to bite between base and large.
- **Transfer ablation** — the full A/B/C/Direct-FT design is replicated on SVAMP (700 train / 300 test) to test whether the cross-condition ranking established on GSM8K generalises to a different word-problem benchmark.

---

## 2. Background and Motivation

CoT prompting reliably elicits step-by-step reasoning in large language models (≥ tens of billions of parameters), but small models (< 10B) typically produce incoherent CoTs and may even lose accuracy when prompted to reason explicitly. Magister et al. (2022) and Ho et al. (2023) close this gap via knowledge distillation: a large teacher (e.g. PaLM 540B, GPT-3 175B) is prompted to generate CoTs on an existing supervised dataset; CoTs whose final answer is incorrect are filtered out (an answer-correctness filter, predating Magister; cf. STaR / Zelikman 2022); a small student (T5 family) is fine-tuned on the surviving (question → CoT + answer) pairs. They report substantial accuracy gains, e.g. on GSM8K, T5-XXL improves from 8.11% to 21.99%.

A natural question follows. The answer-correctness filter accepts any CoT whose **final number** is correct, regardless of whether the reasoning chain is internally consistent, free of hallucinations, or step-wise informative. As a result:

- Some accepted training examples reach the right answer through flawed reasoning.
- The student's reported gain is measured purely on final-answer accuracy — leaving open whether reasoning quality genuinely improved or whether the student learned shortcut patterns.

ReCEval (Prasad et al., 2023) provides a reference-free way to score reasoning chains along two complementary axes: **correctness** (intra-step entailment, inter-step non-contradiction) and **informativeness** (information gain toward the final answer), all computed with off-the-shelf NLI and LM components. Applying ReCEval to the *student's own outputs* — not just to teacher data — directly measures whether reasoning capability has transferred, complementing accuracy.

This project combines the two: a small-scale Ho-et-al.-style distillation pipeline (using Magister's calculator rewrite for Set C), evaluated jointly with accuracy and ReCEval, on a single student family at two scales and two datasets.

---

## 3. Research Question and Hypotheses

### Research question

> When a small language model is trained via CoT knowledge distillation, does final-answer accuracy adequately reflect the quality of its generated reasoning chains, or does ReCEval-based evaluation reveal aspects of reasoning quality that accuracy alone does not capture?

### Sub-questions

- **SQ1 (Reproduction).** Does a Ho-et-al.-style filtered distillation, applied at small scale (FLAN-T5-base/large, free-tier GPU), reproduce the qualitative finding that distillation improves student final-answer accuracy on GSM8K relative to a non-fine-tuned baseline?
- **SQ2 (Filter effect on reasoning quality).** Does the answer-correctness filter (Set B) improve only accuracy, or does it also improve ReCEval-measured reasoning quality? Does process-aware cleaning (Set C, calculator-corrected) widen this gap further on ReCEval without necessarily moving accuracy?
- **SQ3 (Scale).** Does the ranking of conditions (Direct-FT / Set A / Set B / Set C) on accuracy and on ReCEval change between FLAN-T5-base and FLAN-T5-large?
- **SQ4 (Transfer).** Does the cross-condition ranking established on GSM8K reproduce on SVAMP?

### Hypotheses

- **H1.** Distillation improves accuracy relative to the non-fine-tuned baseline (expected; reproduction).
- **H2.** The answer-correctness filter improves accuracy more than it improves ReCEval scores — i.e. the filter is well-targeted at outcomes but only weakly targeted at process. Calculator-cleaning (Set C) further improves ReCEval over Set B while leaving accuracy roughly unchanged.
- **H3 (null acceptable).** A non-trivial fraction of correct-answer outputs receive low ReCEval scores ("right answer, flawed reasoning"), and a non-trivial fraction of incorrect-answer outputs receive high ReCEval scores ("good reasoning, slip at the end"). If true, this directly supports the claim that accuracy alone is an incomplete metric.

A null result for H2 (filter helps both equally) is also reportable and informative — it would suggest that for this particular failure mode, accuracy is in fact a reasonable proxy.

---

## 4. Scope

### In scope

- **Two datasets:** GSM8K (main experiment, 7,473 train / 1,319 test) and SVAMP (transfer probe, 700 train / 300 test). Both have clean automatic correctness checks and multi-step reasoning, so ReCEval is meaningful on both.
- **Two student scales:** FLAN-T5-base (~250M) as the primary student, FLAN-T5-large (~780M) as a scale ablation. Same hyperparameters, same training sets.
- **Four training conditions per dataset:** Direct FT (no CoT), Set A (no filter), Set B (answer-correctness filter), Set C (calculator-cleaned Set B). See [pipeline.md](pipeline.md#student-training-sets) for the precise definitions.
- **Pre-released teacher CoTs:** GPT-3-generated CoTs on GSM8K and SVAMP released by Ho et al. 2022 (`itsnamgyu/reasoning-teacher`). No teacher inference is performed.
- **Two evaluation metrics:** final-answer accuracy and ReCEval (intra-step, inter-step, informativeness).

### Out of scope (and why)

| Excluded | Reason |
|---|---|
| Larger students (T5-XL, T5-XXL, Qwen-3B+) | Will not fit on free-tier T4 with full fine-tuning |
| ReCEval pvi-based informativeness | Requires training T5-large twice; uses too much budget — log-likelihood variant used instead |
| Running the teacher model directly | API cost, reproducibility; pre-released CoTs avoid this |
| ReCEval as a *filter* (not just a metric) | More ambitious variant; deferred to future work |
| Self-consistency, decomposition, RL signals | Distinct research questions |

### Hardware and software

- Free-tier Kaggle notebooks (NVIDIA T4, 16 GB VRAM, ~12-hour sessions)
- HuggingFace Transformers, Datasets, Accelerate
- DeBERTa-v3-small NLI (`cross-encoder/nli-deberta-v3-small`) for ReCEval correctness components
- DistilGPT-2 as the frozen LM for the log-likelihood informativeness variant
- spaCy for sentence segmentation
- Checkpoints saved to `outputs/checkpoints/`; intermediate epoch checkpoints pruned with `python -m src.utils.cleanup_ckpts`

---

## 5. Pipeline

The full pipeline runs in five sequential stages (Stage 5 is split into 5a accuracy and 5b ReCEval). One shell script and one Python module per stage, with a SVAMP-suffixed variant for the transfer pipeline.

The detailed walkthrough — teacher source, set construction, every script and module — lives in [pipeline.md](pipeline.md). What follows is the experimental matrix only.

---

## 6. Experimental Matrix

### GSM8K — FLAN-T5-base (primary)

| Condition | Training data | Evaluated on | Metrics |
|---|---|---|---|
| `baseline` | none (zero-shot) | GSM8K test (1,319) | accuracy + ReCEval |
| `student_direct_ft` | Direct FT (7,473) | GSM8K test | accuracy only (no CoT) |
| `student_set_a` | Set A (7,473) | GSM8K test | accuracy + ReCEval |
| `student_set_b` | Set B (3,389) | GSM8K test | accuracy + ReCEval |
| `student_set_c` | Set C (3,389, 676 calculator-edited) | GSM8K test | accuracy + ReCEval |
| `student_set_c_mix` | Set C + same-sized sample of Direct FT, shuffled (6,778) | GSM8K test | accuracy + ReCEval |

`student_set_c_mix` tests whether interleaving CoT with answer-only targets mitigates catastrophic forgetting at T5-base scale. See [pipeline.md §Variant](pipeline.md#variant--student_set_c_mix-mixed-cot--answer-only-supervision).

### GSM8K — FLAN-T5-large (scale ablation)

| Condition | Training data | Metrics |
|---|---|---|
| `student_direct_ft_large` | Direct FT | accuracy |
| `student_set_a_large` | Set A | accuracy + ReCEval |
| `student_set_b_large` | Set B | accuracy + ReCEval |
| `student_set_c_large` | Set C | accuracy + ReCEval |

### SVAMP — FLAN-T5-base (transfer)

| Condition | Training data | Evaluated on |
|---|---|---|
| `baseline` | none (zero-shot) | SVAMP test (300) |
| `svamp_student_direct_ft` | SVAMP Direct FT (700) | SVAMP test |
| `svamp_student_set_a` | SVAMP Set A (509) | SVAMP test |
| `svamp_student_set_b` | SVAMP Set B (299) | SVAMP test |
| `svamp_student_set_c` | SVAMP Set C (299) | SVAMP test |
| `student_set_b` (GSM8K-trained) | GSM8K Set B (3,389) | SVAMP test | accuracy only — zero-shot cross-dataset transfer probe |

Outputs land in `outputs/eval_results/accuracy.csv`, `outputs/eval_results/receval_summary.csv`, and `outputs/eval_results/svamp/`. Headline plots in `outputs/plots/`: `gsm8k_filter_ablation.png`, `gsm8k_scale_ablation.png`, `svamp_overview.png`, `svamp_transfer_gap.png`, `accuracy_vs_receval.png`, `receval_bars.png`.

---

## 7. Future Work

- ReCEval as a *filter* on teacher CoTs (rather than only as evaluation), comparing against the answer-correctness filter.
- Larger students (T5-XL or Qwen-1.5B with LoRA) on the same dual-metric framework.
- Multiple teacher models, to study whether teacher CoT quality (as measured by ReCEval) predicts student CoT quality.
- The pvi-based informativeness variant of ReCEval, replacing the log-likelihood approximation.
- Extending the transfer matrix beyond SVAMP (StrategyQA, ASDiv, MathQA) to test cross-domain robustness of the dual-metric finding.

---

## References

- Magister, Mallinson, Adamek, Malmi, Severyn. *Teaching Small Language Models to Reason*. ACL 2023 (Short). https://aclanthology.org/2023.acl-short.151/
- Prasad, Saha, Zhou, Bansal. *ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness*. EMNLP 2023. https://arxiv.org/abs/2304.10703
- Ho, Schmid, Yun. *Large Language Models Are Reasoning Teachers*. ACL 2023. https://github.com/itsnamgyu/reasoning-teacher
- Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.
- Cobbe et al. *Training Verifiers to Solve Math Word Problems* (GSM8K). 2021.
- Patel, Bhattamishra, Goyal. *Are NLP Models really able to Solve Simple Math Word Problems?* (SVAMP). NAACL 2021.
- Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning*. NeurIPS 2022.
