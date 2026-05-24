# COT_lab

Small-scale CoT knowledge distillation on GSM8K with FLAN-T5-base, evaluated jointly with final-answer accuracy and ReCEval reasoning-quality scores. Extended with a SVAMP transfer experiment and a FLAN-T5-large scale ablation.

Reproduces Ho et al. (ACL 2023, *Large Language Models Are Reasoning Teachers*); adds Prasad et al. (EMNLP 2023, *ReCEval*) as a reasoning-quality axis.

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

Each stage is one script. All hyperparameters live in [config/config.yaml](config/config.yaml). Data and outputs are gitignored.

### GSM8K (main experiment)

```bash
bash scripts/01_download.sh        # GSM8K + Ho et al. teacher CoTs
bash scripts/02_filter.sh          # Build Set A / B / C / Direct FT
bash scripts/03_train_direct_ft.sh # Fine-tune (one script per condition)
bash scripts/03_train_set_a.sh
bash scripts/03_train_set_b.sh
bash scripts/03_train_set_c.sh
bash scripts/04_inference.sh       # All 5 conditions on the GSM8K test set
bash scripts/05a_accuracy.sh       # Final-answer accuracy
bash scripts/05b_receval.sh        # Intra / inter / informativeness
```

### SVAMP (transfer experiment)

Same A/B/C/Direct-FT design, separate training sets and output directories.

```bash
bash scripts/02_filter_svamp.sh
bash scripts/03_train_svamp_direct_ft.sh
bash scripts/03_train_svamp_set_a.sh
bash scripts/03_train_svamp_set_b.sh
bash scripts/03_train_svamp_set_c.sh
bash scripts/04_inference_svamp.sh
bash scripts/05a_accuracy_svamp.sh
```

### Utilities

```bash
python -m src.status               # Project-wide state from outputs/runs/*.json
python -m src.utils.cleanup_ckpts  # Prune intermediate epoch checkpoints
pytest tests/                      # Unit tests
```

## Notebooks

- [notebooks/Kaggle.ipynb](notebooks/Kaggle.ipynb) — full GSM8K pipeline orchestrator for Kaggle (T4).
- [notebooks/Kaggle_svamp.ipynb](notebooks/Kaggle_svamp.ipynb) — SVAMP variant of the same orchestrator.
- [notebooks/results.ipynb](notebooks/results.ipynb) — loads `outputs/eval_results/` and renders the final tables and plots.

## Docs

- [doc/RESEARCH.md](doc/RESEARCH.md) — research question, hypotheses, scope, experimental matrix.
- [doc/pipeline.md](doc/pipeline.md) — teacher, training sets, and per-stage walkthrough.
- [doc/Related work.md](doc/Related%20work.md) — paper notes.

## Layout

```
config/     single source of truth for hyperparameters
src/        data, train, inference, eval, utils
  eval/receval/   intra / inter / informativeness scorers
scripts/    one shell wrapper per stage (GSM8K + SVAMP variants)
notebooks/  Kaggle orchestrators and results rendering
tests/      pytest unit tests
doc/        research plan, pipeline, findings, related work
outputs/    runs/, generations/, eval_results/, plots/, checkpoints/ (gitignored)
```
