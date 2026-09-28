# Plan: undecided

> Source PRD: [README.md](../README.md). "Q1" to "Q5" are the five questions in *The idea*,
> "M0" to "M7" are the milestones.

## Architectural decisions

Durable decisions that apply across all phases:

- **Record schema**: every example, from any dataset, is
  `{instruction, text, options[], answer_idx, source, group}`. `group` is one of `train`,
  `heldout_labels`, `heldout_task`. Loaders emit raw records with all of a dataset's labels;
  augmentation (option sampling, shuffling, instructions) happens on the fly at training time.
- **Input format**: `[CLS] instruction [SEP] <opt> o1 <opt> o2 ... [SEP] text [SEP]`. The
  instruction segment may be empty. Truncation only ever cuts the text; an example whose
  options alone exceed the max length is an error, never silently truncated.
- **Key models**: `OptionScorer` (encoder plus `Linear(hidden, 1)` at each `<opt>` position,
  padded option slots masked to `-inf`) and `Undecided` (the user-facing wrapper with
  `from_pretrained()` and `classify(text, options, instruction=None) -> {option: probability}`).
  `<opt>` embedding is initialised to the mean of the existing embeddings.
- **Training loop**: plain PyTorch, no Trainer framework. Two AdamW parameter groups (head and
  `<opt>` embedding at 1e-3, rest of the encoder at 3e-5), warmup plus linear decay, gradient
  clipping, length-grouped batches with dynamic padding.
- **Runs are configs**: one YAML file per run. Every ablation and the cloud run are config
  differences, not code branches. A run is launched with a single command taking the config
  path.
- **Artefacts**: checkpoints go to `runs/<run-name>/` (gitignored). Each run writes
  `results/<run-name>/` with a copy of the resolved config, `metrics.json` and `plots/`, all
  committed to git. No external tracking service.
- **Metrics shape**: `metrics.json` holds per-dataset entries (`group`, `n`, `accuracy`,
  `macro_f1`, and later calibration, position and latency fields) plus per-group aggregates.
  Later phases add fields; they never rename existing ones.
- **Evaluation sets**: a fixed-seed sample per dataset, capped at about 2,000 examples, stored
  once and reused by every model, including baselines. Option order in evaluation is shuffled
  with a fixed seed too.
- **Device**: `cuda`, then `mps`, then `cpu`. On MPS: `reference_compile=False`,
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, fp32 by default.
- **Tests**: pytest, written test-first for the deterministic parts (dataset conversion,
  augmentation, tokenisation and `<opt>` positions, collation, model output shapes and masking,
  metrics). Model tests use a tiny randomly initialised ModernBERT config, so the suite needs no
  downloads and runs in seconds. Training quality is checked by runs, not unit tests.
- **Tooling**: `uv`, Python 3.12, `torch`, `transformers>=4.48`, `datasets`, `scikit-learn`,
  `matplotlib`, `pytest`.

---

## Phase 1: Tracer bullet, one dataset through every layer

**User stories**: M0, M2, part of M1

### What to build

The thinnest complete path: AG News is loaded and converted to the record schema, augmented
(option sampling and shuffling only, one fixed instruction), tokenised into the packed format,
collated, scored by `OptionScorer`, trained for a short run on MPS with the two-learning-rate
loop, evaluated on a fixed AG News validation sample, and written to `results/tracer/`.

The same phase measures throughput with a 200-step benchmark and proves the plumbing with an
overfit check on 64 examples. The benchmark result sets the dataset size for the proof of
concept.

### Acceptance criteria

- [ ] Project skeleton with `uv`, the package, the test suite and a gitignore for `runs/`
- [ ] Tests cover: AG News conversion to the record schema, option sampling always includes the
      correct option, shuffling moves `answer_idx` with it, `<opt>` count in the tokenised input
      equals the option count, text truncation never removes options, model logits have shape
      `(batch, max_options)` with `-inf` on padded slots
- [ ] The model reaches near 100% training accuracy on 64 fixed examples
- [ ] A 200-step benchmark on MPS reports examples per second; the number and the chosen
      proof-of-concept dataset size are recorded in `results/tracer/`
- [ ] One command trains the tracer config and writes `config`, `metrics.json` (AG News
      accuracy clearly above 25% chance) to `results/tracer/`

---

## Phase 2: `classify()` on any checkpoint

**User stories**: README *Planned usage*

### What to build

`Undecided.from_pretrained(run_dir)` loads a checkpoint and tokeniser, and `classify()` returns
a probability per option for arbitrary text, options and instruction. A small command-line
entry point does the same from the terminal. Run on the tracer checkpoint, it already accepts
labels AG News never had, with poor accuracy.

### Acceptance criteria

- [ ] `classify()` returns a dict keyed by the given options, values summing to 1
- [ ] Each probability is mapped back to the right option name, covered by a test with a tiny
      model
- [ ] Too many options for the context length raises a clear error
- [ ] Works on `cuda`, `mps` and `cpu`

---

## Phase 3: All training data and the first real run

**User stories**: M1, M3, part of Q1

### What to build

Loaders for the remaining five training datasets (emotion, banking77, TREC-QC, SST-2, MNLI
recast with the hypothesis in the instruction), the full augmentation (label verbalisation and
paraphrases, instruction templates with 20% empty, 2 to 16 options, per-dataset caps), a
preview command that prints sample augmented examples and label distributions, and the
`poc` config. One full proof-of-concept training run on the laptop, evaluated in-distribution.

### Acceptance criteria

- [ ] Tests for each loader's conversion, label verbalisation (e.g. `alarm_set` to
      "alarm set"), MNLI recast, per-dataset caps and the instruction drop rate
- [ ] The preview command prints 5 augmented examples per dataset and the label distribution,
      and a manual read confirms labels match texts
- [ ] `results/poc/metrics.json` has in-distribution accuracy and macro-F1 for all six training
      datasets
- [ ] Run time and peak memory of the proof-of-concept run are recorded

---

## Phase 4: Held-out evaluation

**User stories**: Q1, Q2, M3 exit check

### What to build

Loaders for Yahoo Answers topics, MASSIVE intents and tweet_eval irony, with their fixed
evaluation samples. Evaluation reports every metric per dataset and per group, plus accuracy
by correct-option position. A results command turns any set of `results/*/metrics.json` files
into the Markdown table used in the README.

### Acceptance criteria

- [ ] Tests for the three held-out conversions and for accuracy-by-position on a hand-made
      prediction set
- [ ] MASSIVE is evaluated with all 60 options in one sequence
- [ ] `results/poc/metrics.json` has held-out accuracy and macro-F1 for all three datasets, and
      all three are clearly above random (M3 exit)
- [ ] The results command prints a README-ready table from existing results

---

## Phase 5: Calibration

**User stories**: Q3

### What to build

Expected calibration error (15 bins), Brier score and log loss for every dataset, reliability
diagrams per dataset, and the temperature-scaling experiment: a global T fitted on the training
datasets' validation samples, and a per-dataset T fitted on 200 labelled examples from each
held-out dataset (disjoint from its evaluation sample).

### Acceptance criteria

- [ ] Tests for ECE, Brier and log loss against hand-computed values, and for temperature
      fitting recovering a known T on synthetic logits
- [ ] `metrics.json` holds calibration metrics raw, with global T and with per-dataset T
- [ ] Reliability diagrams for every dataset in `results/poc/plots/`
- [ ] A short written finding: does global T fix held-out calibration, or only per-dataset T?

---

## Phase 6: Latency

**User stories**: Q4

### What to build

A latency benchmark on a trained checkpoint at batch size 1 with 2, 10 and 60 options, on MPS
and CPU, after warmup, reporting p50 and p95.

### Acceptance criteria

- [ ] Latency figures written into `metrics.json` for both devices and all three option counts
- [ ] The results table shows p50 latency at 10 options
- [ ] Machine details (chip, memory, torch version) recorded next to the numbers

---

## Phase 7: Baselines

**User stories**: Q1, M4

### What to build

Random and majority-class baselines, `ModernBERT-base-zeroshot-v2.0`,
`deberta-v3-base-zeroshot-v2.0` and GLiClass modern base, each scored on the same stored
evaluation samples through the same metrics code, writing their own `results/<baseline>/`.
Their latency is measured with the Phase 6 benchmark.

### Acceptance criteria

- [ ] Every baseline has a `metrics.json` with the same fields as `poc`, where applicable
- [ ] The NLI baselines use one pass per option, as they are designed to
- [ ] The results table compares `poc` with all baselines
- [ ] The README results table is filled in for the proof of concept (M4 exit)

---

## Phase 8: Encoder ablations

**User stories**: Q5

### What to build

Two extra configs: head only (encoder frozen, only the head and `<opt>` embedding train) and
LoRA (small adapters on a frozen encoder). Both run through the same loop and evaluation.

### Acceptance criteria

- [ ] Head-only and LoRA runs each have full results
- [ ] Trainable parameter counts recorded for all three variants
- [ ] The results table has the head-only and LoRA rows next to the full fine-tune

---

## Phase 9: Design ablations

**User stories**: M5

### What to build

Configs for option shuffling off, a maximum of 4 training options against 16, no instructions,
and option pooling by the mean of each label's own tokens. Pooling needs label span offsets
from the collator, added behind a config switch.

### Acceptance criteria

- [ ] Test for label span offsets and mean pooling on a hand-made batch
- [ ] Each ablation has full results, including the ones that change nothing
- [ ] An ablation table in the README with the question each one answers and its outcome

---

## Phase 10: Optional full run in the cloud

**User stories**: M6

### What to build

The `full` config (about 200k examples, bf16) on a rented GPU, with the same code. A second look
at the backbone choice, informed by the proof-of-concept results.

### Acceptance criteria

- [ ] The same command runs unchanged on `cuda`
- [ ] Full-run results committed and added as a row in the results table
- [ ] Cost and wall-clock time of the run recorded
- [ ] A short note on whether a different backbone is worth trying

---

## Phase 11: Write-up

**User stories**: M7

### What to build

Rewrite the plan sections of the README as what actually happened: final results, plots,
answers to Q1 to Q5, and what didn't work.

### Acceptance criteria

- [ ] Every question in *The idea* has an answer backed by a number or plot in `results/`
- [ ] The status note no longer says "plan"
- [ ] Failed or surprising results are reported, not dropped
