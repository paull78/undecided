# Plan: undecided

> Source PRD: [README.md](../README.md). "Q1" to "Q5" are the five questions in *The idea*,
> "M0" to "M7" are the milestones.

## Progress

| Phase | Status | Commit |
|---|---|---|
| 1. Tracer bullet | done 2026-09-29: tracer accuracy 0.863, 27.5 examples/s on MPS | `7152d57` |
| 2. `classify()` on any checkpoint | done 2026-09-29: save/load, `classify()`, CLI; `cuda` untested | `12815bb` |
| 3. All training data and the first real run | done 2026-09-29: in-distribution accuracy 0.48 to 0.88, [notes](../results/poc/NOTES.md) | `9f61d0b` |
| 4. Held-out evaluation | done 2026-09-29: Yahoo 0.469, MASSIVE 0.325 (60 options), irony 0.446, below chance; M3 exit partly met | `1a8b3bc` |
| 4b. Unseen task types | step 1 done 2026-10-02 (`4d2bf01`): all four unseen task types fail (AUC 0.37 to 0.56); step 2 first attempt stuck at chance and stopped; cause found (diluted many-option tasks), fix to choose | |
| 5. Calibration | after 4b | |
| 6 to 11 | not started | |

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
  The `<opt>` vector is its own parameter, written into the input embeddings at the marker
  positions (no row in the vocabulary), initialised to the mean pretrained embedding.
- **Training loop**: plain PyTorch, no Trainer framework. Two AdamW parameter groups (head at
  1e-3; encoder and `<opt>` vector at 3e-5, because `<opt>` at 1e-3 collapsed the encoder in
  Phase 1), warmup plus linear decay, gradient clipping, dynamic padding. Length-grouped batches
  arrive in Phase 3, when datasets with different text lengths are mixed.
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
- **Device**: `cuda`, then `mps`, then `cpu`. On MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1`, fp32 by
  default. (transformers 5 no longer has ModernBERT's `reference_compile` option.)
- **Tests**: pytest, written test-first for the deterministic parts (dataset conversion,
  augmentation, tokenisation and `<opt>` positions, collation, model output shapes and masking,
  metrics). Model tests use a tiny randomly initialised ModernBERT config, so the suite runs in
  seconds; the only download is the ModernBERT tokeniser (a few MB, cached after the first run). Training quality is checked by runs, not unit tests.
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

- [x] Project skeleton with `uv`, the package, the test suite and a gitignore for `runs/`
- [x] Tests cover: AG News conversion to the record schema, option sampling always includes the
      correct option, shuffling moves `answer_idx` with it, `<opt>` count in the tokenised input
      equals the option count, text truncation never removes options, model logits have shape
      `(batch, max_options)` with `-inf` on padded slots
- [x] The model reaches near 100% training accuracy on 64 fixed examples
- [x] A 200-step benchmark on MPS reports examples per second; the number and the chosen
      proof-of-concept dataset size are recorded in `results/tracer/`
- [x] One command trains the tracer config and writes `config`, `metrics.json` (AG News
      accuracy clearly above 25% chance) to `results/tracer/`

---

## Phase 2: `classify()` on any checkpoint

**User stories**: README *Usage*

### What to build

`Undecided.from_pretrained(run_dir)` loads a checkpoint and tokeniser, and `classify()` returns
a probability per option for arbitrary text, options and instruction. A small command-line
entry point does the same from the terminal. Run on the tracer checkpoint, it already accepts
labels AG News never had, with poor accuracy.

### Acceptance criteria

- [x] `classify()` returns a dict keyed by the given options, values summing to 1
- [x] Each probability is mapped back to the right option name, covered by a test with a tiny
      model
- [x] Too many options for the context length raises a clear error
- [ ] Works on `cuda`, `mps` and `cpu` (`mps` and `cpu` verified, identical to 5e-7; `cuda`
      waits for the Phase 10 cloud run)

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

- [x] Tests for each loader's conversion, label verbalisation (e.g. `alarm_set` to
      "alarm set"), MNLI recast, per-dataset caps and the instruction drop rate
- [x] The preview command prints 5 augmented examples per dataset and the label distribution,
      and a manual read confirms labels match texts
- [x] `results/poc/metrics.json` has in-distribution accuracy and macro-F1 for all six training
      datasets
- [x] Run time and peak memory of the proof-of-concept run are recorded

---

## Phase 4: Held-out evaluation

**User stories**: Q1, Q2, M3 exit check

### What to build

Loaders for Yahoo Answers topics, MASSIVE intents and tweet_eval irony, with their fixed
evaluation samples. Evaluation reports every metric per dataset and per group, plus accuracy
by correct-option position. A results command turns any set of `results/*/metrics.json` files
into the Markdown table used in the README.

### Acceptance criteria

- [x] Tests for the three held-out conversions and for accuracy-by-position on a hand-made
      prediction set
- [x] MASSIVE is evaluated with all 60 options in one sequence
- [ ] `results/poc/metrics.json` has held-out accuracy and macro-F1 for all three datasets, and
      all three are clearly above random (M3 exit). *Partly:* Yahoo and MASSIVE pass; irony
      scores 0.446, with no signal (AUC 0.46), see [notes](../results/poc/NOTES.md)
- [x] The results command prints a README-ready table from existing results

---

## Phase 4b: Unseen task types

Added on 2026-10-02 after Phase 4, at the user's request, to mitigate the irony failure before
calibration.

**User stories**: Q2, M3 exit check

### Finding that triggered it

On tweet_eval irony, the only held-out *task type*, the poc model scores 0.446: below a coin flip
and below always answering "not ironic" (0.60). It answers "ironic" for 74% of tweets, precision
on "ironic" equals the base rate, and the AUC of P(ironic) is 0.46 (SST-2: 0.89), so it carries no
irony signal at all. Rewording the instruction or the options (diagnostic only) never beat the
majority baseline. New *labels* generalise well (Yahoo 4.7 times chance, MASSIVE 19 times chance),
so the gap is specific to new task types. Details in [the poc notes](../results/poc/NOTES.md).

### Decisions

- **Mitigate before Phase 5.** Calibration has to be measured on the model that is kept; fixing
  afterwards would mean redoing Phase 5.
- **Diagnose before fixing.** One held-out task cannot tell "new task types fail" from "irony
  is too hard". Irony needs reading the opposite of the literal words, on noisy tweets. Step 1
  adds more held-out task types, evaluated on the existing checkpoint, to decide whether step 2
  is needed.
- **Held-out stays held out.** Nothing resembling irony (sarcasm datasets, for example) goes into
  training, and the step 1 datasets become permanent held-out sets, never trained on.
- **Rejected mitigations:**
  - Correcting the label prior at inference (e.g. contextual calibration) only moves the
    threshold; with AUC 0.46 there is no signal to recover, so the best case is the majority
    baseline.
  - Tuning instruction or option wording on the test set would leak, and the diagnostic showed
    it doesn't help anyway.
  - A larger backbone (ModernBERT-large) may help with pragmatics but is slow on the laptop and
    leaves the narrow task mix untouched; it belongs to the optional cloud run (Phase 10).
  - Per-task fine-tuning with a few hundred labels is not zero-shot. It stays a documented
    fallback for users, not a fix.

### Step 1: more held-out task types (no training)

Three task types absent from training, picked to separate task from domain:

| Dataset | Task | Options | Why |
|---|---|---|---|
| `ucirvine/sms_spam` | spam detection | spam, not spam | new task, new domain; 13% spam, so macro-F1 matters |
| `cardiffnlp/tweet_eval` (`hate`) | hate speech | hate speech, not hate speech | tweets like irony: separates "new task" from "Twitter text" |
| `SetFit/subj` | subjectivity | objective, subjective | new task, familiar domain (film sentences, like SST-2) |

SMS spam has only a train split, so its evaluation sample comes from it; the 200 examples for
its per-dataset temperature in Phase 5 must be drawn disjoint from that sample.

Outcome rule: if most of these land well above their majority baseline, irony is the outlier,
the write-up says so, and Phase 5 follows. If they fail too, the model has only learned its six
training task types, and step 2 runs.

### Step 1 outcome (2026-10-02): general failure

Same poc checkpoint, `--eval-only`. The nine earlier datasets reproduced exactly. All four unseen
task types score below their majority baseline, and none ranks better than chance:

| Dataset | Accuracy | Macro-F1 | Majority | Predicted positive / actual | AUC |
|---|---|---|---|---|---|
| tweet_irony | 0.446 | 0.439 | 0.603 | 0.74 / 0.40 | 0.46 |
| sms_spam | 0.839 | 0.511 | 0.851 | 0.03 / 0.15 | 0.56 |
| tweet_hate | 0.494 | 0.493 | 0.563 | 0.59 / 0.44 | 0.51 |
| subjectivity | 0.470 | 0.404 | 0.511 | 0.13 / 0.49 | 0.37 |
| sst2 (trained, for reference) | 0.823 | 0.823 | 0.509 | 0.54 / 0.51 | 0.89 |

So irony is not an outlier: the model learned its six training task types, not to follow an
instruction for a new one. Subjectivity ranks clearly *below* chance; the label mapping was
checked (0 is "objective"), and a plausible reading is that SUBJ's objective sentences are plot
summaries full of charged words ("villainous", "horrifying murders"), which the model takes for
opinion. Step 2 runs.

### Step 2: task diversity (one retrain, about an hour)

The likely cause is the narrow training mix: six task types, mostly "match the text to a label
name", so the instruction carries little weight. Keeping the total at about 30k examples:

- More training task types, none resembling a held-out task. Proposed after checking that each
  loads:

  | Dataset | Task | Options |
  |---|---|---|
  | `nyu-mll/glue` (`qqp`) | are two questions duplicates? | duplicate, not duplicate |
  | `nyu-mll/glue` (`qnli`) | does the sentence answer the question? | answers it, does not answer it |
  | `nyu-mll/glue` (`cola`) | is the sentence grammatical? | grammatical, ungrammatical |
  | `google/boolq` | yes/no question about a passage | yes, no |

  Dropped from the first list: toxicity (too close to the hate-speech held-out set) and stance
  (`tweet_eval` stance has a few hundred tweets per target, and opinion-towards-a-target is close
  to the subjectivity held-out set). Fact checking needs a loader-script-free source; PAWS
  (paraphrase) loads but duplicates QQP's task type.
- Yes/no recasting as an augmentation: some training examples become a property check against
  one of their labels, either "yes / no" or "<label> / not <label>" (the shape of every binary
  held-out task, e.g. "ironic / not ironic"), with the true label and a distractor equally
  often, so the answer is balanced. This also feeds the weak NLI task.
- Mix: the six current datasets at 3,000 each plus the four new ones at 3,000 each (30k, 2 epochs,
  the same budget as `poc`), saved as a new run so `poc` stays as the baseline.

The new run is compared with `poc` on every in-distribution and held-out dataset, to check that
held-out tasks improve without in-distribution scores dropping.

### Step 2, first attempt (2026-10-12): stuck at chance, stopped

`configs/diverse.yaml` as above trained at 29.6 examples per second, but its loss stayed near
uniform guessing (1.003 for this mix): 0.94 averaged over steps 1,200 to 2,000, flat for the last
800 steps while the learning rate was already decaying. It was stopped at step 2,000 of 3,750 (log
kept as `runs/diverse-v1-stopped.log`, no checkpoint).

Diagnosis, with per-dataset loss now logged every 200 steps against each dataset's own chance
level, ln(number of options):

- Diverse, steps 200 to 1,000: every dataset at chance, including AG News (0.99 against 0.98) and
  SST-2 (0.70 against 0.69). Only CoLA, TREC and emotion dip, which label priors alone explain.
- Control, the `poc` config on the current code: training still works, and it shows how. AG
  News and SST-2 are *also* at chance at step 1,000. The plateau breaks on banking77 (loss 1.83,
  1.04, 0.76 at steps 600, 800, 1,000, against 2.07 for guessing), with TREC next. These are the
  many-option tasks where label words appear in the text ("card arrival", "my card hasn't
  arrived"); the model learns that matching first and everything else builds on it.

Cause: the diverse mix diluted the many-option examples that start learning. Multi-option
banking77 went from 1 in 6 training examples (5,000 of 30,000) to 1 in 13 (3,000, a quarter of
them recast to yes/no), and TREC the same. In the diverse probe banking77 never left chance (1.69
against 1.68 at step 1,000). This also explains the "slow start" seen in the tracer and in `poc`:
it is the time to discover word matching between options and text.

### Acceptance criteria

- [x] Tests for the three new held-out conversions
- [x] `results/poc/metrics.json` has all six held-out datasets, with majority baselines next to
      chance in the notes
- [x] A written outcome for step 1: irony outlier or general failure, and whether step 2 runs
- [ ] If step 2 runs: a new run with its own results folder, compared with `poc` on every
      dataset, and the decision on which checkpoint Phase 5 uses

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
