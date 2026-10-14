# Plan: undecided

> Source PRD: [README.md](../README.md). "Q1" to "Q5" are the five questions in *The idea*,
> "M0" to "M7" are the milestones of the original README.

> **Closed on 2026-10-14.** Interest in Jev-style classifiers had dropped, so the project was
> wrapped up after Phase 4b as a learning project. Phases 1 to 4b are done; phases 5 to 11 were
> not run, apart from the parts listed in the table. The README is now the write-up, with the
> results, takeaways and the ideas these phases would have tested.

## Progress

| Phase | Status | Commit |
|---|---|---|
| 1. Tracer bullet | done 2026-09-29: tracer accuracy 0.863, 27.5 examples/s on MPS | `7152d57` |
| 2. `classify()` on any checkpoint | done 2026-09-29: save/load, `classify()`, CLI; `cuda` untested | `12815bb` |
| 3. All training data and the first real run | done 2026-09-29: in-distribution accuracy 0.48 to 0.88, [notes](../results/poc/NOTES.md) | `9f61d0b` |
| 4. Held-out evaluation | done 2026-09-29: Yahoo 0.469, MASSIVE 0.325 (60 options), irony 0.446, below chance; M3 exit partly met | `1a8b3bc` |
| 4b. Unseen task types | done 2026-10-14: unseen task types fail in general; the diverse mix stalled from scratch (diluted many-option tasks) and, warm-started, did not help unseen tasks against a control | `4d2bf01`, `9ade41a`, `bb4e586` |
| 5. Calibration | partly: calibration error and mean confidence for every dataset; temperature scaling not run | `bb4e586` |
| 6. Latency | partly: single-request p50/p95 at 2, 10 and 50 options on MPS and CPU | closing commit |
| 7. Baselines | not done, project closed | |
| 8. Encoder ablations | not done, project closed | |
| 9. Design ablations | not done, project closed | |
| 10. Optional full run in the cloud | not done, project closed | |
| 11. Write-up | done 2026-10-14: README rewritten as the write-up | closing commit |
| Closing day | done 2026-10-14: steps 0 to H; README updated | closing-day commits |

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
against 1.68 at step 1,000). This also explains the long flat start of `poc`: it is the time
to discover word matching between options and text. (The tracer's slow start, on AG News alone,
is not explained by this.)

### Step 2, second attempt (2026-10-14): warm start, against a control

Two runs start from the `poc` checkpoint, which already matches options to text, and train one
more epoch (1,875 steps, about 17 minutes each) with seed 0, so their evaluation samples match
`poc` exactly:

- `diverse-warm`: the ten-task mix with `recast_rate` 0.25.
- `poc-warm`, the control: `poc`'s own six tasks, no recasting.

The warm start removed the stall: at step 200 the six original tasks were far below chance, and
QNLI and QQP started learning within 400 steps. The comparison (full tables in the README):

| | `poc` | `poc-warm` | `diverse-warm` |
|---|---|---|---|
| Six training tasks, mean accuracy | 0.709 | 0.782 | 0.752 |
| Yahoo and MASSIVE, mean accuracy | 0.397 | 0.450 | 0.425 |
| Four unseen task types, mean macro-F1 | 0.462 | 0.455 | 0.429 |
| Four unseen task types, mean AUC | 0.51 | 0.52 | 0.51 |

Task diversity at this scale did not help unseen task types, and the control was better on
everything else. `diverse-warm`'s higher irony accuracy (0.614) is majority-class drift: it
answers "not ironic" for 88% of tweets, "not spam" for every message and "not hate speech" for
95%. Of the new training tasks, QNLI (AUC 0.89) and QQP (0.83) were learned, BoolQ weakly (0.65),
CoLA not at all (0.49, majority answer every time). An observation made mid-run, that the pair
tasks helped MNLI, turned out wrong: the control reached the same MNLI accuracy (0.608).

Decision: `poc-warm` is the final checkpoint, and the project closes here instead of moving to
Phase 5.

### Acceptance criteria

- [x] Tests for the three new held-out conversions
- [x] `results/poc/metrics.json` has all six held-out datasets, with majority baselines next to
      chance in the notes
- [x] A written outcome for step 1: irony outlier or general failure, and whether step 2 runs
- [x] If step 2 runs: a new run with its own results folder, compared with `poc` on every
      dataset, and the decision on which checkpoint Phase 5 uses (`poc-warm`; Phase 5 not run)

---

## Closing day: the questions still open

Added on 2026-10-14, after the write-up, at the user's request: one day of unattended work to
close the project on stronger evidence. It takes the most informative parts of phases 5 to 8,
cheapest first, so a stop at any point still leaves finished, committed results.

**User stories**: Q2, Q3, Q4, Q5

### Rules for running unattended

- Order: inference-only work first (minutes of compute), training runs after.
- Every step ends with its tests passing, its results under `results/`, the README and this plan
  updated, and a commit. Nothing is pushed or published.
- Each step has a time box. A step that fails twice, or runs past its box, is recorded here as
  not done, with the reason, and the day moves on.
- `runs/poc` and `runs/poc-warm` are never overwritten. Every new run gets its own name.
- Before each training run: check free memory, and that no other run is active. Watch the
  per-dataset loss and stop a run that is still at chance by step 1,000 (the stall rule from
  Phase 4b).
- New dependencies go into `pyproject.toml` only if they install cleanly next to transformers
  5.17. Anything that conflicts runs in a separate throwaway environment, or is dropped.
- Findings are reported as measured, including the ones that change nothing.

### Step 0: save intermediate results, and guard the machine (1.5 h)

Added at the user's request before starting. The `poc-warm` run was lost once because the session
ended during its evaluation, after training had finished: `run.py` saved the checkpoint only after
evaluating.

- Save the trained model before evaluating it.
- Write each dataset's metrics as soon as they exist (`results/<run>/metrics.partial.json`), and
  skip datasets already done when a run or `--eval-only` is repeated.
- Mid-training checkpoints every 500 steps (weights, optimiser, schedule, step, random state,
  loss history), written atomically and overwritten each time, and `run.py --resume` to continue
  from the last one with the same data order and augmentation. Skipped batches replay their
  augmentation so the random sequence stays identical. The file is deleted once the final model
  is saved.
- Safeguards for the day: an MPS memory cap (24 GB) so a runaway run fails alone instead of
  pushing the system into swap, a watchdog that stops training if free system memory falls below
  15%, and `caffeinate` against idle sleep.

- [x] Test: a run interrupted after a checkpoint and resumed gives the same losses as an
      uninterrupted run
- [x] Test: evaluation skips datasets already in the partial file
- [x] Watchdog and `caffeinate` running before the first training step of the day

Done 2026-10-14. Also checked on the real model: a run killed at a checkpoint resumed and
finished; a run killed after training, during evaluation, skipped training and the datasets
already scored.

### Step A: does the model read the instruction? (inference only, 45 min)

The likeliest explanation for the unseen-task failure is that the model ignores instructions.
Re-score `poc-warm` on every dataset three ways: the canonical instruction, no instruction, and
a mismatched instruction (another dataset's canonical one). If accuracy barely moves, the model
classifies from options and text alone, which explains why "Is this tweet ironic?" can't steer
it.

- [x] Instruction variants in the evaluation path, with a test that the variant is applied
- [x] A table: accuracy and AUC per dataset under the three variants
- [x] One paragraph in the README findings (step H)

Result (2026-10-14, `results/poc-warm/probe-instructions.json`): the model ignores the
instruction. On the five single-text training datasets and the two new-label sets, removing it
or swapping in another dataset's moves accuracy by at most 0.012 (TREC 0.772, 0.762, 0.760). On
the unseen tasks, irony accuracy rises without its instruction (0.478 to 0.597, AUC 0.53 to
0.55); SMS spam is the only dataset where the instruction carries signal (AUC 0.68 with it,
0.58 without, 0.59 with the hate-speech instruction). A likely reason, not tested: in training
each task's instruction always arrives with that task's label set, so the options already
identify the task and the instruction adds nothing the model needs. That suggests training data
where the same text and options get different answers under different instructions.

### Step B: the calibration experiment (mostly inference, 1.5 h)

Phase 5 as planned, on `poc-warm`: temperature scaling, global against per-dataset.

- Global T: fitted by minimising log loss on fresh validation samples of the six training
  datasets (1,000 each, disjoint from the evaluation samples), then applied unchanged to the
  held-out datasets.
- Per-dataset T: fitted on 200 labelled examples from each held-out dataset's train split. SMS
  spam has a single split, so its 200 are drawn disjoint from its evaluation sample.
- Report calibration error, Brier score and log loss raw, with global T and with per-dataset T,
  plus a reliability diagram per dataset.

- [x] Tests: T fitting recovers a known T on synthetic logits; Brier and log loss against
      hand-computed values
- [x] `results/poc-warm/probe-calibration.json` and reliability diagrams in
      `results/poc-warm/plots/reliability.png`
- [x] The finding in the README: does global T fix held-out calibration, or only per-dataset T?
      This is the direct test of the Molas argument. (step H)

Result (2026-10-14): only per-dataset T. Global T = 1.18, fitted on 5,452 examples (1,000 per
training dataset; TREC had only 452 rows left unused by training). On the training datasets it
helps where they were off (TREC 0.081 to 0.044). On the six held-out datasets, mean calibration
error goes from 0.161 raw to 0.140 with global T and 0.058 with per-dataset T. The fitted
per-dataset temperatures disagree in direction, which is why no single T can work: SMS spam needs
0.55 (the model is underconfident there) while Yahoo needs 2.31 and hate speech 3.87.
Subjectivity's T runs into the search bound of 20, flattening its probabilities to near uniform,
and its error stays at 0.106: with an inverted ranking (accuracy falls as confidence rises), no
temperature fixes it. This is what the Molas argument predicts for a zero-shot model.

### Step C: option-order averaging (inference only, 45 min)

The README's next-step idea, tested. Score each evaluation example under 4 option orders and
average the probabilities. Report accuracy, AUC and calibration error against a single order,
and the cost (4 passes per request).

- [x] An averaging option in the evaluation path, with a test that it averages by option name
- [x] The comparison table (`results/poc-warm/probe-orders.json`)
- [x] The idea moved from "What to try next" to the findings (step H)

Result (2026-10-14): averaging 4 orders helps where there are many options and nowhere else.
Against the mean of the 4 single orders: banking77 0.642 to 0.669 accuracy (calibration error
0.060 to 0.031), TREC 0.774 to 0.792 (0.074 to 0.051), MASSIVE 0.419 to 0.440 (0.125 to 0.084).
Datasets with 2 to 6 options move by less than 0.01, and the unseen task types don't improve
(AUC unchanged within 0.01). Single orders differ by up to 0.018 accuracy on TREC. Cost: 4
forward passes per request. The earlier spam AUC of 0.56 came from one fixed order ("not spam,
spam"); random orders give 0.66 to 0.68.

### Step D: head only (one training run, 45 min)

Question 5, first half. Same data and steps as `poc` (3,750 steps from plain ModernBERT), with
the encoder and the `<opt>` vector frozen, so only `Linear(768, 1)` learns. Compared with `poc`,
not `poc-warm`, since both start from the pretrained backbone. The prediction is near chance:
the comparison between option and text happens inside the encoder.

- [x] `freeze_encoder` in the training config, with a test that encoder weights don't change
- [x] `configs/head-only.yaml`, run, results in `results/head-only/`

Result (2026-10-14): chance everywhere, as predicted. 769 trainable parameters, 3,750 steps in
6.6 minutes (152 examples per second, 1.1 GB), and the per-dataset loss never left chance.
Accuracy: AG News 0.253, banking77 0.013, TREC 0.048, SST-2 0.500 (AUC 0.50), MNLI 0.335,
Yahoo 0.119, MASSIVE 0.011. Emotion (0.337) and SMS spam (0.745) only reflect a fixed preference
for one label. A plausible reading: if the frozen state at each `<opt>` is roughly "label part
plus text part", a linear head's score difference between two options depends on the labels
alone, so it can learn a label prior and nothing that depends on the text.

### Step E: LoRA (one training run, 1.5 h)

Question 5, the middle point: small trainable adapters on the frozen encoder, via `peft`, same
data and steps as `poc`. Time-boxed; dropped if `peft` doesn't work with transformers 5.17 and
ModernBERT within the box.

- [x] LoRA option in the training config (rank 16 on the attention projections, head and `<opt>`
      vector trained), with a test that only adapter, head and `<opt>` parameters change
- [x] `configs/lora.yaml`, run, results in `results/lora/`, trainable parameter count recorded
- [x] Question 5 answered in the README: full fine-tune against LoRA against head only (step H)

Result (2026-10-14): in between, closer to full fine-tuning than to head only. 1,623,553
trainable parameters (about 1% of the model), 3,750 steps in 16.8 minutes (60 examples per
second, 9.9 GB). Mean accuracy on the six training datasets: head only 0.248, LoRA 0.454, full
fine-tune (`poc`) 0.709; new labels 0.065, 0.279, 0.397. The per-dataset loss broke the plateau
about 400 steps later than full fine-tuning (banking77 first again, around step 1,000 to 1,200),
and was still falling at the end, so the shared schedule probably undertrains LoRA; a longer run
or a higher adapter learning rate was not tried. Its biggest gaps are on the many-option datasets
(banking77 0.141 against 0.480 with all 77 options, although its training loss on 2 to 16
options reached 0.57) and on MNLI, which stayed at chance (0.349). Unseen task types: no signal
(AUC 0.46 to 0.51).

### Step F: baselines (inference only, 2.5 h)

Phase 7, cut down. Does an existing zero-shot classifier also fail on unseen task types, and how
much faster is one pass than one pass per option?

- `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`, same backbone, one NLI pass per option, with its
  own hypothesis template, on all 12 shared evaluation sets, plus the same latency benchmark at
  2, 10 and 50 options. MNLI is left out for it, since it was trained on NLI data.
- GLiClass (`knowledgator/gliclass-modern-base-v2.0-init`), packed options like this project.
  Time-boxed to 1 hour, in a separate environment if its package conflicts.

- [x] A baseline runner that writes the same `metrics.json` shape, so the results command and
      the chart work unchanged
- [x] Baseline rows in the README tables and the chart; latency side by side (step H)

Result (2026-10-14): the NLI baseline is far stronger on everything `undecided` didn't train on,
and much slower. Caveat found while writing up: its model card reports per-dataset results under
"Training results" for a mix that includes AG News, emotion, banking77, Yahoo topics, MASSIVE, a
spam dataset and two hate-speech datasets, so on those it is most likely not zero-shot. Irony and
subjectivity are not in that list. `ModernBERT-base-zeroshot-v2.0` (template "This example is {}.", no instruction):
Yahoo 0.645 and MASSIVE 0.701 against `poc-warm`'s 0.478 and 0.422; unseen task AUC irony 0.78,
SMS spam 0.99, subjectivity 0.63, hate speech 0.59 (against 0.53, 0.68, 0.34, 0.54). `undecided`
wins only on two of its own training datasets, emotion (0.901 against 0.763) and TREC (0.772
against 0.330, where question-type labels fit the template badly). Latency p50 on MPS at 2, 10
and 50 options: 15.6, 26.2 and 110.3 ms, against `undecided`'s 9.2, 11.6 and 23.2; on CPU 77.6,
362 and 1,645 ms against 18.7, 28.7 and 49.5. The CPU gap at 2 options (4 times, where two passes
predict about 2) suggests overhead in the baseline runner, not profiled.

GLiClass (`gliclass-modern-base-v2.0-init`, instruction passed as its prompt, scores normalised
per record) is uneven: near chance on the many-option datasets (banking77 0.091, TREC 0.094,
MASSIVE 0.050), weak on AG News (0.643) and emotion (0.271), the best of all models on
subjectivity (0.819, AUC 0.96), good on SST-2 (0.866). Not investigated within the time box:
whether the "-init" checkpoint is meant to be fine-tuned further, and how it handles more than
the 25 classes it defaults to. Its MPS and CPU timings match (18.9 and 19.6 ms at 2 options,
48.2 and 48.6 at 50), so the device setting may not have taken effect.

### Step G: seed variance (one short training run, 30 min)

The README warns that one seed per model can't separate small differences. Repeat the
`poc-warm` step from `poc` with training seed 1, keeping the evaluation samples fixed (a separate
evaluation seed), to show how much a second seed moves each number.

- [x] `eval_seed` in the config, defaulting to `seed` so earlier results reproduce (with
      `data_seed` too, so the training examples stay the same)
- [x] `configs/poc-warm-seed1.yaml`, run, and the spread reported next to the main table

Result (2026-10-14): a second seed moves group averages by about 0.02 (training tasks 0.816 to
0.796, new labels 0.450 to 0.428) and single many-option datasets by more (banking77 0.072,
MASSIVE 0.045). So `poc-warm`'s lead over `diverse-warm` on new labels is within seed noise; the
large differences (unseen-task failure, head only and LoRA against full fine-tuning, the
baselines) stand. Unseen task AUC averages 0.51 for the second seed too.

### Step H: write-up (1 h)

- [x] README: findings, results tables, chart and questions 3 to 5 updated; "What to try next"
      trimmed to what is still untested
- [x] This plan: each step marked done or not done, with the reason; memory updated
- [x] Final commit and a notification

Done 2026-10-14. Before committing, a fact-check workflow (4 section checkers, each reported
issue then tested by an independent refuter) compared every number and claim in the README with
the results files: 17 issues reported, 16 confirmed and fixed (among them "a quarter of the
examples" recast where it was 15% overall, an overstated tracer accuracy jump, a latency claim
true only on the GPU, and subjectivity "below chance in every model" where LoRA sat at chance).
Writing up also surfaced the NLI baseline's training mix, recorded under step F.

Rough total: about 10.5 hours, of which about 3.5 are compute.

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
