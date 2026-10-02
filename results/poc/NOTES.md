# Proof-of-concept notes

Phase 3 of the plan: all six training datasets, 5,000 examples each, 2 epochs, on the laptop.
Machine: Apple M4 Pro, 48 GB, MPS, torch 2.14, transformers 5.17, fp32.

## In-distribution results

Evaluation uses each dataset's own test or validation split, with **every** label of the dataset
as an option, in a fixed-seed shuffled order. Training only ever showed 2 to 16 options at once,
so banking77 (77 options) and TREC (50) are harder at evaluation than anything seen in training.

| Dataset | Options | n | Accuracy | Macro-F1 | Chance |
|---|---|---|---|---|---|
| ag_news | 4 | 1,000 | 0.884 | 0.884 | 0.25 |
| emotion | 6 | 1,000 | 0.857 | 0.789 | 0.17 |
| banking77 | 77 | 1,000 | 0.480 | 0.463 | 0.01 |
| trec | 50 | 500 | 0.716 | 0.531 | 0.02 |
| sst2 | 2 | 872 | 0.823 | 0.823 | 0.50 |
| mnli | 3 | 1,000 | 0.495 | 0.490 | 0.33 |

Where macro-F1 falls well below accuracy (emotion, TREC), the rare labels are the problem:
emotion's "surprise" had 189 training examples against 1,691 for "joy", and some TREC labels
had 4. MNLI is the weakest relative to chance; it is the only task where the answer depends on
comparing two texts, and 5,000 examples is a small slice of its 393k.

These are in-distribution numbers. The held-out results are below.

## Held-out results (Phase 4)

Same checkpoint, re-evaluated with `run.py --eval-only`. The six in-distribution numbers came back
identical to the ones above, so the saved checkpoint reproduces the trained model exactly.

| Dataset | Group | Options | n | Accuracy | Macro-F1 | Chance |
|---|---|---|---|---|---|---|
| yahoo_answers | new labels | 10 | 1,000 | 0.469 | 0.464 | 0.10 |
| massive | new labels | 60 | 1,000 | 0.325 | 0.312 | 0.017 |
| tweet_irony | new task | 2 | 784 | 0.446 | 0.439 | 0.50 (0.60 majority) |

**New labels work.** Yahoo is 4.7 times chance and MASSIVE 19 times chance, with all 60 intents
offered in one sequence, nearly four times the 16 options seen in training.

**Irony fails.** 0.446 is below a coin flip and below always answering "not ironic" (0.60). The
model says "ironic" for 582 of 784 tweets, though only 40% are ironic, and precision on "ironic"
(0.39) equals the base rate. The ranking tells the same story: the AUC of P(ironic) is 0.46,
against 0.89 for P(positive) on SST-2. The model finds no irony signal at all, and the sub-50%
accuracy comes from preferring the minority label. The loader was checked (0 is "non_irony" in
the dataset's own label names).

Wording doesn't rescue it. As a diagnostic only (the reported number stays on the canonical
wording, since picking wording on the test set would leak), five other instructions and option
pairs all gave 0.40 to 0.58. Each variant simply shifts which option the model prefers:
"literal / ironic" picks "ironic" 99% of the time, "sincere / sarcastic" picks "sincere" 86% of
the time. None beats the majority baseline.

So milestone M3 passes for new labels and fails for a new task type. Irony is a hard task:
nothing in training asks whether a text means the opposite of what it says, and sentiment
(SST-2, emotion) is the closest thing it saw. The baselines in Phase 7 will show whether
irony is hard for zero-shot models in general or only for this one.

### Position of the correct option

No first-option bias. On the small-option datasets accuracy is flat or slightly rising across
positions (AG News 0.87, 0.86, 0.89, 0.91 for positions 1 to 4; emotion 0.80 to 0.89). With many
options the picture is mixed: banking77 drops in the middle third of its 77 options (0.54, 0.30,
0.57 for first, middle and last third, about 330 examples each), while MASSIVE is flat (0.34,
0.30, 0.33) and TREC peaks in the middle (0.65, 0.78, 0.71). The banking77 dip is not explained
yet; the option-shuffling ablation (Phase 8) is the natural place to look again.

## Labels it never saw

The same five prompts, before (tracer, AG News only) and after (poc):

| Text | Options | tracer | poc |
|---|---|---|---|
| NASA's rover found traces of water... | space exploration, cooking, football, stock market | space exploration 31.6% | space exploration 98.1% |
| The striker scored twice... | same | cooking 40.7% | football 99.7% |
| My card was charged twice... | billing, bug report, praise | praise 98.5% | billing 97.5% |
| I absolutely loved this film... | positive, negative | negative 65.1% | positive 99.1% |
| The app crashes every time I open the settings page. | billing, bug report, praise | praise 95.6% | bug report 99.2% |

All five are right after Phase 3. The confidence (97 to 99.7%) is high for a model that is
wrong about half the time on MNLI, so calibration (Phase 5) is worth a close look. The held-out
numbers above are a sterner test: easy hand-written examples overstate what it can do.

## Run

3,750 steps in 3,155 s (52.6 min), 19.0 examples per second, peak MPS driver memory 11.5 GB.

The first attempt was stopped by Claude Code at step 860 because the whole system ran low on
memory. A 300-step check afterwards showed live tensor memory flat at 2.4 GB and the MPS
allocator cache levelling off at 9.7 GB, so training itself was not leaking; the likely cause
was a stale diagnostic process plus other applications. The run now clears the MPS cache every
200 steps and logs driver, tensor and system memory on every progress line.

Open question: the 300-step check ran at 30.6 examples per second, the full run at about 21.
Candidates are the cache clearing (buffers reallocated every 200 steps) and CPU contention from
other work on the machine. The latency benchmark in Phase 6 is the place to settle it.
