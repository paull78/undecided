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

These are in-distribution numbers. Whether the model generalises to new labels is Phase 4.

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
wrong about half the time on MNLI, so calibration (Phase 5) is worth a close look.

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
