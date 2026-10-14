# undecided

A small, open reproduction of the idea behind Jev: a classifier that never generates text and
instead returns probabilities over options you choose at request time. This version is built on
[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M parameters) with a
linear head that scores each option, and it was fine-tuned on a laptop GPU.

> **Status:** complete. This is a learning project, closed on 2026-10-14. The model works for new
> labels on familiar kinds of task, answers in about 12 ms at 10 options on a laptop, and fails on
> kinds of task it never trained on. The numbers, what went wrong and what I would try next are
> below. The day-by-day record is in the [implementation plan](plans/undecided.md).
>
> Not affiliated with TypeSafe. "Jev" refers to their product and is used here only to
> describe the behaviour being reproduced.

## In short

- **The interface works.** Text, a list of options and an optional instruction go in; one
  probability per option comes out, from a single forward pass. A request takes 9 ms with 2
  options, 12 ms with 10 and 23 ms with 50 on an M4 Pro GPU.
- **New labels transfer, new kinds of task don't.** The final model scores 4.8 times chance on
  Yahoo Answers topics and 25 times chance on MASSIVE's 60 assistant intents, none of which it
  saw in training. On irony, spam, hate speech and subjectivity, task types it never trained on,
  its ranking of texts is weak or worse than random (AUC 0.34 to 0.68, where 0.5 is a coin flip).
- **Ten training task types weren't enough.** Adding four more task types and yes/no versions of
  every task did not help on the unseen ones. The same amount of extra training on the original
  six tasks did as well or better everywhere else.
- **Learning starts with word overlap.** The model first learns to match option words to the
  text, on many-option intent datasets. A training mix that diluted those datasets stayed at
  chance for 2,000 steps.
- **Calibration doesn't carry over to new data.** Confidence matches accuracy on the training tasks (calibration
  error 0.02 to 0.08) and drifts on new data: on Yahoo the final model is 72% sure on average and
  right 48% of the time.

![Ranking AUC per two-option dataset, for the three models](results/auc.png)

## What Jev does

According to the public write-ups ([MindStudio](https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark),
[Towards Data Science](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/),
[XenoSpectrum](https://xenospectrum.com/en/jev-typesafe-bert-classifier-decomposition/)):

- TypeSafe calls it a "System One" model. It returns probabilities over the options you define
  and does not produce text or a chain of thought.
- It answers in a single pass, in 70 to 500 ms.
- It is trained with reinforcement learning for calibrated decisions (RL-CD), meant to make a
  90% confidence score correspond to 90% real accuracy.
- In published comparisons, locally trained classifiers win on raw accuracy for a fixed task,
  and Jev wins on zero-shot flexibility and on instructions that change at runtime.

The calibration claim is disputed. [Alex Molas argues](https://www.alexmolas.com/2026/09/23/jev-cant-be-calibrated.html)
that calibration depends on the data distribution, which a zero-shot model never sees, so
Jev's probabilities can't be calibrated for your data.

## The idea

The target is Jev's interface. Its training recipe and architecture stay out of scope, and
what gets reproduced is the contract:

```text
input:   text, a list of options, an optional instruction
output:  one probability per option, summing to 1
cost:    one forward pass of a small encoder
```

The case study set out to answer five questions. Three got answers:

1. **How much zero-shot behaviour does a 149M encoder fine-tuned on public classification
   datasets get?** Some. It handles new labels for kinds of task it trained on, and nothing
   beyond that.
2. **How well does it handle labels and task types it never saw?** Labels well, task types not
   at all. See [Results](#results).
3. **Are its probabilities calibrated?** On the training tasks, yes; on new data, no. The planned
   temperature-scaling experiment was not run, so whether a global fix helps is still open.
4. **How fast is it, and how does latency grow with the number of options?** 12 ms at 10
   options, about 2.5 times the 2-option cost at 50 options.
5. **What does fine-tuning the encoder buy compared with training only the head?** Not measured;
   the ablations were dropped when the project closed.

## Results

### The three models

| Model | Starts from | Training data | Steps | Time on the laptop |
|---|---|---|---|---|
| `poc` | ModernBERT-base | 6 datasets, 5,000 examples each, 2 epochs | 3,750 | 53 min |
| `poc-warm` (final) | `poc` | the same 6 datasets, 1 more epoch | 1,875 | 17 min |
| `diverse-warm` | `poc` | 10 datasets, 3,000 each, a quarter asked as yes/no checks, 1 epoch | 1,875 | 17 min |

`poc-warm` is the control for `diverse-warm`: same start, same number of steps, only the task mix
differs. It turned out to be the best model, so it is the one the usage examples load.

### Summary

Every dataset counts once in each average, whatever its size. For the unseen task types the
average accuracy is misleading (see [per dataset](#per-dataset)), so the table also gives the
average ranking AUC, where 0.5 means no signal.

| Model | Training tasks, accuracy | New labels, accuracy | Unseen task types, macro-F1 | Unseen task types, AUC |
|---|---|---|---|---|
| `poc` | 0.709 | 0.397 | 0.462 | 0.51 |
| `poc-warm` | **0.782** | **0.450** | 0.455 | 0.52 |
| `diverse-warm` | 0.752 | 0.425 | 0.429 | 0.51 |

"Training tasks" are the six datasets all three models trained on.

### Per dataset

Accuracy and macro-F1. Every evaluation offers all of a dataset's labels at once, in a shuffled
order fixed by the seed, so MASSIVE has 60 options in one sequence although training never showed
more than 16.

| Dataset | Options | Chance or majority | `poc` | `poc-warm` | `diverse-warm` |
|---|---|---|---|---|---|
| **Trained by all three** | | | | | |
| AG News topics | 4 | 0.25 | 0.884 / 0.884 | 0.899 / 0.899 | 0.886 / 0.886 |
| emotion | 6 | 0.17 | 0.857 / 0.789 | 0.901 / 0.867 | 0.870 / 0.813 |
| banking77 intents | 77 | 0.01 | 0.480 / 0.463 | 0.651 / 0.637 | 0.559 / 0.553 |
| TREC question types | 50 | 0.02 | 0.716 / 0.531 | 0.772 / 0.581 | 0.748 / 0.523 |
| SST-2 sentiment | 2 | 0.51 | 0.823 / 0.823 | 0.858 / 0.858 | 0.842 / 0.842 |
| MNLI entailment | 3 | 0.33 | 0.495 / 0.490 | 0.608 / 0.606 | 0.608 / 0.602 |
| **Trained by `diverse-warm` only** | | | | | |
| QQP duplicate questions | 2 | 0.61 | | | 0.743 / 0.725 |
| QNLI answer in sentence | 2 | 0.50 | | | 0.816 / 0.816 |
| CoLA grammaticality | 2 | 0.69 | | | 0.689 / 0.408 |
| BoolQ yes/no questions | 2 | 0.62 | | | 0.636 / 0.564 |
| **New labels, never trained on** | | | | | |
| Yahoo Answers topics | 10 | 0.10 | 0.469 / 0.464 | 0.478 / 0.464 | 0.447 / 0.440 |
| MASSIVE assistant intents | 60 | 0.02 | 0.325 / 0.312 | 0.422 / 0.391 | 0.403 / 0.360 |
| **Unseen task types** | | | | | |
| tweet irony | 2 | 0.60 | 0.446 / 0.439 | 0.478 / 0.478 | 0.614 / 0.496 |
| SMS spam | 2 | 0.85 | 0.839 / 0.511 | 0.837 / 0.473 | 0.851 / 0.460 |
| tweet hate speech | 2 | 0.56 | 0.494 / 0.493 | 0.517 / 0.506 | 0.558 / 0.398 |
| subjectivity | 2 | 0.51 | 0.470 / 0.404 | 0.402 / 0.364 | 0.409 / 0.362 |

For two-option datasets the third column is the majority baseline, the score of always giving
the more common answer. Only one score clears it on an unseen task type, `diverse-warm`'s 0.614
on irony against 0.603, and that model gets there by answering "not ironic" for 88% of tweets. It
also answers "not spam" for every message (0.851, exactly the baseline) and "not hate speech" for
95% of tweets, which is why its macro-F1 on spam and hate speech is the lowest of the three. Among the new training tasks, QNLI and QQP were learned
(AUC 0.89 and 0.83), BoolQ barely (0.65), and CoLA not at all (AUC 0.49; it calls every sentence
grammatical).

The model shows no preference for the first option: on AG News, accuracy for the correct option
in positions 1 to 4 was 0.87, 0.86, 0.89 and 0.91 for `poc`.

### Calibration

Expected calibration error (15 bins) is the average gap between how sure the model says it is
and how often it is right. For `poc-warm`:

| Data | Calibration error | Example |
|---|---|---|
| Training tasks | 0.02 to 0.08 | SST-2: 86.5% sure on average, 85.8% right |
| New labels | 0.12 to 0.26 | Yahoo: 72.5% sure, 47.8% right |
| Unseen task types | 0.09 to 0.25 | subjectivity: 64.8% sure, 40.2% right |

`poc` was underconfident on the many-option training tasks (banking77: 36% sure, 48% right), and
the extra epoch fixed that. Nothing fixed held-out calibration: the three models average 0.15
to 0.17 calibration error across the six held-out datasets. This is what the Molas critique
predicts for a zero-shot model, but the experiment that would test it properly (a global
temperature against a per-dataset one) was not run.

### Speed

One request at a time through `classify()`, a 20-word text, 100 timed requests after warm-up,
on an Apple M4 Pro (macOS 26.5, torch 2.14):

| Options | MPS p50 | MPS p95 | CPU p50 | CPU p95 |
|---|---|---|---|---|
| 2 | 9.2 ms | 10.5 ms | 18.7 ms | 19.5 ms |
| 10 | 11.6 ms | 13.0 ms | 28.7 ms | 29.8 ms |
| 50 | 23.2 ms | 25.4 ms | 49.5 ms | 54.5 ms |

Going from 2 to 50 options costs 2.5 times more on the GPU, because every option shares one
forward pass. An NLI-style zero-shot model runs one pass per option, so 50 options mean 50
passes. The raw numbers are in [`results/diverse-warm/latency.json`](results/diverse-warm/latency.json).

### Five hand-written examples

The same five prompts were tried after every stage. Except for the sentiment row, the labels
appear in no training dataset.

| Text | Options | `poc-warm` |
|---|---|---|
| NASA's rover found traces of water beneath the Martian surface. | space exploration, cooking, football, stock market | space exploration 99.0% |
| The striker scored twice in the second half. | same | football 99.9% |
| My card was charged twice. | billing, bug report, praise | billing 92.6% |
| I absolutely loved this film, the acting was superb. | positive, negative | positive 98.8% |
| The app crashes every time I open the settings page. | billing, bug report, praise | bug report 99.3% |

All five were right from `poc` onwards. They are also the kind of example that flatters a model:
everyday topics with obvious keywords. The held-out evaluation is what showed the limits.

## Takeaways

**One forward pass for any set of options is cheap and works.** Packing the options into the
input and scoring each `<opt>` marker with a shared `Linear(768, 1)` gives a real probability
distribution over the options of each request, and 50 options cost 2.5 times what 2 do. This
part of Jev's contract is easy to reproduce with a small encoder.

**Matching a text to a new label is a different skill from doing a new kind of task.** The model
learned to read label words: "turn the lights off" and "health" work although it never saw them.
It did not learn to follow an instruction for a judgement it was never trained to make. Irony,
spam, hate speech and subjectivity all need a property of the text ("does it mean the opposite
of what it says?"), and the instruction alone didn't get it there. Subjectivity even ranks below
chance in all three models, probably because SUBJ's "objective" sentences are plot summaries full
of charged words ("villainous", "horrifying murders"), and the model reads emotional vocabulary
as opinion.

**Learning starts with word overlap between options and text.** Logging the loss per dataset
showed the order: in the first 1,000 steps the clear movers were banking77 and TREC, the
many-option intent and question-type tasks where label words appear in the text ("card arrival",
"my card hasn't arrived"), with emotion a distant third. AG News and SST-2 stayed at chance
until that skill existed. A mix that cut banking77
from 1 in 6 training examples to 1 in 13 never got started; the same mix trained from the `poc`
checkpoint learned immediately. The long flat start of the `poc` run was the same effect.

**Four more task types didn't help.** Ten task types, with a quarter of the examples
asked as yes/no checks shaped like the unseen tasks ("ironic / not ironic"), did not move unseen
task AUC (0.51 against 0.52 for the control). The extra tasks also cost accuracy on everything
else, probably because each original task got 3,000 examples instead of 5,000.

**Accuracy misled on the unseen binary tasks.** SMS spam at 0.839 looks fine until the majority
baseline (0.851) and macro-F1 (0.51) sit next to it. `diverse-warm`'s irony accuracy rose from
0.446 to 0.614 while its ranking stayed at chance. Every two-option result here is reported with
the majority baseline, macro-F1 and AUC for that reason.

**Option order changes the answer.** The options are part of the input, so their order changes
the probabilities, even with two options: `poc`'s spam AUC is 0.56 with the options always in
the order "not spam, spam" and 0.65 with the shuffled order the evaluation uses.

## What didn't work

**The `<opt>` vector at the head's learning rate.** `<opt>` started as a fast-moving new
parameter (1e-3) next to an encoder at 3e-5. Because it feeds every attention layer, moving it
that fast collapsed the encoder within 200 steps: every token ended up with nearly the same hidden
state (cosine similarity 0.994) and accuracy stayed at chance. Moving it with the encoder fixed
it (0.308 to 0.863 accuracy on the AG News tracer). Details in
[`results/tracer/NOTES.md`](results/tracer/NOTES.md).

**Training the diverse mix from scratch.** The first `diverse` run sat at the loss of uniform
guessing for 2,000 steps and was stopped, for the dilution reason above. The diagnosis, with the
per-dataset losses of a 1,000-step probe and a control run, is in the
[plan, Phase 4b](plans/undecided.md#phase-4b-unseen-task-types).

**Rewording as a fix for irony.** As a diagnostic on the first model, six other instructions
and option pairs for irony scored 0.40 to 0.58, none above always answering "not ironic". Each
wording only changed which answer the model preferred.

## What I would try next

- **Hundreds of task types instead of ten.** Ten weren't enough for unseen task types to
  transfer. Collecting hundreds of labelled datasets is impractical; generating them is not. An
  LLM can write task descriptions, label sets and example texts for invented tasks, and the
  held-out evaluation here can tell whether that transfers.
- **A curriculum.** Start with many-option intent data until option-to-text matching exists, then
  add harder task types. The warm-start runs show this works for keeping training alive.
- **Order averaging.** Score each request under a few option orders and average the
  probabilities. Order demonstrably changes the scores, and with a 9 to 23 ms pass a few extra
  passes are affordable.
- **The calibration experiment as planned.** A global temperature fitted on the training tasks
  against a per-dataset temperature from 200 labelled examples, to test the Molas argument
  directly.
- **Baselines.** [`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)
  (same backbone, one pass per option) and
  [GLiClass](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init) (packed options)
  on the same evaluation sets, to see whether unseen task types defeat them too.

Also planned and not done: ablations (head only, LoRA, no shuffling, 4 against 16 training
options, no instructions, mean-pooled label tokens instead of `<opt>`) and a larger run on a
rented GPU.

## Architecture

### Input format

Instruction, options and text go into **one sequence**. Each option is preceded by a new
special token, `<opt>`:

```text
[CLS] Classify the support ticket. [SEP] <opt> billing <opt> bug report <opt> praise [SEP] My card was charged twice. [SEP]
```

The instruction is optional. When the sequence is too long, only the text is truncated, never
the options.

### Model

```mermaid
flowchart LR
    I["instruction + options + text"] --> T["tokeniser"]
    T --> E["ModernBERT-base encoder<br/>(fully fine-tuned)"]
    E --> G["hidden state at each &lt;opt&gt; token"]
    G --> H["Linear(768, 1)<br/>one logit per option"]
    H --> S["softmax over options"]
    S --> P["probabilities"]
```

The encoder reads the whole sequence, a single `Linear(768, 1)` scores the hidden state at each
`<opt>` token, and a softmax across the options of that example turns the scores into
probabilities. [Why this design](#why-this-design) explains how that works and why the options
can change at runtime.

The whole model is a few lines on top of `transformers`
([`src/undecided/model.py`](src/undecided/model.py)):

```python
class OptionScorer(nn.Module):
    def __init__(self, encoder, opt_token_id):
        super().__init__()
        self.encoder = encoder                                       # AutoModel.from_pretrained(...)
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.opt_token_id = opt_token_id
        pretrained = encoder.get_input_embeddings().weight.detach()
        self.opt_embedding = nn.Parameter(pretrained.mean(dim=0).clone())

    def forward(self, input_ids, attention_mask):
        is_opt = input_ids == self.opt_token_id                      # (batch, seq)
        safe_ids = input_ids.masked_fill(is_opt, self.encoder.config.pad_token_id)
        embeds = self.encoder.get_input_embeddings()(safe_ids)
        embeds = torch.where(is_opt[..., None], self.opt_embedding, embeds)

        hidden = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        counts = is_opt.sum(dim=1)                                   # options per example
        max_opts = int(counts.max())
        opt_mask = torch.arange(max_opts, device=hidden.device)[None] < counts[:, None]
        opt_hidden = hidden.new_zeros(hidden.size(0), max_opts, hidden.size(-1))
        opt_hidden[opt_mask] = hidden[is_opt]                        # (batch, max_opts, hidden)
        logits = self.head(opt_hidden).squeeze(-1)
        return logits.masked_fill(~opt_mask, float("-inf"))          # padded options get p = 0
```

Training loss is plain cross-entropy between these logits and the index of the correct option.

### Why this design

A head is the small layer on top of the encoder that turns its output into an answer. The
three common designs differ in where the label names live, and that decides whether options
can change at runtime.

| Design | Options at runtime | Cost per request | Used by |
|---|---|---|---|
| Fixed head, `Linear(768, N)` | No, the N classes are frozen at training time | 1 pass | Classic fine-tuned BERT |
| Natural-language inference (NLI), one pass per option | Yes | N passes | `ModernBERT-base-zeroshot-v2.0`, most zero-shot pipelines |
| **Packed options, per-option head** | Yes | 1 pass | This project, GLiClass |

#### Fixed head: labels live in the weights

```text
[CLS] My card was charged twice. [SEP]
        │
     encoder
        │
   CLS vector (768 numbers)
        │
  Linear(768, 3)        one row of weights per class
        │
  [billing, bug report, praise]  →  softmax
```

`Linear(768, 3)` has exactly 3 rows of weights, and each row is one class. Row 0 learned what
billing looks like from thousands of billing examples. The model never reads the word
"billing": label names exist only in the calling code, as a lookup from row 0 to a string.

So at request time:

- An option can't be added. There is no row for "refund request".
- An option can't be renamed. Calling class 0 "payment issue" changes nothing, the model still
  means whatever row 0 was trained on.
- The model can't be reused for another task. Sentiment needs a new head and new training.

Passing the options with every request, including options nobody trained on, is the point of
Jev. A fixed head can't do it, so a fixed head can't be a Jev clone.

#### One pass per option: labels live in the input, one at a time

NLI-based zero-shot models read labels as text, but they run the encoder once per option:
"My card was charged twice" paired with "This is about billing" gives a yes/no score, then the
same again for every other option. That handles new labels, with two drawbacks:

- 60 options cost 60 forward passes.
- Each option is scored alone, so the model never learns to weigh one option against another,
  and the probabilities aren't trained as a real distribution.

#### Packed options: labels live in the input, all together

```text
[CLS] Classify the support ticket. [SEP] <opt> billing <opt> bug report <opt> praise [SEP] My card was charged twice. [SEP]
                                         │             │                │
                              the encoder reads everything together, with attention
                                         │             │                │
                                     vector at     vector at        vector at
                                     1st <opt>     2nd <opt>        3rd <opt>
                                         │             │                │
                                 Linear(768, 1), the same layer applied to each
                                         │             │                │
                                        score         score            score       one per option
                                         └────────── softmax ───────────┘
                                       0.926         0.072            0.002     (poc-warm's answer)
```

1. **The head has one output.** `Linear(768, 1)` answers a single, generic question: how well
   does this option fit? It knows nothing about billing or praise. The same layer is applied to
   every option, so 3 options or 60 make no difference to it.
2. **The encoder does the comparison.** Each `<opt>` token sits right before its label, and
   attention lets it look at its own label words, the text and the instruction. After
   fine-tuning, the vector at each `<opt>` summarises "this label against this text". This is
   why the encoder has to be fine-tuned: a frozen encoder was never trained to make that
   comparison, so the vector would hold nothing useful for the head to score.
3. **The softmax runs over the options of each request.** Probabilities are relative: 0.926
   means "billing beats the other two options it was given". Different options give a different
   distribution, which matches how Jev behaves.

Compared with one pass per option, packing keeps a single forward pass however many options
there are, and the softmax is trained over the options that actually compete.

#### Why unseen labels can work

Training shows the model many (text, option set, correct option) triples from different
datasets, with label names in plain words. What it has to learn is the general skill of matching
a text to a label description, since memorising 77 banking intents doesn't help on the other
datasets. At test time, "set an alarm" or "health" are words it reads, like the text. The results
show this transfers for new labels and stops short of new kinds of task.

#### What packing costs

- **Position bias.** Options can see each other and their order, so the model could learn "the
  first option is usually right". Training shuffles the option order every time, and the
  evaluation found no first-option preference. Order still changes the probabilities, as the
  spam AUC above shows.
- **Context budget.** Every option costs tokens. MASSIVE's 60 intent names take at most 340
  tokens with the text, which fits easily in ModernBERT's 8,192-token context, but very long
  option descriptions would not scale.

### Backbone choice

The backbone is ModernBERT-base, an encoder, for four reasons:

- **Bidirectional attention.** An encoder reads the whole input in both directions at once, so
  every `<opt>` token sees the text, the instruction and the other options, wherever they sit
  in the sequence. A decoder LLM reads left to right, so an option could only see what comes
  before it.
- **Size.** 149M parameters keeps inference fast and makes a full fine-tune on a laptop
  practical. Single-pass speed is also the property Jev is sold on.
- **ModernBERT over older BERTs.** The 8,192-token context (classic BERT and DeBERTa stop at
  512) fits 60 or more options in one sequence, and it has a more recent tokeniser, more
  pretraining data and faster inference.
- **Prior art.** GLiClass and the zero-shot v2.0 models already use ModernBERT for zero-shot
  classification.

The expected weakness was instruction following, and the unseen task types confirmed it. Whether
[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) or a small decoder with the
same per-option head would do better on them is untested; the training data looks like the
bigger limit.

## Data

### Unified format

Every source dataset is converted into the same record:

```json
{
  "instruction": "What does this banking customer want?",
  "text": "I still haven't received my new card, it's been two weeks.",
  "options": ["card arrival", "lost or stolen card", "top up failed", "exchange rate"],
  "answer_idx": 0,
  "source": "banking77",
  "group": "train"
}
```

### Training datasets

| Dataset | Task | Labels | Used by |
|---|---|---|---|
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | news topic | 4 | all |
| [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | emotion | 6 | all |
| [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | banking intent | 77 | all |
| [`SetFit/TREC-QC`](https://huggingface.co/datasets/SetFit/TREC-QC) | question type | 50 (fine-grained) | all |
| [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2) | sentiment | 2 | all |
| [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) | entailment | 3 | all |
| [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`qqp`) | duplicate questions | 2 | `diverse-warm` |
| [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`qnli`) | does the sentence answer the question | 2 | `diverse-warm` |
| [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue) (`cola`) | grammaticality | 2 | `diverse-warm` |
| [`google/boolq`](https://huggingface.co/datasets/google/boolq) | yes/no question about a passage | 2 | `diverse-warm` |

Tasks over two texts (MNLI, QQP, QNLI, BoolQ) put the second text inside the instruction, for
example `Does this sentence answer the question "Who wrote Hamlet?"`, with the template picked
per example when loading. The original `PolyAI/banking77` and `CogComp/trec` repos rely on
loading scripts, which recent versions of `datasets` no longer run, so the mirrors above are used
instead.

### Held-out datasets (never trained on)

| Dataset | Task | Labels | What it tests |
|---|---|---|---|
| [`community-datasets/yahoo_answers_topics`](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | question topic | 10 | new labels, familiar task type |
| [`SetFit/amazon_massive_intent_en-US`](https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US) | assistant intent | 60 | new labels, far more options than seen in training |
| [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`irony`) | irony | 2 | a task type absent from training |
| [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) | spam | 2 | a task type absent from training, in a new domain |
| [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`hate`) | hate speech | 2 | a task type absent from training, on tweets like irony |
| [`SetFit/subj`](https://huggingface.co/datasets/SetFit/subj) | subjectivity | 2 | a task type absent from training, in a familiar domain |

Irony was the only unseen task type at first. When it failed, the last three were added to tell
a hard task apart from unseen task types failing in general; they failed too. Nothing resembling
them (sarcasm, toxicity, stance) was ever added to training.

Options are worded as a user would type them. MASSIVE's raw intents are terse
("iot_hue_lightoff", "email_querycontact"), so each of the 60 has a hand-written name ("turn the
lights off", "look up a contact"), fixed in [`tasks.py`](src/undecided/tasks.py) before the first
held-out evaluation. Evaluation samples are 1,000 examples per dataset (all 784 irony test tweets,
500 for TREC, the 872 of SST-2's validation split), drawn once with a fixed seed.

### Augmentation

Each training example is rebuilt on the fly, so every epoch sees different variants:

- **Option sampling.** 2 to 16 options per example: the correct label plus distractors drawn
  from the same dataset's other labels. Label sets of 3 or fewer always use all labels.
- **Shuffling.** Option order is random every time.
- **Readable labels.** `alarm_set` becomes "alarm set", and a few labels per dataset have
  hand-written paraphrases ("science and technology" can become "sci-tech"), swapped in 30% of
  the time.
- **Instructions.** 4 to 8 templates per task, plus no instruction at all in 20% of examples
  (except the two-text tasks, whose instruction carries the second text).
- **Yes/no recasting** (`diverse-warm` only). A quarter of the single-text examples become a
  check on one label, as "yes / no" or "<label> / not <label>", with the true label asked about
  half the time.

## Training

### What gets trained

The **whole model**: all 149M encoder parameters plus the head. The comparison between option
and text happens inside the encoder, so a frozen encoder would leave the head nothing to score.

`<opt>` is a new token. It is added to the tokeniser, but its vector is a separate parameter
rather than a new row in ModernBERT's embedding matrix: the model looks up every other token
normally and writes the `<opt>` vector in at the marker positions. The vector starts as the mean
of the pretrained token embeddings.

| Parameters | Learning rate |
|---|---|
| head | 1e-3 |
| encoder, including the `<opt>` vector | 3e-5 |

AdamW with weight decay 0.01, 6% warm-up then linear decay, gradient clipping at 1.0, batch size
16, maximum length 512, fp32. Sequences are grouped by length before batching, so a batch of
short tweets doesn't pay for one long news article. The warm-start runs load the `poc`
checkpoint through `init_from` in their config and restart the schedule.

### Hardware

All training ran on an Apple M4 Pro (20-core GPU, 48 GB unified memory) through PyTorch's MPS
backend, with `PYTORCH_ENABLE_MPS_FALLBACK=1` and no flash attention (unavailable on macOS).
Throughput was 29 to 32 examples per second at batch size 16; the `poc` run averaged 19,
probably because other work was running on the machine at the time. The MPS allocator cache grows as new batch shapes appear,
so training clears it every 200 steps; driver memory peaked at 11.5 GB for `poc` and 18.7 GB for
the runs with BoolQ's longer passages. The code picks `cuda`, `mps` or `cpu` automatically, but
a cloud GPU was never needed.

## Evaluation

| Metric | What it answers |
|---|---|
| Accuracy, macro-F1 | Does it pick the right option? Macro-F1 stops frequent classes from hiding failures on rare ones. |
| Majority baseline, ranking AUC (two-option datasets) | Is there any signal at all, whatever the decision threshold? |
| Expected calibration error (15 bins), mean confidence | When it says 80%, is it right 80% of the time? |
| Accuracy by correct-option position | Is the first option favoured? |
| Latency p50 / p95 at 2, 10 and 50 options | How long does one request take, on MPS and on CPU? |
| Loss per dataset against ln(number of options), every 200 steps | Which tasks is training actually learning? |

Every metric is reported per dataset and averaged per group: training datasets, held-out
datasets with new labels, and unseen task types. Each run's numbers are in
`results/<run>/metrics.json`, its config in `results/<run>/config.yaml` and its loss curve in
`results/<run>/plots/loss.png`.

## Usage

```python
from undecided import Undecided

clf = Undecided.from_pretrained("runs/poc-warm")           # any run directory under runs/
clf.classify(
    "My card was charged twice.",
    options=["billing", "bug report", "praise"],
    instruction="Classify the support ticket.",
)
# {"billing": 0.926, "bug report": 0.072, "praise": 0.002}
```

The same from the terminal:

```text
uv run python -m undecided.classify runs/poc-warm "My card was charged twice." \
    --options billing "bug report" praise --instruction "Classify the support ticket."
```

`classify()` rejects duplicate options, fewer than 2 options, and more options than fit in the
checkpoint's max length. The device is picked automatically (`cuda`, then `mps`, then `cpu`), or
passed as `from_pretrained(path, device=...)`. Checkpoints are not published (600 MB each, and the
dataset licences differ), so `runs/` has to be produced locally:

```text
uv run python -m undecided.run configs/poc.yaml                # train, evaluate, save runs/poc (53 min)
uv run python -m undecided.run configs/poc-warm.yaml           # one more epoch from runs/poc (17 min)
uv run python -m undecided.run configs/poc.yaml --eval-only    # re-evaluate a checkpoint, no training
uv run python -m undecided.results results/poc results/poc-warm  # Markdown tables
uv run python -m undecided.latency runs/poc-warm               # single-request timings
uv run python -m undecided.figure                              # results/auc.png
uv run pytest                                                  # 49 tests, about 3 s
```

## Layout

```text
undecided/
├── src/undecided/
│   ├── data.py        # conversion of each dataset into the unified record format
│   ├── sources.py     # where each dataset is downloaded from, sampling with a fixed seed
│   ├── tasks.py       # instruction templates, label paraphrases, MASSIVE intent names
│   ├── augment.py     # instructions, option sampling, paraphrasing, shuffling, yes/no recasting
│   ├── encoding.py    # packed <opt> input format, collation
│   ├── model.py       # OptionScorer and the Undecided wrapper (save, load, classify)
│   ├── train.py       # training loop, two learning-rate groups, per-dataset loss
│   ├── evaluate.py    # accuracy, macro-F1, AUC, calibration error, accuracy by position
│   ├── run.py         # one experiment from a YAML config, or re-evaluation of a checkpoint
│   ├── results.py     # Markdown tables from results/*/metrics.json
│   ├── latency.py     # single-request timings
│   ├── figure.py      # the AUC chart in this README
│   ├── preview.py     # prints augmented examples and label counts for a config
│   └── classify.py    # command-line classification with a checkpoint
├── tests/
├── configs/           # one YAML file per run
├── plans/             # the implementation plan and its progress log
├── results/           # metrics, configs, plots and notes per run, committed
└── runs/              # checkpoints and logs, gitignored
```

Tooling: `uv`, Python 3.12, `torch` 2.14, `transformers` 5.17, `datasets`, `scikit-learn`,
`matplotlib`.

## Limitations

- It shares Jev's interface and none of TypeSafe's methods, RL-CD included.
- English only, single-label only.
- Each run was trained once, with one seed. Differences of a point or two between models are
  within what another seed could change.
- Calibration was measured, not fixed: no temperature scaling.
- No baseline models were run, so how these numbers compare with existing zero-shot classifiers
  on the same data is unknown.
- Dataset licences differ. Check each one before publishing trained weights.

## References

- ModernBERT: [model card](https://huggingface.co/answerdotai/ModernBERT-base)
- Jev coverage: [MindStudio benchmark](https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark),
  [Towards Data Science](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/),
  [XenoSpectrum](https://xenospectrum.com/en/jev-typesafe-bert-classifier-decomposition/),
  [AWS Builder Center, Kyoto-ben intents](https://builder.aws.com/content/3JcBTYjvqJbHO2D6e72S9pPZ41d/jev-vs-bert-on-kyoto-ben-intent-classification-measured)
- Calibration critique: [Jev can't be calibrated](https://www.alexmolas.com/2026/09/23/jev-cant-be-calibrated.html)
- Prior art: [GLiClass](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init),
  [zero-shot v2.0 models](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)
