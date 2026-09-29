# Tracer notes

Phase 1 of the plan: AG News only, through every layer. Machine: Apple M4 Pro, 48 GB, MPS,
torch 2.14, transformers 5.17, fp32.

## Throughput

27.5 examples per second at batch size 16 (600 steps, 348.8 s). AG News texts average 59
tokens, plus 2 to 4 short options. The 64-example overfit run gave the same rate (27.0).

## Proof-of-concept size

Kept at the planned 5,000 examples per training dataset, 30,000 in total. Two epochs are
60,000 examples, about 36 minutes at this rate. The other training datasets have texts of a
similar length or shorter (MNLI adds a hypothesis to the instruction), so 35 to 50 minutes per
run is the working estimate. That's about 3,750 steps, well past the slow start described
below.

## Results

| Run | Setup | Result |
|---|---|---|
| `tracer-overfit` | 64 examples, fixed option order, 30 epochs | training accuracy 1.0 |
| `tracer-v1-collapse` | 3,200 examples, 1 epoch, `<opt>` vector at 1e-3 | accuracy 0.308, encoder collapsed |
| `tracer` | 3,200 examples, 3 epochs, `<opt>` vector at 3e-5 | accuracy 0.863 |

Evaluation is 1,000 AG News test examples with all 4 options in a fixed-seed shuffled order.
Chance is 0.25.

## What went wrong in v1

The first design trained the `<opt>` vector at the head's learning rate (1e-3). After 200 steps
the loss sat at the level of a uniform guess, and the whole encoder output had collapsed:

| | Untrained | After v1 training |
|---|---|---|
| Cosine similarity between ordinary text tokens | 0.594 | 0.994 |
| Cosine similarity between `[CLS]` of different articles | 0.760 | 0.998 |
| Cosine similarity between two `<opt>` vectors of one input | 0.930 | 0.999 |

At 3e-5, Adam can move an encoder weight by at most about 0.006 in 200 steps, too little to
explain this. The `<opt>` vector was the only input moving fast (its direction rotated about 23°),
and it feeds every attention layer. Training it at the encoder's rate, with nothing else changed,
removed the collapse (text-token similarity 0.68) and gave 0.428 after the same 200 steps.

## Slow start

Even without the collapse, the loss stays near chance for about the first 200 steps and then
drops (see `plots/loss.png`). At initialisation the `<opt>` vectors mostly encode position rather
than label: two `<opt>` vectors at the same position with different labels have cosine similarity
0.986, the same label at different positions 0.937. The encoder has to learn to pull label
information into the marker before the head has anything to score. The option pooling ablation
(mean of the label's own tokens) is a direct test of whether this start can be skipped.
