from undecided.train import build_optimizer


def test_only_the_head_trains_at_head_lr_and_the_opt_vector_moves_with_the_encoder(tiny_scorer):
    # Training <opt> at the head LR collapsed the whole encoder output in the tracer run
    # (every token state ended up with cosine similarity 0.994), so it moves at the encoder LR.
    optimizer = build_optimizer(tiny_scorer, head_lr=1e-3, encoder_lr=3e-5, weight_decay=0.01)

    lr_of = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    assert len(lr_of) == sum(len(g["params"]) for g in optimizer.param_groups)  # no duplicates
    assert lr_of[id(tiny_scorer.opt_embedding)] == 3e-5
    assert all(lr_of[id(p)] == 1e-3 for p in tiny_scorer.head.parameters())
    assert all(lr_of[id(p)] == 3e-5 for p in tiny_scorer.encoder.parameters())
    assert len(lr_of) == len(list(tiny_scorer.parameters()))


def test_training_memorises_a_small_fixed_set(tokenizer, tiny_scorer):
    import random

    import torch

    from undecided.data import Record
    from undecided.evaluate import evaluate
    from undecided.train import TrainConfig, train

    rng = random.Random(0)
    labels = ["world", "sports", "business", "science and technology"]
    words = ["apple", "river", "engine", "violet", "castle", "pepper", "falcon", "marble"]
    records = [
        Record("", " ".join(rng.choices(words, k=4)), list(labels), i % 4, "test", "train")
        for i in range(32)
    ]
    assert evaluate(tiny_scorer, tokenizer, records, max_length=64, batch_size=16, device=torch.device("cpu"))["accuracy"] < 0.6

    cfg = TrainConfig(epochs=60, batch_size=8, head_lr=3e-3, encoder_lr=3e-3, max_length=64, augment=False)
    train(tiny_scorer, tokenizer, records, cfg, device=torch.device("cpu"))

    metrics = evaluate(tiny_scorer, tokenizer, records, max_length=64, batch_size=16, device=torch.device("cpu"))
    assert (metrics["n"], metrics["accuracy"], metrics["macro_f1"]) == (32, 1.0, 1.0)


def test_batches_cover_every_record_once_and_group_similar_lengths():
    import random

    from undecided.data import Record
    from undecided.train import length_grouped_batches

    rng = random.Random(0)
    records = [
        Record("", "word " * rng.choice([3, 5, 200, 250]), ["a", "b"], 0, "test", "train") for _ in range(640)
    ]

    batches = length_grouped_batches(records, batch_size=16, rng=random.Random(1))

    flat = [r for batch in batches for r in batch]
    assert sorted(map(id, flat)) == sorted(map(id, records))
    assert all(len(b) <= 16 for b in batches)
    mixed = sum(min(len(r.text) for r in b) * 10 < max(len(r.text) for r in b) for b in batches)
    assert mixed <= 2  # short and long texts rarely share a batch


def test_training_reports_each_examples_loss_with_its_source_and_option_count(tokenizer, tiny_scorer):
    import math
    import random

    import torch

    from undecided.data import Record
    from undecided.train import TrainConfig, train

    rng = random.Random(0)
    records = [Record("", f"text {i}", ["a", "b", "c", "d"][: 2 + i % 3], 0, ["x", "y"][i % 2], "train")
               for i in range(24)]
    calls = []

    train(tiny_scorer, tokenizer, records, TrainConfig(epochs=1, batch_size=8, augment=False, max_length=64),
          torch.device("cpu"), on_step=lambda step, loss, examples: calls.append((loss, examples)))

    assert len(calls) == 3
    for loss, examples in calls:
        assert len(examples) == 8
        assert {source for source, _, _ in examples} <= {"x", "y"}
        assert all(n in (2, 3, 4) for _, _, n in examples)
        assert math.isclose(loss, sum(l for _, l, _ in examples) / 8, rel_tol=1e-4)


def test_losses_are_summarised_per_source_against_uniform_guessing():
    import math

    from undecided.train import loss_by_source

    examples = [("x", 0.2, 2), ("x", 0.4, 4), ("y", 1.0, 3)]

    summary = loss_by_source(examples)

    assert summary["x"]["n"] == 2 and math.isclose(summary["x"]["loss"], 0.3)
    assert math.isclose(summary["x"]["chance"], (math.log(2) + math.log(4)) / 2)
    assert math.isclose(summary["y"]["chance"], math.log(3))


def test_a_resumed_run_continues_exactly_where_an_interrupted_one_stopped(tokenizer, tiny_scorer, tmp_path):
    import copy
    import random

    import pytest
    import torch

    from undecided.data import Record
    from undecided.train import TrainConfig, train

    rng = random.Random(0)
    words = ["apple", "river", "engine", "violet", "castle", "pepper", "falcon", "marble"]
    labels = ["world", "sports", "business", "science", "health", "music"]
    records = [Record("", " ".join(rng.choices(words, k=3 + i % 5)), list(labels), i % 6, "test", "train")
               for i in range(48)]
    cfg = TrainConfig(epochs=2, batch_size=8, max_length=64, augment=True, max_options=4)  # 12 steps
    cpu = torch.device("cpu")
    initial = copy.deepcopy(tiny_scorer.state_dict())

    uninterrupted = train(tiny_scorer, tokenizer, records, cfg, cpu)

    class Interrupted(Exception):
        pass

    def stop_at_step_9(step, loss, examples):
        if step == 9:
            raise Interrupted

    tiny_scorer.load_state_dict(initial)
    with pytest.raises(Interrupted):
        train(tiny_scorer, tokenizer, records, cfg, cpu, on_step=stop_at_step_9,
              checkpoint=tmp_path / "train-state.pt", checkpoint_every=4)  # last checkpoint: step 8

    tiny_scorer.load_state_dict(initial)  # the resume must come from the file, not from memory
    resumed = train(tiny_scorer, tokenizer, records, cfg, cpu,
                    checkpoint=tmp_path / "train-state.pt", checkpoint_every=4, resume=True)

    assert len(uninterrupted) == 12
    assert resumed == pytest.approx(uninterrupted, rel=1e-5)
