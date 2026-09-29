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
    assert metrics == {"n": 32, "accuracy": 1.0, "macro_f1": 1.0}


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
