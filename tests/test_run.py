import json

import torch

from undecided.data import Record
from undecided.run import evaluate_all


def records(n, source):
    return [Record("", f"text {i}", ["x", "y"], i % 2, source, "heldout_task") for i in range(n)]


def test_evaluation_reuses_datasets_already_in_the_partial_file_and_saves_the_new_ones(
        tokenizer, tiny_scorer, tmp_path):
    partial = tmp_path / "metrics.partial.json"
    done_before = {"group": "heldout_task", "options": 2, "n": 4, "accuracy": 0.123, "macro_f1": 0.1}
    stale = {"group": "heldout_task", "options": 2, "n": 99, "accuracy": 0.456, "macro_f1": 0.4}  # other sample size
    partial.write_text(json.dumps({"a": done_before, "c": stale}))
    eval_sets = {"a": records(4, "a"), "b": records(6, "b"), "c": records(4, "c")}

    result = evaluate_all(tiny_scorer, tokenizer, eval_sets, 64, torch.device("cpu"), partial=partial)

    assert result["datasets"]["a"] == done_before
    assert result["datasets"]["b"]["n"] == 6
    assert result["datasets"]["c"]["n"] == 4 and result["datasets"]["c"]["accuracy"] != 0.456
    assert set(json.loads(partial.read_text())) == {"a", "b", "c"}
