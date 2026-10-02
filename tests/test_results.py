from undecided.results import group_summary, markdown_tables


def metrics(accuracy, macro_f1, group, options):
    return {"n": 100, "accuracy": accuracy, "macro_f1": macro_f1, "group": group, "options": options}


def test_each_group_averages_its_datasets_equally_whatever_their_size():
    datasets = {
        "a": {**metrics(0.8, 0.6, "train", 4), "n": 1000},
        "b": {**metrics(0.6, 0.4, "train", 2), "n": 10},
        "c": metrics(0.5, 0.45, "heldout_task", 2),
    }

    assert group_summary(datasets) == {
        "train": {"datasets": ["a", "b"], "accuracy": 0.7, "macro_f1": 0.5},
        "heldout_task": {"datasets": ["c"], "accuracy": 0.5, "macro_f1": 0.45},
    }


def test_runs_become_a_summary_table_and_a_per_dataset_table():
    poc = {"run": "poc", "datasets": {"ag_news": metrics(0.8841, 0.884, "train", 4),
                                      "massive": metrics(0.4, 0.35, "heldout_labels", 60)}}
    small = {"run": "small", "datasets": {"ag_news": metrics(0.5, 0.45, "train", 4)}}

    summary, per_dataset = markdown_tables([poc, small])

    assert summary.splitlines() == [
        "| Run | In-distribution acc. | Held-out labels acc. | Held-out task acc. |",
        "|---|---|---|---|",
        "| poc | 0.884 | 0.400 |  |",
        "| small | 0.500 |  |  |",
    ]
    assert per_dataset.splitlines() == [
        "| Dataset | Group | Options | Chance | poc acc. / F1 | small acc. / F1 |",
        "|---|---|---|---|---|---|",
        "| ag_news | train | 4 | 0.250 | 0.884 / 0.884 | 0.500 / 0.450 |",
        "| massive | heldout_labels | 60 | 0.017 | 0.400 / 0.350 |  |",
    ]


def test_older_runs_without_macro_f1_or_groups_still_print():
    old = {"run": "tracer", "datasets": {"ag_news": {"n": 1000, "accuracy": 0.863}}}

    summary, per_dataset = markdown_tables([old])

    assert "| tracer | 0.863 |  |  |" in summary
    assert "| ag_news | train |  |  | 0.863 |" in per_dataset
