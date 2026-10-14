import pytest

from undecided.data import Record, convert_ag_news


def test_ag_news_rows_become_records_with_all_labels_as_readable_options():
    rows = [
        {"text": "Stocks rally as rates fall", "label": 2},
        {"text": "Rover finds water on Mars", "label": 3},
    ]

    records = convert_ag_news(rows)

    assert records[0] == Record(
        instruction="What is this news article about?",
        text="Stocks rally as rates fall",
        options=["world", "sports", "business", "science and technology"],
        answer_idx=2,
        source="ag_news",
        group="train",
    )
    assert records[1].options[records[1].answer_idx] == "science and technology"


def test_banking77_labels_become_readable_phrases_and_every_label_is_an_option():
    from undecided.data import convert_banking77

    labels = ["card_arrival", "Refund_not_showing_up", "apple_pay_or_google_pay"]
    rows = [{"text": "Where is my refund?", "label_text": "Refund_not_showing_up"}]

    [record] = convert_banking77(rows, labels)

    assert record.options == ["card arrival", "refund not showing up", "apple pay or google pay"]
    assert record.options[record.answer_idx] == "refund not showing up"
    assert record.source == "banking77"
    assert record.instruction  # canonical instruction for the task


@pytest.mark.parametrize(
    ("convert", "row", "answer", "source", "n_options"),
    [
        ("convert_emotion", {"text": "im so happy today", "label": 1}, "joy", "emotion", 6),
        ("convert_sst2", {"sentence": "a charming journey", "label": 1}, "positive", "sst2", 2),
        ("convert_trec", {"text": "How far is Denver ?", "label_text": "linear measures"},
         "linear measures", "trec", 3),
    ],
)
def test_each_training_dataset_converts_to_records_with_readable_options(convert, row, answer, source, n_options):
    import undecided.data as data

    kwargs = {"labels": ["dates", "linear measures", "cities"]} if source == "trec" else {}
    [record] = getattr(data, convert)([row], **kwargs)

    assert record.options[record.answer_idx] == answer
    assert len(record.options) == n_options
    assert record.source == source and record.group == "train"
    assert record.text == next(iter(row.values()))
    assert record.instruction


def test_mnli_is_recast_with_the_hypothesis_inside_a_varied_instruction():
    from undecided.data import convert_mnli

    rows = [
        {"premise": f"Premise {i}.", "hypothesis": f"Hypothesis {i}.", "label": i % 3} for i in range(60)
    ] + [{"premise": "No gold label.", "hypothesis": "Skip me.", "label": -1}]

    records = convert_mnli(rows, seed=0)

    assert len(records) == 60
    first = records[0]
    assert first.text == "Premise 0."
    assert "Hypothesis 0." in first.instruction
    assert first.options == ["true", "cannot tell", "false"]
    assert [r.options[r.answer_idx] for r in records[:3]] == ["true", "cannot tell", "false"]
    assert len({r.instruction.replace(f"Hypothesis {i}.", "") for i, r in enumerate(records)}) > 3
    assert records == convert_mnli(rows, seed=0)  # same seed, same templates


def test_yahoo_questions_join_title_and_body_and_are_marked_as_held_out_labels():
    from undecided.data import convert_yahoo_answers

    rows = [{"question_title": "Why is the sky blue?", "question_content": "Asked by my kid.", "topic": 1},
            {"question_title": "Best striker ever?", "question_content": "", "topic": 5}]

    records = convert_yahoo_answers(rows)

    assert records[0].text == "Why is the sky blue? Asked by my kid."
    assert records[1].text == "Best striker ever?"
    assert records[0].options[records[0].answer_idx] == "science & mathematics"
    assert records[1].options[records[1].answer_idx] == "sports"
    assert len(records[0].options) == 10
    assert {r.group for r in records} == {"heldout_labels"}


def test_massive_intents_get_plain_english_names_and_every_intent_is_an_option():
    from undecided.data import convert_massive
    from undecided.tasks import MASSIVE_NAMES

    labels = sorted(MASSIVE_NAMES)
    rows = [{"text": "wake me up at five am this week", "label_text": "alarm_set"},
            {"text": "lights off please", "label_text": "iot_hue_lightoff"}]

    records = convert_massive(rows, labels)

    assert len(labels) == 60 and len(set(MASSIVE_NAMES.values())) == 60
    assert all(len(r.options) == 60 for r in records)
    assert [r.options[r.answer_idx] for r in records] == ["set an alarm", "turn the lights off"]
    assert {r.group for r in records} == {"heldout_labels"}


def test_irony_is_a_held_out_task_with_two_readable_options():
    from undecided.data import convert_tweet_irony

    plain, ironic = convert_tweet_irony([{"text": "Nice weather", "label": 0},
                                         {"text": "Great, Monday again", "label": 1}])

    assert plain.options == ironic.options == ["not ironic", "ironic"]
    assert ironic.options[ironic.answer_idx] == "ironic"
    assert plain.group == ironic.group == "heldout_task"


@pytest.mark.parametrize(
    ("convert", "rows", "options", "answers"),
    [
        ("convert_sms_spam", [{"sms": "WINNER! Claim your prize now\n", "label": 1}, {"sms": "See you at 5", "label": 0}],
         ["not spam", "spam"], ["spam", "not spam"]),
        ("convert_tweet_hate", [{"text": "@user have a nice day", "label": 0}],
         ["not hate speech", "hate speech"], ["not hate speech"]),
        ("convert_subjectivity", [{"text": "the film is set in 1920s paris .", "label": 0, "label_text": "objective"},
                                  {"text": "a gorgeous , witty film .", "label": 1, "label_text": "subjective"}],
         ["objective", "subjective"], ["objective", "subjective"]),
    ],
)
def test_extra_held_out_tasks_are_binary_with_readable_options(convert, rows, options, answers):
    import undecided.data as data

    records = getattr(data, convert)(rows)

    assert [r.options for r in records] == [options] * len(rows)
    assert [r.options[r.answer_idx] for r in records] == answers
    assert {r.group for r in records} == {"heldout_task"}
    assert all(r.instruction and r.text == r.text.strip() for r in records)


@pytest.mark.parametrize(
    ("convert", "row", "text", "inside", "options", "answer"),
    [
        ("convert_qqp", {"question1": "How do I learn Python?", "question2": "Best way to learn Python?", "label": 1},
         "How do I learn Python?", "Best way to learn Python?", ["not duplicate", "duplicate"], "duplicate"),
        ("convert_qnli", {"question": "Who wrote Hamlet?", "sentence": "Hamlet is a play by Shakespeare.", "label": 0},
         "Hamlet is a play by Shakespeare.", "Who wrote Hamlet?", ["answers it", "does not answer it"], "answers it"),
        ("convert_boolq", {"question": "is the sun a star", "passage": "The Sun is the star at the centre.", "answer": True},
         "The Sun is the star at the centre.", "is the sun a star", ["no", "yes"], "yes"),
    ],
)
def test_pair_tasks_put_the_second_text_inside_a_seeded_instruction(convert, row, text, inside, options, answer):
    import undecided.data as data

    records = getattr(data, convert)([row] * 40, seed=0)

    assert records[0].text == text
    assert all(inside in r.instruction for r in records)
    assert records[0].options == options
    assert records[0].options[records[0].answer_idx] == answer
    assert {r.group for r in records} == {"train"}
    assert len({r.instruction for r in records}) > 1  # templates vary
    assert records == getattr(data, convert)([row] * 40, seed=0)


def test_cola_sentences_are_judged_grammatical_or_not():
    from undecided.data import convert_cola

    bad, good = convert_cola([{"sentence": "Him went home.", "label": 0}, {"sentence": "She went home.", "label": 1}])

    assert bad.options == ["ungrammatical", "grammatical"]
    assert (bad.options[bad.answer_idx], good.options[good.answer_idx]) == ("ungrammatical", "grammatical")
    assert bad.source == "cola" and bad.group == "train"
