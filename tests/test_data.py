from undecided.data import Record, convert_ag_news


def test_ag_news_rows_become_records_with_all_labels_as_readable_options():
    rows = [
        {"text": "Stocks rally as rates fall", "label": 2},
        {"text": "Rover finds water on Mars", "label": 3},
    ]

    records = convert_ag_news(rows)

    assert records[0] == Record(
        instruction="",
        text="Stocks rally as rates fall",
        options=["world", "sports", "business", "science and technology"],
        answer_idx=2,
        source="ag_news",
        group="train",
    )
    assert records[1].options[records[1].answer_idx] == "science and technology"
