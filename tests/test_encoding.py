import pytest


from undecided.data import Record
from undecided.encoding import OPT_TOKEN, encode


def record(text="My card was charged twice.", options=("billing", "bug report", "praise"), instruction=""):
    return Record(
        instruction=instruction,
        text=text,
        options=list(options),
        answer_idx=0,
        source="test",
        group="train",
    )


def test_packed_sequence_has_one_opt_marker_per_option_and_decodes_to_the_packed_format(tokenizer):
    encoded = encode(tokenizer, record(instruction="Classify the support ticket."), max_length=512)

    opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
    assert encoded["input_ids"].count(opt_id) == 3
    assert encoded["label"] == 0
    assert tokenizer.decode(encoded["input_ids"]) == (
        "[CLS]Classify the support ticket.[SEP]<opt> billing<opt> bug report<opt> praise[SEP]"
        "My card was charged twice.[SEP]"
    )


def test_long_text_is_truncated_to_max_length_and_every_option_survives(tokenizer):
    encoded = encode(tokenizer, record(text="word " * 2000), max_length=64)

    ids = encoded["input_ids"]
    assert len(ids) == 64
    assert ids.count(tokenizer.convert_tokens_to_ids(OPT_TOKEN)) == 3
    assert ids[-1] == tokenizer.sep_token_id


def test_options_that_do_not_fit_raise_instead_of_being_truncated(tokenizer):
    many = [f"option number {i}" for i in range(100)]

    with pytest.raises(ValueError, match="options"):
        encode(tokenizer, record(options=many), max_length=64)
