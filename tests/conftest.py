import pytest

from undecided.encoding import build_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    # Real ModernBERT tokeniser: a few MB, cached by Hugging Face after the first run.
    return build_tokenizer("answerdotai/ModernBERT-base")


@pytest.fixture
def tiny_scorer(tokenizer):
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from undecided.encoding import OPT_TOKEN
    from undecided.model import OptionScorer

    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    return OptionScorer(ModernBertModel(config), tokenizer.convert_tokens_to_ids(OPT_TOKEN))
