import math

import torch

from undecided.data import Record
from undecided.encoding import collate, encode


def record(options, answer_idx=0):
    return Record("", "My card was charged twice.", list(options), answer_idx, "test", "train")


def test_batch_with_different_option_counts_gives_one_logit_per_option_and_masks_padding(tokenizer, tiny_scorer):
    examples = [
        encode(tokenizer, record(["billing", "bug report", "praise"]), max_length=128),
        encode(tokenizer, record(["a", "b", "c", "d", "e"], answer_idx=4), max_length=128),
    ]
    batch = collate(examples, pad_token_id=tokenizer.pad_token_id)

    logits = tiny_scorer(batch["input_ids"], batch["attention_mask"])

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits[0, :3]).all() and torch.isfinite(logits[1]).all()
    assert (logits[0, 3:] == -math.inf).all()
    assert batch["labels"].tolist() == [0, 4]
    probs = logits.softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2))


def test_opt_vector_starts_at_the_mean_embedding_and_needs_no_vocabulary_slot(tokenizer):
    from transformers import ModernBertConfig, ModernBertModel

    from undecided.encoding import OPT_TOKEN
    from undecided.model import OptionScorer

    base_vocab = len(tokenizer) - 1  # the pretrained vocabulary: <opt> has no row in it
    encoder = ModernBertModel(
        ModernBertConfig(vocab_size=base_vocab, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, pad_token_id=tokenizer.pad_token_id)
    )
    original_mean = encoder.get_input_embeddings().weight.detach().mean(dim=0)

    scorer = OptionScorer(encoder, tokenizer.convert_tokens_to_ids(OPT_TOKEN))

    assert torch.allclose(scorer.opt_embedding.detach(), original_mean)
    batch = collate([encode(tokenizer, record(["billing", "praise"]), max_length=64)], tokenizer.pad_token_id)
    assert scorer(batch["input_ids"], batch["attention_mask"]).shape == (1, 2)
