"""OptionScorer: an encoder plus a Linear(hidden, 1) head applied at every <opt> position.
Undecided: the user-facing wrapper that classifies a text against options chosen at call time."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from undecided.data import Record
from undecided.device import pick_device
from undecided.encoding import OPT_TOKEN, collate, encode


class OptionScorer(nn.Module):
    def __init__(self, encoder: nn.Module, opt_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.opt_token_id = opt_token_id
        # <opt> lives in its own parameter rather than a row of the embedding matrix, so the
        # pretrained vocabulary stays untouched. It starts at the mean pretrained embedding.
        pretrained = encoder.get_input_embeddings().weight.detach()
        self.opt_embedding = nn.Parameter(pretrained.mean(dim=0).clone())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        is_opt = input_ids == self.opt_token_id  # (batch, seq)
        # <opt> may be outside the pretrained vocabulary, so look it up as padding and overwrite.
        safe_ids = input_ids.masked_fill(is_opt, self.encoder.config.pad_token_id)
        embeds = self.encoder.get_input_embeddings()(safe_ids)
        embeds = torch.where(is_opt[..., None], self.opt_embedding, embeds)

        hidden = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        counts = is_opt.sum(dim=1)  # options per example
        max_opts = int(counts.max())
        opt_mask = torch.arange(max_opts, device=hidden.device)[None] < counts[:, None]
        opt_hidden = hidden.new_zeros(hidden.size(0), max_opts, hidden.size(-1))
        opt_hidden[opt_mask] = hidden[is_opt]  # (batch, max_opts, hidden)
        logits = self.head(opt_hidden).squeeze(-1)
        return logits.masked_fill(~opt_mask, float("-inf"))  # padded options get p = 0


class Undecided:
    def __init__(self, scorer: OptionScorer, tokenizer, max_length: int, device: torch.device):
        self.scorer = scorer.to(device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def save_pretrained(self, path: str | Path) -> None:
        """Weights, encoder config and tokeniser: everything needed to reload without the hub."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), path / "scorer.pt")
        self.scorer.encoder.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "undecided.json").write_text(json.dumps({"max_length": self.max_length}) + "\n")

    @classmethod
    def from_pretrained(cls, path: str | Path, device: torch.device | None = None) -> "Undecided":
        path = Path(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path))
        scorer = OptionScorer(encoder, tokenizer.convert_tokens_to_ids(OPT_TOKEN))
        scorer.load_state_dict(torch.load(path / "scorer.pt", map_location="cpu"))
        settings = json.loads((path / "undecided.json").read_text())
        return cls(scorer, tokenizer, settings["max_length"], device or pick_device())

    @torch.no_grad()
    def classify(self, text: str, options: list[str], instruction: str | None = None) -> dict[str, float]:
        if len(options) < 2:
            raise ValueError(f"need at least 2 options, got {len(options)}")
        if len(set(options)) != len(options):
            raise ValueError(f"duplicate options: {sorted({o for o in options if options.count(o) > 1})}")
        record = Record(instruction or "", text, list(options), 0, "classify", "inference")
        batch = collate([encode(self.tokenizer, record, self.max_length)], self.tokenizer.pad_token_id)
        logits = self.scorer(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
        probs = logits[0].softmax(-1).tolist()
        return dict(zip(options, probs))
