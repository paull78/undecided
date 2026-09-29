"""OptionScorer: an encoder plus a Linear(hidden, 1) head applied at every <opt> position."""

import torch
from torch import nn


class OptionScorer(nn.Module):
    def __init__(self, encoder: nn.Module, opt_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.opt_token_id = opt_token_id
        # <opt> lives in its own parameter rather than a row of the embedding matrix, so it can
        # train at the head's learning rate. It starts at the mean pretrained embedding.
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
