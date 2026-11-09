from __future__ import annotations
import math
from typing import Optional, Dict
import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    Decoder-only causal Transformer.
    Forward inputs:
      - seq_tokens: LongTensor [B, L]  (concatenation of src_history + tgt_input)
      - seq_age:    FloatTensor [B, L] (aligned ages in years)
      - key_padding_mask: BoolTensor [B, L] True at PAD positions
    Returns:
      - logits: FloatTensor [B, L, V]  (vocab logits per position)
      - age_preds: FloatTensor [B, L] (legacy head, not trained by src.train)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.linear = torch.nn.Linear(d_model,d_model, bias=False)

        # Age encoding scale terms
        # vector of frequency scales
        age_div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("age_div_term", age_div_term)

        # encoder with causal mask
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # output heads
        self.code_head = nn.Linear(d_model, vocab_size)
        self.age_head = nn.Linear(d_model, 1)

        # Init
        self.reset_parameters()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Accept historical checkpoints containing unused model components."""
        legacy_keys = {
            "pos_emb.weight",
            "age_proj.weight", "age_proj.bias",
            "age_proj.0.weight", "age_proj.0.bias",
            "age_proj.2.weight", "age_proj.2.bias",
            "pool_ffn.0.weight", "pool_ffn.0.bias",
            "pool_ffn.2.weight", "pool_ffn.2.bias",
        }
        for key in legacy_keys:
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.code_head.weight)
        nn.init.zeros_(self.code_head.bias)
        # init age head
        nn.init.xavier_uniform_(self.age_head.weight)
        nn.init.zeros_(self.age_head.bias)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular (1s above diagonal) boolean mask of shape [T, T]
        True where positions are masked
        """
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _embed_tokens_with_age(
        self,
        tokens: torch.LongTensor,
        ages: torch.FloatTensor,
    ) -> torch.Tensor:
        # tokens: [B, L]
        # ages:   [B, L] (float)
        tok = self.tok_emb(tokens) # [B, L, D]
        age = self._age_encoding(ages) # [B, L, D]

        return tok + age

    def _age_encoding(self, ages: torch.Tensor) -> torch.Tensor:
        """
        ages: [B, L]
        returns: [B, L, d_model]
        """
        B, L = ages.shape
        t = ages.unsqueeze(-1)

        # self.age_div_term: [d_model/2]
        # arg: [B, L, d_model/2]
        arg = t * self.age_div_term.view(1, 1, -1)

        # build [B, L, d_model]
        y = torch.zeros(B, L, self.d_model, device=ages.device, dtype=arg.dtype)
        y[..., 0::2] = torch.sin(arg)
        y[..., 1::2] = torch.cos(arg)

        y = self.linear(y)
        return y


    def forward(
        self,
        seq_tokens: torch.LongTensor,           # [B, L] -> history concat decoder-input
        seq_age: torch.FloatTensor,             # [B, L]
        key_padding_mask: Optional[torch.BoolTensor] = None,  # [B, L] True at PAD
    ) -> Dict[str, torch.Tensor]:

        B, L = seq_tokens.shape
        x = self._embed_tokens_with_age(seq_tokens, seq_age)  # [B, L, D]
        causal = self._causal_mask(L, device=x.device)        # [L, L] mask
        out = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)  # [B, L, D]

        logits = self.code_head(out)       # [B, L, V]
        age_preds = self.age_head(out).squeeze(-1)  # [B, L]

        if key_padding_mask is not None:
            # key_padding_mask: True at PAD → we want False as "keep"
            keep = ~key_padding_mask                    # [B, L]
            hidden_masked = out * keep.unsqueeze(-1)    # [B, L, D], PAD→0
            lengths = keep.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            pooled = hidden_masked.sum(dim=1) / lengths           # [B, D]
        else:
            pooled = out.mean(dim=1)                                # [B, D]

        pooled_logits = self.code_head(pooled)          # [B, V]
        pooled_age = self.age_head(pooled).squeeze(-1)  # [B]
        # pooled logits are usually not used. for prediction logits of 'last valid token' are used
        # age prediction unused now
        return {"logits": logits, "age_preds": age_preds, "hidden": out, "pooled_logits": pooled_logits, "pooled_age": pooled_age}
