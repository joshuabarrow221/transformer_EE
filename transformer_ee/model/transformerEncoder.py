"""
Transformer Encoder based energy estimator models (backward-compatible, extended).
"""

import torch
from torch import nn
from torch.nn import functional as F


def _choose_d_model(nvec: int, nhead: int, min_head_dim: int = 8, base_multiple: int = 4) -> int:
    """
    Choose a d_model that:
      - is at least base_multiple * nvec (so you can widen vs. raw feature count)
      - gives each head at least min_head_dim dims
      - is divisible by nhead
    """
    d = max(base_multiple * nvec, nhead * min_head_dim)
    # ceil to nearest multiple of nhead
    return ((d + nhead - 1) // nhead) * nhead


class Transformer_EE_MV(nn.Module):
    """
    Information of slice and prongs are concatenated together.

    parameters expected in config:
        config["vector"]: list of vector feature names (per-token features)
        config["scalar"]: list of global scalar feature names
        config["target"]: list of target names

    config["model"]["kwargs"] (all optional; defaults preserve legacy behavior):
        nhead: int (default 4)
        dim_feedforward: int (default 2048)
        dropout: float (default 0.1)
        num_layers: int (default 6)
        linear_hidden: int (default 16)              # legacy scalar MLP width
        post_linear_hidden: int (default 24)         # legacy head hidden width

        # New / extended options:
        d_model: int                                 # set explicit transformer width (must be % nhead == 0)
        auto_d_model: bool = False                   # if True, compute d_model automatically (ignored if d_model is set)
        min_head_dim: int = 8                        # per-head min dim for auto_d_model
        base_multiple: int = 4                       # widen baseline for auto_d_model

        scalar_as_token: bool = False                # if True, project scalars to a "[CLS]-like" token and prepend
        use_new_head: bool = False                   # if True, use LayerNorm+GELU MLP on pooled token (new head)
        norm_first: bool = True                      # set norm_first on TransformerEncoderLayer
    """

    def __init__(self, config):
        super().__init__()
        kwargs = dict(config.get("model", {}).get("kwargs", {}))

        # --- basic sizes
        self.nvec = len(config["vector"])
        self.nsca = len(config["scalar"])
        self.ntgt = len(config["target"])

        # --- transformer hyperparams (legacy defaults preserved)
        self.nhead = kwargs.get("nhead", 4)
        self.num_layers = kwargs.get("num_layers", 6)
        self.dropout = kwargs.get("dropout", 0.1)
        self.dim_ff = kwargs.get("dim_feedforward", 2048)
        self.norm_first = bool(kwargs.get("norm_first", True))

        # --- extended toggles
        self.scalar_as_token = bool(kwargs.get("scalar_as_token", False))
        self.use_new_head = bool(kwargs.get("use_new_head", False))

        # --- choose d_model (backward-compatible default is raw feature count)
        auto_d_model = bool(kwargs.get("auto_d_model", False))
        explicit_d_model = kwargs.get("d_model", None)

        if explicit_d_model is not None:
            d_model = int(explicit_d_model)
        elif auto_d_model:
            d_model = _choose_d_model(
                nvec=self.nvec,
                nhead=self.nhead,
                min_head_dim=int(kwargs.get("min_head_dim", 8)),
                base_multiple=int(kwargs.get("base_multiple", 4)),
            )
        else:
            # Legacy width: require nvec % nhead == 0 (exactly your original constraint)
            d_model = self.nvec

        if d_model % self.nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={self.nhead}")

        self.d_model = d_model

        # --- input projections (only used if widening)
        self.use_vec_proj = (self.d_model != self.nvec)
        if self.use_vec_proj:
            self.vec_proj = nn.Sequential(
                nn.Linear(self.nvec, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
            )

        # If we may use a scalar token, define its projection. We'll keep the legacy
        # scalar MLP too so you can mix & match heads without breaking.
        if self.scalar_as_token:
            self.sca_proj = nn.Sequential(
                nn.Linear(self.nsca, self.d_model),
                nn.GELU(),
                nn.LayerNorm(self.d_model),
            )

        # --- transformer encoder (keep the legacy attribute name!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff if self.dim_ff is not None else max(4 * self.d_model, 128),
            dropout=self.dropout,
            batch_first=True,
            norm_first=self.norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # --- legacy scalar MLP (preserved for backward compatibility)
        linear_hidden = kwargs.get("linear_hidden", 16)
        self.linear_scalar1 = nn.Linear(self.nsca, linear_hidden)
        self.linear_scalar2 = nn.Linear(linear_hidden, linear_hidden)

        # --- legacy post head (preserved for backward compatibility)
        post_linear_hidden = kwargs.get("post_linear_hidden", 24)
        # NOTE: legacy expects concat([pooled_x, scalar_hidden]) as input
        self.linear1 = nn.Linear(self.d_model + linear_hidden, post_linear_hidden)
        self.linear2 = nn.Linear(post_linear_hidden, post_linear_hidden)
        self.linear3 = nn.Linear(post_linear_hidden, self.ntgt)

        # --- new head (optional)
        if self.use_new_head:
            self.new_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.ntgt),
            )

    def forward(self, x, y, mask):
        """
        x:     (B, L, nvec)   per-prong vector features (padded)
        y:     (B, nsca)      global scalar features
        mask:  (B, L)  bool   True where positions are PAD
        """
        B, L, nfeat = x.shape  # ← was F
        if nfeat != self.nvec:
            raise ValueError(f"Expected x feature size {self.nvec}, got {nfeat}")
    
        # (1) Project vectors to d_model if widened; else pass through
        if self.use_vec_proj:
            v = self.vec_proj(x)               # (B, L, d_model)
        else:
            v = x                              # (B, L, d_model==nvec)
    
        # (2) Build sequence (optionally with scalar-as-token)
        if self.scalar_as_token:
            s_tok = self.sca_proj(y).unsqueeze(1)    # (B, 1, d_model)
            seq = torch.cat([s_tok, v], dim=1)       # (B, 1+L, d_model)
            if mask is not None:
                pad0 = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
                enc_mask = torch.cat([pad0, mask], dim=1)  # (B, 1+L)
            else:
                enc_mask = None
        else:
            seq = v
            enc_mask = mask
    
        if seq.size(-1) != self.d_model:
            raise ValueError(f"Transformer got width {seq.size(-1)}, expected {self.d_model}")
    
        # (3) Encode
        z = self.transformer_encoder(seq, src_key_padding_mask=enc_mask)  # (B, S, d_model)
    
        # (4) Pool
        if self.scalar_as_token:
            pooled = z[:, 0, :]  # (B, d_model)
        else:
            if mask is not None:
                z = z.masked_fill(torch.unsqueeze(mask, -1), 0)
            pooled = torch.sum(z, dim=1)  # (B, d_model)
    
        # (5) Heads
        y_legacy = F.relu(self.linear_scalar1(y))
        y_legacy = F.relu(self.linear_scalar2(y_legacy))
    
        if self.use_new_head:
            out = self.new_head(pooled)
            return out
    
        h = torch.cat((pooled, y_legacy), dim=1)  # (B, d_model + linear_hidden)
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        out = self.linear3(h)
        return out

