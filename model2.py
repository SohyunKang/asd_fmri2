from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeripherySeedness(nn.Module):
    """
    Learnable seedness vector s in [0,1]^N.
    """
    def __init__(self, n_rois: int):
        super().__init__()
        self.u_per = nn.Parameter(torch.zeros(n_rois))  # logits

    def forward(self) -> torch.Tensor:
        # (N,)
        return torch.sigmoid(self.u_per)


class HierarchicalTrajectoryEncoder(nn.Module):
    """
    Hop-recurrent encoder.
    - A: (B, N, N) propagation operator
    - H^(0): diag(s) * I_N  (B, N, N)
    Produces:
    - H_list: list of hop-wise node states, each (B, N, d)
      where row corresponds to seed-specific latent feature (interpretation per your design).
    """
    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_hops: int,
        use_residual: bool = True,
        norm_type: str = "layer",  # "layer" or "none"
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.n_hops = n_hops
        self.use_residual = use_residual
        self.norm_type = norm_type

        # First projection from (N) -> (d) after A @ H0 gives (N,N)
        self.W_in = nn.Parameter(torch.empty(n_rois, d_model))
        nn.init.xavier_uniform_(self.W_in)

        # Shared projection for subsequent hops: (d) -> (d)
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()

    def forward(self, A: torch.Tensor, seedness: torch.Tensor) -> List[torch.Tensor]:
        """
        A: (B, N, N)
        seedness: (N,) or (B, N)
        returns: list H^(1..K), each (B, N, d)
        """
        B, N, _ = A.shape
        assert N == self.n_rois

        if seedness.dim() == 1:
            s = seedness.unsqueeze(0).expand(B, -1)  # (B, N)
        else:
            s = seedness  # (B, N)

        # H0 = diag(s) * I => (B, N, N) with s on diagonal
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        H0 = I * s.unsqueeze(-1)  # broadcast along columns => diagonal gets s

        H_list: List[torch.Tensor] = []

        # Hop 1: (B,N,N) -> (B,N,N), then project to (B,N,d)
        Z = A @ H0  # (B, N, N)
        H = Z @ self.W_in  # (B, N, d)

        if self.use_residual:
            # Residual from a projected identity (seed-specific basis)
            # Project H0 to same dimension:
            H_res = H0 @ self.W_in  # (B, N, d)
            H = H + H_res

        H = self.norm(H)
        H_list.append(H)

        # Hops 2..K: recurrent with shared W
        for _k in range(2, self.n_hops + 1):
            Z = A @ H               # (B, N, d)
            H_new = Z @ self.W      # (B, N, d)
            if self.use_residual:
                H_new = H_new + H
            H = self.norm(H_new)
            H_list.append(H)

        return H_list


from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDeepBlock(nn.Module):
    """
    Node-wise deep block (does NOT apply A). Depth is added here, not via hop.
    Transformer-ish FFN block: PreNorm -> FFN -> Residual
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        norm_type: str = "layer",
        act: str = "gelu",
        use_gated_residual: bool = False,
    ):
        super().__init__()
        d_ff = d_ff or (4 * d_model)

        self.norm = nn.LayerNorm(d_model) if norm_type == "layer" else nn.Identity()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

        self.use_gated_residual = use_gated_residual
        if use_gated_residual:
            # learnable residual gate in [0,1]
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)

        if self.use_gated_residual:
            g = torch.sigmoid(self.gate)
            return x + g * h
        else:
            return x + h


class HierarchicalTrajectoryEncoderDeep(nn.Module):
    """
    Hop-recurrent encoder with depth inside each hop update (WITHOUT increasing hop count).
    - Each hop does: Z = A @ H  (message passing)
      then apply L deep blocks: H = DeepBlocks(Z)
    """
    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_hops: int,
        depth: int = 2,              # <- NEW: number of layers per hop
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_residual: bool = True,   # hop-level residual (H_new += H)
        norm_type: str = "layer",
        act: str = "gelu",
        use_gated_residual: bool = False,
        share_depth_across_hops: bool = True,  # 공유/비공유
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.n_hops = n_hops
        self.depth = depth
        self.use_residual = use_residual
        self.norm_type = norm_type
        self.share_depth_across_hops = share_depth_across_hops

        # Hop1 projection: (N) -> (d) after A @ H0 gives (N,N)
        self.W_in = nn.Parameter(torch.empty(n_rois, d_model))
        nn.init.xavier_uniform_(self.W_in)

        # For hop>=2: (d)->(d) right-multiply after A@H
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

        # Optional final norm per hop output (you had self.norm)
        self.out_norm = nn.LayerNorm(d_model) if norm_type == "layer" else nn.Identity()

        def make_depth_stack():
            return nn.Sequential(*[
                GraphDeepBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_type=norm_type,
                    act=act,
                    use_gated_residual=use_gated_residual,
                )
                for _ in range(depth)
            ])

        if share_depth_across_hops:
            # same deep stack for all hops (clean & parameter efficient)
            self.deep = make_depth_stack()
        else:
            # separate deep stack per hop (more expressive)
            self.deep_per_hop = nn.ModuleList([make_depth_stack() for _ in range(n_hops)])

    def _apply_deep(self, x: torch.Tensor, hop_idx: int) -> torch.Tensor:
        if self.share_depth_across_hops:
            return self.deep(x)
        else:
            return self.deep_per_hop[hop_idx](x)

    def forward(self, A: torch.Tensor, seedness: torch.Tensor) -> List[torch.Tensor]:
        """
        A: (B, N, N)
        seedness: (N,) or (B, N)
        returns: list H^(1..K), each (B, N, d)
        """
        B, N, _ = A.shape
        assert N == self.n_rois

        if seedness.dim() == 1:
            s = seedness.unsqueeze(0).expand(B, -1)  # (B, N)
        else:
            s = seedness  # (B, N)

        # H0 = diag(s) * I => (B, N, N)
        I = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        H0 = I * s.unsqueeze(-1)

        H_list: List[torch.Tensor] = []

        # Hop 1
        Z = A @ H0            # (B, N, N)
        H = Z @ self.W_in     # (B, N, d)

        if self.use_residual:
            H_res = H0 @ self.W_in
            H = H + H_res

        # Depth inside hop 1 (node-wise)
        H = self._apply_deep(H, hop_idx=0)
        H = self.out_norm(H)
        H_list.append(H)

        # Hop 2..K
        for k in range(1, self.n_hops):
            Z = A @ H              # (B, N, d)
            H_new = Z @ self.W     # (B, N, d)

            if self.use_residual:
                H_new = H_new + H  # hop-level residual

            # Depth inside hop k+1
            H_new = self._apply_deep(H_new, hop_idx=k)
            H = self.out_norm(H_new)
            H_list.append(H)

        return H_list


class TransmodalArrival(nn.Module):
    """
    Arrival weighting at final hop K:
      a = softmax(HK @ w_tm) over nodes
      HK_tm = diag(a) HK
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.w_tm = nn.Parameter(torch.zeros(d_model, 1))
        nn.init.xavier_uniform_(self.w_tm)

    def forward(self, HK: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HK: (B, N, d)
        returns:
          HK_tm: (B, N, d)
          a: (B, N) arrival distribution
        """
        # (B, N, 1)
        score = HK @ self.w_tm
        a = F.softmax(score.squeeze(-1), dim=-1)  # (B, N)

        HK_tm = HK * a.unsqueeze(-1)  # node-wise scaling
        return HK_tm, a


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiTraNet(nn.Module):
    """
    HiTraNet = PeripherySeedness (learnable) + HierarchicalTrajectoryEncoder + TransmodalArrival + MLP.

    Outputs:
      logits: (B, C)
      aux: dict for interpretability (seedness, arrival, hop embeddings)
    """
    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_hops: int,
        mlp_hidden: int,
        n_classes: int,
        dropout: float = 0.2,
        pool: str = "mean",  # mean or sum
        use_residual: bool = True,
        norm_type: str = "layer",
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.n_hops = n_hops
        self.pool = pool

        self.periphery_seedness = PeripherySeedness(n_rois)
        # self.encoder = HierarchicalTrajectoryEncoder(
        #     n_rois=n_rois,
        #     d_model=d_model,
        #     n_hops=n_hops,
        #     use_residual=use_residual,
        #     norm_type=norm_type,
        # )
        self.encoder = HierarchicalTrajectoryEncoderDeep(
            n_rois=n_rois,
            d_model=d_model,
            n_hops=n_hops,
            depth=3,
            dropout=0.2,
            use_residual=True,
            norm_type="layer",
            act="gelu",
            use_gated_residual=True,
            share_depth_across_hops=True,
        )
        self.transmodal = TransmodalArrival(d_model)

        # We will pool each hop to get g^(k) in R^d, then concat => K*d
        self.head = MLPHead(in_dim=n_hops * d_model, hidden_dim=mlp_hidden, n_classes=n_classes, dropout=dropout)

    def _pool(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, N, d) -> (B, d)
        if self.pool == "mean":
            return H.mean(dim=1)
        if self.pool == "sum":
            return H.sum(dim=1)
        raise ValueError(f"Unknown pool: {self.pool}")

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        A: (B, N, N) normalized FC operator
        """
        s = self.periphery_seedness()  # (N,)
        H_list = self.encoder(A, s)    # list of (B, N, d), length K

        # Apply transmodal arrival to the final hop only
        HK = H_list[-1]
        HK_tm, a = self.transmodal(HK)

        # Use all hops for prediction; replace last hop with transmodal-weighted one
        g_list = []
        for k in range(self.n_hops - 1):
            g_list.append(self._pool(H_list[k]))
        g_list.append(self._pool(HK_tm))

        g = torch.cat(g_list, dim=-1)  # (B, K*d)
        logits = self.head(g)

        aux = {
            "seedness": s.detach(),        # (N,)
            "arrival": a.detach(),         # (B, N)
            "H_list": [h.detach() for h in H_list],  # for visualization if needed
            "g": g.detach(),               # (B, K*d)
        }
        return logits, aux
