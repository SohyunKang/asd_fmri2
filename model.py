from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeripherySeedness(nn.Module):
    """Learnable seedness vector s in [0,1]^N."""
    def __init__(self, n_rois: int):
        super().__init__()
        self.u_per = nn.Parameter(torch.zeros(n_rois))  # logits
        # randomness added
        nn.init.normal_(self.u_per, std=0.1)

    def forward(self) -> torch.Tensor:
        # return torch.sigmoid(self.u_per)  # (N,)
        return self.u_per  # (N,)


class GraphDeepBlock(nn.Module):
    """Node-wise deep block (no A), PreNorm-FFN-Residual."""
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        act: str = "gelu",
        use_gated_residual: bool = False,
    ):
        super().__init__()
        d_ff = d_ff or (4 * d_model)

        self.norm = nn.LayerNorm(d_model)
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
            self.gate = nn.Parameter(torch.zeros(1))  # scalar gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)

        if self.use_gated_residual:
            g = torch.sigmoid(self.gate)
            return x + g * h
        return x + h


class EdgeGatedPropagation(nn.Module):
    """
    Hop-wise learned propagation P^(k) from node states (attention-like edge gating).

    We compute scores from node states H_bar (B,N,d):
      score_ij = (q_i · k_j)/sqrt(dk) + log(A_ij + eps)
    then row-softmax over j => P (B,N,N).

    This makes propagation hop-dependent and learned, while respecting FC structure via log(A).
    """
    def __init__(self, d_model: int, d_k: Optional[int] = None, eps: float = 1e-8):
        super().__init__()
        self.d_k = d_k or d_model
        self.eps = eps
        self.Wq = nn.Linear(d_model, self.d_k, bias=False)
        self.Wk = nn.Linear(d_model, self.d_k, bias=False)

    def forward(self, A: torch.Tensor, H_bar: torch.Tensor) -> torch.Tensor:
        B, N, _ = A.shape
        q = self.Wq(H_bar)
        k = self.Wk(H_bar)

        # A = torch.abs(A)
        A = A + 1
        # A = A / (A.sum(-1, keepdim=True) + 1e-8)

        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)  # (B,N,N)
        
        # 1. attention
        # attn = F.softmax(score, dim=-1)                                 # (B,N,N)

        # # FC 구조 반영: attn을 A로 gate하고 row-renorm
        # P = attn * A
        # P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 2. 
        score = score + torch.log(A + self.eps)

        P = F.softmax(score, dim=-1)

        return P


class SeedHopEncoder(nn.Module):
    """
    Seed-conditioned hop-recurrent encoder with:
      - learned edge-gated P^(k)
      - explicit seed flows p^(k) (destination decomposition)
      - seed-conditioned node states X^(k)

    Shapes:
      X: (B, M, N, d)
      p: (B, M, N)
      P: (B, N, N)
    """
    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_hops: int,
        depth: int = 2,
        dropout: float = 0.2,
        seed_std: float = 0.5,
        act: str = "gelu",
        use_gated_residual: bool = True,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.n_hops = n_hops

        # shared feature mixing per hop
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

        # node feature projection
        self.fc_proj = nn.Linear(n_rois, d_model)

        # node identity embedding
        self.id_embed = nn.Embedding(n_rois, d_model)

        # seed token (shared) used to initialize seed-specific latent at the seed node
        self.seed_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.seed_token, mean=0.0, std=seed_std)

        self.propagation = EdgeGatedPropagation(d_model=d_model)

        self.deep = nn.Sequential(*[
            GraphDeepBlock(
                d_model=d_model,
                dropout=dropout,
                act=act,
                use_gated_residual=use_gated_residual,
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _bmm_seed_p(p: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        p: (B,M,N) row vectors
        P: (B,N,N)
        returns p_new: (B,M,N) where p_new = p @ P
        """
        B, M, N = p.shape
        p2 = p.reshape(B * M, 1, N)            # (BM,1,N)
        P2 = P.unsqueeze(1).expand(B, M, N, N) # (B,M,N,N)
        P2 = P2.reshape(B * M, N, N)           # (BM,N,N)
        out = torch.bmm(p2, P2).squeeze(1)     # (BM,N)
        return out.reshape(B, M, N)

    @staticmethod
    def _bmm_seed_X(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        P: (B,N,N)
        X: (B,M,N,d)
        returns: (B,M,N,d) where X_new = P @ X (node-dimension)
        """
        B, M, N, d = X.shape
        X2 = X.reshape(B * M, N, d)            # (BM,N,d)
        P2 = P.unsqueeze(1).expand(B, M, N, N) # (B,M,N,N)
        P2 = P2.reshape(B * M, N, N)           # (BM,N,N)
        out = torch.bmm(P2, X2)                # (BM,N,d)
        return out.reshape(B, M, N, d)

    def forward(
        self,
        A: torch.Tensor,
        seed_indices: torch.Tensor,     # (M,)
        seed_weights: torch.Tensor,     # (M,)  (already normalized alpha)
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        A: (B,N,N) normalized FC (row-norm recommended)
        seed_indices: (M,) top-m seed node indices
        seed_weights: (M,) alpha weights (sum=1), fixed from seedness
        returns:
          g: (B, K*d) subject representation
          aux: dict with P_list, p_list, g_seed_list, etc.
        """
        device = A.device
        B, N, _ = A.shape
        M = seed_indices.numel()

        # ----- init p^(0): one-hot per seed -----
        p = torch.zeros(B, M, N, device=device, dtype=A.dtype)  # (B,M,N)
        p[:, torch.arange(M, device=device), seed_indices] = 1.0

        # ----- init X^(0): A 기반 node feature 사용 -----
        # connectivity profile
        fc_feat = self.fc_proj(A)            # (B,N,d)

        # node identity embedding
        node_ids = torch.arange(N, device=A.device)
        id_feat = self.id_embed(node_ids)    # (N,d)
        id_feat = id_feat.unsqueeze(0).expand(B, -1, -1)   # (B,N,d)

        # combine
        deg_feat = fc_feat + id_feat

        # 3️⃣ seed 채널 차원 추가
        X = deg_feat.unsqueeze(1).repeat(1, M, 1, 1)  # (B,M,N,d)

        # 4️⃣ seed 위치에 seed_token 추가 (subject-specific 강조)
        for j in range(M):
            idx = int(seed_indices[j].item())
            X[:, j, idx, :] += self.seed_token

        # optional: scale initial token by alpha (fixed seedness weight)
        # X = X * seed_weights.view(1, M, 1, 1)

        P_list: List[torch.Tensor] = []
        p_list: List[torch.Tensor] = []
        g_seed_list: List[torch.Tensor] = []
        g_bar_list: List[torch.Tensor] = []
        H_list: List[torch.Tensor] = []

        # ----- init p_all^(0): ALL-seed arrival, p_all[b, i, j] = reach j from seed i -----
        I = torch.eye(N, device=device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)  # (B,N,N)
        p_all = I.clone()
        p_all_list: List[torch.Tensor] = []

        # hop recurrence
        for k in range(1, self.n_hops + 1):
            # H_bar = X.mean(dim=1)  # (B,N,d) aggregate across seeds (keeps hop-dependence)
            alpha = seed_weights.view(1, M, 1, 1)        # (1,M,1,1)
            H_bar = (X * alpha).sum(dim=1)               # (B,N,d)

            # (선택) 정규화해주고 싶으면 아래 한 줄 추가 (안 해도 됨)
            # H_bar = H_bar / (alpha.sum(dim=1) + 1e-8)

            P = self.propagation(A, H_bar)  # (B,N,N)
            # update ALL-seed arrivals: p_all = p_all @ P  => (B,N,N)
            p_all = torch.bmm(p_all, P)
            p_all_list.append(p_all)

            P_list.append(P)

            # update flow p: (B,M,N)
            p = self._bmm_seed_p(p, P)
            p_list.append(p)

            # propagate node features with the same P
            Z = self._bmm_seed_X(P, X)          # (B,M,N,d)
            # X_new = torch.matmul(Z, self.W)     # (B,M,N,d)

            # residual 추가
            X_new = 0.3 * torch.matmul(Z, self.W) + 0.7 * X

            # residual
            # X_new = X_new + X
            # deep blocks (node-wise)
            X_new = self.deep(X_new)
            X = self.out_norm(X_new)

            H_list.append(X.detach())

            # flow-weighted pooling: g_{b,s}^{(k)} = sum_j p_{b,s}[j] * X_{b,s}[j,:]
            g_seed = torch.einsum("bmn,bmnd->bmd", p, X)  # (B,M,d)
            g_seed_list.append(g_seed)

            # fixed seed integration: \bar g^{(k)} = sum_s alpha_s g_s^{(k)}
            g_bar = torch.einsum("m,bmd->bd", seed_weights, g_seed)  # (B,d)
            g_bar_list.append(g_bar)

        # concat hop-level integrated embeddings -> subject rep
        g = torch.cat(g_bar_list, dim=-1)  # (B,K*d)

        aux: Dict[str, Any] = {
            "seed_indices": seed_indices.detach().cpu(),      # (M,)
            # "seed_weights": seed_weights.detach().cpu(),      # (M,)
            "seed_weights": seed_weights,      # (M,)
            "P_list": [t.detach() for t in P_list],           # list of (B,N,N)
            "p_list": [t.detach() for t in p_list],           # list of (B,M,N)
            "g_seed_list": [t.detach() for t in g_seed_list], # list of (B,M,d)
            "g_bar_list": [t.detach() for t in g_bar_list],   # list of (B,d)
            "g": g.detach(),                                  # (B,K*d)
            "p_all_list": [t.detach() for t in p_all_list],     # list of (B,N,N), length K
            "arrival_all": p_all_list[-1].detach() if len(p_all_list) > 0 else None,  # (B,N,N),
            "H_list": H_list,

        }
        return g, aux


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
    HiTraNet (new):
      - PeripherySeedness -> select top-m seeds
      - SeedHopEncoder with edge-gated P^(k), explicit flow p^(k)
      - fixed seed integration weights alpha (from seedness)
      - classifier on concat_{k} \bar g^{(k)}

    """
    def __init__(
        self,
        n_rois: int,                    # of regions, eg. aal=116
        d_model: int,                   # of latent node feature dimension
        n_hops: int,                    # of recurrent hops
        top_m: int,                     # of seeds
        mlp_hidden: int,                # of node in classifier
        n_classes: int,                 # of class, eg. 2 (ASD/TD)
        depth: int = 3,                 # of modules. @@
        dropout: float = 0.2,           # 
        seed_std: float = 0.5,          # Random Seed Standard Dev.
        act: str = "gelu",              #
        use_gated_residual: bool = True,# @@
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.n_hops = n_hops
        self.top_m = top_m

        self.periphery_seedness = PeripherySeedness(n_rois)
        self.encoder = SeedHopEncoder(
            n_rois=n_rois,
            d_model=d_model,
            n_hops=n_hops,
            depth=depth,
            dropout=dropout,
            seed_std=seed_std,
            act=act,
            use_gated_residual=use_gated_residual,
        )
        self.head = MLPHead(in_dim=n_hops * d_model, hidden_dim=mlp_hidden, n_classes=n_classes, dropout=dropout)

    def topm_seeds_st(self, tau: float = 1.0):
        """
        returns:
          seed_indices: (M,)
          seed_weights: (M,) normalized alpha (sum=1)
        """
        s = self.periphery_seedness()  # (N,)
        M = min(self.top_m, s.numel())

        # Gumbel TopK
        seed_indices = self.gumbel_topk(s, M, tau)  # (M,)

        # weights
        w = s[seed_indices]
        seed_weights = F.softmax(w, dim=0)

        return seed_indices, seed_weights

    def gumbel_topk(self, logits, k, tau=1.0):
        U = torch.rand_like(logits) # [0,1)
        g = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        
        y = (logits + g) / tau
        _, idx = torch.topk(y, k=k)

        return idx

    @torch.no_grad()
    def topm_seeds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          seed_indices: (M,)
          seed_weights: (M,) normalized alpha (sum=1)
        """
        s = self.periphery_seedness()  # (N,)
        M = min(self.top_m, s.numel())

        # Just TopK
        seed_indices = torch.topk(s, k=M, largest=True).indices  # (M,)
        
        # weights
        w = s[seed_indices]
        seed_weights = F.softmax(w, dim=0)

        return seed_indices, seed_weights

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.training:
            seed_indices, seed_weights = self.topm_seeds_st()
        else:
            seed_indices, seed_weights = self.topm_seeds()

        g, aux = self.encoder(A, seed_indices=seed_indices, seed_weights=seed_weights)
        logits = self.head(g)
        # also expose seedness vector for logging
        aux["seedness"] = self.periphery_seedness().detach()
        return logits, aux
