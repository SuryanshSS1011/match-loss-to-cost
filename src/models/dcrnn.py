"""DCRNN forecaster (Li, Yu, Shahabi, Liu, ICLR 2018).

Reference: Y. Li, R. Yu, C. Shahabi, Y. Liu, "Diffusion Convolutional
Recurrent Neural Network: Data-Driven Traffic Forecasting," ICLR 2018.
Original repo: liyaguang/DCRNN.

Self-contained port. Two pieces:
  1. Diffusion convolution: K-step random walk on a transition matrix
     plus the same on its transpose, with learnable per-step weights.
  2. DCGRU cell: standard GRU but the gate computations are diffusion
     convolutions over the graph instead of plain Linear layers.

We stack `num_layers` DCGRU cells and fold the recurrent hidden state
through the full input window. For our `horizon=1` use case the
encoder-only variant suffices: read the final hidden state and project
to (batch, num_links) via a per-link Linear head. (The published DCRNN
adds a separate decoder for multi-step; not needed here.)

Adjacency. Built from the routing matrix `R ∈ R^{num_links × num_od}`:
two links are connected if they appear on the route of any common OD
pair, i.e. `A[i, j] = 1[(R @ R.T)[i, j] > 0]` with the diagonal zeroed.
We then row-normalise to a transition matrix. For datasets where R is
absent or `R = I` (e.g. CESNET), DCRNN reduces to a per-link GRU — the
diffusion conv does no spatial mixing and the cell behaves like a
standard GRU per channel. Documented; that's still a reasonable baseline.

Shape contract:
    forward(x: (batch, window_size, num_links)) -> (batch, num_links)

No torch_geometric dependency. The diffusion conv is `(P @ ... @ X) @ W`
which is fully expressible in plain pytorch ops.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def adjacency_from_routing(R: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Boolean link-link adjacency from a routing matrix R.

    Args:
        R: (num_links, num_od) routing matrix. Entry (l, k) is 1 (or a
            fractional weight in [0, 1]) iff link l carries demand k.
    Returns:
        A: (num_links, num_links) symmetric boolean adjacency with the
            diagonal zeroed. `A[i, j] = 1` iff links i and j share at
            least one routed OD pair.
    """
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D (num_links, num_od); got {R.shape}")
    overlap = (R > 0).astype(np.float32) @ (R > 0).astype(np.float32).T
    A = (overlap > 0).astype(dtype)
    np.fill_diagonal(A, 0.0)
    return A


def random_walk_transition(A: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Row-normalised random-walk transition matrix `D^{-1} A`.

    Rows that sum to zero (isolated nodes) become all-zero rows — the
    diffusion conv leaves them untouched at every step.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D; got {A.shape}")
    row_sum = A.sum(axis=1, keepdims=True)
    P = np.where(row_sum > 0, A / np.maximum(row_sum, 1e-12), 0.0)
    return P.astype(dtype)


# ---------------------------------------------------------------------------
# Diffusion convolution
# ---------------------------------------------------------------------------

class _DiffusionConv(nn.Module):
    """Bidirectional K-step diffusion convolution.

    Computes
        Y = Σ_{k=0..K-1} (P^k X) W_k^fwd  +  Σ_{k=0..K-1} (P_T^k X) W_k^bwd
    where `X ∈ R^{B×N×F_in}` and the output is `R^{B×N×F_out}`. The
    constant term `k=0` is `X W_0`, which is just a linear projection
    independent of the graph.

    `P` and `P_T` are registered buffers (no gradient) of shape (N, N).
    The weights `W_k` are stored stacked as `(2K, F_in, F_out)`.
    """

    def __init__(self, P_fwd: torch.Tensor, P_bwd: torch.Tensor,
                 in_dim: int, out_dim: int, K: int):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        if P_fwd.shape != P_bwd.shape or P_fwd.dim() != 2:
            raise ValueError(
                f"P_fwd / P_bwd must be square 2-D; got {tuple(P_fwd.shape)}, "
                f"{tuple(P_bwd.shape)}"
            )
        self.K = int(K)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.register_buffer("P_fwd", P_fwd)
        self.register_buffer("P_bwd", P_bwd)
        # Two directions × K steps → 2K weight matrices.
        self.weight = nn.Parameter(torch.empty(2 * K, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in)
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, num_nodes, in_dim); got {tuple(x.shape)}"
            )
        out = self.bias[None, None, :].expand(x.shape[0], x.shape[1], -1).clone()
        # Forward direction.
        y = x  # (B, N, F_in)
        for k in range(self.K):
            if k > 0:
                y = self.P_fwd @ y
            out = out + y @ self.weight[k]
        # Backward direction.
        y = x
        for k in range(self.K):
            if k > 0:
                y = self.P_bwd @ y
            out = out + y @ self.weight[self.K + k]
        return out


# ---------------------------------------------------------------------------
# DCGRU cell
# ---------------------------------------------------------------------------

class _DCGRUCell(nn.Module):
    """One layer of DCGRU.

    Standard GRU update with diffusion conv replacing the gate Linears:
        r = σ(DC([x ; h]))
        u = σ(DC([x ; h]))
        c = tanh(DC([x ; r⊙h]))
        h' = u⊙h + (1 - u)⊙c
    """

    def __init__(self, P_fwd: torch.Tensor, P_bwd: torch.Tensor,
                 in_dim: int, hidden_dim: int, K: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        # Each gate's diffusion conv consumes the concat of x and h, so
        # input dim is `in_dim + hidden_dim`.
        self.dc_r = _DiffusionConv(P_fwd, P_bwd,
                                    in_dim + hidden_dim, hidden_dim, K)
        self.dc_u = _DiffusionConv(P_fwd, P_bwd,
                                    in_dim + hidden_dim, hidden_dim, K)
        self.dc_c = _DiffusionConv(P_fwd, P_bwd,
                                    in_dim + hidden_dim, hidden_dim, K)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: (B, N, in_dim or hidden_dim)
        xh = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.dc_r(xh))
        u = torch.sigmoid(self.dc_u(xh))
        xrh = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.dc_c(xrh))
        return u * h + (1.0 - u) * c


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class DCRNNForecaster(nn.Module):
    """Encoder-only DCRNN with a per-link linear head.

    Args:
        input_size: number of nodes / channels (= num_links).
        window_size: input length.
        adjacency: (num_links, num_links) numpy adjacency matrix. If None,
            falls back to the identity, which makes DCRNN reduce to a
            per-link GRU (no spatial mixing).
        horizon: forecast horizon. Default 1.
        hidden_dim: hidden state size per node.
        num_layers: number of stacked DCGRU layers.
        K: number of diffusion steps per direction. Total spatial filter
            size is K (forward) + K (backward).
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        adjacency: Optional[np.ndarray] = None,
        horizon: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        K: int = 2,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1; got {input_size}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        self.input_size = int(input_size)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.K = int(K)

        if adjacency is None:
            A = np.eye(input_size, dtype=np.float32)
        else:
            A = np.asarray(adjacency, dtype=np.float32)
            if A.shape != (input_size, input_size):
                raise ValueError(
                    f"adjacency shape {A.shape} != "
                    f"({input_size}, {input_size})"
                )
        P_fwd = torch.from_numpy(random_walk_transition(A))
        P_bwd = torch.from_numpy(random_walk_transition(A.T))

        cells = []
        in_dim = 1  # univariate signal per node
        for layer in range(self.num_layers):
            cells.append(_DCGRUCell(P_fwd, P_bwd,
                                    in_dim=in_dim,
                                    hidden_dim=self.hidden_dim,
                                    K=self.K))
            in_dim = self.hidden_dim
        self.cells = nn.ModuleList(cells)

        # Per-node linear head from final hidden state to horizon. We share
        # weights across nodes (one Linear applied independently to each
        # node's hidden vector) — keeps the parameter count flat in num_links.
        self.head = nn.Linear(self.hidden_dim, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, window, channels); got {tuple(x.shape)}"
            )
        if x.shape[1] != self.window_size:
            raise ValueError(
                f"window_size mismatch: model={self.window_size}, "
                f"input={x.shape[1]}"
            )
        if x.shape[2] != self.input_size:
            raise ValueError(
                f"input_size mismatch: model={self.input_size}, "
                f"input={x.shape[2]}"
            )

        b, t, n = x.shape
        # (B, T, N) → (T, B, N, 1) for the recurrent loop.
        x_seq = x.permute(1, 0, 2).unsqueeze(-1).contiguous()

        # Initialise per-layer hidden states to zeros.
        h_states = [
            x_seq.new_zeros(b, n, self.hidden_dim)
            for _ in range(self.num_layers)
        ]

        for step in range(t):
            inp = x_seq[step]  # (B, N, 1)
            for layer in range(self.num_layers):
                h_states[layer] = self.cells[layer](inp, h_states[layer])
                inp = h_states[layer]

        last = h_states[-1]  # (B, N, hidden_dim)
        out = self.head(last)  # (B, N, horizon)

        if self.horizon == 1:
            return out.squeeze(-1)  # (B, N)
        return out.permute(0, 2, 1)  # (B, horizon, N)
