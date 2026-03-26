"""Unit tests for src/models/dcrnn.py.

Pure tensor / numpy math. We verify:
  - adjacency_from_routing produces a symmetric, zero-diag boolean matrix.
  - random_walk_transition row-normalises (rows sum to 1 or 0 for isolates).
  - DiffusionConv shape contracts and that K=1 (no diffusion) differs from
    K=2 (graph-aware) in parameter count.
  - DCRNNForecaster shape contracts, identity-adjacency reduction (no
    cross-link signal), gradient flow, validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.dcrnn import (
    DCRNNForecaster,
    _DiffusionConv,
    adjacency_from_routing,
    random_walk_transition,
)


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

class TestAdjacency:
    def test_symmetric_zero_diag(self):
        # Shared route → both links connected.
        R = np.array([
            [1, 1, 0],   # link 0 carries demands 0 and 1
            [1, 0, 1],   # link 1 carries demands 0 and 2
            [0, 1, 0],   # link 2 carries demand 1
        ], dtype=np.float32)
        A = adjacency_from_routing(R)
        assert A.shape == (3, 3)
        # Symmetric.
        assert np.allclose(A, A.T)
        # Diagonal zeroed even though every link "shares routes with itself".
        assert np.all(np.diag(A) == 0)

    def test_disjoint_routes_no_edges(self):
        R = np.array([[1, 0], [0, 1]], dtype=np.float32)
        A = adjacency_from_routing(R)
        # No shared routes → no edges.
        assert A.sum() == 0

    def test_identity_routing_yields_no_edges(self):
        # CESNET case: R = I means each "link" is its own demand.
        R = np.eye(5, dtype=np.float32)
        A = adjacency_from_routing(R)
        assert A.sum() == 0

    def test_invalid_R_shape(self):
        with pytest.raises(ValueError):
            adjacency_from_routing(np.array([1, 2, 3]))


class TestRandomWalk:
    def test_row_sums(self):
        A = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
        P = random_walk_transition(A)
        # Row 0: sums to 1 (split between cols 1 and 2).
        # Row 1: sums to 1.
        # Row 2: isolated, sums to 0.
        assert P[0].sum() == pytest.approx(1.0)
        assert P[1].sum() == pytest.approx(1.0)
        assert P[2].sum() == 0.0

    def test_uniform_for_complete_graph(self):
        N = 4
        A = np.ones((N, N), dtype=np.float32) - np.eye(N, dtype=np.float32)
        P = random_walk_transition(A)
        # Each row has 3 non-zero entries each = 1/3.
        for i in range(N):
            assert P[i, i] == 0.0
            for j in range(N):
                if i != j:
                    assert P[i, j] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Diffusion convolution
# ---------------------------------------------------------------------------

class TestDiffusionConv:
    def test_shape(self):
        N, F_in, F_out, K = 5, 3, 7, 2
        P = torch.eye(N)
        dc = _DiffusionConv(P, P, in_dim=F_in, out_dim=F_out, K=K)
        x = torch.randn(2, N, F_in)
        out = dc(x)
        assert out.shape == (2, N, F_out)

    def test_param_count_grows_with_K(self):
        N, F_in, F_out = 5, 3, 7
        P = torch.eye(N)
        dc1 = _DiffusionConv(P, P, in_dim=F_in, out_dim=F_out, K=1)
        dc3 = _DiffusionConv(P, P, in_dim=F_in, out_dim=F_out, K=3)
        n1 = sum(p.numel() for p in dc1.parameters())
        n3 = sum(p.numel() for p in dc3.parameters())
        # weight is (2K, F_in, F_out) = 2K*F_in*F_out per K.
        # K=1: 2*3*7 = 42 + 7 (bias) = 49
        # K=3: 6*3*7 = 126 + 7 (bias) = 133
        assert n3 - n1 == 4 * F_in * F_out  # (2*3 - 2*1) * F_in * F_out = 84

    def test_K_zero_rejected(self):
        N = 3
        P = torch.eye(N)
        with pytest.raises(ValueError, match="K"):
            _DiffusionConv(P, P, in_dim=1, out_dim=1, K=0)

    def test_diffusion_uses_neighbors(self):
        # On a 2-node fully-connected graph, the K=2 forward pass should
        # mix the two nodes' inputs even with simple weights.
        torch.manual_seed(0)
        A = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        # row-normalised: each row has 1 neighbour with weight 1.0.
        P = A
        dc = _DiffusionConv(P, P, in_dim=1, out_dim=1, K=2)
        x = torch.tensor([[[1.0], [0.0]]])  # node 0 = 1, node 1 = 0
        with torch.no_grad():
            out = dc(x)
        # Both nodes' outputs should be non-zero (graph mixed the signal).
        assert out[0, 0, 0].abs().item() > 1e-6
        assert out[0, 1, 0].abs().item() > 1e-6


# ---------------------------------------------------------------------------
# DCRNNForecaster
# ---------------------------------------------------------------------------

class TestDCRNNForecaster:
    def test_shape_one_step(self):
        m = DCRNNForecaster(input_size=5, window_size=24, horizon=1,
                             hidden_dim=8, num_layers=1, K=1)
        x = torch.randn(3, 24, 5)
        out = m(x)
        assert out.shape == (3, 5)

    def test_shape_multi_step(self):
        m = DCRNNForecaster(input_size=4, window_size=24, horizon=6,
                             hidden_dim=8, num_layers=1, K=1)
        x = torch.randn(2, 24, 4)
        out = m(x)
        assert out.shape == (2, 6, 4)

    def test_finite_output(self):
        m = DCRNNForecaster(input_size=5, window_size=24,
                             hidden_dim=8, num_layers=1, K=1)
        out = m(torch.randn(2, 24, 5))
        assert torch.isfinite(out).all()

    def test_identity_adjacency_no_cross_link_signal(self):
        # With identity adjacency, K=1, perturbing channel 0 must not
        # change channel 1's output (no spatial mixing).
        torch.manual_seed(0)
        N = 4
        A = np.eye(N, dtype=np.float32)
        m = DCRNNForecaster(input_size=N, window_size=12,
                             adjacency=A,
                             hidden_dim=8, num_layers=1, K=1)
        m.eval()
        x = torch.randn(1, 12, N)
        x_perturbed = x.clone()
        x_perturbed[..., 0] += 5.0
        with torch.no_grad():
            out_orig = m(x)
            out_pert = m(x_perturbed)
        # Channel 1's output must be identical (within fp32 noise).
        diff_ch1 = (out_pert[..., 1] - out_orig[..., 1]).abs().max().item()
        assert diff_ch1 < 1e-5

    def test_real_adjacency_spreads_signal(self):
        # With a real adjacency where node 0 connects to node 1, perturbing
        # channel 0 must shift channel 1's output.
        torch.manual_seed(0)
        N = 3
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32)
        m = DCRNNForecaster(input_size=N, window_size=12,
                             adjacency=A,
                             hidden_dim=8, num_layers=1, K=2)
        m.eval()
        x = torch.randn(1, 12, N)
        x_perturbed = x.clone()
        x_perturbed[..., 0] += 5.0
        with torch.no_grad():
            out_orig = m(x)
            out_pert = m(x_perturbed)
        # Channel 1 should pick up the perturbation through one diffusion hop.
        diff_ch1 = (out_pert[..., 1] - out_orig[..., 1]).abs().mean().item()
        assert diff_ch1 > 1e-3

    def test_gradient_flow(self):
        m = DCRNNForecaster(input_size=4, window_size=12,
                             hidden_dim=8, num_layers=1, K=1)
        x = torch.randn(2, 12, 4)
        target = torch.randn(2, 4)
        loss = (m(x) - target).pow(2).mean()
        loss.backward()
        any_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in m.parameters()
        )
        assert any_grad

    def test_no_adjacency_falls_back_to_identity(self):
        # adjacency=None should still build a working model (identity graph).
        m = DCRNNForecaster(input_size=4, window_size=12,
                             adjacency=None,
                             hidden_dim=8, num_layers=1, K=1)
        out = m(torch.randn(2, 12, 4))
        assert out.shape == (2, 4)

    def test_wrong_adjacency_shape_raises(self):
        with pytest.raises(ValueError, match="adjacency"):
            DCRNNForecaster(
                input_size=4, window_size=12,
                adjacency=np.ones((3, 3), dtype=np.float32),
                hidden_dim=8, num_layers=1, K=1,
            )

    def test_rejects_wrong_window(self):
        m = DCRNNForecaster(input_size=4, window_size=12,
                             hidden_dim=8, num_layers=1, K=1)
        with pytest.raises(ValueError, match="window_size"):
            m(torch.randn(2, 16, 4))

    def test_rejects_wrong_channels(self):
        m = DCRNNForecaster(input_size=4, window_size=12,
                             hidden_dim=8, num_layers=1, K=1)
        with pytest.raises(ValueError, match="input_size"):
            m(torch.randn(2, 12, 8))

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            DCRNNForecaster(input_size=0, window_size=12)
        with pytest.raises(ValueError):
            DCRNNForecaster(input_size=4, window_size=0)
        with pytest.raises(ValueError):
            DCRNNForecaster(input_size=4, window_size=12, num_layers=0)
