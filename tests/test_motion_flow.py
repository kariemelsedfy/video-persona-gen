from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from avagen.models.motion_flow import (  # noqa: E402
    MotionFlowConfig,
    MotionFlowModel,
    sample_motion,
    sinusoidal_time_embedding,
)


def test_time_embedding_shape_and_varies_with_t() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_time_embedding(t, 128)
    assert emb.shape == (3, 128)
    assert not torch.allclose(emb[0], emb[2])


def test_forward_shapes() -> None:
    cfg = MotionFlowConfig(motion_size=205, audio_size=768, hidden_size=64, num_layers=2)
    model = MotionFlowModel(cfg).eval()
    b, tlen = 2, 30
    xt = torch.randn(b, tlen, 205)
    audio = torch.randn(b, tlen, 768)
    t = torch.rand(b)
    v = model(xt, audio, t)
    assert v.shape == (b, tlen, 205)


def test_sample_motion_shape_and_finite() -> None:
    cfg = MotionFlowConfig(motion_size=205, audio_size=768, hidden_size=64, num_layers=2)
    model = MotionFlowModel(cfg).eval()
    audio = torch.randn(1, 40, 768)
    out = sample_motion(model, audio, steps=5)
    assert out.shape == (1, 40, 205)
    assert torch.isfinite(out).all()


def test_can_overfit_one_window_flow_loss_decreases() -> None:
    # sanity: flow-matching loss should drop when overfitting a single pair
    torch.manual_seed(0)
    cfg = MotionFlowConfig(motion_size=8, audio_size=5, hidden_size=64, num_layers=2, dropout=0.0)
    model = MotionFlowModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x1 = torch.randn(1, 20, 8)
    audio = torch.randn(1, 20, 5)
    first = None
    for step in range(150):
        x0 = torch.randn_like(x1)
        t = torch.rand(1)
        xt = (1 - t).view(-1, 1, 1) * x0 + t.view(-1, 1, 1) * x1
        v_pred = model(xt, audio, t)
        loss = ((v_pred - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = float(loss)
    assert float(loss) < first
