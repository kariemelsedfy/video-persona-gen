from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from avagen.models.sync_net import SyncNet, SyncNetConfig  # noqa: E402


def _model():
    return SyncNet(SyncNetConfig(audio_size=32, motion_size=16, hidden_size=32, embed_dim=32, num_layers=1))


def test_embeddings_normalized_and_shapes() -> None:
    model = _model().eval()
    a = torch.randn(4, 20, 32)
    m = torch.randn(4, 20, 16)
    ea = model.encode_audio(a)
    em = model.encode_motion(m)
    assert ea.shape == (4, 32) and em.shape == (4, 32)
    assert torch.allclose(ea.norm(dim=-1), torch.ones(4), atol=1e-4)
    score = model.sync_score(a, m)
    assert score.shape == (4,)
    assert float(score.min()) >= -1.001 and float(score.max()) <= 1.001


def test_contrastive_loss_learns_alignment() -> None:
    torch.manual_seed(0)
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # fixed set of aligned (audio, motion) pairs the net should learn to match
    a = torch.randn(16, 20, 32)
    m = torch.randn(16, 20, 16)
    first = float(model.contrastive_loss(a, m))
    for _ in range(200):
        opt.zero_grad(); loss = model.contrastive_loss(a, m); loss.backward(); opt.step()
    assert float(loss) < first
    # after training, aligned pairs should score higher than mismatched (rolled)
    model.eval()
    with torch.no_grad():
        aligned = model.sync_score(a, m).mean()
        mismatched = model.sync_score(a, torch.roll(m, 1, 0)).mean()
    assert float(aligned) > float(mismatched)
