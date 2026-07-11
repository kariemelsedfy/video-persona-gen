from __future__ import annotations

import numpy as np

from avagen.features.motion_features import _project_to_rotation


def _is_rotation(mat: np.ndarray, tol: float = 1e-4) -> bool:
    identity_err = float(np.max(np.abs(mat @ mat.T - np.eye(3))))
    det_err = abs(float(np.linalg.det(mat)) - 1.0)
    return identity_err < tol and det_err < tol


def test_project_leaves_valid_rotation_essentially_unchanged() -> None:
    theta = 0.4
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    projected = _project_to_rotation(rot[None])[0]
    assert _is_rotation(projected)
    np.testing.assert_allclose(projected, rot, atol=1e-5)


def test_project_repairs_non_orthonormal_matrix() -> None:
    # A slightly perturbed rotation (like a regressed prediction: det != 1).
    theta = 0.3
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    noisy = rot * 1.03 + np.array([[0.02, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    assert abs(np.linalg.det(noisy) - 1.0) > 0.02  # invalid before
    projected = _project_to_rotation(noisy[None])[0]
    assert _is_rotation(projected)  # valid after


def test_project_handles_batch_and_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    mats = rng.normal(size=(5, 3, 3)).astype(np.float32) + np.eye(3, dtype=np.float32)
    projected = _project_to_rotation(mats)
    assert projected.shape == (5, 3, 3)
    for i in range(5):
        assert _is_rotation(projected[i])
