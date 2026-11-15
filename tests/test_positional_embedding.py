import math

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from flowpolicy_jittor.model.positional_embedding import (  # noqa: E402
    SinusoidalPosEmb,
)


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = previous


def _numpy_reference(timesteps: np.ndarray, dim: int) -> np.ndarray:
    half_dim = dim // 2
    frequency_step = math.log(10000.0) / (half_dim - 1)
    frequencies = np.exp(
        np.arange(half_dim, dtype=np.float32) * -frequency_step
    )
    angles = timesteps.astype(np.float32)[:, None] * frequencies[None, :]
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)


def test_matches_upstream_numpy_formula():
    timesteps = np.array([0.0, 0.25, 1.0, 7.5], dtype=np.float32)
    actual = SinusoidalPosEmb(8)(jt.array(timesteps)).numpy()
    expected = _numpy_reference(timesteps, dim=8)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_shape_zero_timestep_and_determinism():
    timesteps = jt.array(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    layer = SinusoidalPosEmb(256)
    first = layer(timesteps).numpy()
    second = layer(timesteps).numpy()

    assert first.shape == (3, 256)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[0, :128], 0.0, atol=0.0)
    np.testing.assert_allclose(first[0, 128:], 1.0, atol=0.0)


@pytest.mark.parametrize("dim", [0, 2, 3, 5, -4])
def test_rejects_unsupported_dimension(dim):
    with pytest.raises(ValueError, match="even integer >= 4"):
        SinusoidalPosEmb(dim)


@pytest.mark.parametrize("dim", [True, 8.0, "8"])
def test_rejects_non_integer_dimension(dim):
    with pytest.raises(TypeError, match="must be an integer"):
        SinusoidalPosEmb(dim)


@pytest.mark.parametrize(
    "shape",
    [(2, 1), (1, 2, 1)],
)
def test_rejects_non_vector_timestep_input(shape):
    values = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \[batch\]"):
        SinusoidalPosEmb(8)(jt.array(values))


def test_jittor_scalar_is_treated_as_single_item_batch():
    # Jittor represents a scalar array with shape [1], so it already satisfies
    # the module's one-dimensional batch contract.
    scalar = jt.array(np.float32(0.5))
    assert tuple(scalar.shape) == (1,)
    assert tuple(SinusoidalPosEmb(8)(scalar).shape) == (1, 8)


def test_integer_input_uses_documented_float32_output():
    output = SinusoidalPosEmb(8)(jt.array([0, 1, 2]))
    assert output.dtype == jt.float32
    expected = _numpy_reference(np.array([0, 1, 2]), dim=8)
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_timestep_gradient_is_finite():
    timesteps = jt.array(np.array([0.25, 0.75], dtype=np.float32))
    timesteps.start_grad()
    gradient = jt.grad(SinusoidalPosEmb(8)(timesteps).sum(), timesteps)
    assert gradient.shape == timesteps.shape
    assert np.isfinite(gradient.numpy()).all()

