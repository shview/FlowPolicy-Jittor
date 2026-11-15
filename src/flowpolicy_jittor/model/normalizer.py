"""Linear normalizers used by the released FlowPolicy data path.

This module ports the ``mode='limits'``, ``last_n_dims=1`` path from
FlowPolicy commit ``86d10c1`` (``model/common/normalizer.py``).  Statistics
are registered as stopped-gradient Jittor buffers so they are included in a
checkpoint but never optimized.  Gaussian mode and multi-axis feature
flattening are deliberately left out until they are needed and tested.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Dict, Optional, Union

import jittor as jt
import numpy as np


ArrayLike = Union[np.ndarray, jt.Var]
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STATE_NAMES = ("scale", "offset", "input_min", "input_max", "input_mean", "input_std")


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, jt.Var):
        array = value.numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:  # pragma: no cover - defensive error context.
            raise TypeError(f"{name} must be array-like") from exc
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain numeric values, got {array.dtype}")
    return np.asarray(array, dtype=np.float32)


def _buffer(value: Any, name: str) -> jt.Var:
    array = _as_numpy(value, name).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return jt.array(array).float32().stop_grad()


class SingleFieldLinearNormalizer(jt.Module):
    """Affine feature normalizer ``y = x * scale + offset``.

    Feature statistics correspond to the final input dimension.  A one-value
    identity normalizer is allowed to broadcast over any shape, matching the
    upstream ``create_identity`` behavior.
    """

    def __init__(
        self,
        scale: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
        input_stats: Optional[Mapping[str, ArrayLike]] = None,
    ) -> None:
        super().__init__()
        self._is_initialized = False
        if scale is not None or offset is not None or input_stats is not None:
            if scale is None or offset is None or input_stats is None:
                raise ValueError("scale, offset, and input_stats must be supplied together")
            self._set_state(scale, offset, input_stats)

    @property
    def is_initialized(self) -> bool:
        return bool(self._is_initialized)

    @property
    def feature_dim(self) -> int:
        self._require_initialized()
        return int(self.scale.shape[0])

    def _require_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("normalizer is not initialized")

    def _set_state(
        self,
        scale: ArrayLike,
        offset: ArrayLike,
        input_stats: Mapping[str, ArrayLike],
    ) -> None:
        required = {"min", "max", "mean", "std"}
        if set(input_stats) != required:
            raise KeyError(
                "input_stats must contain exactly min, max, mean, and std"
            )
        arrays = {
            "scale": _buffer(scale, "scale"),
            "offset": _buffer(offset, "offset"),
            "input_min": _buffer(input_stats["min"], "input_stats.min"),
            "input_max": _buffer(input_stats["max"], "input_stats.max"),
            "input_mean": _buffer(input_stats["mean"], "input_stats.mean"),
            "input_std": _buffer(input_stats["std"], "input_stats.std"),
        }
        sizes = {name: int(value.shape[0]) for name, value in arrays.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"all normalizer arrays must have one size, got {sizes}")
        if np.any(np.abs(arrays["scale"].numpy()) < 1e-12):
            raise ValueError("scale must be non-zero so unnormalize is defined")

        for name, value in arrays.items():
            self.register_buffer(name, value, persistent=True)
        self._is_initialized = True

    def fit(
        self,
        data: ArrayLike,
        *,
        last_n_dims: int = 1,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> "SingleFieldLinearNormalizer":
        if last_n_dims != 1:
            raise NotImplementedError("only last_n_dims=1 is validated")
        if mode != "limits":
            raise NotImplementedError("only mode='limits' is validated")
        if not fit_offset:
            raise NotImplementedError("only fit_offset=True is validated")
        if not np.isfinite(output_min) or not np.isfinite(output_max):
            raise ValueError("output limits must be finite")
        if output_max <= output_min:
            raise ValueError("output_max must be larger than output_min")
        if range_eps <= 0 or not np.isfinite(range_eps):
            raise ValueError("range_eps must be positive and finite")

        array = _as_numpy(data, "data")
        if array.ndim < 1 or array.size == 0 or array.shape[-1] <= 0:
            raise ValueError("data must be non-empty with a feature dimension")
        if not np.all(np.isfinite(array)):
            raise ValueError("data must contain only finite values")
        flat = array.reshape(-1, array.shape[-1])
        input_min = flat.min(axis=0)
        input_max = flat.max(axis=0)
        input_mean = flat.mean(axis=0)
        # torch.std used by the upstream code is sample std.  A single row is
        # assigned zero instead of NaN; std is metadata in limits mode.
        ddof = 1 if flat.shape[0] > 1 else 0
        input_std = flat.std(axis=0, ddof=ddof)

        input_range = input_max - input_min
        constant = input_range < range_eps
        safe_range = input_range.copy()
        safe_range[constant] = output_max - output_min
        scale = (output_max - output_min) / safe_range
        offset = output_min - scale * input_min
        offset[constant] = (output_max + output_min) / 2.0 - input_min[constant]

        self._set_state(
            scale,
            offset,
            {
                "min": input_min,
                "max": input_max,
                "mean": input_mean,
                "std": input_std,
            },
        )
        return self

    @classmethod
    def create_fit(cls, data: ArrayLike, **kwargs: Any) -> "SingleFieldLinearNormalizer":
        return cls().fit(data, **kwargs)

    @classmethod
    def create_manual(
        cls,
        scale: ArrayLike,
        offset: ArrayLike,
        input_stats_dict: Mapping[str, ArrayLike],
    ) -> "SingleFieldLinearNormalizer":
        return cls(scale, offset, input_stats_dict)

    @classmethod
    def create_identity(cls) -> "SingleFieldLinearNormalizer":
        one = np.ones((1,), dtype=np.float32)
        zero = np.zeros((1,), dtype=np.float32)
        return cls.create_manual(
            one,
            zero,
            {"min": -one, "max": one, "mean": zero, "std": one},
        )

    def _check_input(self, value: ArrayLike) -> jt.Var:
        self._require_initialized()
        if isinstance(value, jt.Var):
            result = value.float32()
        else:
            result = jt.array(_as_numpy(value, "value")).float32()
        if len(result.shape) < 1 or int(np.prod(result.shape)) == 0:
            raise ValueError("value must be a non-empty tensor")
        if self.feature_dim != 1 and int(result.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"expected final feature dimension {self.feature_dim}, "
                f"got {int(result.shape[-1])}"
            )
        return result

    def normalize(self, value: ArrayLike) -> jt.Var:
        value = self._check_input(value)
        original_shape = tuple(map(int, value.shape))
        flattened = value.reshape((-1, self.feature_dim))
        return (flattened * self.scale + self.offset).reshape(original_shape)

    def unnormalize(self, value: ArrayLike) -> jt.Var:
        value = self._check_input(value)
        original_shape = tuple(map(int, value.shape))
        flattened = value.reshape((-1, self.feature_dim))
        return ((flattened - self.offset) / self.scale).reshape(original_shape)

    def execute(self, value: ArrayLike) -> jt.Var:
        return self.normalize(value)

    def get_input_stats(self) -> Dict[str, jt.Var]:
        self._require_initialized()
        return {
            "min": self.input_min,
            "max": self.input_max,
            "mean": self.input_mean,
            "std": self.input_std,
        }

    def get_output_stats(self) -> Dict[str, jt.Var]:
        return {
            name: self.normalize(value)
            for name, value in self.get_input_stats().items()
        }

    def load_state_dict(self, params: Mapping[str, Any]) -> None:
        expected = set(_STATE_NAMES)
        if set(params) != expected:
            raise KeyError(
                f"normalizer state keys must be {sorted(expected)}, "
                f"got {sorted(params)}"
            )
        values = {name: _as_numpy(params[name], name) for name in _STATE_NAMES}
        loaded_sizes = {name: int(value.reshape(-1).shape[0]) for name, value in values.items()}
        if len(set(loaded_sizes.values())) != 1:
            raise ValueError(
                f"loaded normalizer arrays must have one size, got {loaded_sizes}"
            )
        loaded_feature_dim = next(iter(loaded_sizes.values()))
        if self.is_initialized and loaded_feature_dim != self.feature_dim:
            raise ValueError(
                "loaded feature dimension does not match initialized normalizer: "
                f"loaded={loaded_feature_dim}, existing={self.feature_dim}"
            )
        self._set_state(
            values["scale"],
            values["offset"],
            {
                "min": values["input_min"],
                "max": values["input_max"],
                "mean": values["input_mean"],
                "std": values["input_std"],
            },
        )


class LinearNormalizer(jt.Module):
    """A registered collection of named single-field normalizers."""

    _PREFIX = "normalizer_"

    def __init__(self) -> None:
        super().__init__()
        self._field_attributes: Dict[str, str] = {}

    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str) or not _FIELD_RE.fullmatch(key):
            raise ValueError(
                "normalizer field names must be Python identifiers, "
                f"got {key!r}"
            )
        return key

    @classmethod
    def _attribute_name(cls, key: str) -> str:
        return cls._PREFIX + key

    @classmethod
    def _key_from_attribute(cls, attribute: str) -> str:
        if not attribute.startswith(cls._PREFIX):
            raise ValueError(f"invalid normalizer state prefix {attribute!r}")
        return attribute[len(cls._PREFIX) :]

    def keys(self):
        return self._field_attributes.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._field_attributes

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        if key not in self._field_attributes:
            raise KeyError(key)
        return getattr(self, self._field_attributes[key])

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        key = self._validate_key(key)
        if not isinstance(value, SingleFieldLinearNormalizer):
            raise TypeError("value must be a SingleFieldLinearNormalizer")
        if not value.is_initialized:
            raise RuntimeError("cannot register an uninitialized normalizer")
        attribute = self._attribute_name(key)
        setattr(self, attribute, value)
        self._field_attributes[key] = attribute

    def fit(
        self,
        data: Union[Mapping[str, ArrayLike], ArrayLike],
        **kwargs: Any,
    ) -> "LinearNormalizer":
        if isinstance(data, Mapping):
            if not data:
                raise ValueError("data mapping cannot be empty")
            for key, value in data.items():
                self[key] = SingleFieldLinearNormalizer.create_fit(value, **kwargs)
        else:
            self["_default"] = SingleFieldLinearNormalizer.create_fit(data, **kwargs)
        return self

    def _apply_fields(
        self,
        value: Union[Mapping[str, ArrayLike], ArrayLike],
        *,
        forward: bool,
    ):
        if isinstance(value, Mapping):
            missing = set(value) - set(self._field_attributes)
            if missing:
                raise KeyError(f"normalizer has no fields: {sorted(missing)}")
            function = "normalize" if forward else "unnormalize"
            return {
                key: getattr(self[key], function)(field_value)
                for key, field_value in value.items()
            }
        if "_default" not in self:
            raise RuntimeError("default normalizer is not initialized")
        return (
            self["_default"].normalize(value)
            if forward
            else self["_default"].unnormalize(value)
        )

    def normalize(self, value):
        return self._apply_fields(value, forward=True)

    def unnormalize(self, value):
        return self._apply_fields(value, forward=False)

    def execute(self, value):
        return self.normalize(value)

    def get_input_stats(self):
        if not self._field_attributes:
            raise RuntimeError("normalizer is not initialized")
        if set(self._field_attributes) == {"_default"}:
            return self["_default"].get_input_stats()
        return {key: self[key].get_input_stats() for key in self.keys()}

    def get_output_stats(self):
        if not self._field_attributes:
            raise RuntimeError("normalizer is not initialized")
        if set(self._field_attributes) == {"_default"}:
            return self["_default"].get_output_stats()
        return {key: self[key].get_output_stats() for key in self.keys()}

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if not parameter.is_stop_grad()]

    def load_state_dict(self, params: Mapping[str, Any]) -> None:
        if not isinstance(params, Mapping) or not params:
            raise ValueError("normalizer state must be a non-empty mapping")
        grouped: Dict[str, Dict[str, Any]] = {}
        for full_name, value in params.items():
            if "." not in full_name:
                raise KeyError(f"invalid normalizer state key {full_name!r}")
            attribute, state_name = full_name.split(".", 1)
            key = self._key_from_attribute(attribute)
            self._validate_key(key)
            grouped.setdefault(key, {})[state_name] = value

        if self._field_attributes and set(grouped) != set(self._field_attributes):
            raise KeyError(
                "loaded normalizer fields do not match existing fields: "
                f"loaded={sorted(grouped)}, existing={sorted(self._field_attributes)}"
            )
        for key, state in grouped.items():
            if key in self:
                self[key].load_state_dict(state)
            else:
                field = SingleFieldLinearNormalizer()
                field.load_state_dict(state)
                self[key] = field

    def load_prefixed_state_dict(
        self,
        params: Mapping[str, Any],
        *,
        prefix: str = "normalizer.",
    ) -> set[str]:
        """Strictly rebuild this normalizer from a parent checkpoint state.

        Jittor's parent ``Module.load_state_dict`` does not call a child's
        custom loader, so an empty dynamic normalizer cannot create fields
        during ordinary recursive loading.  A future top-level policy loader
        must call this method first, then strictly load the remaining model
        state.  The returned keys are the parent keys consumed here.
        """

        if not isinstance(params, Mapping) or not params:
            raise ValueError("parent state must be a non-empty mapping")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        normalized_prefix = prefix if prefix.endswith(".") else prefix + "."
        extracted = {
            key[len(normalized_prefix) :]: value
            for key, value in params.items()
            if isinstance(key, str) and key.startswith(normalized_prefix)
        }
        if not extracted:
            raise KeyError(f"no normalizer state found under prefix {normalized_prefix!r}")
        self.load_state_dict(extracted)
        return {
            key
            for key in params
            if isinstance(key, str) and key.startswith(normalized_prefix)
        }

