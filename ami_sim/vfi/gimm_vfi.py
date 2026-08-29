"""Arbitrary-timestep bridge for an official GIMM-VFI checkout."""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from .base import VFIBackend, validate_targets


def _load_callable(specification: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    if ":" not in specification:
        raise ValueError("GIMM adapter must be written as 'package.module:function'.")
    module_name, function_name = specification.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"GIMM adapter {specification!r} is not callable.")
    return function


class GIMMVFIBackend(VFIBackend):
    name = "gimm_vfi"

    def __init__(self, adapter: str) -> None:
        self.adapter_spec = adapter
        self._adapter: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
        targets = validate_targets(target_timestamps)
        if self._adapter is None:
            self._adapter = _load_callable(self.adapter_spec)
        result = np.asarray(self._adapter(np.asarray(frame0), np.asarray(frame1), targets))
        expected_shape = (len(targets), *np.asarray(frame0).shape)
        if result.shape != expected_shape:
            raise ValueError(f"GIMM adapter returned {result.shape}; expected {expected_shape}.")
        return result

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "adapter": self.adapter_spec}
