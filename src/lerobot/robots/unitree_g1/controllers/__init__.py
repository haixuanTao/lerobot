#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unitree G1 locomotion controllers (Groot, Holosoma, SONIC, Zealot).

Re-exports are lazy: the ONNX-backed controllers pull `onnxruntime` at import
time, and this package's `__init__` runs on *any* submodule import. Eager
re-exports would therefore make a controller that needs no ONNX (Zealot is pure
numpy) unimportable on a robot that has no onnxruntime installed.
"""

import importlib

_CONTROLLER_MODULES = {
    "GrootLocomotionController": "gr00t_locomotion",
    "HolosomaLocomotionController": "holosoma_locomotion",
    "SonicWholeBodyController": "sonic_whole_body",
    "ZealotLocomotionController": "zealot_locomotion",
}

__all__ = list(_CONTROLLER_MODULES)


def __getattr__(name: str):
    module_name = _CONTROLLER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module_name}", __name__), name)
