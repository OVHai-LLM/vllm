# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-latency (small batch) CPU-path switches of the GPU model runner.

At one or two decode requests the engine step is bound by host CPU time
(~2-3 ms of Python per step vs ~1.6 ms of GPU work on B300), so these trim
per-step host work. ``VLLM_LOWLAT=0`` turns them all off; each flag can be
flipped at runtime for A/B measurements."""

import os
from dataclasses import dataclass


@dataclass
class _LowLat:
    staged_writes: bool = True
    sampler_graph: bool = True


_on = os.environ.get("VLLM_LOWLAT", "1") != "0"
LOWLAT = _LowLat(**{f: _on for f in _LowLat.__dataclass_fields__})
