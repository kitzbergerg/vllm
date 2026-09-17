# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Helion paged-attention kernels.

One kernel serving decode rows, prefill rows and mixed batches in a single
launch, with no dispatch on query length. Two interchangeable grid forms,
selected with `VLLM_HELION_UNIFIED_IMPL` (default `flat`):

  flat  (default)  flat per-row q-tile grid, sized by a sync-free host-side bound
  grid             3-D grid sized by `max_seqlen_q`, guarded per row

`flat` is the more robust of the two. `grid` wins on pure
decode and above qlen 512, but no clean shape predicate separates them in the
middle of the range and a graph-capturable switch must be a step function on
capture sizes, so the choice is a static knob rather than a runtime heuristic.

Configs come from the AOT decision trees in
`_helion_aot_kernel_{flat,grid}_cuda_sm90.py`, which Helion discovers by name
next to each kernel's source file.
"""

import os

import helion

from .kernel_common import unified_factory

IMPL_ENV = "VLLM_HELION_UNIFIED_IMPL"
DEFAULT_IMPL = "flat"
IMPLS = ("flat", "grid")


def _build_unified(is_flat: bool):
    # Each phase-1 module carries its own `_kernel_combine`: Helion merges every
    # kernel in a source file into that file's AOT artifact, so this gives each
    # form a combine heuristic tuned next to its own phase 1.
    if is_flat:
        from .kernel_flat import _kernel_combine
        from .kernel_flat import _kernel_unified_flat as kernel_fn
    else:
        from .kernel_grid import _kernel_combine
        from .kernel_grid import _kernel_unified as kernel_fn

    return unified_factory(
        helion.aot_kernel(static_shapes=False)(kernel_fn),
        helion.aot_kernel(static_shapes=False)(_kernel_combine),
        is_flat,
    )


def get_impl(name: str | None = None):
    """Return the attention callable for `name`, matching the paged form of
    `flash_attn_varlen_func`."""
    name = (name or os.environ.get(IMPL_ENV) or DEFAULT_IMPL).strip().lower()
    if name not in IMPLS:
        raise ValueError(f"{IMPL_ENV}={name!r} not in {IMPLS}")
    return _build_unified(name == "flat")


__all__ = ["DEFAULT_IMPL", "IMPLS", "IMPL_ENV", "get_impl"]
