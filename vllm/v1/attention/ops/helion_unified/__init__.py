# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Helion paged-attention kernels.

One kernel serving decode rows, prefill rows and mixed batches in a single
launch, with no dispatch on query length. Two interchangeable grid forms, each
registered as its own attention backend:

  flat  flat per-row q-tile grid, sized by a sync-free host-side bound
  grid  3-D grid sized by `max_seqlen_q`, guarded per row

`flat` is the stronger default on measurement (geomean FA3/impl 0.631 vs 0.595
over 132 shape cells, H100/Qwen3) and the more robust of the two. `grid` wins on
pure decode and above qlen 512, but no clean shape predicate separates them in
the middle of the range.

Configs come from the AOT decision trees in
`_helion_aot_kernel_{flat,grid}_cuda_sm90.py`, which Helion discovers by name
next to each kernel's source file. Hence `helion.aot_kernel` with no `config=`:
a pinned config would override the tree.
"""

import helion

from .kernel_common import unified_factory


def build_flat():
    from .kernel_flat import _kernel_combine, _kernel_unified_flat

    return unified_factory(
        helion.aot_kernel(static_shapes=False)(_kernel_unified_flat),
        helion.aot_kernel(static_shapes=False)(_kernel_combine),
        True,
    )


def build_grid():
    from .kernel_grid import _kernel_combine, _kernel_unified

    return unified_factory(
        helion.aot_kernel(static_shapes=False)(_kernel_unified),
        helion.aot_kernel(static_shapes=False)(_kernel_combine),
        False,
    )


__all__ = ["build_flat", "build_grid"]
