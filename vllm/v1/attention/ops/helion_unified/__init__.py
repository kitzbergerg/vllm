# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Helion paged-attention kernels.

One kernel serving vLLM's whole workload -- decode rows, prefill rows and mixed
batches -- in a single launch, with no dispatch on query length. Three
interchangeable implementations:

  flat  (default)  flat per-row q-tile grid, sized by a sync-free host-side bound
  grid             3-D grid sized by `max_seqlen_q`, guarded per row
  ref              the upstream Helion kernel this rewrites; internal baseline

Select with `VLLM_HELION_UNIFIED_IMPL` (default `flat`).

`flat` is the default on measurement: geomean FA3/impl 0.631 for flat vs 0.595
for grid vs 0.359 for ref over 132 shape cells (H100, Qwen3). `grid` wins on pure
decode (0.857 vs 0.734) and above qlen 512 (1.14x on 18/21 cells), so it stays
selectable -- but no clean shape predicate separates them in the middle of the
range, and a graph-capturable switch must be a step function on capture sizes, so
the choice is a static knob rather than a runtime heuristic.

Configs come from the AOT decision trees in
`_helion_aot_kernel_{flat,grid,ref}_cuda_sm90.py`, which Helion discovers
automatically: `aot_cache.py` looks for
`_helion_aot_<source-stem>_<device>_<compute>.py` NEXT TO the kernel's source
file, so the trees sitting beside `kernel_flat.py` here need no registration. The
trees cut on continuous shape features (56 leaves for flat, 68 for grid) and pick
block sizes, warps and `num_splits` per call. Hence `helion.aot_kernel` with NO
`config=`: a pinned config would override the tree.
"""

import os

import helion

from .kernel_common import unified_factory
from .kernel_ref import kernel as _kernel_ref
from .kernel_ref import kernel_ref_factory

IMPL_ENV = "VLLM_HELION_UNIFIED_IMPL"
DEFAULT_IMPL = "flat"
IMPLS = ("flat", "grid", "ref")


def _build_unified(is_flat: bool):
    """Phase 1 + phase 2, wrapped in the vLLM-shaped callable.

    Each phase-1 file carries its OWN `_kernel_combine`: Helion names an AOT
    artifact after the source-file stem and merges every kernel in that file into
    it, so this gives each form a combine heuristic tuned next to its own phase 1.
    A single shared combine module put both into one artifact, which whichever
    tuning run finished last overwrote.
    """
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
    """Return the attention callable for `name`.

    Defaults to `VLLM_HELION_UNIFIED_IMPL`, else `flat`. The returned callable
    matches `flash_attn_varlen_func`'s paged form:

        (query, key_cache, value_cache, block_table, seqused_k,
         query_start_loc=None, max_seqlen_q=1, max_seqlen_k=0,
         softmax_scale=None, causal=True, out=None) -> Tensor
    """
    name = (name or os.environ.get(IMPL_ENV) or DEFAULT_IMPL).strip().lower()
    if name not in IMPLS:
        raise ValueError(f"{IMPL_ENV}={name!r} not in {IMPLS}")
    if name == "ref":
        return kernel_ref_factory(helion.aot_kernel(static_shapes=False)(_kernel_ref))
    return _build_unified(name == "flat")


__all__ = ["DEFAULT_IMPL", "IMPLS", "IMPL_ENV", "get_impl"]
