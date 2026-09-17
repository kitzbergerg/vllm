# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared non-kernel parts of the unified paged-attention package.

Contains no `@helion.kernel`: Helion names an AOT artifact after the source-file
stem and merges every kernel in that file into it, so each grid form needs its
own module (`kernel_flat.py`, `kernel_grid.py`) and everything artifact-free
lives here.
"""

from __future__ import annotations

import torch
from torch import Tensor

_LOG2_E = 1.44269504

# Registered minimum of `q_block`. Must equal the literal in both kernel bodies:
# the flat grid bound divides by it, and `floor(N/b) + S` is non-increasing in b,
# so a larger tuned `q_block` only widens the slack while a larger divisor would
# silently drop query tiles. 1, not 16: at decode the M extent is
# `q_block * num_groups` with only `num_groups` rows live, so a floor of 16 gave
# M=128 with 8 real rows (1.68-1.78x slower on saturated decode).
_Q_BLOCK_MIN = 1

# `m_budget`'s floor, deliberately not `_Q_BLOCK_MIN`: this keeps the registered
# max from collapsing below the M extent prefill needs (at ng=8, 128 // 8 == 16),
# while the min above must reach 1 for decode.
_M_BUDGET_FLOOR = 16


def _reject_unsupported(**kwargs) -> None:
    window_size = kwargs.pop("window_size", (-1, -1))
    # `num_splits` is popped and ignored: it is a `hl.register_tunable` read in
    # the kernel body, so the autotuner / AOT tree owns it, not the caller.
    kwargs.pop("num_splits", None)
    for name, value in kwargs.items():
        bad = value is not None if isinstance(value, torch.Tensor) else bool(value)
        assert not bad, f"{name} is unimplemented in this paged_attention kernel"
    assert window_size == (-1, -1), "sliding window is unimplemented"
    raise AssertionError("unsupported argument combination")


def p_budget_for(device, page_size: int, head_dim: int, itemsize: int) -> int:
    """`block_p`'s registered max: largest pow2 whose live K+V tile fits smem."""
    smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    per_p = 2 * page_size * head_dim * itemsize
    return max(2, min(32, 1 << (max(smem // per_p, 1).bit_length() - 1)))


def unified_factory(kernel_fn, combine_fn, grid_flat: bool):
    """Wrap phase 1 + phase 2 in the `flash_attn_varlen_func` paged signature.

    `grid_flat` must agree with which body `kernel_fn` is: the grid extent means
    `max_seqlen_q` for `kernel_grid` and `num_q_tiles` for `kernel_flat`, and a
    mismatch under-tiles silently.
    """

    def helion_unified_attention(
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        query_start_loc: Tensor | None = None,
        max_seqlen_q: int = 1,
        max_seqlen_k: int = 0,
        softmax_scale: float | None = None,
        causal: bool = True,
        out: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            _reject_unsupported(**kwargs)
        del max_seqlen_k

        num_tokens, num_q_heads, head_dim = query.shape
        num_kv_heads = key_cache.size(2)
        num_groups = num_q_heads // num_kv_heads
        page_size = key_cache.size(1)

        assert causal, "non-causal is unimplemented (needs an unbounded KV loop)"
        assert query_start_loc is not None, "unified attention requires query_start_loc"
        # The gather reshape only factors for a pow2 head_dim (96/160/192 fail).
        assert head_dim & (head_dim - 1) == 0, f"head_dim={head_dim} must be pow2"
        assert num_q_heads % num_kv_heads == 0
        # The M extent is `q_block * num_groups`; a non-pow2 product is not a
        # legal tile shape.
        assert num_groups & (num_groups - 1) == 0, (
            f"num_groups={num_groups} must be pow2"
        )
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        # Two separate 0/1-specialization predicates. A leading dim of runtime
        # size 1 loses its stride; the pad row gets `query_len == 0`, which the
        # kernel's `q_start < q_end` guard skips.
        if block_table.size(0) == 1:
            seqused_k = seqused_k.repeat(2)
            block_table = block_table.repeat(2, 1)
            query_start_loc = torch.cat([query_start_loc, query_start_loc[-1:]])
        # `query` is varlen-packed, so its specialized dim is `num_tokens`, not
        # `num_seqs`; using the latter zeroes every row but row 0.
        if query.size(0) == 1:
            query = query.repeat(2, 1, 1)
        if block_table.size(1) < 2:
            block_table = block_table[:, -1:].repeat(1, 2)

        rows = block_table.size(0)

        # `m_budget` / `p_budget` are `hl.constexpr` arguments rather than inline
        # literals in the kernel body, and must stay that way: the AOT tree keys
        # only on argument features, and `config_spec._normalize` never clamps a
        # replayed `block_sizes` entry DOWN to `max_size`. As constexprs they
        # enter the specialization key, so a leaf tuned at one (num_groups,
        # page_size, head_dim) cannot be replayed at another where its tile is
        # unlaunchable (`OutOfResources`).
        m_budget = max(_M_BUDGET_FLOOR, 128 // num_groups)
        p_budget = p_budget_for(query.device, page_size, head_dim, query.element_size())

        if grid_flat:
            # Sync-free upper bound on the q-block count:
            #   sum_i ceil(qlen_i / B) <= floor(sum_i qlen_i / B) + num_seqs
            # Depends only on static shapes; reading the query lengths for the
            # exact count is a D2H sync and would break graph capture. Divides by
            # `_Q_BLOCK_MIN`, never the tuned `q_block`, which the host cannot
            # read (the AOT tree keys on this very argument).
            grid_extent = max(query.size(0) // _Q_BLOCK_MIN + rows, 2)
        else:
            # `max(query_lens)`, never `capture_max_query_len` -- the latter
            # inflated launched q-tiles ~3x, invisibly (masking hides it).
            grid_extent = max(max_seqlen_q, 2)

        partial_out, partial_l, partial_m = kernel_fn(
            query,
            key_cache,
            value_cache,
            block_table,
            seqused_k,
            query_start_loc,
            softmax_scale,
            grid_extent,
            block_table.size(1),
            m_budget,
            p_budget,
        )
        if out is None:
            out = torch.empty(
                [num_tokens, num_q_heads, head_dim],
                dtype=query.dtype,
                device=query.device,
            )
        # Phase 2 merges the split-KV partials and mutates `out` in place.
        # `n_keep` slices the padded query row off the flat `[splits, token*head]`
        # partials, so the write lands exactly `num_tokens` output rows.
        combine_fn(out, partial_out, partial_l, partial_m, num_tokens * num_q_heads)
        return out

    return helion_unified_attention
