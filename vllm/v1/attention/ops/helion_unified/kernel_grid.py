# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified paged attention, 3-D grid form.

Grid `[num_splits, num_seqs, num_kv_heads, max_seqlen_q]`, so `(seq_idx,
q_offset)` come straight from the grid and no device-side search is needed. The
cost is that every sequence gets `max_seqlen_q` q-tiles regardless of its own
length, which is mostly empty blocks on decode-heavy batches.

Body-identical to `kernel_flat.py` from the `q_start + tile_q.begin < q_end`
guard down; the performance rationale for the loop split, the base-2 softmax and
`num_splits` is documented there. Any correctness-bearing change must be made in
both. Needs its own AOT artifact because `grid_extent` is arg 7 and the tree cuts
on it, but it means `max_seqlen_q` here and `num_q_tiles` in the flat form.
"""

from __future__ import annotations

import helion.language as hl
import torch
from helion.autotuner import PowerOfTwoFragment
from torch import Tensor

from .kernel_common import _LOG2_E


def _kernel_unified(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    seqused_k: Tensor,
    query_start_loc: Tensor,
    scale: float,
    grid_extent: int,
    max_pages: hl.constexpr,
    m_budget: hl.constexpr,
    p_budget: hl.constexpr,
) -> tuple[Tensor, Tensor, Tensor]:
    num_tokens, num_q_heads, head_dim = query.size()
    num_q_heads = hl.specialize(num_q_heads)
    head_dim = hl.specialize(head_dim)
    page_size = hl.specialize(key_cache.size(1))
    num_kv_heads = hl.specialize(key_cache.size(2))
    num_groups = num_q_heads // num_kv_heads
    max_block_idx = max_pages - 1
    num_seqs = block_table.size(0)

    # Maxes are the constexpr budgets; see `kernel_common` for why they cannot be
    # inline literals.
    q_block = hl.register_block_size(1, m_budget)
    block_p = hl.register_block_size(2, p_budget)
    # Split-KV: occupancy on decode, neutral on prefill. See `kernel_flat.py`.
    num_splits = hl.register_tunable("num_splits", PowerOfTwoFragment(1, 32))

    n_rows = num_tokens * num_q_heads
    dev = query.device
    partial_out = torch.zeros(
        [num_splits, n_rows, head_dim], dtype=torch.float32, device=dev
    )
    partial_l = torch.zeros([num_splits, n_rows], dtype=torch.float32, device=dev)
    partial_m = torch.full(
        [num_splits, n_rows], float("-inf"), dtype=torch.float32, device=dev
    )

    for tile_s, s_idx, tile_g, tile_q in hl.tile(
        [num_splits, num_seqs, num_kv_heads, grid_extent],
        block_size=[1, 1, 1, q_block],
    ):
        seq_idx = s_idx.begin
        split_idx = tile_s.begin
        seq_len = seqused_k[seq_idx]
        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        context_len = torch.clamp(seq_len - (q_end - q_start), min=0)

        if q_start + tile_q.begin < q_end:
            block_m = q_block * num_groups
            q_idx = q_start + tile_q.begin + hl.arange(q_block)
            q_head = tile_g.begin * num_groups + hl.arange(num_groups)
            kv_head = tile_g.begin
            q_load_mask = q_idx[:, None, None] < q_end
            q_i = hl.load(
                query, [q_idx, q_head, hl.arange(head_dim)], extra_mask=q_load_mask
            ).flatten(start_dim=0, end_dim=1)
            m_pos = hl.arange(q_block)[:, None].expand(q_block, num_groups).flatten()
            m_valid = q_start + tile_q.begin + m_pos < q_end
            q_valid = m_valid[:, None]

            m_i = hl.full([block_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([block_m], dtype=torch.float32)
            acc = hl.zeros([block_m, head_dim], dtype=torch.float32)

            n_tok_max = torch.minimum(seq_len, context_len + tile_q.begin + q_block + 1)
            n_page_max = (n_tok_max + page_size - 1) // page_size

            pages_per_split = (n_page_max + num_splits - 1) // num_splits
            split_lo = torch.minimum(pages_per_split * split_idx, n_page_max)
            split_hi = torch.minimum(split_lo + pages_per_split, n_page_max)

            n_page_split = torch.clamp((context_len + tile_q.begin) // page_size, min=0)
            n_page_split = torch.clamp(n_page_split, min=split_lo)
            n_page_split = torch.minimum(n_page_split, split_hi)
            # Rounded DOWN to a whole `block_p` relative to `split_lo`: the
            # gather reshape uses the full block size, so an unrounded boundary
            # would load real KV past the causal range.
            n_page_split = split_lo + ((n_page_split - split_lo) // block_p) * block_p

            # Interior pages: strictly below the diagonal, so no mask -- and K/V
            # are gathered directly into the `hl.dot` operands rather than
            # materialized into a padded tile.
            for tile_p in hl.tile(split_lo, n_page_split, block_size=block_p):
                p_idx = torch.clamp(
                    tile_p.begin + hl.arange(tile_p.block_size), max=max_block_idx
                )
                physical = block_table[seq_idx, p_idx]
                n = tile_p.block_size * page_size
                k = key_cache[physical, :, kv_head, :].reshape(n, head_dim)
                v = value_cache[physical, :, kv_head, :].reshape(n, head_dim)
                qk = hl.dot(q_i, k.T, out_dtype=torch.float32) * (scale * _LOG2_E)
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                p = torch.exp2(qk - m_ij[:, None])
                alpha = torch.exp2(torch.where(m_i == -float("inf"), m_ij, m_i) - m_ij)
                m_i = m_ij
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = hl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])

            # Diagonal pages: the only ones that need the causal mask.
            for tile_p in hl.tile(n_page_split, split_hi, block_size=block_p):
                p_idx = torch.clamp(
                    tile_p.begin + hl.arange(tile_p.block_size), max=max_block_idx
                )
                physical = block_table[seq_idx, p_idx]
                n = tile_p.block_size * page_size
                k = key_cache[physical, :, kv_head, :].reshape(n, head_dim)
                v = value_cache[physical, :, kv_head, :].reshape(n, head_dim)
                qk = hl.dot(q_i, k.T, out_dtype=torch.float32) * scale
                bcast_n = (
                    tile_p.begin * page_size + hl.arange(tile_p.block_size * page_size)
                )[None, :]
                causal = bcast_n <= context_len + (tile_q.begin + m_pos)[:, None]
                keep = (
                    causal
                    & (bcast_n < seq_len)
                    & (bcast_n < split_hi * page_size)
                    & q_valid
                )
                qk = torch.where(keep, qk * _LOG2_E, -float("inf"))
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                m_safe = torch.where(m_ij == -float("inf"), 0.0, m_ij)
                p = torch.exp2(qk - m_safe[:, None])
                alpha = torch.exp2(
                    torch.where(m_i == -float("inf"), m_safe, m_i) - m_safe
                )
                m_i = m_ij
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = hl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])

            row = (q_idx[:, None] * num_q_heads + q_head[None, :]).flatten()
            hl.store(
                partial_out,
                [split_idx, row, hl.arange(head_dim)],
                acc,
                extra_mask=q_valid,
            )
            hl.store(partial_l, [split_idx, row], l_i, extra_mask=m_valid)
            hl.store(partial_m, [split_idx, row], m_i, extra_mask=m_valid)

    return partial_out, partial_l, partial_m


def _kernel_combine(
    out: Tensor,
    partial_out: Tensor,
    partial_l: Tensor,
    partial_m: Tensor,
    num_live_rows: int,
) -> None:
    hl.specialize(partial_out.size(0))
    head_dim = hl.specialize(partial_out.size(2))

    out_flat = out.view(-1, head_dim)

    for row in hl.grid(num_live_rows):
        m_i = partial_m[:, row]
        l_i = partial_l[:, row]
        o_i = partial_out[:, row, :]

        m_final = torch.amax(m_i, dim=0)
        m_safe = torch.where(m_final == float("-inf"), 0.0, m_final)
        alpha = torch.exp2(m_i - m_safe)
        l_final = torch.sum(l_i * alpha, dim=0)
        o_final = torch.sum(o_i * alpha[:, None], dim=0)
        o_final = o_final / torch.where(l_final == 0, 1.0, l_final)
        out_flat[row, :] = o_final.to(out_flat.dtype)
