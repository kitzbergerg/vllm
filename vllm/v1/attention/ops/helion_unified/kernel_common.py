"""
Shared, NON-KERNEL parts of the unified paged-attention package.

Deliberately contains no `@helion.kernel`: Helion derives an AOT artifact name from the
kernel's SOURCE FILE stem, not from the function name
(`helion/autotuner/aot_cache.py:132` `base_name = source_path.stem`, and
`heuristic_generator.py:1004`), and it COMBINES the heuristics of every kernel sharing a
source file into that one artifact. Two grid forms in one file would therefore write one
`_helion_aot_kernel_*.py` holding both trees -- with `grid_extent` meaning
`max_seqlen_q` in one and `num_q_tiles` in the other. Hence `kernel_grid.py` /
`kernel_flat.py`, and hence everything artifact-free lives here.

Design (plan §1): FA3 has one kernel, not two. `should_pack_gqa` returns true
unconditionally for varlen (`hopper/heuristics.h:9-11`), so the GQA group is folded into
M for prefill rows as well as decode rows, and a qlen=1 row differs from a qlen=241 row
only in its per-row `num_m_blocks` (`hopper/pack_gqa.h:64-67`). That is already what
`paged_attention_prefill/kernel_prefill.py:113-124` does with `block_m = q_block *
num_groups`. So the split into two kernels buys nothing and costs the R1 regime (255
decodes + 1 prefill), where the current `max_query_len > 1` dispatch runs all 255 decode
rows at prefill's M utilization.

Two launches, not one: `hl.barrier()` inside an `if`-wrapped tile body makes Helion emit
a host-side `_SHAPE_DIM = block_m` naming a device variable -> NameError
(`kernel_flash.py:11-16`).
"""

from __future__ import annotations

import torch
from torch import Tensor

_LOG2_E = 1.44269504


def _reject_unsupported(**kwargs) -> None:
    """
    Raise naming whichever vLLM kwarg was passed an unsupported value.

    Raising rather than ignoring is what keeps an unimplemented feature from silently
    producing wrong numerics. Copied from `kernel_prefill.py:15-30`.
    """
    window_size = kwargs.pop("window_size", (-1, -1))
    # `num_splits` is popped and IGNORED, not asserted. Stage 3 made it a
    # `hl.register_tunable` read inside the kernel body, so the split count is chosen by
    # the autotuner / AOT tree from shape features -- there is nothing a caller-supplied
    # value could set. vLLM only passes one under graph capture anyway
    # (`flash_attn.py:543-553`), where it is FA3's pre-allocation bound, and ours is
    # sized from the tunable instead (see either phase-1 file's partials).
    for name, value in kwargs.items():
        # `is not None` rather than truthiness: a multi-element tensor (alibi_slopes)
        # raises on bool() instead of naming the bad kwarg.
        bad = value is not None if isinstance(value, torch.Tensor) else bool(value)
        assert not bad, f"{name} is unimplemented in this paged_attention kernel"
    assert window_size == (-1, -1), "sliding window is unimplemented"
    raise AssertionError("unsupported argument combination")


def _combine_ref(partial_out: Tensor, partial_l: Tensor, partial_m: Tensor) -> Tensor:
    """Eager-torch merge over the split axis. Mirrors `kernel_flash._combine_ref`."""
    m_final = partial_m.amax(dim=0)
    m_safe = torch.where(m_final == float("-inf"), 0.0, m_final)
    alpha = torch.exp2(partial_m - m_safe)
    l_final = (partial_l * alpha).sum(dim=0)
    o_final = (partial_out * alpha[..., None]).sum(dim=0)
    return o_final / torch.where(l_final == 0, 1.0, l_final)[..., None]


def _baseline_partials(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    seqused_k: Tensor,
    query_start_loc: Tensor,
    scale: float,
    grid_extent: int = 1,
    max_pages: int = 0,
    m_budget: int = 0,
    p_budget: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    flash_attn as a ONE-SPLIT partial set, in phase 1's flat layout.

    Args arrive positionally via `Kernel.normalize_args` (`runtime/kernel.py:966`), so
    this mirrors `_kernel_unified`'s signature exactly. At one split `_combine_ref`
    reduces to `p_out / p_l` (`alpha = exp2(m - m) = 1`), so the exact partial set for a
    known answer `O` is `p_out = O`, `p_l = 1`, `p_m = 0` -- the true softmax statistics
    are never needed. `p_m` must be 0, NOT `-inf`, which `_combine_ref` treats as an
    empty split. Mirrors `kernel_flash.baseline_flash_split`.

    `max_seqlen_q` is DERIVED here, never taken from `grid_extent`. `grid_extent` is
    dual-meaning (`:124`): `max_seqlen_q` under GRID_FLAT=False but `num_q_tiles` under
    True, an unrelated quantity. Handing the flat value to FA3 as `max_seqlen_q`
    silently corrupts the very baseline the tuner scores every candidate against -- and
    `q_load_mask`-style masking hides a wrong `max_seqlen_q` from any correctness check,
    so it would never surface as a failure. The `.max()` is a D2H sync, which is fine in
    a baseline and would NOT be in the kernel's host path (that is exactly why
    `unified_factory` uses the shape-derived bound instead).
    """
    del grid_extent, max_pages, m_budget, p_budget
    from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

    max_seqlen_q = int((query_start_loc[1:] - query_start_loc[:-1]).max())
    out = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        page_table=block_table,
        cache_seqlens=seqused_k,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=scale,
        causal=True,
    )
    num_tokens, num_q_heads, head_dim = out.shape
    n_rows = num_tokens * num_q_heads
    p_out = out.reshape(1, n_rows, head_dim).to(torch.float32)
    p_l = torch.ones([1, n_rows], dtype=torch.float32, device=out.device)
    p_m = torch.zeros([1, n_rows], dtype=torch.float32, device=out.device)
    return p_out, p_l, p_m


# The LOW bound of `q_block`'s `hl.register_block_size` (`kernel_grid.py` ==
# `kernel_flat.py`), duplicated here because the host needs it to size the GRID_FLAT
# grid and the in-body bound must stay an inline literal (a name load in a kernel body
# traces to an unbacked SymInt). MUST equal that literal: the flat bound is only safe
# while every `q_block` the tuner can pick is >= this value. Raising it here without
# raising it there silently drops query tiles. 1, not 16. At decode the M extent is
# `q_block * num_groups` with only `num_groups` rows real, so a floor of 16 made M=128
# with 8 real rows -- 6.2% utilization -- and made `kernel_tune`'s and `kernel_ref`'s
# packing (M == num_groups) UNREACHABLE rather than merely unchosen: a supplied
# `block_sizes[0] < 16` is clamped UP by `BlockSizeSpec._normalize`
# (`config_spec.py:2902-2906`), the same never-range-checked branch INVARIANT 14
# documents for the max direction. Measured on Qwen3-32B decode, `num_splits=1`, pinned
# configs (`logs/qb_real.log`, registered min patched to 1 in a module copy):
#     nseq/ctx      grid | kernel_ref | q_block=1 | q_block=16 | ratio
#      16 /  4956    128 |   0.621    |   0.638   |   0.550    | 1.16x
#      64 /  4956    512 |   0.752    |   0.846   |   0.502    | 1.68x
#     256 /  4956   2048 |   0.898    |   0.846   |   0.486    | 1.74x
#     256 / 18618   2048 |   0.898    |   0.854   |   0.480    | 1.78x
#       1 / 18618      8 |   0.145    |   0.123   |   0.114    | 1.08x
#
# Monotone in `q_block` and the gain grows with occupancy; the starved cell is flat
# (that one is the `num_splits` axis, not this one). This is a RANGE change, not a
# heuristic: which value wins is left to the tuner and the AOT tree, because it is
# exactly the decision `kernel_occ.py` made host-side from a wave heuristic and thereby
# collapsed its tree to a single leaf (INVARIANT 8's neighbour, plan Sec 3). `q_block`
# is a legitimate tunable under the INVARIANT 8 test -- its legal range does not depend
# on the model (1 is always legal); only its MAX does, and that is what `m_budget`
# carries into the specialization key. Cost, measured over the real pools: the GRID form
# pays NOTHING (its extent is `max(max_seqlen_q, 2)` and never reads this). The
# GRID_FLAT bound `floor(N/B) + S` stays provably safe (`floor(N/1) + S = N + S`
# dominates every `b >= 1`) but loses slack, and the loss is NOT a corner case: measured
# over the real pools it is geomean 5.3x on MEASURE (n=288, median 4.6x) and 6.9x on
# COLLECT (median 11x). Only the four DECODE cells sit in the mild 1.9-2.0x band (there
# `floor(N/16)` is already degenerate and `+S` dominates); 15-16x is the NORM on the
# nseq=1 prefill cells (qlen=4096: 257 -> 4097), and 11-12x on the mid cells. Every
# over-provisioned FLAT block still runs the full binary search and its four tensor
# loads before reaching the guard, so this is real prologue work on ~5x more blocks
# pool-wide -- the very waste stage 2 exists to remove. A real regression for FLAT
# prefill, and why the two forms are measured separately (INVARIANT 15).
_Q_BLOCK_MIN = 1
# `m_budget`'s FLOOR, deliberately NOT `_Q_BLOCK_MIN`. These were one constant and must
# not be again: the floor keeps the registered max from collapsing below the M extent
# that prefill needs (at ng=8, `128 // 8 == 16`), while the min above must reach 1 for
# decode. Tying them made `q_block`'s range degenerate (16..16) at ng=8 -- the geometry
# the decode pool measures.
_M_BUDGET_FLOOR = 16


def p_budget_for(device, page_size: int, head_dim: int, itemsize: int) -> int:
    """
    `block_p`'s registered max: the largest power of two whose live K+V tile fits smem.

    Exported because `_phase_a.check_split_counts` pins a config rather than autotuning,
    and a pinned `block_p` above this bound is unlaunchable (`OutOfResources`) on the
    shapes the bound exists for. Two copies of the arithmetic would be the INVARIANT 14
    drift this guards against.
    """
    smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    per_p = 2 * page_size * head_dim * itemsize
    return max(2, min(32, 1 << (max(smem // per_p, 1).bit_length() - 1)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def unified_factory(kernel_fn, combine_fn, grid_flat: bool):
    """
    Returns the vLLM-shaped callable. Mirrors `kernel_prefill_factory`
    (`kernel_prefill.py:213-300`) and `kernel_flash_factory`.

    def helion_unified_attention(
        query, key_cache, value_cache, block_table, seqused_k,
        query_start_loc=None, max_seqlen_q=1, max_seqlen_k=0,
        softmax_scale=None, causal=True, out=None, **kwargs,
    ) -> Tensor

    Host-side responsibilities, in order:
      1. asserts: causal; query_start_loc is not None; pow2 head_dim; num_q_heads %
         num_kv_heads == 0; pow2 num_groups.
      2. softmax_scale default head_dim ** -0.5.
      3. 0/1-specialization padding, on TWO SEPARATE predicates -- this is the bug that
         silently zeroed every row but row 0 on 40/86 cells:
           - `block_table.size(0) == 1` -> repeat seqused_k / block_table / query_start_loc
           - `query.size(0) == 1`       -> pad query, because query is varlen-packed so the
             0/1-specialized dim is num_tokens, NOT num_seqs (`kernel_prefill.py:274-275`)
      4. grid_extent: `max_seqlen_q` if not `grid_flat` else
         `query.size(0) // _Q_BLOCK_MIN + rows` (computed AFTER the padding in step 3, so both
         terms describe the tensors actually passed).
         NOTE: max_seqlen_q must be max(query_lens), NOT capture_max_query_len -- the latter
         inflated launched q-tiles 2.97x over 86 fullcg entries (16x worst case), invisible in
         correctness because q_load_mask discards the surplus.
      5. call phase 1, then phase 2 into `out` (allocating `out` if None).
      6. un-pad and return.

    `grid_flat` is an explicit PARAMETER, not a module global. It must agree with which
    body `kernel_fn` is -- the grid extent means `max_seqlen_q` for
    `kernel_grid._kernel_unified` and `num_q_tiles` for
    `kernel_flat._kernel_unified_flat`, and a mismatch under-tiles silently (INVARIANT
    12). Passing it alongside `kernel_fn` at the single construction site in
    `__init__.py` is what keeps the two from drifting; a global read from another module
    is exactly how they drift (and, before this was a parameter, a NameError at every
    call).
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
        # The M extent is `q_block * num_groups`; a non-pow2 product is not a legal tile
        # shape.
        assert num_groups & (num_groups - 1) == 0, (
            f"num_groups={num_groups} must be pow2"
        )
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        # TWO SEPARATE 0/1-specialization predicates (INVARIANT 4). A leading dim of
        # runtime size 1 loses its stride; the pad row gets `query_len == 0`, which the
        # kernel's `q_start < q_end` guard skips.
        if block_table.size(0) == 1:
            seqused_k = seqused_k.repeat(2)
            block_table = block_table.repeat(2, 1)
            query_start_loc = torch.cat([query_start_loc, query_start_loc[-1:]])
        # `query` is varlen-packed, so ITS 0/1-specialized dim is `num_tokens`, NOT
        # `num_seqs` (`kernel_prefill.py:274-275`). Copying decode's `num_seqs`
        # predicate here silently zeroes every row but row 0 (40/86 cells, ascending
        # order only).
        if query.size(0) == 1:
            query = query.repeat(2, 1, 1)
        # A single-page cache collapses the gather's page dim.
        if block_table.size(1) < 2:
            block_table = block_table[:, -1:].repeat(1, 2)

        # `rows`, not `num_tokens`: the key must describe the grid the kernel is
        # launched with (`kernel_tune.py:257-262`).
        rows = block_table.size(0)

        # `q_block`'s registered max, computed HERE and passed as an `hl.constexpr`
        # rather than written inline in the kernel body (INVARIANT 14). The M extent is
        # the product `q_block * num_groups` and 128 is the budget for the product, so
        # the bound depends on the model -- and the AOT heuristic cannot see
        # `num_groups` (it extracts only `dim0`/`dim1`/`numel` per tensor arg), while
        # `config_spec.py:2902-2906` never clamps a supplied `block_sizes` entry DOWN to
        # `max_size`. As a constexpr it is in the specialization key, so an ng=4 leaf
        # and an ng=8 leaf cannot be selected for each other.
        m_budget = max(_M_BUDGET_FLOOR, 128 // num_groups)
        # `block_p`'s registered max, same mechanism and same reason (INVARIANT 14). The
        # live K+V tile is `block_p * page_size` tokens with K and V both resident, and
        # that PRODUCT is the measured smem invariant -- it does NOT scale with the M
        # extent (`experiment/details/kernel-prefill.md`, "largest launchable block_p at
        # q_block=32"):
        #     num_groups | page 16 | page 32 | page 64        product
        #        1        |   32    |   16    |    8            512
        #       >=4       |   16    |    8    |    4            256
        #
        # That table was measured at `head_dim=128` ONLY, and the token product is not
        # the whole invariant: the live tile is `2 * block_p * page_size * head_dim *
        # itemsize` bytes (K and V both resident), so head_dim scales it linearly.
        # Verified against the H100 opt-in limit (232448 B): the byte formula reproduces
        # the table's 16/8/4 frontier at head_dim 128 exactly, and at head_dim 256 the
        # frontier halves to 8/4/2 -- which is why a page-64 cell that launches on Qwen3
        # (hd 128) raised `OutOfResources: Required 294912` on Gemma-2-9B (hd 256).
        # Deriving from bytes rather than from the token product covers both. The
        # `num_groups` term of the table (512 at ng=1 vs 256 at ng>=4) is NOT reproduced
        # here. It is not a property of the KV tile -- the doc's own reading is that
        # `block_p * page_size` "does NOT scale with the M extent" -- so it is headroom
        # left by a smaller M, which `m_budget` now holds constant at 128 for every ng.
        # Taking the ng>=4 row for all ng is the conservative direction: it can only
        # under-use the ng=1 headroom, never over-commit. NOT a second tunable:
        # `experiment/details/kernel-prefill.md` rejects
        # `register_tunable("block_p_bound", ...)` explicitly -- two independent knobs
        # for one physical limit lets the tuner pick `bound=32, block_p=32` at page 64,
        # which is the `ng_block` silent-garbage pattern. This is a DERIVED constexpr:
        # the caller cannot choose it, so that failure is unrepresentable. Derived in
        # `p_budget_for` (shared with `_phase_a`, which pins a config): the limit is a
        # per-device `torch.cuda` property (H100 232448 B), not a literal. Floored at 2
        # (`register_block_size`'s own min) and capped at 32 (the previous flat max,
        # which no shape reaches anyway) so an unlaunchable cell still fails LOUD rather
        # than silently registering a degenerate 1-page tile.
        p_budget = p_budget_for(query.device, page_size, head_dim, query.element_size())

        if grid_flat:
            # Sync-free upper bound on the number of q-blocks, from
            # `triton_unified_attention.py:730-741`:
            #   sum_i ceil(qlen_i / B) <= sum_i (floor(qlen_i / B) + 1)
            #                          <= floor(sum_i qlen_i / B) + num_seqs
            # It depends only on `query.size(0)` and `num_seqs` -- both static shapes --
            # which is the whole point: reading the query lengths to get the EXACT count
            # is a D2H sync (the reference says so at `:734`) and would break graph
            # capture in stage 5. `_Q_BLOCK_MIN`, not the tuned `q_block`: `q_block` is
            # an `hl.register_block_size` resolved INSIDE the kernel (by the autotuner,
            # or by the AOT heuristic, which keys on this very argument --
            # `_helion_aot_kernel_cuda_sm90.py` `_arg7_scalar` -- so the host cannot
            # read it without a circular dependency). Using the registered MINIMUM is
            # what makes the bound safe for whatever the kernel actually picks:
            # `floor(N/B)` is non-increasing in B, so for any actual `b >= _Q_BLOCK_MIN`
            # actual(b) = sum_i ceil(qlen_i/b) <= floor(N/b) + S <=
            # floor(N/_Q_BLOCK_MIN) + S i.e. a LARGER actual `q_block` only widens the
            # slack; it can never under-provision. Verified exhaustively over random
            # (num_seqs, qlens) x b in {16,32,64,128}: the launched set of (seq, local)
            # pairs is exactly the required set, never short. A too-small bound would
            # silently DROP query tiles, so this direction is the one that must be
            # provable.
            grid_extent = max(query.size(0) // _Q_BLOCK_MIN + rows, 2)
        else:
            # `max(query_lens)`, never `capture_max_query_len` -- the latter inflated
            # launched q-tiles 2.97x, invisibly (`q_load_mask` discards the surplus).
            # Floored at 2 for the same 0/1-specialization reason as the tensors above:
            # at `max_seqlen_q == 1` Helion emits `tl.arange(0, 1)` for
            # `hl.arange(q_block)`, collapsing the M tile to ONE query row, and the
            # specialization key records only `<class 'int'>` for a bare numeric arg
            # (`kernel_prefill.py:324-338`), so that degenerate kernel is then reused
            # for every later value.
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
        # Phase 2. Stage 1 has ONE split, where the merge reduces to
        # `partial_out / partial_l` (`alpha = exp2(m - m) = 1`) -- i.e. exactly
        # `_kernel_combine`'s epilogue, including the `where(l == 0, 1.0, l)` that makes
        # a padded slot read zero rather than NaN (INVARIANT 5). So at one split the
        # compiled kernel and `_combine_ref` are the same function, which is what makes
        # this swap verifiable. `n_keep` slices the padded query row off the PARTIALS --
        # they are flat `[splits, token*head]`, so the slice is in ROWS, not tokens --
        # which is what makes the write land exactly `num_tokens` output rows
        # (`kernel_tune.py:277-283`). The compiled kernel takes it as `num_live_rows`
        # and BOUNDS ITS GRID by it instead of slicing: a stage-5 builder-owned partial
        # buffer is sized to the capture-time maximum, so the surplus rows are stale
        # rather than absent and a slice would not be enough (plan Sec 4).
        n_keep = num_tokens * num_q_heads
        if combine_fn is not None:
            # Mutates `out` in place -- no full-output copy, which is plan Sec 5 defect
            # 5.
            combine_fn(out, partial_out, partial_l, partial_m, n_keep)
        else:
            merged = _combine_ref(
                partial_out[:, :n_keep], partial_l[:, :n_keep], partial_m[:, :n_keep]
            )
            out.copy_(merged.view(num_tokens, num_q_heads, head_dim).to(out.dtype))
        return out

    return helion_unified_attention


# ---------------------------------------------------------------------------
# Autotune baselines (required, not optional)
# ---------------------------------------------------------------------------
def _baseline_unified(*args, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
    """
    Launchable baseline for `autotune_baseline_fn`.

    Required because the tuner's baseline ignores the pinned config and compiles the
    DEFAULT, which takes both registered maxes and cannot launch (294912 B at ng=4 /
    page 16). Mirrors `kernel_prefill.baseline_prefill`.
    """
    return _baseline_partials(*args, **kwargs)


def _check_partials_close(got, expected) -> bool:
    """
    `autotune_baseline_accuracy_check_fn` for the 3-tuple return.

    The default check cannot compare (out, l, m) triples; m is -inf on empty splits.
    Mirrors `paged_attention/kernel_tune.check_partials_close`.
    """
    a_out, a_l, a_m = got
    e_out, e_l, e_m = expected
    torch.testing.assert_close(
        _combine_ref(a_out, a_l, a_m),
        _combine_ref(e_out, e_l, e_m),
        atol=1e-2,
        rtol=1e-2,
    )
    return True


# ---------------------------------------------------------------------------
# INVARIANTS -- each of these was a silent-wrong-answer bug. Do not regress.
# ---------------------------------------------------------------------------
# 1. Gather block index clamped to a STATIC bound (`kernel_tune.py:143-145`).
#    Data-dependent TMA extent faults and poisons the whole CUDA process.
# 2. Interior/diagonal page split rounded DOWN to a whole block_p
# (`kernel_prefill.py:135-146`). hl.tile's tail tile is partial but tile_p.block_size in
# the gather reshape is the FULL block size, so the overhang loads real KV past the
# causal
#    range. Correctness condition, not an optimization. Cost 6/16 shapes when absent.
# 3. BOTH ends of a split range clamped to n_page_max (`kernel_flash.py:229-238`). A
# reversed
# hl.tile range is NOT a no-op like an empty one -- it silently scans KV and contributes
#    garbage (18/80 (shape,split) pairs wrong).
# 4. The 0/1-specialized dim for varlen query is num_tokens, NOT num_seqs
#    (`kernel_prefill.py:302` and `:318`, rationale `:307-315`). Two separate padding
#    predicates, not an elif.
# 5. Padded slots (seq_len == 0) read exactly zero, not NaN
# (`kernel_prefill.py:245-246`). 6. Never let a head index be a SCALAR MID coordinate in
# a store: Helion folds the offset into
# the descriptor base and corrupts it (`kernel_tune.py:214-217`). `merge_tune` satisfies
# this
# with a 1-wide head slice `out[t, h:h+1, :]`; `_kernel_combine` satisfies it by a
# different
# route -- its row axis is FLAT `token*head`, so it stores `out.view(-1, head_dim)[row,
# :]` and
# no mid coordinate exists. Do not "simplify" that to a 2-D grid over the 3-D `out`;
# that
#    reintroduces the scalar-mid form. Same requirement, two admissible shapes.
# 7. context_len floored at 0. HARDENING WITH NO PREDECESSOR: both
# `kernel_prefill.py:153`
# and `kernel_flash.py:193` compute `seq_len - (q_end - q_start)` UNCLAMPED. Added here
# at
# `kernel_grid.py:150` / `kernel_flat.py` because a negative context is a silent NaN,
# not an error. 8. A knob whose legal range depends on the MODEL cannot be a knob
# (prefill README).
# Note this is about a bound that varies with a value the CONFIG cannot see. A bound
# that is
#    merely unlaunchable on some hardware is a different, weaker problem, fixed by
# `autotune_baseline_fn` + OutOfResources skipping rather than by narrowing the bound.
# 9. `autotune_baseline_fn` is REQUIRED, not optional (`__init__.py`). Both relaxed
# bounds above
# depend on it. `paged_attention/kernel_tune.py:103` keeps block_p at 16 precisely
# because the
# decode package sets only `autotune_baseline_accuracy_check_fn` and never `baseline_fn`
# (`paged_attention/__init__.py:214, 287`), so its default config IS still compiled. Do
# not
#    port a relaxed bound without porting the precondition.
# 10. STAGE 2. The GRID_FLAT grid bound divides by `_Q_BLOCK_MIN` (the REGISTERED
# MINIMUM
# q_block), never by the tuned one. `floor(N/B) + S` is non-increasing in B, so a larger
# actual q_block only widens the slack; a divisor larger than the actual q_block would
# UNDER-provision and silently drop query tiles. `_Q_BLOCK_MIN` must track the literal
# at
# `q_block = hl.register_block_size(1, ...)` -- now 1, so the bound is `N + S` and holds
# for
# every `b >= 1` by the same argument. Do NOT reintroduce `_M_BUDGET_FLOOR` here: the
# MAX's
#     floor and the MIN are separate constants (see their definitions).
# 15. The two forms are NOT interchangeable on the `q_block` range. Lowering the
# registered
# minimum to 1 (decode parity, worth 1.68-1.78x saturated) is free for GRID but costs
# GRID_FLAT geomean 5.3x grid extent over MEASURE (15-16x on nseq=1 prefill), because
# its
# bound divides by that same minimum. Report GRID and FLAT separately per REGIME; a
# single
#     geomean hides a win in one
#     form paid for by a loss in the other.
# 11. STAGE 2. `_baseline_partials` must DERIVE `max_seqlen_q` from `query_start_loc`,
# never take
#     `grid_extent` -- that argument means `num_q_tiles` under GRID_FLAT, and a wrong
# `max_seqlen_q` corrupts the tuner's accuracy baseline invisibly (masking hides it).
# 12. STAGE 2. `GRID_FLAT` selects a kernel FUNCTION at `__init__` IMPORT time. It
# cannot be read
#     in a kernel body (unbacked SymInt) and Helion forbids an `if` around a grid loop
# (NestedGridLoop), so there is no runtime branch. Flipping the switch after import
# leaves
# the 3-D body compiled while the factory computes the flat extent -- wrong rows, no
# error.
#     Reload the package after changing it.
# 13. STAGE 2. In the flat body `block_table[seq_idx, p_idx]` does NOT lower when
# `seq_idx` comes
# out of the `while` search (InductorLoweringError "list index out of range"); the 1-D
# gather
#     `hl.load(block_table_flat, [seq_idx * max_pages + p_idx])` does.
# 14. STAGE 3. A `register_block_size` MAX that varies with the model or the layout must
# be passed
# in as an `hl.constexpr` (`m_budget`, `p_budget`), never written inline in the body.
#     Two independent facts make an inline bound wrong, and both are needed to see why:
# (a) `config_spec.py:2902-2906` `_normalize` clamps a supplied `block_sizes` entry UP
# to
# `min_size` and has NO `max_size` branch, so the max constrains only the autotuner's
#           SEARCH -- never a replayed config.
# (b) the AOT heuristic generator builds its key from ARGUMENT features only (tensor
# dim0/dim1/numel and scalar args). `hl.specialize` results live in `extra_results`,
# which forks the COMPILED KERNEL cache but is invisible to the generated tree. So
# `num_groups` (derived from arg shapes) and `page_size` (specialized) could not be cut
#           on, and one leaf served every model.
#     Consequence measured on the 2026-09-08 AOT eval: 28 failures, all AOT-only, all
# `OutOfResources` at REPLAY -- not wrong numerics. 12 cells at ng=8/page-16 selected
# `block_sizes=[32,8]` (tuned at ng=4, M=256 there: Required 262208, limit 232448) and 3
# at
# ng=2/page-64 selected `[32,4]` (256 tokens/tile: Required 409632). Both leaves are
# perfectly
# launchable at the ng/page they were tuned at, which is why nothing failed during
# tuning.
# A constexpr budget is in the specialization key by construction, so those leaves
# become
# unreachable for the models they are illegal on. Verified: ng4/p16, ng8/p16, ng4/p64
# and
# ng2/p64-hd256 now produce four distinct keys with (m_budget, p_budget) = (32,16),
# (16,16),
#     (32,4), (64,2).
# NOT a fixed literal instead: capping `q_block` at 16 would fix ng=8 and throw away the
#     measured-best tile at every smaller ng (the legal range is 1..128 at ng=1).
#     NOT a second TUNABLE either: `experiment/details/kernel-prefill.md` rejects
# `register_tunable("block_p_bound", ...)` -- two knobs for one physical limit lets the
# tuner
# pick `bound=32, block_p=32` at page 64, which is INVARIANT 8's pattern again. A
# derived
#     constexpr is not choosable by the caller, so that is unrepresentable.
# `p_budget` is derived from BYTES (`2 * block_p * page_size * head_dim * itemsize` vs
# the
#     device's `shared_memory_per_block_optin`), not from the token product
# `block_p * page_size <= 256` recorded in `kernel-prefill.md`: that table was measured
# at
# head_dim 128 only. The byte form reproduces its 16/8/4 frontier exactly there and
# correctly
#     halves to 8/4/2 at head_dim 256, which is the Gemma-2-9B page-64 cell.
