"""
Unified paged attention, flat grid form (plan stage 2).

Grid `[num_q_tiles, num_kv_heads]`: only the QUERY axis is flattened, over the sync-free
host-side bound `query.size(0) // _Q_BLOCK_MIN + num_seqs`
(`triton_unified_attention.py:730-741`), with `(seq_idx, q_offset)` recovered on device
by binary search over `query_start_loc` (`triton_attention_helpers.py:45-105`). Removes
the 3-D form's 87% empty tiles in R1; whether the search costs less than the tiles it
removes is the measurement that decides stage 2.

Output-identical to `kernel_grid.py` by construction -- everything from the `q_start +
q_offset < q_end` guard down is the same computation, with `q_offset` in place of
`tile_q.begin`. Any INVARIANT-bearing change must be made in BOTH files.

Its own AOT artifact is `_helion_aot_kernel_flat_cuda_sm90.py`, which is REQUIRED and
not a convenience: `grid_extent` is arg 7 and the generated tree cuts on it
(`_arg7_scalar <= 2.0 / 8.0 / 64.0`), but it means `max_seqlen_q` there and
`num_q_tiles` here -- so the 3-D tree would mis-select on every call made through this
file.
"""

from __future__ import annotations

import helion.language as hl
import torch
from helion.autotuner import PowerOfTwoFragment
from torch import Tensor

from .kernel_common import _LOG2_E


def _kernel_unified_flat(
    query: Tensor,  # [num_tokens, num_q_heads, head_dim] varlen-packed, fp16/bf16
    key_cache: Tensor,  # [num_blocks, page_size, num_kv_heads, head_dim] LBNHC
    value_cache: Tensor,  # same
    block_table: Tensor,  # [num_seqs, max_pages] int32
    seqused_k: Tensor,  # [num_seqs] int32 -- TOTAL context (vLLM `seq_lens`)
    query_start_loc: Tensor,  # [num_seqs + 1] int32, tail-padded
    scale: float,
    grid_extent: int,  # max_seqlen_q (GRID_FLAT=False) | num_q_tiles (True)
    max_pages: hl.constexpr,  # block_table.size(1): static gather clamp + KV len in the key
    m_budget: hl.constexpr,  # `q_block`'s registered max: max(_M_BUDGET_FLOOR, 128 // num_groups)
    p_budget: hl.constexpr,  # `block_p`'s registered max: 512 // page_size, floored at 2
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns (partial_out, partial_l, partial_m). Does NOT take `out=`.

    Returning keeps `mutated_arg_indices` empty so subprocess benchmarking stays
    enabled, which turns an IMA from a run-killing TritonUnrecoverableRuntimeError into
    a skipped config (`kernel_prefill.py:44-51`). vLLM's buffer is written by phase 2
    instead, which also removes the redundant full-output copy the current decode path
    pays.

    `grid_extent` is a plain host int, exactly as `max_seqlen_q: int` already is
    (`kernel_prefill.py:41`). Under GRID_FLAT it is the sync-free upper bound
    `query.size(0) // _Q_BLOCK_MIN + num_seqs >= sum_i ceil(qlen_i / q_block)`, from
    `triton_unified_attention.py:730-741` -- shape-derived, so no D2H sync and stable
    under graph capture. The divisor is the REGISTERED MINIMUM q_block, not the tuned
    one, which is what keeps the bound valid for whatever the tuner picks (see
    `unified_factory`).

    constexpr set is deliberately minimal. `static_shapes=False` buckets every dim >= 2
    to a literal 2, so anything that must fork the specialization key has to be
    constexpr:
      - `max_pages` puts KV length in the key AND is the static clamp bound for the gather.
        Must be static: a data-dependent TMA extent faults and poisons the whole CUDA process
        (`paged_attention/kernel_tune.py:143-145`).
    num_q_heads / head_dim / page_size / num_kv_heads come from `hl.specialize` on the
    tensor shapes in the body -- model constants, not call-site variables.

    Body sections:
      1. specialize the model dims; derive num_groups.
      2. tunables: `q_block = hl.register_block_size(1, m_budget)` and
         `block_p = hl.register_block_size(2, p_budget)`. Both maxes arrive as `hl.constexpr`
         ARGUMENTS, computed host-side in `kernel_common.unified_factory`, because they are
         model-dependent and a bound written inline cannot be (a name load in a kernel body
         traces to an unbacked SymInt, so an inline bound must be a literal). Making them
         arguments puts them in the specialization key, which is the point -- see INVARIANT 14:
         `config_spec.py:2902-2906` never clamps a supplied `block_sizes` entry DOWN to
         `max_size`, so a registered max bounds only the autotuner's SEARCH, and the AOT tree
         reads argument features only. A model-dependent max that is NOT an argument therefore
         lets the tree replay a leaf on a model the leaf is unlaunchable on -- measured as 28
         AOT-only `OutOfResources` failures. Neither a fixed literal (it cannot hold at every
         page_size/head_dim) nor a second tunable (`experiment/details/kernel-prefill.md` rejects
         `register_tunable("block_p_bound", ...)`: two knobs for one physical limit is the
         `ng_block` silent-garbage pattern) is acceptable. Precedent: `kernel_ref.py:59,78`.
      3. allocate partials (stage 5 moves this out to the builder; see module docstring of
         `_kernel_combine`).
      4. grid. GRID_FLAT=False: `hl.tile([num_seqs, num_kv_heads, grid_extent],
         block_size=[1, 1, q_block])`. True: 2-D `hl.tile([grid_extent, num_kv_heads],
         block_size=[1, 1])` -- only the QUERY axis is flattened, `num_kv_heads` stays a real
         grid dim (what `triton_unified_attention.py:730-741` launches), so the head indexing is
         unchanged. Then recover (seq_idx, q_block_local_idx) by binary search over
         `query_start_loc`, mirroring `triton_attention_helpers.py:44-106`
         (`q_block_start_idx = qsl[seq]//q_block + seq`).
      5. per-row lengths + tail-pad guard `if q_start + m_block*q_block < q_end` -- both grid
         forms need it: the flat bound over-provisions, the 3-D grid over-tiles.
      6. load Q 3-D then flatten to M = (q_pos, q_head) pairs; `m_pos` tile-relative.
      7. causal bound, FA3 bottom-right (`hopper/block.h:23-33`), no offset fixup:
         `context_len = seqused_k - (q_end - q_start)`, floored at 0.
      8. interior/diagonal page split, rounded DOWN to a whole block_p -- a correctness
         condition, not an optimization (see INVARIANTS below).
      9. interior KV loop: no mask, no -inf guard (~96.7% of visited pages at chunked prefill,
         +13.9% geomean).
    10. diagonal KV loop: causal mask + full m_safe NaN guard. 11. write partials.
    """
    # `GRID_FLAT` is NOT read here, in either kernel body. A module-global name load in
    # a kernel body traces to an unbacked SymInt, so `if not GRID_FLAT:` becomes `Eq(u3,
    # 1)`, a GuardOnDataDependentSymNode -- and `hl.constexpr` is no escape either,
    # because Helion requires grid loops at a function's TOP LEVEL (NestedGridLoop), so
    # no `if` can wrap one. Hence two separate kernel functions sharing this prologue
    # and `_attend_tile`, selected host-side in `unified_factory`.
    num_tokens, num_q_heads, head_dim = query.size()
    num_q_heads = hl.specialize(num_q_heads)
    head_dim = hl.specialize(head_dim)
    page_size = hl.specialize(key_cache.size(1))
    num_kv_heads = hl.specialize(key_cache.size(2))
    num_groups = num_q_heads // num_kv_heads
    # Static clamp bound for the gather (INVARIANT 1): a data-dependent TMA extent
    # faults and poisons the whole CUDA process
    # (`paged_attention/kernel_tune.py:143-145`).
    max_block_idx = max_pages - 1
    num_seqs = block_table.size(0)
    block_table_flat = block_table.reshape(-1)

    # `m_budget`, NOT the inline `max(16, 128 // num_groups)` this used to be (INVARIANT
    # 14). The M extent is the PRODUCT `q_block * num_groups` (`kernel_prefill.py:122`)
    # and 128 is the budget for that product, so the bound is model-dependent -- 32 at
    # ng=4, 16 at ng=8. But a `register_block_size` max is NOT enforced on a supplied
    # config: `config_spec.py:2902-2906` `_normalize` clamps UP to `min_size` and has no
    # `max_size` branch at all. And `num_groups` is invisible to the AOT heuristic
    # generator, which extracts only `dim0`/`dim1`/`numel` per tensor arg -- never
    # `arg0_dim1`/`arg1_dim2`. So a leaf tuned at ng=4 carrying `q_block=32` was
    # replayed unchanged at ng=8 for an M extent of 256 against the 128 budget (22 of 28
    # AOT eval failures). Passing the bound as an `hl.constexpr` is what fixes it: a
    # constexpr is in the specialization key by construction, so the ng=4 and ng=8
    # leaves become structurally distinct and neither can be selected for the other.
    # This is the escape the plan names for `block_p` ("acceptable only because page
    # size is in the specialization key"), and the mechanism `kernel_ref.py:59,78`
    # already uses for `q_block_padded_size`. NOT a fixed literal: 16 would cap the M
    # extent at 128 only at ng=8 and would throw away the measured-best `q_block` at
    # every smaller ng (at ng=1 the legal range is 1..128). NOT
    # `hl.specialize(num_groups)` either -- that is already done at `:113` and does not
    # help, because the value is hidden from the TREE, not from the compiler. The MIN is
    # 1, not 16, and must stay equal to `kernel_common._Q_BLOCK_MIN` (INVARIANT 10). At
    # decode only `num_groups` of the `q_block * num_groups` M rows are real, so a floor
    # of 16 forced M=128 for 8 real rows and made `kernel_tune`'s packing unreachable --
    # not unchosen, UNREACHABLE, since `config_spec.py:2902-2906` clamps a supplied
    # value UP to `min_size`. Worth 1.68-1.78x on saturated decode pinned, and the fully
    # tuned kernel reaches 0.974 vs FA3 there (vs 0.48-0.50 at the old minimum); table
    # at `kernel_common.py:_Q_BLOCK_MIN`. Which value wins is the TUNER's call, not a
    # host heuristic (plan Sec 3).
    q_block = hl.register_block_size(1, m_budget)
    # `p_budget`, NOT the flat literal 32 this used to be (INVARIANT 14). The live K+V
    # tile is `block_p * page_size` tokens, so the smem cost is that PRODUCT -- the same
    # shape of constraint as `q_block * num_groups`, on the other axis. A flat max was
    # affordable for the AUTOTUNER (unlaunchable candidates raise OutOfResources at
    # precompile and are skipped, INVARIANT 9), but an AOT leaf is REPLAYED, not
    # re-searched: `block_p=4` tuned at page 16 (64 tokens/tile) was selected at page 64
    # (256 tokens/tile) and raised `OutOfResources: Required 409632, limit 232448` -- 3
    # of 28 eval failures. `page_size` is `hl.specialize`d, which forks the compiled
    # kernel but NOT the AOT tree: the heuristic generator reads argument features only
    # (tensor dim0/dim1/numel and scalar args), so a specialized value is invisible to
    # it. As a constexpr the budget becomes a real feature.
    block_p = hl.register_block_size(2, p_budget)
    # The K axis (stage 3). `1` degenerates to stage 1's grid exactly, so the split is
    # retired as a TUNING outcome rather than a host-side heuristic -- which is the
    # point: `kernel_occ.py` decided it host-side from a wave heuristic and that
    # collapsed its AOT tree to a single leaf. Literal bounds, not a name (a name load
    # traces to an unbacked SymInt), and 32 matches vLLM's own
    # `flash_attn_max_num_splits_for_cuda_graph` (`vllm/config/attention.py:45`).
    num_splits = hl.register_tunable("num_splits", PowerOfTwoFragment(1, 32))

    # Sized at `num_splits`, NOT at a constant 32, and allocated INSIDE the body --
    # which is legal precisely because `hl.register_tunable` is readable here.
    # Allocating host-side would force the constant 32 (the caller cannot see the
    # tunable), and that is `paged_attention/kernel_split.py:153-157`'s "largest known
    # unfixed cost": 0.618x -> 0.873x when sized correctly. Here `num_tokens` reaches
    # 8192, where a constant-32 allocation is 4.4 GB of pure HBM round-trip against
    # target times of 0.4-1.5 ms. In-body allocation IS capturable: PyTorch's allocator
    # switches to a graph-private pool during capture, and Helion constant-folds the
    # tunable into a literal in the generated host wrapper, so the split count is a
    # compile-time constant per config (`_cudagraph_probe.py`, 4/4; plan Sec 4's
    # 2026-09-07 note, and the checks it still asks for). `zeros`/`-inf` is
    # load-bearing, not hygiene (`kernel_flash.py:174-180`): the merge weights each
    # split by `alpha`, which is 0 for an `m == -inf` split, and NaN*0 == NaN. A
    # (q-tile, split) pair that writes nothing -- reachable whenever `num_splits >
    # n_page_max` -- must read as an EMPTY split, not as garbage.
    n_rows = num_tokens * num_q_heads
    dev = query.device
    partial_out = torch.zeros(
        [num_splits, n_rows, head_dim], dtype=torch.float32, device=dev
    )
    partial_l = torch.zeros([num_splits, n_rows], dtype=torch.float32, device=dev)
    partial_m = torch.full(
        [num_splits, n_rows], float("-inf"), dtype=torch.float32, device=dev
    )

    # Stage 2 grid. 2-D `[grid_extent, num_kv_heads]`: only the QUERY axis is flattened;
    # `num_kv_heads` stays a real grid dim, so the head indexing inside `_attend_tile`
    # is unchanged. This is the shape the reference launches
    # (`triton_unified_attention.py:730-741`). `grid_extent` is the sync-free UPPER
    # bound `query.size(0) // _Q_BLOCK_MIN + num_seqs`, computed host-side in
    # `unified_factory`. The divisor is the REGISTERED MINIMUM (now 1, and NOT the
    # literal 16 this comment used to name), never the tuned `q_block` -- see INVARIANT
    # 10 for why that direction is the safe one. It over-provisions; the guard below is
    # that slack, and at min=1 the slack is widest (INVARIANT 15). The split axis goes
    # FIRST, as in the 3-D form and both predecessors (`kernel_tune.py:120`,
    # `kernel_flash.py:182-184`). At `num_splits == 1` this degenerates to the stage-2
    # 2-D grid exactly.
    for tile_s, tile_t, tile_g in hl.tile(
        [num_splits, grid_extent, num_kv_heads], block_size=[1, 1, 1]
    ):
        target_idx = tile_t.begin
        split_idx = tile_s.begin
        # Binary search over the cumulative query-length prefix in q-block units, ported
        # from `find_seq_idx` / `resolve_seq_and_query_len`
        # (`paged_attention/_helpers/triton_attention_helpers.py:45-105`). int64
        # throughout: `(left + right) // 2` promotes, and a branch that redefines an
        # int32 loop variable as int64 is a Helion codegen assert. Seeded off
        # `query_start_loc[0]` so both are device-scoped -- a host int makes `right =
        # mid` a CannotModifyHostVariableOnDevice.
        left = (query_start_loc[0] * 0).to(torch.int64)
        right = (query_start_loc[0] * 0 + num_seqs).to(torch.int64)
        while left < right:
            mid = (left + right) // 2
            # `// q_block + mid` must be the SAME transform as `q_block_start` below, or
            # the recovered (seq, local) pair is inconsistent with the host bound.
            if query_start_loc[mid] // q_block + mid <= target_idx:
                left = mid + 1
            else:
                right = mid
        # `left - 1`, NOT `left` (helpers `:105`). Always >= 0: `qsl[0] // q_block + 0
        # == 0` satisfies the `<= target_idx` test for every `target_idx >= 0`, so `left
        # >= 1`.
        seq_idx = left - 1
        q_block_start = query_start_loc[seq_idx] // q_block + seq_idx
        # The flat path's `tile_q.begin`: in-sequence first query row of this tile.
        q_offset = (target_idx - q_block_start) * q_block
        # `block_table[seq_idx, p_idx]` will NOT lower when `seq_idx` comes out of the
        # `while` (InductorLoweringError "list index out of range" -- the 3-D path's
        # `s_idx.begin` is a tile index, this is a device value). A 1-D gather off the
        # flattened table lowers fine, and `max_pages` is already the constexpr row
        # stride.
        bt_base = seq_idx * max_pages

        seq_len = seqused_k[seq_idx]
        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        # `q_end - q_start` is the WHOLE sequence's query length, so `context_len` is
        # per-sequence and independent of which q-tile this is -- it carries over from
        # the 3-D path unchanged (INVARIANT 7).
        context_len = torch.clamp(seq_len - (q_end - q_start), min=0)

        # The over-provision guard the reference's own docstring requires ("callers must
        # still early-return when `q_block_local_idx * BLOCK_Q >=
        # cur_batch_query_len`"). Identical predicate to the 3-D path, with `q_offset`
        # for `tile_q.begin`.
        if q_start + q_offset < q_end:
            block_m = q_block * num_groups
            # M rows are (query position, q head) pairs. Load 3-D then flatten: a
            # pre-flattened row index paired with a head index makes `qk` rank-2 in M.
            q_idx = q_start + q_offset + hl.arange(q_block)
            q_head = tile_g.begin * num_groups + hl.arange(num_groups)
            kv_head = tile_g.begin
            q_load_mask = q_idx[:, None, None] < q_end
            q_i = hl.load(
                query, [q_idx, q_head, hl.arange(head_dim)], extra_mask=q_load_mask
            ).flatten(start_dim=0, end_dim=1)
            m_pos = hl.arange(q_block)[:, None].expand(q_block, num_groups).flatten()
            # Kept 1-D as well: `q_valid[:, 0]` is a narrowing kernel-tensor subscript,
            # which the triton backend rejects (`_compiler/backend.py:447`).
            m_valid = q_start + q_offset + m_pos < q_end
            q_valid = m_valid[:, None]

            m_i = hl.full([block_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([block_m], dtype=torch.float32)
            acc = hl.zeros([block_m, head_dim], dtype=torch.float32)

            # `q_block`, not `block_m`: only `q_block` M rows are distinct query
            # positions, and FA3 divides identically under PackGQA
            # (`hopper/block.h:27`).
            n_tok_max = torch.minimum(seq_len, context_len + q_offset + q_block + 1)
            n_page_max = (n_tok_max + page_size - 1) // page_size

            # Split this q-tile's OWN causal range `[0, n_page_max)`, not the global
            # `[0, klen)` -- ported from `kernel_flash.py:216-239`, which is FA3's
            # `hopper/block.h:47-56`. This is the one place the DECODE split
            # (`kernel_tune.py:126-129`, which cuts the row's whole `own_pages`) does
            # not port, and it is what load-balances the split under a causal mask: with
            # `ctx >> qlen` every q-tile scans nearly the same length so either range is
            # balanced, but with `ctx == 0` q-tile i needs only `i*q_block` tokens, and
            # cutting the GLOBAL range would give split 0 all the work and the last
            # split none. Ceiling division, so `num_splits * pages_per_split >=
            # n_page_max` and the tail is never dropped.
            pages_per_split = (n_page_max + num_splits - 1) // num_splits
            # BOTH ends clamped to `n_page_max` (INVARIANT 3). Clamping only `split_hi`
            # leaves a surplus split with `split_lo > split_hi`, and a REVERSED
            # `hl.tile` range is not a no-op the way an empty one is -- it silently
            # scans KV and the split contributes garbage instead of `m=-inf, l=0`.
            # Reachable whenever `num_splits > n_page_max`, i.e. on any q-tile whose
            # causal range is shorter than the split count: exactly the early tiles of a
            # fresh prompt. Measured as a silent wrong answer on 18 of 80 (shape, split)
            # pairs (`kernel_flash.py:229-237`).
            split_lo = torch.minimum(pages_per_split * split_idx, n_page_max)
            split_hi = torch.minimum(split_lo + pages_per_split, n_page_max)

            # Where the diagonal starts; the tile's FIRST query row, floor-divided,
            # since the interior must be safe for every row. Clamped into THIS split's
            # share: a split entirely below the diagonal is all interior, one entirely
            # above is all diagonal, and the one straddling it runs both loops.
            n_page_split = torch.clamp((context_len + q_offset) // page_size, min=0)
            n_page_split = torch.clamp(n_page_split, min=split_lo)
            n_page_split = torch.minimum(n_page_split, split_hi)
            # Round DOWN to a whole `block_p` RELATIVE TO `split_lo`, not to absolute 0
            # (INVARIANT 2): `hl.tile`'s tail tile is partial but `tile_p.block_size` in
            # the gather reshape is the FULL block size, so the overhang would load real
            # KV past the causal range. The offset matters -- the interior loop now
            # starts at `split_lo`, so it is `n_page_split - split_lo` that must be a
            # whole number of `block_p` tiles.
            n_page_split = split_lo + ((n_page_split - split_lo) // block_p) * block_p

            # Interior: every token here is attended by every M row and is in bounds, so
            # no mask and no `-inf` guard (nothing masks `qk`, so `m_i` is finite from
            # step 1).
            for tile_p in hl.tile(split_lo, n_page_split, block_size=block_p):
                p_idx = torch.clamp(
                    tile_p.begin + hl.arange(tile_p.block_size), max=max_block_idx
                )
                physical = hl.load(block_table_flat, [bt_base + p_idx])
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

            # Diagonal: the masked tail of THIS split. A qlen=1 row is all diagonal
            # (`n_page_split` rounds back to `split_lo` whenever this split's share is
            # under one `block_p` of pages), and with a split a (q-tile, split) pair can
            # be fully masked or entirely empty -- which is why the m_safe guard below
            # is not optional here.
            for tile_p in hl.tile(n_page_split, split_hi, block_size=block_p):
                p_idx = torch.clamp(
                    tile_p.begin + hl.arange(tile_p.block_size), max=max_block_idx
                )
                physical = hl.load(block_table_flat, [bt_base + p_idx])
                n = tile_p.block_size * page_size
                k = key_cache[physical, :, kv_head, :].reshape(n, head_dim)
                v = value_cache[physical, :, kv_head, :].reshape(n, head_dim)
                qk = hl.dot(q_i, k.T, out_dtype=torch.float32) * scale
                bcast_n = (
                    tile_p.begin * page_size + hl.arange(tile_p.block_size * page_size)
                )[None, :]
                # Bottom-right causal: row i sees token j iff `j <= context_len + i`.
                # The `< split_hi * page_size` term keeps splits DISJOINT and is
                # load-bearing at every count > 1 (`kernel_flash.py:277-280`):
                # `hl.tile`'s tail tile is partial but the gather reshape above uses the
                # FULL `tile_p.block_size`, so this split's last tile overhangs into the
                # next split's pages -- and those tokens would then be counted by BOTH
                # splits and double-weighted by the merge. Measured: without it, split=1
                # is exact (4.9e-04) while every count > 1 fails on all three
                # `check_split_counts` shapes, maxdiff growing with the count (1.3e-01
                # at 2 -> 5.5e-01 at 8) and saturating once every split is one tile
                # wide. The INTERIOR loop needs no such term: it ends at `n_page_split`,
                # rounded down to a whole `block_p` relative to `split_lo`, so it cannot
                # overhang.
                causal = bcast_n <= context_len + (q_offset + m_pos)[:, None]
                keep = (
                    causal
                    & (bcast_n < seq_len)
                    & (bcast_n < split_hi * page_size)
                    & q_valid
                )
                qk = torch.where(keep, qk * _LOG2_E, -float("inf"))
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                # `-inf - -inf` is NaN at both sites, and one NaN row fails EVERY
                # autotune config (`equal_nan=False`). Reachable on a row past `q_end`
                # AND, since stage 3, on a split whose every token is masked -- the case
                # that does not exist in decode (`kernel_flash.py:288-290`).
                m_safe = torch.where(m_ij == -float("inf"), 0.0, m_ij)
                p = torch.exp2(qk - m_safe[:, None])
                alpha = torch.exp2(
                    torch.where(m_i == -float("inf"), m_safe, m_i) - m_safe
                )
                m_i = m_ij
                l_i = l_i * alpha + torch.sum(p, -1)
                acc = hl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])

            # NO epilogue here. The `acc / where(l_i == 0, 1.0, l_i)` division is PHASE
            # 2's, per `_kernel_combine`'s docstring, and it must stay there:
            # `partial_l` is the online softmax denominator, so dividing here as well
            # would make `_combine_ref` -- and hence `_check_partials_close` and stage
            # 3's real split -- divide twice (measured: 0.200 maxdiff against FA3
            # through the merge, while the pre-divided `partial_out[0]` alone matched at
            # 7.4e-05). Un-normalized partials are also what `kernel_flash.py:315-322`
            # writes (the store is un-normalized; both merge paths apply the division,
            # `:49` and `:362`). Flat `[splits, token*head]` row index, mirroring
            # `kernel_flash.py:313`: storing a `[block_m]` value into a 3-D partial
            # needs a reshape of a `block_m`-shaped tensor, which trips the `_SHAPE_DIM
            # = block_m` codegen bug.
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


# ---------------------------------------------------------------------------
# Phase 2 -- online-softmax merge over the split axis
# ---------------------------------------------------------------------------
# Grid-form INDEPENDENT (it reads only the partials, never the grid extent), yet
# DUPLICATED here and in `kernel_grid.py` rather than shared from one file. Helion names
# an AOT artifact after the source-file stem and merges every kernel in that file into
# it (`heuristic_generator.py:1003-1005`), so a copy in each phase-1 file gives each
# form its OWN combine heuristic, tuned alongside its own phase 1, in
# `_helion_aot_kernel_flat_*`. One shared file instead put both in
# `_helion_aot_kernel_combine_*`, which the second AOT run overwrote. The price is a
# second body that must be kept identical to `kernel_grid.py`'s -- the same rule the
# phase-1 tile body already carries. Any INVARIANT-bearing change goes in BOTH.


def _kernel_combine(
    out: Tensor,  # [num_tokens, num_q_heads, head_dim] -- MUTATED (vLLM's buffer)
    partial_out: Tensor,  # [num_splits, num_rows, head_dim] fp32, num_rows = token*head
    partial_l: Tensor,  # [num_splits, num_rows] fp32
    partial_m: Tensor,  # same
    num_live_rows: int,  # bound the grid; see below
) -> None:
    """
    Online-softmax merge over the split axis, writing `out` in place.

    Flat grid over ROWS, one cell per output row, no inner tile -- `merge_tune`'s
    geometry (`kernel_tune.py:202`), which won all 15 points of an nt 2..32 x splits
    4..16 sweep at 1.55-2.55us against 15.2us and 9.9us for the two tiled forms
    (`kernel_occ.py:207-224`). The live tile is only `[num_splits, head_dim]` fp32, so
    the reduction fits in registers. Hence no `block_sizes` in its config; `num_warps`
    is the only knob.

    **The row axis is FLAT, unlike `merge_tune`'s.** Phase 1 allocates
    `[splits, num_tokens*num_q_heads, head_dim]` (`kernel_grid.py:136-138`) rather than
    a 4-D `[splits, tokens, heads, dim]`, because storing a `[block_m]`-shaped value
    into a 3-D partial needs a reshape of a `block_m`-shaped tensor and that trips the
    `_SHAPE_DIM = block_m` codegen bug (`kernel_grid.py:249-251`). So this grid is 1-D
    over rows, and `(token, head)` is recovered by divmod for the store.

    That also satisfies INVARIANT 6 by construction rather than by care: `merge_tune`
    needs the 1-WIDE head slice `out[t, h:h+1, :]` because a scalar mid coordinate makes
    Helion fold the head offset into the descriptor base and corrupt the store
    (`kernel_tune.py:214-217`). Here the head offset is part of the flat row index and
    `out` is addressed through a flat view, so no mid coordinate exists to fold. Do NOT
    "simplify" this into a 2-D `hl.grid([num_tokens, num_q_heads])` store on the 3-D
    `out` -- that reintroduces exactly the scalar-mid form.

    `num_live_rows` bounds the GRID, and is the one genuinely new constraint in this
    design (plan Sec 4, stage 5). `kernel_tune.py:105-118` gets away with `torch.empty`
    partials only because its grid writes EVERY slot the merge reads; the -inf still
    matters inside the kernel (an empty split keeps `m_i == -inf`, neutralized via
    `alpha == 0`) but need not be prefilled from the host, which saved a third
    FillFunctor launch worth 1.63x at nseq=1. A builder-owned buffer sized to the
    capture-time maximum breaks that precondition -- a smaller live batch leaves the
    tail untouched and stale -- so the reduction must be bounded rather than filled.
    Bounding is cheaper than filling. Contrast `kernel_flash.py:130-137`, where the fill
    IS load-bearing (NaN*0 == NaN).

    It bounds the ROW extent, not the split extent. The split axis is `hl.specialize`d
    to a compile-time constant for the register reduction, so it cannot be a runtime
    bound; unused splits are neutralized by `alpha == 0` as in both predecessors. Rows
    are what a short batch leaves stale.

    Epilogue: `acc / where(l == 0, 1.0, l)` -- padded slots (seq_len == 0) must read
    exactly zero, not NaN (INVARIANT 5). Phase 1 stores UN-NORMALIZED `acc`/`l_i`/`m_i`
    (mirroring `kernel_flash.py:315-322`), so this division happens here and exactly
    once; normalizing in phase 1 as well measured 0.200 maxdiff through the merge.
    """
    # Constant extents for the register reduction: the split axis and head_dim. The
    # split extent is config-dependent upstream, so this recompiles per tuned split
    # count -- correct, since it is separately tuned. Mirrors `kernel_tune.py:196-199`.
    hl.specialize(partial_out.size(0))
    head_dim = hl.specialize(partial_out.size(2))

    # Flat view of vLLM's buffer. The store target is `[num_rows, head_dim]`, so the
    # head offset rides in the row index and never becomes a scalar mid coordinate (see
    # docstring).
    out_flat = out.view(-1, head_dim)

    for row in hl.grid(num_live_rows):
        m = partial_m[:, row]  # [num_splits]
        l = partial_l[:, row]
        o = partial_out[:, row, :]  # [num_splits, head_dim]

        m_final = torch.amax(m, dim=0)
        # An all-empty row (padded slot, or every split past this row's pages) is -inf
        # throughout, and `-inf - -inf` is NaN. It has no valid keys, so any finite
        # substitute works: every alpha becomes 1 and, with l == 0, l_final stays 0.
        m_safe = torch.where(m_final == float("-inf"), 0.0, m_final)
        alpha = torch.exp2(m - m_safe)
        l_final = torch.sum(l * alpha, dim=0)
        o_final = torch.sum(o * alpha[:, None], dim=0)
        # Guard 0/0 for an all-empty row: INVARIANT 5, vLLM reads padded slots back.
        o_final = o_final / torch.where(l_final == 0, 1.0, l_final)
        out_flat[row, :] = o_final.to(out_flat.dtype)


# No `autotune_baseline_fn` here, matching the decode merge
# (`paged_attention/__init__.py:298`, which passes config + static_shapes only).
# INVARIANT 9's reason does not reach phase 2: it exists because phase 1's DEFAULT
# config takes both registered block-size maxes and cannot launch, and this kernel
# registers no block size at all -- `num_warps` is its only knob, so every candidate is
# launchable and the tuner's default-config baseline compiles fine.
