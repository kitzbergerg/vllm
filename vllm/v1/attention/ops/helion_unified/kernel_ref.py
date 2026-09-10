"""
The UPSTREAM reference kernel -- what the unified kernel is a rewrite of.

`kernel` is a **verbatim copy** of `helion-aot-research/kernels/kernel.py` except for
one line: `num_pages_at_once`'s registered max is 16, not 32, because 32 does not launch
(`out of resource: shared memory, Required: 263168, Hardware limit: 232448`,
`experiment/details/kernel-ref.md:30`). Verified: the body here is byte-identical to
`paged_attention/kernel_ref.py`'s and `paged_attention_prefill/kernel_ref.py`'s, and
diffs against the upstream source in that one line alone. Keep all three identical -- a
fork stops being a comparable baseline. `baseline_flash_attn` and the adapter
(`_next_power_of_2`, `_mix_buckets`, `kernel_ref_factory`) are taken from
`paged_attention_prefill/kernel_ref.py` unchanged; only this docstring is local. Copied
rather than imported so the package stands alone: importing another package's submodule
executes ITS `__init__`, which evaluates that package's shape pools at module scope and
dies on this package's `TUNING_SHAPES` values.

Why it is here: it is the baseline the unified kernel must beat on its OWN pools, and
the one that shows what the rewrite bought -- same DSL and same constraints, so a gap to
it is attributable to the design rather than to Helion. `CHECK_REF` stays `flash_attn`,
the external competitor; this is an internal reference.

What it does NOT have, which is the whole point of the comparison:
  - No split-KV, no partials, no combine -- it is single-phase and takes `out=`. Stage 3's
    `num_splits` axis is the thing this kernel cannot express.
  - Grid `[num_seqs, num_query_heads, max_query_len]`: the batch MAXIMUM query extent, i.e.
    exactly the 87%-empty-tile grid `kernel_flat.py` replaces (plan Sec 1).
  - Natural-`exp` softmax rather than base-2, and no interior/diagonal loop split, so every KV
    tile pays the causal mask (the unified kernel's interior loop pays none, +13.9% geomean).
  - Specialization from four constexpr buckets (`q_block_padded_size`, `batch_size_padded`,
    `decode_frac_bucket`, `prefill_skew_bucket`) where the unified kernel uses `max_pages` +
    `rows_bucket`.
So a cell where this wins is a cell where none of those changes paid.

Config constraint: large tiles overflow smem. The KV tile is dense (`block_p *
page_size` rows live for both K and V) and `S` is a full fp32 `block_m x block_n`, so at
page_size 16 / head_dim 128 / 8 queries-per-kv the default `block_sizes=[32, 16]` needs
557,056 B against H100's 232,448 B and **fails to launch**. That default is
`config_spec.py`'s heuristic clamped by the registered maxes, and the autotuner builds
its accuracy baseline from it (`benchmark_provider.py:583`), so on prefill shapes
`InvalidConfig` killed the search before it began. Fixed with
`autotune_baseline_fn=baseline_flash_attn` rather than by lowering the maxes: a lower
max shrinks the search space on every shape, including the ones where a large tile does
fit. See `_REF_CONFIG` in `__init__.py`.
"""

import helion.language as hl
import torch


def kernel(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    t_query_start_lens,  # [num_seqs+1]
    max_query_len,  # must be on CPU
    num_seqs,  # must be on cpu
    # to trigger re-compilation (and re-tuning) for decodes only
    q_block_padded_size: hl.constexpr,
    # to trigger re-compilation (and re-tuning) for small and large batches
    batch_size_padded: hl.constexpr,
    # decode/prefill token-ratio bucket: triggers re-tuning for decode-heavy
    # vs prefill-heavy batches (0 = pure prefill .. MIX_NUM_BUCKETS = pure decode)
    decode_frac_bucket: hl.constexpr,
    # prefill query-length skew bucket: triggers re-tuning when the prefill
    # length distribution changes (uniform vs mixed short/long prompts)
    prefill_skew_bucket: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    q_block_size = hl.register_block_size(1, q_block_padded_size)
    num_pages_at_once = hl.register_block_size(1, 16)

    for seq_tile, tile_m, tile_q in hl.tile(
        [num_seqs, num_query_heads, max_query_len],
        block_size=[1, num_queries_per_kv, q_block_size],
    ):
        seq_idx = seq_tile.begin  # is scalar
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        if query_start + tile_q.begin < query_end:
            block_m_size = num_queries_per_kv * q_block_size
            kv_head_idx = tile_m.begin // num_queries_per_kv

            # cannot use tile_q.index directly, since tile_q.index is dynamic
            adjusted_tile_q_index = query_start + tile_q.begin + hl.arange(q_block_size)
            query_head_offset = tile_m.begin + hl.arange(num_queries_per_kv)
            q_load_mask = adjusted_tile_q_index[:, None, None] < query_end
            # (tile_q, tile_m, HEAD_SIZE)
            q = hl.load(
                t_query,
                [adjusted_tile_q_index, query_head_offset, hl.arange(head_size)],
                extra_mask=q_load_mask,
            )
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)

            M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)

            # adjust for causal mask
            max_seq_prefix_len = context_len + tile_q.begin + block_m_size + 1
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)
            num_blocks = torch.ceil(max_seq_prefix_len / page_size)
            for tile_n in hl.tile(num_blocks, block_size=num_pages_at_once):
                block_n_size = num_pages_at_once * page_size
                # explicit load due to wrong if tile_n is partial
                blk_idxs = hl.load(
                    t_block_tables,
                    [seq_idx, tile_n.begin + hl.arange(num_pages_at_once)],
                )
                blk_idxs = blk_idxs.view([num_pages_at_once]).to(torch.int64)

                # (tile_n, PAGE_SIZE, 1, HEAD_SIZE)
                k_load = t_key_cache[blk_idxs, :, kv_head_idx, :]
                k_load = k_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                k = hl.zeros([block_n_size, head_size], dtype=k_load.dtype)
                absolute_tile_token_offsets = tile_n.begin * page_size + hl.arange(
                    block_n_size
                )
                k = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, k_load, k
                )
                # (HEAD_SIZE, tile_n)
                k = k.transpose(0, 1)

                # (tile_n, PAGE_SIZE, HEAD_SIZE)
                v_load = t_value_cache[blk_idxs, :, kv_head_idx, :]
                v_load = v_load.flatten(start_dim=0, end_dim=1)
                # (tile_n, HEAD_SIZE)
                v = hl.zeros([block_n_size, head_size], dtype=v_load.dtype)
                v = torch.where(
                    absolute_tile_token_offsets[:, None] < seq_len, v_load, v
                )

                # (tile_m, tile_n)
                # use S with float32 as acc to enforce higher precision
                #  for the additions of the dot operation?
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = hl.dot(q, k, out_dtype=torch.float32, acc=S) * scale
                block_m_query_mask = tile_q.begin + hl.arange(
                    q_block_size
                ).repeat_interleave(num_queries_per_kv, dim=0)
                # construct 2d causal mask
                causal_mask = (
                    absolute_tile_token_offsets[None, :]
                    < context_len + block_m_query_mask[:, None] + 1
                )
                S = torch.where(causal_mask, S, float("-inf"))

                # (tile_m)
                M_j = torch.maximum(M, torch.amax(S, 1))
                # (tile_m, tile_n)
                P = torch.exp(S - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp(M - M_j)
                # (tile_m, HEAD_SIZE)
                acc = acc * alpha[:, None]
                L = (L * alpha) + L_j
                M = M_j

                # (tile_m, HEAD_SIZE)
                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # epilogue
            acc = acc / L[:, None]
            hl.store(
                t_output,
                [adjusted_tile_q_index, tile_m.index, hl.arange(head_size)],
                acc.view([q_block_size, num_queries_per_kv, head_size]),
                extra_mask=q_load_mask,
            )


def baseline_flash_attn(
    t_output,
    t_query,
    t_key_cache,
    t_value_cache,
    t_block_tables,
    t_seq_lens,
    scale,
    t_query_start_lens,
    max_query_len,
    num_seqs,
    q_block_padded_size=None,
    batch_size_padded=None,
    decode_frac_bucket=None,
    prefill_skew_bucket=None,
):
    """
    `autotune_baseline_fn`: flash_attn in place of the default config.

    The autotuner's accuracy baseline ignores any pinned `config=` and compiles
    `config_spec.default_config()` (`autotuner/benchmark_provider.py:583`). Here that
    default is `block_sizes=[32, 16]`, whose fp32 `S` tile alone is `(32*8) x (16*16) x
    4 = 262144` B; the whole config needs 557056 B against H100's 232448, so
    `InvalidConfig` killed the search before it started. Supplying this baseline
    bypasses that compile entirely, which keeps both registered maxes intact -- lowering
    them would shrink the search space and cost performance on the shapes where a large
    tile does fit.

    Args arrive positionally via `Kernel.normalize_args` (`runtime/kernel.py:966`, a
    `signature.bind` + `apply_defaults`), so this must mirror the kernel signature
    exactly. Must WRITE into `t_output`: the harness diffs the cloned args to find
    mutated tensors and compares candidates against those
    (`benchmark_provider.py:608-632`), so a return value alone would leave nothing to
    check.
    """
    from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

    t_output.copy_(
        flash_attn_with_kvcache(
            t_query,
            t_key_cache,
            t_value_cache,
            page_table=t_block_tables,
            cache_seqlens=t_seq_lens,
            cu_seqlens_q=t_query_start_lens,
            max_seqlen_q=max_query_len,
            softmax_scale=scale,
            causal=True,
        )
    )


# --- adapter: this repo's call convention -> the signature above ---------------
# The reference reaches these via vLLM helpers (`_vllm_compat.next_power_of_2`,
# `helion_unified_attention._compute_mix_buckets`); reproduced here so the port
# has no cross-repo import.

MIX_NUM_BUCKETS = 8


def _next_power_of_2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def _mix_buckets(
    num_total_query_tokens: int,
    num_decode_tokens: int,
    num_seqs: int,
    max_query_len: int,
) -> tuple[int, int]:
    if num_total_query_tokens <= 0:
        return 0, 0
    decode_frac_bucket = round(
        MIX_NUM_BUCKETS * num_decode_tokens / num_total_query_tokens
    )
    num_prefill_seqs = num_seqs - num_decode_tokens
    prefill_tokens = num_total_query_tokens - num_decode_tokens
    if num_prefill_seqs > 0 and max_query_len > 0:
        skew = round(
            MIX_NUM_BUCKETS * (prefill_tokens / num_prefill_seqs) / max_query_len
        )
        prefill_skew_bucket = max(0, min(MIX_NUM_BUCKETS, skew))
    else:
        prefill_skew_bucket = 0
    return decode_frac_bucket, prefill_skew_bucket


def kernel_ref_factory(kernel_fn):
    """
    Wrap the ported kernel in this repo's `flash_attn_varlen_func`-style signature (see
    `kernel.kernel_factory`).
    """

    def paged_attention(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        max_seqlen_q: int = 1,
        max_seqlen_k: int = 0,
        softmax_scale: float | None = None,
        causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        assert causal, "Only causal attention is supported"
        assert query_start_loc is not None, "this kernel needs cu_seqlens_q"
        num_tokens, _num_q_heads, head_dim = query.shape
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        num_seqs = block_table.size(0)
        out = torch.empty_like(query)

        # The reference's own specialization padding (`prepare_helion_inner`,
        # eager branch): query-length bucket, batch bucket, and the two mix
        # buckets. `torch.version.cuda` is truthy here, so the coarse batch
        # bucket applies.
        q_pad = (
            _next_power_of_2(max_seqlen_q)
            if _next_power_of_2(max_seqlen_q) in [1, 8, 16, 32, 64]
            else 128
        )
        batch_pad = min(256, _next_power_of_2(num_seqs))
        num_decode_tokens = num_seqs if max_seqlen_q == 1 else 0
        decode_frac_bucket, prefill_skew_bucket = _mix_buckets(
            num_tokens, num_decode_tokens, num_seqs, max_seqlen_q
        )

        kernel_fn(
            t_output=out,
            t_query=query,
            t_key_cache=key_cache,
            t_value_cache=value_cache,
            t_block_tables=block_table,
            t_seq_lens=seqused_k,
            scale=softmax_scale,
            t_query_start_lens=query_start_loc,
            max_query_len=max_seqlen_q,
            num_seqs=num_seqs,
            q_block_padded_size=q_pad,
            batch_size_padded=batch_pad,
            decode_frac_bucket=decode_frac_bucket,
            prefill_skew_bucket=prefill_skew_bucket,
        )
        return out

    return paged_attention
