# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Helion paged-attention backend.

ONE backend serving vLLM's whole workload: decode rows, prefill rows and mixed batches
all go through a single kernel launch, with no dispatch on query length. All three
kernel forms are reachable through it, selected by `VLLM_VLLM_HELION_UNIFIED_IMPL` in
{flat, grid, ref}; `flat` is the default.

Default rationale (132 shape cells, 7 pools, H100/Qwen3, geomean FA3/impl): flat 0.631 >
grid 0.595 > ref 0.359. `ref` is the upstream kernel kept as an internal baseline only
-- it is also wrong on 10 of those cells, so it must never be the default. `grid` wins
pure decode (0.857 vs 0.734) and qlen>512 prefill; `flat` wins the mid-range and mixed
batches. No shape predicate separates them (win rates ~50% in the middle qlen buckets),
and any shape-dependent branch must be a step function on cudagraph capture sizes to
stay capturable -- so the choice is a static env knob, not a runtime dispatch.

Usage:
    vllm serve <model> --attention-backend HELION_UNIFIED_ATTN
    VLLM_VLLM_HELION_UNIFIED_IMPL=grid vllm serve <model> \
        --attention-backend HELION_UNIFIED_ATTN
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.helion_unified import (
    DEFAULT_IMPL,
    IMPL_ENV,
    IMPLS,
    get_impl,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

logger = init_logger(__name__)


def _selected_impl() -> str:
    """The implementation name from `VLLM_HELION_UNIFIED_IMPL`, validated."""
    name = (os.environ.get(IMPL_ENV) or DEFAULT_IMPL).strip().lower()
    if name not in IMPLS:
        raise ValueError(f"{IMPL_ENV}={name!r} not in {IMPLS}")
    return name


def _load_kernel(name: str | None = None):
    """Resolve the selected implementation's vLLM-shaped callable."""
    return get_impl(name)


@dataclass
class HelionAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class HelionAttentionMetadataBuilder(AttentionMetadataBuilder[HelionAttentionMetadata]):
    """Defect 6 -- `_cudagraph_support` is a ClassVar on the BUILDER
    (`vllm/v1/attention/backend.py:584-600`), not on the backend.

    ALWAYS, i.e. plan §4's stage 3, reached directly. It is documented as "supports
    mixed-prefill-decode" (`backend.py:567-582`), which one unified kernel gives by
    construction -- there is no `max_query_len > 1` dispatch to make a second code path.

    Measured, not assumed (`_cudagraph_stage5.py`, all three impls): capture and replay
    are BIT-IDENTICAL to eager (maxdiff exactly 0) at all 9 capture sizes 1..256, on the
    four mixed shapes including the plan's R1 case (255 decodes + one 241-token
    prefill), and on a stale tail. Partials stay allocated IN-BODY: the plan's premise
    that in-kernel allocation cannot be captured was already retired by
    `_cudagraph_probe.py` and is now confirmed on the real kernel, so the builder-owned
    fallback (`triton_attn.py:156-175`) is NOT needed. PyTorch's caching allocator
    serves the capture from a graph-private pool, and Helion constant-folds the tunable,
    so the split count is a compile-time literal per config.

    `rows_bucket` (`min(256, 1 << (rows-1).bit_length())`) is the only shape-dependent
    fork, and it is already a step function on powers of two, which the 1,2,4,..,256
    capture ladder matches exactly -- the property `triton_attn.py:143-152` snaps
    `seq_threshold_3D` to get.

    Resident cost: ~4.1 GiB across 9 simultaneously-captured decode graphs at nseq<=256,
    Qwen3-8B geometry -- the number vLLM's own comment flags (`flash_attn.py:543-553`).
    Worth re-measuring per deployment before enabling the full ladder.

    `reorder_batch_threshold` is left None, as `flash_attn.py` and `triton_attn.py` both
    do (`backend.py:591`): with one kernel there is nothing to reorder for (plan §1).
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.impl_name = _selected_impl()

        logger.info_once(
            f"Using HelionAttention version {self.impl_name}",
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HelionAttentionMetadata:
        """Pure repackaging: no D2H sync, no host-side data-dependent branch.

        The grid extent is deliberately NOT computed here. `unified_factory` derives it
        inside the call (`kernel_common.py:317-347`) -- `max(max_seqlen_q, 2)` for grid,
        and for flat the sync-free bound `query.size(0) // _Q_BLOCK_MIN + rows` using
        the REGISTERED MINIMUM (`floor(N/b)` is non-increasing in b, so a larger actual
        `q_block` only widens the slack and can never under-provision). It must be
        computed AFTER the factory's 0/1-specialization padding, so both terms describe
        the tensors actually passed; recomputing it here from the unpadded metadata
        would drift from that silently, and the grid form's extent means a different
        thing than the flat form's (INVARIANT 12).

        `max_query_len` is vLLM's `max(query_lens)` -- NOT `capture_max_query_len`,
        which inflated launched q-tiles 2.97x, invisibly (`q_load_mask` discards the
        surplus).
        """
        del common_prefix_len, fast_build
        return HelionAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            # "the number of computed tokens for each request" (`backend.py:381`). The
            # kernel reads exact per-row lengths from this, so do NOT substitute
            # `max_seq_len`, which is documented as possibly an upper bound
            # (`backend.py:392`).
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class HelionAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        # Each of these is a capability the kernel does not have, and the backend's
        # `supports_*` classmethods already decline them. Assert rather than ignore:
        # silently dropping a soft cap or a sliding window is a wrong-answer bug, not a
        # slow path.
        if alibi_slopes is not None:
            raise NotImplementedError("alibi is unimplemented")
        if sliding_window is not None:
            raise NotImplementedError("sliding window is unimplemented")
        if logits_soft_cap is not None:
            raise NotImplementedError("logits soft cap is unimplemented")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"attn_type={attn_type} is unimplemented")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.impl_name = _selected_impl()
        self.kernel = _load_kernel(self.impl_name)
        # `kernel_ref` is single-phase and allocates its own output
        # (`kernel_ref.py:298,331`) -- it has no `out=` at all, so it cannot write
        # vLLM's buffer and must pay the copy defect 5 removes from the other two. That
        # is a property of the upstream kernel this package rewrites, not something to
        # fix here; it is one more reason `ref` is the internal baseline and never the
        # default.
        self.writes_out = self.impl_name != "ref"

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """`forward_includes_kv_cache_update = False` means vLLM issues the cache
        write as a
        SEPARATE op that calls back into here (`attention.py:711-722`) -- it does not
        mean vLLM writes the cache for us. The base implementation (`backend.py:1052`)
        is MLA-only, so a non-MLA impl must provide this or the assert at
        `attention.py:713` fires.

        Mirrors `triton_attn.py:783-795`, unquantized path only: quantized KV is already
        rejected in `__init__`, and the same `transpose(1, 2).split` view `forward` uses
        is zero-copy because we declare LBNHC.
        """
        if kv_cache.numel() == 0:
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HelionAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Defect 7 -- NO `max_query_len > 1` dispatch. That is the point of one
        kernel: the
        old dispatch ran all 255 decode rows of a mixed batch at prefill's M
        utilization.

        Defect 5 -- no `output[:n] = out` copy. `out=` is passed straight through and
        phase 2 writes vLLM's buffer in place (`kernel_common.py:381-385`).

        vLLM's CPU-overhead warning applies here: `view` and `slice` are surprisingly
        slow even when they invoke no GPU ops (`triton_attn.py:577-584`). Keep this body
        minimal.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is unimplemented")
        if attn_metadata is None:
            return output.fill_(0)  # profiling run

        # `forward_includes_kv_cache_update = False`, so vLLM has already written
        # key/value into the cache via `unified_kv_cache_update`
        # (`model_executor/layers/attention/attention.py:531-541`); key/value are unused
        # here. The cache arrives in LOGICAL (B, H, N, 2*hs) order regardless of layout
        # -- the layout only permutes the STRIDES (`kv_cache_interface.py:267-303`).
        # Declaring LBNHC is what makes this transpose a contiguous zero-copy view of
        # exactly the kernel's native [num_blocks, page_size, num_kv_heads, head_dim]
        # (defect 2). Same shape as `flex_attention.py:1361`.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        n = attn_metadata.num_actual_tokens
        args = (
            query[:n],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            0,
            self.scale,
            True,
        )
        if self.writes_out:
            self.kernel(*args, output[:n])
        else:
            output[:n].copy_(self.kernel(*args))
        return output


class HelionUnifiedAttentionBackend(AttentionBackend):
    """`get_name`/`get_impl_cls`/`get_builder_cls` are the only three abstract statics
    (`vllm/v1/attention/backend.py:76-89`)."""

    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        # Must be an AttentionBackendEnum MEMBER name, not a descriptive string:
        # `attention.py:423` does `AttentionBackendEnum[backend.get_name()]`.
        return AttentionBackendEnum.HELION_UNIFIED_ATTN.name

    @staticmethod
    def get_impl_cls() -> type[HelionAttentionImpl]:
        return HelionAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[HelionAttentionMetadataBuilder]:
        return HelionAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        """Defect 2 -- neither predecessor declared this, so both paid a strided
        `kv_cache.transpose(1, 2)`.

        LBNHC is `[num_blocks, block_size, num_kv_heads, head_size]`
        (`vllm/v1/kv_cache_layout.py:24`) -- the kernel's native layout, making that
        transpose contiguous. Returns a TUPLE: the signature is `tuple[KVCacheLayout,
        ...] | None` (`backend.py:350-354`). Precedent: `flex_attention.py:126-129`.
        """
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Defect 4 -- `[MultipleOf(16)]` admits page 32/64, where the Helion
        kernels lose
        ~35% and FA3 loses nothing (`prefill-shape-space.md` §3). Declare [16].

        `get_preferred_block_size` (`backend.py:150-158`) is not the lever: it only
        applies when the default is outright unsupported, so it cannot say "16 is
        fastest but 64 is legal". This is a real restriction and the honest one until
        the page-size regression is diagnosed (plan Open item); page 16 is vLLM's
        default and the traced target (`VLLM_TARGET.md`), so it costs nothing there.
        """
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == 16

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Defect 3 -- both predecessors returned True while the factory asserts causal
        (`kernel_common.py:242`), advertising a capability the kernel rejects at
        runtime."""
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # The gather reshape only factors for a pow2 head_dim (`kernel_common.py:244`),
        # which is why Phi-3's 96 is excluded.
        return head_size > 0 and head_size & (head_size - 1) == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Defect 8 -- `get_kv_cache_shape` no longer exists. The hooks are
        `compute_layer_kv_cache_shape_bytes` (`vllm/v1/kv_cache_interface.py:248-264`),
        which derives the shape from the spec, and this one (`backend.py:136-147`). The
        default (B, H, N, 2*hs) spec is already what the kernel wants, so there is
        nothing to
        customize."""
        return spec


# Defect 9 -- do NOT port the predecessors' AOT artifacts. The unified signature changed
# the specialization key (`grid_extent`, `rows_bucket`, `m_budget`, `p_budget`), and
# Helion names an artifact after the SOURCE-FILE STEM, merging every kernel in that file
# into one (`heuristic_generator.py:1003-1005`) -- which is why `grid` and `flat` are
# separate files with separate trees. Their trees cut on arg 7 (`grid_extent`), which
# means `max_seqlen_q` for grid and `num_q_tiles` for flat, so one shared tree would
# mis-select on every call. Treat any inherited artifact as invalid.
