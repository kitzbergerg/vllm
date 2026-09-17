# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Helion paged-attention backend.

One backend serving decode rows, prefill rows and mixed batches through a single
kernel launch, with no dispatch on query length. Two kernel grid forms are
reachable through it, selected by `VLLM_HELION_UNIFIED_IMPL` in {flat, grid};
`flat` is the default. See `vllm/v1/attention/ops/helion_unified/__init__.py`.

Usage:
    vllm serve <model> --attention-backend HELION_UNIFIED_ATTN
    VLLM_HELION_UNIFIED_IMPL=grid vllm serve <model> \
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
    name = (os.environ.get(IMPL_ENV) or DEFAULT_IMPL).strip().lower()
    if name not in IMPLS:
        raise ValueError(f"{IMPL_ENV}={name!r} not in {IMPLS}")
    return name


@dataclass
class HelionAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class HelionAttentionMetadataBuilder(AttentionMetadataBuilder[HelionAttentionMetadata]):
    # ALWAYS: one unified kernel supports mixed prefill/decode by construction, and
    # the split-KV partials are allocated in-body, which captures cleanly from the
    # graph-private pool. `reorder_batch_threshold` stays None -- with one kernel
    # there is nothing to reorder for.
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
        logger.info_once(f"Using HelionAttention version {self.impl_name}")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HelionAttentionMetadata:
        # Pure repackaging: no D2H sync, no host-side data-dependent branch. The
        # grid extent is deliberately not computed here -- `unified_factory`
        # derives it after its 0/1-specialization padding, and it means a
        # different quantity in each grid form.
        del common_prefix_len, fast_build
        return HelionAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            # vLLM's `max(query_lens)`, not `capture_max_query_len`.
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            # Exact per-row computed-token counts; do not substitute
            # `max_seq_len`, which is documented as possibly an upper bound.
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
        # The backend's `supports_*` classmethods already decline these. Raise
        # rather than ignore: silently dropping a soft cap or a sliding window is
        # a wrong-answer bug, not a slow path.
        if alibi_slopes is not None:
            raise NotImplementedError("alibi is unimplemented")
        if sliding_window is not None:
            raise NotImplementedError("sliding window is unimplemented")
        if logits_soft_cap is not None:
            raise NotImplementedError("logits soft cap is unimplemented")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"attn_type={attn_type} is unimplemented")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                f"kv_cache_dtype={kv_cache_dtype} is unimplemented"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.impl_name = _selected_impl()
        self.kernel = get_impl(self.impl_name)

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # `forward_includes_kv_cache_update = False` means vLLM issues the cache
        # write as a separate op that calls back into here; the base
        # implementation is MLA-only, so a non-MLA impl must provide this.
        # Unquantized path only -- quantized KV is rejected in `__init__`.
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
        # No `max_query_len > 1` dispatch: that is the point of one kernel. `out=`
        # is passed straight through and phase 2 writes vLLM's buffer in place, so
        # there is no full-output copy. `view`/`slice` are surprisingly costly on
        # the CPU side even with no GPU work, so keep this body minimal.
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is unimplemented")
        if attn_metadata is None:
            return output.fill_(0)  # profiling run

        # key/value are unused: vLLM has already written them into the cache. The
        # cache arrives in logical (B, H, N, 2*hs) order and the declared LBNHC
        # layout makes this transpose a contiguous zero-copy view of exactly the
        # kernel's native [num_blocks, page_size, num_kv_heads, head_dim].
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        n = attn_metadata.num_actual_tokens
        self.kernel(
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
            output[:n],
        )
        return output


class HelionUnifiedAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # No descale path in the kernel, so no quantized KV.
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return AttentionBackendEnum.HELION_UNIFIED_ATTN.name

    @staticmethod
    def get_impl_cls() -> type[HelionAttentionImpl]:
        return HelionAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[HelionAttentionMetadataBuilder]:
        return HelionAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The AOT trees are tuned at page 16 only; page 32/64 select leaves whose
        # tiles can exceed the smem limit at replay. `get_preferred_block_size` is
        # not the lever -- it only applies when the default is unsupported.
        return [16]

    @classmethod
    def supports_non_causal(cls) -> bool:
        # The factory asserts causal; a non-causal path needs an unbounded KV loop.
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # The gather reshape only factors for a pow2 head_dim, which excludes 96.
        return head_size > 0 and head_size & (head_size - 1) == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # ENCODER_ONLY needs non-causal; ENCODER_DECODER breaks the kernel's
        # `context_len = seq_len - query_len` identity.
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        return spec
