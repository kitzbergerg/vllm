# Unified Helion paged attention

One Helion kernel for vLLM's whole attention workload — decode rows, prefill rows and
mixed batches in a single launch, with no dispatch on query length.

## Files

| file | role |
|---|---|
| `kernel_common.py` | shared invariants + `unified_factory` (the vLLM-shaped callable) |
| `kernel_flat.py` | **default.** Flat per-row q-tile grid + its `_kernel_combine` |
| `kernel_grid.py` | 3-D grid sized by `max_seqlen_q` + its `_kernel_combine` |
| `kernel_ref.py` | upstream kernel this rewrites; internal baseline, **wrong on 10 cells** |
| `_helion_aot_kernel_{flat,grid,ref}_cuda_sm90.py` | autotuned heuristics (see below) |

The backend that drives these lives at
`vllm/v1/attention/backends/helion_unified_attn.py`.

## Use

```bash
vllm serve <model> --attention-backend HELION_UNIFIED_ATTN
VLLM_HELION_UNIFIED_IMPL=grid vllm serve <model> \
    --attention-backend HELION_UNIFIED_ATTN
```

Or offline: `LLM(model=..., attention_backend="HELION_UNIFIED_ATTN")`.

Requires the `helion` extra (`helion==1.4.0`, `setup.py:1538`). The registry stores only a
dotted path, so the import is lazy: without Helion installed vLLM still starts, and only
selecting this backend raises `ImportError`.

Confirm it took, in the server log:

```
Using AttentionBackendEnum.HELION_UNIFIED_ATTN backend.
Using LBNHC KV cache layout.
```

## Autotuned heuristics

`helion.aot_kernel` is used with **no `config=`** — a pinned config would override the tree.
Helion finds the trees itself: `aot_cache.py` searches for
`_helion_aot_<source-stem>_<device>_<compute>.py` **next to the kernel's source file**, so
the files here need no registration. They cut on continuous shape features (56 leaves for
flat, 68 for grid) and select block sizes, warps and `num_splits` per call.

These are `sm90` (H100). On another compute capability Helion falls back to older
capabilities in the same family, and failing that autotunes at runtime — so a fresh tuning
run is needed for a different GPU.

Note this is an *alternative* to `vllm/kernels/helion/`, not additive: that package injects
configs by overriding Helion's `key` and `autotuner_fn` hooks, which is exactly what
`aot_kernel` uses. It also keys on discrete `CaseKey` dicts, so porting these continuous
cuts into it would be lossy.

## Which implementation

Measured over 132 shape cells (H100, Qwen3), geomean FA3/impl:

| impl | geomean | notes |
|---|---|---|
| `flat` | **0.631** | default; wins mid-range and mixed batches |
| `grid` | 0.595 | wins pure decode (0.857 vs 0.734) and qlen>512 (1.14x on 18/21) |
| `ref` | 0.359 | baseline only; also wrong on 10 cells |

Best-of-both would be 0.700, but win rates sit near 50% through the middle qlen buckets so
no clean shape predicate separates them — and a graph-capturable branch must be a step
function on capture sizes. Hence a static env knob rather than a runtime heuristic.

## Restrictions

- **Page size 16 only** (`get_supported_kernel_block_sizes() -> [16]`). Both Helion kernels
  lose ~35% at page 32/64 where FA3 loses nothing; undiagnosed, so the honest declaration is
  16 — which is vLLM's default anyway.
- **Causal only** (`supports_non_causal() -> False`), asserted in the factory.
- No sliding window, no sink, no cascade, decoder self-attention only.
- Unquantized KV only.
