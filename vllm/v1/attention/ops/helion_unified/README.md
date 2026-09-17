# Helion attention backend

One Helion kernel for vLLM's whole attention workload — decode rows, prefill rows and
mixed batches in a single launch, with no dispatch on query length.

## Files

| file | role |
| --- | --- |
| `kernel_common.py` | shared invariants + `unified_factory` (the vLLM-shaped callable) |
| `kernel_flat.py` | **default.** Flat per-row q-tile grid + its `_kernel_combine` |
| `kernel_grid.py` | 3-D grid sized by `max_seqlen_q` + its `_kernel_combine` |
| `_helion_aot_kernel_{flat,grid}_cuda_sm90.py` | autotuned heuristics (see below) |

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

## Autotuned heuristics

`helion.aot_kernel` is used with **no `config=`** — a pinned config would override the tree.
Helion finds the trees itself: `aot_cache.py` searches for
`_helion_aot_<source-stem>_<device>_<compute>.py` **next to the kernel's source file**, so
the files here need no registration. They cut on continuous shape features (56 leaves for
flat, 68 for grid) and select block sizes, warps and `num_splits` per call.

These are `sm90` (H100). On another compute capability Helion falls back to older
capabilities in the same family, and failing that autotunes at runtime — so a fresh tuning
run is needed for a different GPU.
