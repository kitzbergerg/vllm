"""
Auto-generated heuristics for kernels: _kernel_unified, _kernel_combine
Backend: decision_tree

Provides for each kernel:
- key_<kernel>(*args): Cache key function
- autotune_<kernel>(*args): Config selection function
"""

import torch

# === Kernel: _kernel_unified ===


def key__kernel_unified(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = (
        int(args[0].shape[0])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0
        else 0
    )
    _arg0_dim1 = (
        int(args[0].shape[1])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1
        else 0
    )
    _arg0_numel = (
        int(args[0].numel())
        if len(args) > 0 and isinstance(args[0], torch.Tensor)
        else 0
    )
    _arg10_scalar = (
        args[10] if len(args) > 10 and isinstance(args[10], (int, float)) else 0
    )
    _arg1_dim0 = (
        int(args[1].shape[0])
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim > 0
        else 0
    )
    _arg1_dim1 = (
        int(args[1].shape[1])
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim > 1
        else 0
    )
    _arg1_dim2 = (
        int(args[1].shape[2])
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim > 2
        else 0
    )
    _arg1_numel = (
        int(args[1].numel())
        if len(args) > 1 and isinstance(args[1], torch.Tensor)
        else 0
    )
    _arg3_dim0 = (
        int(args[3].shape[0])
        if len(args) > 3 and isinstance(args[3], torch.Tensor) and args[3].ndim > 0
        else 0
    )
    _arg3_dim1 = (
        int(args[3].shape[1])
        if len(args) > 3 and isinstance(args[3], torch.Tensor) and args[3].ndim > 1
        else 0
    )
    _arg7_scalar = args[7] if len(args) > 7 and isinstance(args[7], (int, float)) else 0
    _arg9_scalar = args[9] if len(args) > 9 and isinstance(args[9], (int, float)) else 0
    if _arg0_numel <= 294912.0:
        if _arg10_scalar <= 4.0:
            if _arg0_numel <= 16384.0:
                if _arg1_dim0 <= 521.0:
                    if _arg9_scalar <= 32.0:
                        return 5
                    else:
                        return 0
                else:
                    return 5
            else:
                if _arg1_dim2 <= 8.0:
                    return 0
                else:
                    if _arg7_scalar <= 2.0:
                        return 0
                    else:
                        if _arg0_dim0 <= 11.0:
                            return 5
                        else:
                            return 0
        else:
            if _arg3_dim1 <= 46.0:
                if _arg0_dim0 <= 4.0:
                    if _arg9_scalar <= 32.0:
                        if _arg1_dim0 <= 5.0:
                            return 0
                        else:
                            return 5
                    else:
                        return 0
                else:
                    if _arg0_dim1 <= 16.0:
                        return 0
                    else:
                        if _arg1_dim2 <= 16.0:
                            return 9
                        else:
                            return 6
            else:
                if _arg0_numel <= 32768.0:
                    if _arg3_dim1 <= 363.0:
                        if _arg1_numel <= 168468480.0:
                            return 5
                        else:
                            return 0
                    else:
                        if _arg1_numel <= 537985024.0:
                            return 2
                        else:
                            return 0
                else:
                    if _arg0_dim1 <= 32.0:
                        if _arg9_scalar <= 32.0:
                            return 3
                        else:
                            return 6
                    else:
                        if _arg1_dim0 <= 1041.0:
                            return 3
                        else:
                            return 2
    else:
        if _arg9_scalar <= 32.0:
            if _arg7_scalar <= 32.0:
                if _arg10_scalar <= 4.0:
                    return 0
                else:
                    if _arg1_dim0 <= 8449.0:
                        if _arg0_dim0 <= 238.0:
                            return 3
                        else:
                            return 1
                    else:
                        if _arg7_scalar <= 2.0:
                            return 6
                        else:
                            return 3
            else:
                if _arg0_dim1 <= 40.0:
                    if _arg1_dim1 <= 16.0:
                        if _arg3_dim1 <= 32.0:
                            return 1
                        else:
                            return 7
                    else:
                        return 1
                else:
                    if _arg1_dim1 <= 16.0:
                        return 1
                    else:
                        return 3
        else:
            if _arg7_scalar <= 2.0:
                if _arg1_dim0 <= 257.0:
                    if _arg0_dim0 <= 94.0:
                        return 9
                    else:
                        return 0
                else:
                    if _arg3_dim1 <= 725.0:
                        if _arg1_dim1 <= 16.0:
                            return 6
                        else:
                            return 1
                    else:
                        if _arg0_dim0 <= 100.0:
                            return 0
                        else:
                            return 1
            else:
                if _arg3_dim0 <= 4.0:
                    if _arg0_dim0 <= 87.0:
                        return 3
                    else:
                        if _arg3_dim1 <= 2050.0:
                            return 4
                        else:
                            return 8
                else:
                    if _arg1_dim2 <= 16.0:
                        if _arg10_scalar <= 4.0:
                            return 0
                        else:
                            return 1
                    else:
                        if _arg1_dim0 <= 16384.0:
                            return 4
                        else:
                            return 3


def autotune__kernel_unified(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {
            "block_sizes": [2, 2],
            "loop_orders": [[3, 0, 1, 2]],
            "l2_groupings": [2],
            "range_unroll_factors": [0, 1],
            "range_warp_specializes": [],
            "range_num_stages": [0, 2],
            "range_multi_buffers": [None, False],
            "range_flattens": [None, None],
            "load_eviction_policies": [
                "first",
                "first",
                "last",
                "last",
                "first",
                "first",
                "last",
                "",
                "",
                "last",
            ],
            "num_warps": 4,
            "num_stages": 4,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 4,
        },
        {
            "block_sizes": [32, 4],
            "loop_orders": [[1, 0, 3, 2]],
            "l2_groupings": [4],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 0],
            "range_multi_buffers": [None, None],
            "range_flattens": [None, True],
            "load_eviction_policies": [
                "",
                "",
                "",
                "",
                "",
                "first",
                "last",
                "last",
                "last",
                "",
            ],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 1,
        },
        {
            "block_sizes": [2, 2],
            "loop_orders": [[0, 3, 2, 1]],
            "l2_groupings": [2],
            "range_unroll_factors": [0, 4],
            "range_warp_specializes": [],
            "range_num_stages": [0, 4],
            "range_multi_buffers": [None, None],
            "range_flattens": [None, None],
            "load_eviction_policies": [
                "first",
                "last",
                "last",
                "last",
                "last",
                "",
                "",
                "",
                "",
                "last",
            ],
            "num_warps": 2,
            "num_stages": 2,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 32,
        },
        {
            "block_sizes": [16, 2],
            "loop_orders": [[0, 3, 2, 1]],
            "l2_groupings": [16],
            "range_unroll_factors": [0, 2],
            "range_warp_specializes": [],
            "range_num_stages": [0, 4],
            "range_multi_buffers": [None, None],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "last",
                "first",
                "first",
                "",
                "first",
                "last",
                "first",
                "last",
                "first",
                "",
            ],
            "num_warps": 4,
            "num_stages": 8,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 2,
        },
        {
            "block_sizes": [64, 2],
            "loop_orders": [[3, 0, 1, 2]],
            "l2_groupings": [16],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [4, 4],
            "range_multi_buffers": [False, None],
            "range_flattens": [None, True],
            "load_eviction_policies": [
                "",
                "first",
                "",
                "",
                "first",
                "first",
                "",
                "",
                "last",
                "",
            ],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 32,
            "num_splits": 1,
        },
        {
            "block_sizes": [1, 4],
            "loop_orders": [[0, 3, 2, 1]],
            "l2_groupings": [4],
            "range_unroll_factors": [0, 1],
            "range_warp_specializes": [],
            "range_num_stages": [0, 2],
            "range_multi_buffers": [None, False],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "first",
                "first",
                "first",
                "last",
                "last",
                "first",
                "first",
                "",
                "last",
                "last",
            ],
            "num_warps": 4,
            "num_stages": 1,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 16,
        },
        {
            "block_sizes": [4, 4],
            "loop_orders": [[2, 3, 0, 1]],
            "l2_groupings": [1],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 4],
            "range_multi_buffers": [None, False],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "",
                "",
                "last",
                "last",
                "last",
                "last",
                "",
                "last",
                "",
                "",
            ],
            "num_warps": 4,
            "num_stages": 3,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 2,
        },
        {
            "block_sizes": [32, 8],
            "loop_orders": [[0, 3, 2, 1]],
            "l2_groupings": [32],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 1],
            "range_multi_buffers": [None, False],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "last",
                "first",
                "last",
                "last",
                "first",
                "last",
                "first",
                "first",
                "first",
                "last",
            ],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 1,
        },
        {
            "block_sizes": [64, 4],
            "loop_orders": [[0, 3, 2, 1]],
            "l2_groupings": [8],
            "range_unroll_factors": [0, 4],
            "range_warp_specializes": [],
            "range_num_stages": [0, 2],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, True],
            "load_eviction_policies": [
                "",
                "first",
                "last",
                "last",
                "",
                "last",
                "",
                "",
                "first",
                "last",
            ],
            "num_warps": 8,
            "num_stages": 3,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 2,
        },
        {
            "block_sizes": [1, 4],
            "loop_orders": [[2, 1, 0, 3]],
            "l2_groupings": [16],
            "range_unroll_factors": [0, 1],
            "range_warp_specializes": [],
            "range_num_stages": [0, 4],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, None],
            "load_eviction_policies": [
                "last",
                "last",
                "first",
                "last",
                "",
                "first",
                "first",
                "first",
                "",
                "",
            ],
            "num_warps": 4,
            "num_stages": 2,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
            "num_splits": 1,
        },
    ]
    return _C[key__kernel_unified(*args)]


# === Kernel: _kernel_combine ===


def key__kernel_combine(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = (
        int(args[0].shape[0])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0
        else 0
    )
    _arg0_dim1 = (
        int(args[0].shape[1])
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1
        else 0
    )
    _arg0_numel = (
        int(args[0].numel())
        if len(args) > 0 and isinstance(args[0], torch.Tensor)
        else 0
    )
    _arg1_dim0 = (
        int(args[1].shape[0])
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim > 0
        else 0
    )
    _arg1_dim1 = (
        int(args[1].shape[1])
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim > 1
        else 0
    )
    _arg1_numel = (
        int(args[1].numel())
        if len(args) > 1 and isinstance(args[1], torch.Tensor)
        else 0
    )
    if _arg0_numel <= 1048576.0:
        if _arg0_numel <= 138240.0:
            if _arg1_dim0 <= 16.0:
                if _arg1_dim0 <= 4.0:
                    if _arg0_dim1 <= 16.0:
                        if _arg0_dim0 <= 30.0:
                            return 4
                        else:
                            return 0
                    else:
                        if _arg0_dim0 <= 21.0:
                            return 0
                        else:
                            return 0
                else:
                    if _arg0_dim1 <= 40.0:
                        if _arg0_dim1 <= 16.0:
                            return 0
                        else:
                            return 4
                    else:
                        if _arg0_dim0 <= 1.0:
                            return 0
                        else:
                            return 0
            else:
                return 0
        else:
            if _arg1_numel <= 1310720.0:
                if _arg1_numel <= 798720.0:
                    return 4
                else:
                    if _arg1_numel <= 819200.0:
                        return 0
                    else:
                        if _arg0_dim0 <= 32.0:
                            return 0
                        else:
                            return 4
            else:
                if _arg1_dim1 <= 4096.0:
                    return 0
                else:
                    if _arg0_dim0 <= 192.0:
                        return 1
                    else:
                        return 4
    else:
        if _arg0_dim0 <= 512.0:
            if _arg0_numel <= 2416640.0:
                return 1
            else:
                if _arg0_dim0 <= 312.0:
                    if _arg1_dim0 <= 2.0:
                        return 2
                    else:
                        return 1
                else:
                    return 1
        else:
            if _arg0_numel <= 5427200.0:
                if _arg1_dim0 <= 2.0:
                    return 2
                else:
                    return 0
            else:
                if _arg1_dim1 <= 79872.0:
                    if _arg1_numel <= 10878976.0:
                        return 3
                    else:
                        return 2
                else:
                    if _arg1_dim0 <= 2.0:
                        return 3
                    else:
                        return 0


def autotune__kernel_combine(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {
            "block_sizes": [],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": [
                "first",
                "last",
                "",
                "first",
                "first",
                "",
                "last",
                "last",
                "",
                "last",
            ],
            "num_warps": 1,
            "num_stages": 2,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [],
            "reduction_loops": [None],
            "range_unroll_factors": [1],
            "range_warp_specializes": [],
            "range_num_stages": [3],
            "range_multi_buffers": [None],
            "range_flattens": [False],
            "load_eviction_policies": ["", "", "", ""],
            "num_warps": 1,
            "num_stages": 2,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 32,
            "maxnreg": 128,
        },
        {
            "block_sizes": [],
            "reduction_loops": [None],
            "range_unroll_factors": [3],
            "range_warp_specializes": [],
            "range_num_stages": [4],
            "range_multi_buffers": [None],
            "range_flattens": [False],
            "load_eviction_policies": ["first", "", "first", "first"],
            "num_warps": 1,
            "num_stages": 3,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 64,
            "maxnreg": 32,
        },
        {
            "block_sizes": [],
            "reduction_loops": [None],
            "range_unroll_factors": [3],
            "range_warp_specializes": [],
            "range_num_stages": [3],
            "range_multi_buffers": [False],
            "range_flattens": [True],
            "load_eviction_policies": ["", "", "first", "last"],
            "num_warps": 1,
            "num_stages": 1,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 128,
            "maxnreg": 256,
        },
        {
            "block_sizes": [],
            "reduction_loops": [None],
            "range_unroll_factors": [1],
            "range_warp_specializes": [],
            "range_num_stages": [3],
            "range_multi_buffers": [False],
            "range_flattens": [True],
            "load_eviction_policies": [
                "last",
                "last",
                "first",
                "first",
                "last",
                "first",
                "",
                "",
                "",
                "last",
            ],
            "num_warps": 1,
            "num_stages": 3,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 8,
            "maxnreg": 64,
        },
    ]
    return _C[key__kernel_combine(*args)]
