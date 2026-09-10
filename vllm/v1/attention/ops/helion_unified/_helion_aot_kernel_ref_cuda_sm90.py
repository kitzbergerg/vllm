"""
Auto-generated heuristic for kernel: kernel
Backend: decision_tree

Provides:
- key_kernel(*args): Returns config index (cache key)
- autotune_kernel(*args): Returns config dict for the given arguments
"""

import torch


def key_kernel(*args) -> int:
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
    _arg13_scalar = (
        args[13] if len(args) > 13 and isinstance(args[13], (int, float)) else 0
    )
    _arg2_dim0 = (
        int(args[2].shape[0])
        if len(args) > 2 and isinstance(args[2], torch.Tensor) and args[2].ndim > 0
        else 0
    )
    _arg2_dim1 = (
        int(args[2].shape[1])
        if len(args) > 2 and isinstance(args[2], torch.Tensor) and args[2].ndim > 1
        else 0
    )
    _arg2_dim2 = (
        int(args[2].shape[2])
        if len(args) > 2 and isinstance(args[2], torch.Tensor) and args[2].ndim > 2
        else 0
    )
    _arg2_numel = (
        int(args[2].numel())
        if len(args) > 2 and isinstance(args[2], torch.Tensor)
        else 0
    )
    _arg4_dim1 = (
        int(args[4].shape[1])
        if len(args) > 4 and isinstance(args[4], torch.Tensor) and args[4].ndim > 1
        else 0
    )
    _arg4_numel = (
        int(args[4].numel())
        if len(args) > 4 and isinstance(args[4], torch.Tensor)
        else 0
    )
    _arg8_scalar = args[8] if len(args) > 8 and isinstance(args[8], (int, float)) else 0
    if _arg0_dim0 <= 480.0:
        if _arg2_dim1 <= 16.0:
            if _arg0_dim0 <= 8.0:
                if _arg0_dim1 <= 16.0:
                    if _arg2_dim0 <= 5.0:
                        return 3
                    else:
                        return 4
                else:
                    if _arg2_dim0 <= 17.0:
                        return 3
                    else:
                        if _arg2_dim2 <= 8.0:
                            return 7
                        else:
                            return 1
            else:
                if _arg4_dim1 <= 33.0:
                    if _arg0_dim0 <= 64.0:
                        return 3
                    else:
                        if _arg4_dim1 <= 1.0:
                            return 2
                        else:
                            return 0
                else:
                    if _arg8_scalar <= 2.0:
                        if _arg0_dim1 <= 16.0:
                            return 3
                        else:
                            return 1
                    else:
                        if _arg0_dim1 <= 32.0:
                            return 6
                        else:
                            return 0
        else:
            if _arg2_dim1 <= 32.0:
                if _arg0_dim0 <= 6.0:
                    if _arg4_dim1 <= 17.0:
                        if _arg2_dim2 <= 16.0:
                            return 3
                        else:
                            return 5
                    else:
                        return 4
                else:
                    if _arg2_dim2 <= 10.0:
                        if _arg2_numel <= 270565376.0:
                            return 3
                        else:
                            return 3
                    else:
                        if _arg0_dim0 <= 256.0:
                            return 0
                        else:
                            return 2
            else:
                if _arg0_dim1 <= 16.0:
                    return 2
                else:
                    if _arg4_dim1 <= 1.0:
                        return 2
                    else:
                        if _arg0_dim1 <= 40.0:
                            return 3
                        else:
                            return 3
    else:
        if _arg8_scalar <= 32.0:
            if _arg2_dim1 <= 16.0:
                if _arg13_scalar <= 4.0:
                    if _arg0_dim1 <= 40.0:
                        if _arg0_dim1 <= 16.0:
                            return 0
                        else:
                            return 2
                    else:
                        return 3
                else:
                    if _arg2_numel <= 1107361792.0:
                        return 6
                    else:
                        return 0
            else:
                if _arg4_numel <= 16448.0:
                    return 2
                else:
                    return 8
        else:
            if _arg2_dim2 <= 10.0:
                if _arg0_dim1 <= 16.0:
                    if _arg0_dim0 <= 512.0:
                        return 8
                    else:
                        return 5
                else:
                    if _arg2_dim1 <= 16.0:
                        if _arg2_dim0 <= 8705.0:
                            return 6
                        else:
                            return 2
                    else:
                        if _arg0_dim1 <= 40.0:
                            return 8
                        else:
                            return 2
            else:
                if _arg13_scalar <= 5.0:
                    if _arg0_dim0 <= 1038.0:
                        return 6
                    else:
                        if _arg0_dim0 <= 1866.0:
                            return 5
                        else:
                            return 6
                else:
                    return 5


def autotune_kernel(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {
            "block_sizes": [16, 2],
            "loop_orders": [[1, 0, 2]],
            "l2_groupings": [32],
            "range_unroll_factors": [0, 3],
            "range_warp_specializes": [],
            "range_num_stages": [0, 1],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "",
                "last",
                "last",
                "",
                "last",
                "first",
                "first",
            ],
            "num_warps": 4,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [2, 8],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [32],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 1],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, None],
            "load_eviction_policies": ["first", "last", "last", "last", "", "", ""],
            "num_warps": 8,
            "num_stages": 5,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [16, 2],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [32],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 1],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, False],
            "load_eviction_policies": ["last", "last", "", "first", "", "", ""],
            "num_warps": 4,
            "num_stages": 1,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1, 2],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [1],
            "range_unroll_factors": [0, 4],
            "range_warp_specializes": [],
            "range_num_stages": [0, 2],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, False],
            "load_eviction_policies": [
                "last",
                "last",
                "",
                "last",
                "first",
                "last",
                "first",
            ],
            "num_warps": 4,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1, 4],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [1],
            "range_unroll_factors": [0, 4],
            "range_warp_specializes": [],
            "range_num_stages": [1, 3],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, False],
            "load_eviction_policies": ["", "first", "last", "last", "first", "", ""],
            "num_warps": 4,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 4,
        },
        {
            "block_sizes": [64, 2],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [32],
            "range_unroll_factors": [0, 1],
            "range_warp_specializes": [],
            "range_num_stages": [0, 4],
            "range_multi_buffers": [None, None],
            "range_flattens": [None, None],
            "load_eviction_policies": [
                "last",
                "first",
                "",
                "last",
                "first",
                "",
                "first",
            ],
            "num_warps": 8,
            "num_stages": 5,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [32, 8],
            "loop_orders": [[0, 1, 2]],
            "l2_groupings": [16],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 1],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, None],
            "load_eviction_policies": [
                "first",
                "",
                "first",
                "first",
                "first",
                "last",
                "last",
            ],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1, 8],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [4],
            "range_unroll_factors": [1, 0],
            "range_warp_specializes": [],
            "range_num_stages": [4, 4],
            "range_multi_buffers": [True, None],
            "range_flattens": [None, True],
            "load_eviction_policies": ["first", "first", "", "", "first", "last", ""],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 16,
            "maxnreg": 256,
        },
        {
            "block_sizes": [32, 2],
            "loop_orders": [[2, 0, 1]],
            "l2_groupings": [8],
            "range_unroll_factors": [0, 0],
            "range_warp_specializes": [],
            "range_num_stages": [0, 0],
            "range_multi_buffers": [None, True],
            "range_flattens": [None, False],
            "load_eviction_policies": ["", "last", "first", "first", "", "first", ""],
            "num_warps": 8,
            "num_stages": 6,
            "indexing": [
                "tensor_descriptor",
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
        },
    ]
    return _C[key_kernel(*args)]
