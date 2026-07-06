//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"


void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Fused MoE expert GEMV over all (token, expert-slot) routed pairs of a decode-sized batch.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1), one row per (token, src1-row)
    const int32_t *    ids_dev,             // device-side int32 [n_ids, n_tokens] with strides ids_s0/ids_s1
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_ids,               // experts used per token
    int                n_tokens,
    int                ne11,                // src1 rows per token: 1 (shared) or n_ids (per-slot)
    size_t             ids_s0,              // ids stride between slots, in elements
    size_t             ids_s1,              // ids stride between tokens, in elements
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_slot_stride,      // bytes between dst rows of one token (dst->nb[1])
    size_t             dst_token_stride,     // bytes between dst token planes (dst->nb[2])
    size_t             src1_qrow_stride,     // bytes per quantized src1 row
    dpct::queue_ptr    stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_ids,
    int                n_tokens,
    int                ne11,
    size_t             ids_s0,
    size_t             ids_s1,
    size_t             expert_weight_stride,
    size_t             dst_slot_stride,
    size_t             dst_token_stride,
    size_t             src1_qrow_stride,
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
