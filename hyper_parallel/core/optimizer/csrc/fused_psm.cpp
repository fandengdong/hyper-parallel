/*
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */

// Fused PSM (Power-Sign Momentum) update kernel for host-offloaded parameters.
//
// The Python update needs a chain of elementwise tensor ops per parameter
// (m = gamma * m + g, sign(m) * |m| ** beta, + wd * theta, theta -= lr * update).
// On host-offloaded parameters that chain runs on the CPU and is pure memory
// traffic: every op walks the whole tensor again, so the update costs roughly seven
// read/write passes over the model plus several full-size temporaries -- the
// momentum, a sign, an abs, a power and the combined update tensor.
//
// This translation unit replaces it with a single pass that reads grad, exp_avg and
// param once and writes exp_avg and param once, with no temporaries -- the same shape
// as torch._fused_adamw_, which is what makes the AdamW path fast. It deliberately
// exposes a plain C ABI over raw pointers so it can be built by any host compiler
// without Torch headers, Python headers or a specific ABI, and loaded with ctypes.
//
// Arithmetic for both float32 and bfloat16 storage is done in float, matching the
// opmath promotion torch uses for its bfloat16 elementwise kernels, and every store
// rounds once. Together with -ffp-contract=off (set by the Python builder) the
// momentum buffer is therefore bit-identical to the two-op torch sequence, while the
// update term can differ by one unit in the last place because the power is evaluated
// by the host libm instead of torch's vectorized kernel.
//
// Threading: when a group holds many parameters the outer loop is parallelised and
// each tensor is walked serially; for a handful of very large tensors the inner loop
// is parallelised instead. OpenMP picks the thread count from OMP_NUM_THREADS
// (hp_psm_set_threads overrides it process-wide).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

/** Read/write helpers for one storage type; all arithmetic happens in float. */
struct Float32IO {
    using storage_t = float;

    static float load(const storage_t *base, long index) { return base[index]; }
    static void store(storage_t *base, long index, float value) { base[index] = value; }
    static float round_trip(float value) { return value; }
};

/** bfloat16 is the top half of a float32; conversion is a shift plus one rounding. */
struct BFloat16IO {
    using storage_t = uint16_t;

    static float load(const storage_t *base, long index) { return from_bf16(base[index]); }

    static void store(storage_t *base, long index, float value) { base[index] = to_bf16(value); }

    /**
     * Round ``value`` to bfloat16 and back.
     *
     * The fallback path is a chain of separate torch kernels, so every intermediate
     * result is stored back into the bfloat16 tensor and rounded. Reproducing that
     * here keeps the two paths on the same numerical trajectory -- the momentum buffer
     * in particular stays bit-identical -- instead of silently running a higher
     * precision update under the same name.
     */
    static float round_trip(float value) { return from_bf16(to_bf16(value)); }

private:
    static float from_bf16(uint16_t value) {
        const uint32_t bits = static_cast<uint32_t>(value) << 16;
        float out = 0.0F;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    static uint16_t to_bf16(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t exponent = bits & 0x7F800000U;
        const uint32_t mantissa = bits & 0x007FFFFFU;
        if (exponent == 0x7F800000U && mantissa != 0U) {
            // NaN: keep the payload verbatim instead of rounding it into an infinity.
            return static_cast<uint16_t>(bits >> 16);
        }
        bits += 0x7FFFU + ((bits >> 16) & 1U);  // round to nearest, ties to even
        return static_cast<uint16_t>(bits >> 16);
    }
};

/**
 * Apply the PSM update to element ``index`` of one tensor, matching the fallback's
 * op sequence and rounding exactly.
 *
 * For bfloat16 every intermediate is rounded back to bfloat16 because the fallback
 * writes each step of the chain into the tensor; for float32 the round trip is a
 * no-op and the computation stays in registers.
 */
template <typename IO>
inline void psm_update_element(typename IO::storage_t *param, const typename IO::storage_t *grad,
                               typename IO::storage_t *exp_avg, long index, float lr, float gamma,
                               float beta, float weight_decay) {
    const float scaled = IO::round_trip(gamma * IO::load(exp_avg, index));
    const float momentum = IO::round_trip(scaled + IO::load(grad, index));
    IO::store(exp_avg, index, momentum);
    // sign(m) * |m| ** beta keeps the momentum direction while the power law rescales
    // its magnitude; |m| ** 0 degenerates to plain signSGD.
    const float powered =
        IO::round_trip(std::copysign(std::pow(std::fabs(momentum), beta), momentum));
    const float update = IO::round_trip(powered + weight_decay * IO::load(param, index));
    IO::store(param, index, IO::round_trip(IO::load(param, index) - lr * update));
}

/** Apply the PSM update to one tensor inside a single pass, no temporaries. */
template <typename IO>
void psm_update_serial(typename IO::storage_t *param, const typename IO::storage_t *grad,
                       typename IO::storage_t *exp_avg, long numel, float lr, float gamma,
                       float beta, float weight_decay) {
    for (long i = 0; i < numel; ++i) {
        psm_update_element<IO>(param, grad, exp_avg, i, lr, gamma, beta, weight_decay);
    }
}

/** Inner-parallel variant, used when the group holds few but very large tensors. */
template <typename IO>
void psm_update_parallel(typename IO::storage_t *param, const typename IO::storage_t *grad,
                         typename IO::storage_t *exp_avg, long numel, float lr, float gamma,
                         float beta, float weight_decay) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < numel; ++i) {
        psm_update_element<IO>(param, grad, exp_avg, i, lr, gamma, beta, weight_decay);
    }
}

/** Run the whole group for one storage type. */
template <typename IO>
void psm_update_group(void *const *params, const void *const *grads, void *const *exp_avgs,
                      const long *sizes, int count, float lr, float gamma, float beta,
                      float weight_decay) {
    auto *const *typed_params = reinterpret_cast<typename IO::storage_t *const *>(params);
    auto *const *typed_grads = reinterpret_cast<const typename IO::storage_t *const *>(grads);
    auto *const *typed_exp_avgs = reinterpret_cast<typename IO::storage_t *const *>(exp_avgs);
    // torch's scalar multiply rounds the scalar to the tensor type before multiplying,
    // so a bfloat16 group really uses bfloat16(gamma). Rounding once here reproduces
    // that and keeps the momentum buffer bit-identical to the fallback (float32 is
    // unaffected, its round trip is a no-op).
    const float stored_gamma = IO::round_trip(gamma);
    // Many tensors: parallelise across them so the whole group costs one fork/join.
    // Few tensors: parallelise inside each one, otherwise a single huge tensor would
    // be walked by a single thread.
    if (count >= 8) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int k = 0; k < count; ++k) {
            psm_update_serial<IO>(typed_params[k], typed_grads[k], typed_exp_avgs[k], sizes[k],
                                  lr, stored_gamma, beta, weight_decay);
        }
    } else {
        for (int k = 0; k < count; ++k) {
            psm_update_parallel<IO>(typed_params[k], typed_grads[k], typed_exp_avgs[k], sizes[k],
                                    lr, stored_gamma, beta, weight_decay);
        }
    }
}

}  // namespace

/**
 * Update a list of tensors in place with the PSM rule.
 *
 * Args:
 *     params: Array of ``count`` writable parameter buffers.
 *     grads: Array of ``count`` read-only gradient buffers.
 *     exp_avgs: Array of ``count`` momentum buffers, read and written.
 *     sizes: Array of ``count`` element counts.
 *     count: Number of tensors in the group; must be positive.
 *     dtype_code: Storage type of every buffer; 0 is float32, 1 is bfloat16.
 *     lr: Learning rate.
 *     gamma: Momentum factor.
 *     beta: Power-law exponent applied to the momentum magnitude.
 *     weight_decay: Decoupled weight-decay coefficient.
 *
 * Returns:
 *     0 on success, -1 when an argument is null or ``count`` is not positive,
 *     -2 when ``dtype_code`` is not a supported storage type.
 */
extern "C" int hp_psm_update(void *const *params, const void *const *grads, void *const *exp_avgs,
                             const long *sizes, int count, int dtype_code, float lr, float gamma,
                             float beta, float weight_decay) {
    if (params == nullptr || grads == nullptr || exp_avgs == nullptr || sizes == nullptr ||
        count <= 0) {
        return -1;
    }
    if (dtype_code == 0) {
        psm_update_group<Float32IO>(params, grads, exp_avgs, sizes, count, lr, gamma, beta,
                                    weight_decay);
    } else if (dtype_code == 1) {
        psm_update_group<BFloat16IO>(params, grads, exp_avgs, sizes, count, lr, gamma, beta,
                                     weight_decay);
    } else {
        return -2;
    }
    return 0;
}

/** Override the OpenMP thread count used by this kernel (0 or less is ignored). */
extern "C" void hp_psm_set_threads(int threads) {
#if defined(_OPENMP)
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif
}
