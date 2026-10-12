// Compare the direct cell-list Metal attention path with the original dense-mask path.
// Exercises both decode and prefill kernels, GQA, padded rows, and a 128k KV cache.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static bool check(ggml_backend_t backend, int head, int kv, int queries, int selected, bool strided, bool benchmark = false) {
    ggml_context * ctx = ggml_init({2*1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head, queries, 6, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head, kv, 2, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head, kv, 2, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, queries);
    auto * cells_base = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected + (strided ? 7 : 0), queries);
    auto * cells = strided ? ggml_view_2d(ctx, cells_base, selected, queries, cells_base->nb[1], 0) : cells_base;
    auto * ref = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(head), 0.0f, 0.0f);
    auto * got = ggml_flash_attn_ext(ctx, q, k, v, cells, 1.0f/sqrtf(head), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(ref, selected);
    ggml_flash_attn_ext_set_n_kv_max(got, selected);
    ggml_prec_set_acc(ref, GGML_PREC_F32);
    ggml_prec_set_acc(got, GGML_PREC_F32);
    ggml_set_output(ref);
    ggml_set_output(got);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ref);
    ggml_build_forward_expand(graph, got);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    std::mt19937 rng(37 + queries);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> qdata(ggml_nelements(q));
    for (auto & x : qdata) x = value(rng);
    ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> data(ggml_nelements(k));
    for (auto & x : data) x = ggml_fp32_to_fp16(value(rng));
    ggml_backend_tensor_set(k, data.data(), 0, ggml_nbytes(k));
    for (auto & x : data) x = ggml_fp32_to_fp16(value(rng));
    ggml_backend_tensor_set(v, data.data(), 0, ggml_nbytes(v));
    std::vector<ggml_fp16_t> masks((size_t)kv*queries, ggml_fp32_to_fp16(-INFINITY));
    std::vector<int32_t> indices(ggml_nelements(cells_base), -1);
    std::vector<int32_t> order(kv), row(selected);
    std::iota(order.begin(), order.end(), 0);
    for (int iq = 0; iq < queries; ++iq) {
        std::shuffle(order.begin(), order.end(), rng);
        // Include one valid cell, a full row, and partially populated rows.
        const int live = iq % 3 == 0 ? 1 : iq % 3 == 1 ? std::min(kv, selected) : std::min(kv, selected*3/4);
        for (int j = 0; j < selected; ++j) {
            row[j] = j < live ? order[j] : j % 2 == 0 ? kv + j : -1;
            if (j < live) masks[(size_t)iq*kv + row[j]] = ggml_fp32_to_fp16(0.0f);
        }
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), indices.begin() + iq*cells_base->ne[0]);
    }
    ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(cells_base, indices.data(), 0, ggml_nbytes(cells_base));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> a(ggml_nelements(ref)), b(a.size());
    ggml_backend_tensor_get(ref, a.data(), 0, ggml_nbytes(ref));
    ggml_backend_tensor_get(got, b.data(), 0, ggml_nbytes(got));
    size_t different = 0;
    float max_error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++different;
        max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
    std::printf("head=%d kv=%d queries=%d selected=%d strided=%d: %zu differing floats, max error %.9g\n",
                head, kv, queries, selected, strided, different, max_error);
    bool timed_ok = true;
    if (benchmark && computed && different == 0) {
        auto * perf_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(perf_graph, got);
        for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(backend, perf_graph);
        const auto start = std::chrono::steady_clock::now();
        constexpr int repeats = 100;
        for (int i = 0; i < repeats; ++i) {
            if (ggml_backend_graph_compute(backend, perf_graph) != GGML_STATUS_SUCCESS) {
                timed_ok = false;
                break;
            }
        }
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        std::printf("perf head=%d kv=%d queries=%d selected=%d: %.3f us/run\n",
                    head, kv, queries, selected, elapsed / repeats);
    }
    std::fflush(stdout);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return computed && timed_ok && different == 0;
}

int main(int argc, char ** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc != 1 && !benchmark) return 2;
    auto backend = ggml_backend_metal_init();
    if (!backend) return 2;
    bool ok = true;
    for (int head : {128, 256}) {
        if (benchmark) {
            for (int queries : {1, 65, 256}) ok &= check(backend, head, 131072, queries, 2051, true, true);
            continue;
        }
        for (int selected : {1, 7, 32, 64}) ok &= check(backend, head, 256, 3, selected, true);
        for (int queries : {1, 7, 33, 65}) {
            ok &= check(backend, head, 256, queries, 127, queries % 2 == 1);
            ok &= check(backend, head, 5000, queries, 2051, false);
        }
        ok &= check(backend, head, 131072, 3, 2051, true);
        ok &= check(backend, head, 8192, 3, 4096, false);
    }
    ggml_backend_free(backend);
    std::puts(ok ? "all indexed attention outputs are bit-identical" : "FAILED");
    return ok ? 0 : 1;
}
