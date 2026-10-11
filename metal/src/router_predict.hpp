// metal/src/router_predict.hpp - which experts the next layers will route to, before they are computed.
//
// A layer's router is a matrix product: its 512 expert scores are ffn_gate_inp x the layer's MoE input, and the ten
// largest are the experts.  The MoE input changes little from one layer to the next (the layers add to a residual
// stream), so the NEXT layer's router applied to THIS layer's input already names most of what the next layer will
// pick.  Measured on IQ3_S (100 tokens of a chat, docs/APPLE_METAL.md): of the next layer's ten experts, the ten
// predicted this way hold 67%, and 60% of those not used in the last 8 tokens - the ones the cache misses.
//
// The engine uses it to read those experts from the SSD while the GPU computes, instead of after it has stopped to
// wait for them (llama_moe_cache_set_prefetch).  A wrong prediction costs a read; it never changes what is computed.
//
// The prediction is corrected as it goes: when a layer has routed, the experts that were named for it are scored
// with the layer's real input, and how far each was off is remembered per expert (a running average).  Some experts
// are named too readily by the layer before and some too rarely; with the correction the same number of reads
// brought in 16% more of the experts that were then routed to (measured on the same 100 tokens).
#pragma once

#include "expert_io.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::metal {

struct RouterPredict {
    int n_embd = 0, n_expert = 0;
    int top = 10;     // experts named per token and layer, at most
    int depth = 1;    // layers ahead
    float margin = 0.0f;   // ... and only those whose predicted score is this much above the predicted 10th score
    float learn = 0.1f;    // how fast the per-expert correction follows what was seen (0: no correction)
    static constexpr int kLook = 16;   // experts looked at per layer: the ones with the highest predicted scores
    struct Named {
        int32_t expert;
        float margin;   // predicted score over the predicted 10th score, before the correction
    };
    std::vector<std::vector<Named>> named;      // per layer: what the last one-token pass's prediction looked at
    std::vector<std::vector<float>> bias;       // per layer and expert: the correction of the margin
    std::vector<std::vector<int32_t>> routed;   // per layer: what the last one-token pass routed to (the engine sets it)
    std::vector<std::vector<uint16_t>> gate;   // per layer: ffn_gate_inp as BF16, [n_expert][n_embd]
    std::vector<float> score;
    std::vector<int32_t> order;
    double ms = 0.0;
    uint64_t calls = 0;

    // the routers' weights, read from the model file (2.5 MiB a layer for IQ3_S)
    bool load(const ModelFiles& files, std::string& err) {
        for (int il = 0;; ++il) {
            const auto it = files.tensors.find("blk." + std::to_string(il) + ".ffn_gate_inp.weight");
            if (it == files.tensors.end()) break;
            const TensorAt& t = it->second;
            if (t.type != GGML_TYPE_BF16 && t.type != GGML_TYPE_F32) {
                err = "the routers' weights are neither BF16 nor F32";
                return false;
            }
            std::vector<uint16_t> w(t.size / 2);
            size_t got = 0;
            while (got < t.size) {
                const ssize_t r = ::pread(files.fds[(size_t) t.file], (char*) w.data() + got, t.size - got, (off_t) (t.offs + got));
                if (r <= 0) {
                    err = "the routers' weights could not be read";
                    return false;
                }
                got += (size_t) r;
            }
            if (t.type == GGML_TYPE_F32) {   // a prediction does not need the lower half
                for (size_t i = 0; i < w.size() / 2; ++i) w[i] = w[2 * i + 1];
                w.resize(w.size() / 2);
            }
            gate.push_back(std::move(w));
        }
        if (gate.empty() || files.n_expert <= 0 || gate[0].size() % (size_t) files.n_expert != 0) {
            err = "the model has no routers this engine knows";
            return false;
        }
        n_expert = (int) files.n_expert;
        n_embd = (int) (gate[0].size() / (size_t) n_expert);
        score.resize((size_t) n_expert);
        order.resize((size_t) n_expert);
        named.assign(gate.size(), {});
        routed.assign(gate.size(), {});
        bias.assign(gate.size(), std::vector<float>((size_t) n_expert, 0.0f));
        return n_embd % 16 == 0;
    }

    // one expert's score: a row of a layer's router (BF16, the upper half of a float) times the router's input
    float dot(const uint16_t* row, const float* xt) const {
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int j = 0; j < n_embd; j += 16) {
            const uint16x8_t h0 = vld1q_u16(row + j), h1 = vld1q_u16(row + j + 8);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)), vld1q_f32(xt + j));
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)), vld1q_f32(xt + j + 4));
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)), vld1q_f32(xt + j + 8));
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)), vld1q_f32(xt + j + 12));
        }
        return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }

    // the engine: layer il of a one-token pass routed to these experts
    void saw(int32_t il, const int32_t* ids, size_t n) {
        if (il >= 0 && (size_t) il < routed.size()) routed[(size_t) il].assign(ids, ids + n);
    }

    // layer il has routed and its router's real input is here: how far off was what was predicted for it?
    void correct(int32_t il, const float* x) {
        std::vector<Named>& nm = named[(size_t) il];
        const std::vector<int32_t>& ids = routed[(size_t) il];
        if (learn <= 0.0f || nm.empty() || ids.size() < 10) {
            nm.clear();
            return;
        }
        const uint16_t* w = gate[(size_t) il].data();
        float tenth = 0.0f;   // the lowest score that was routed to
        for (size_t i = 0; i < ids.size(); ++i) {
            const float v = dot(w + (size_t) ids[i] * (size_t) n_embd, x);
            tenth = i == 0 ? v : std::min(tenth, v);
        }
        for (const Named& c : nm) {
            const float real = dot(w + (size_t) c.expert * (size_t) n_embd, x) - tenth;
            float& b = bias[(size_t) il][(size_t) c.expert];
            b += learn * (real - c.margin - b);
        }
        nm.clear();
    }

    static size_t callback(int32_t il, const float* x, int64_t n_embd, int64_t n_tokens, int32_t* layers, int32_t* experts,
                           size_t cap, void* ud) {
        return static_cast<RouterPredict*>(ud)->predict(il, x, n_embd, n_tokens, layers, experts, cap);
    }

    size_t predict(int32_t il, const float* x, int64_t ne, int64_t n_tokens, int32_t* layers, int32_t* experts, size_t cap) {
        if (ne != n_embd || top <= 0 || il < 0 || (size_t) il >= gate.size()) return 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (n_tokens == 1) correct(il, x);
        else named[(size_t) il].clear();
        size_t n = 0;
        for (int d = 1; d <= depth; ++d) {
            const int it = il + d;
            if (it >= (int) gate.size()) break;
            const uint16_t* w = gate[(size_t) it].data();
            if (d == 1) named[(size_t) it].clear();
            for (int64_t tk = 0; tk < n_tokens; ++tk) {
                const float* xt = x + tk * n_embd;
                for (int e = 0; e < n_expert; ++e) {
                    score[(size_t) e] = dot(w + (size_t) e * (size_t) n_embd, xt);
                    order[(size_t) e] = e;
                }
                const int look = std::min(kLook, n_expert);
                std::partial_sort(order.begin(), order.begin() + look, order.end(),
                                  [this](int32_t a, int32_t b) { return score[(size_t) a] > score[(size_t) b]; });
                const float tenth = score[(size_t) order[(size_t) std::min(look, 10) - 1]];
                // the experts looked at, by their corrected margin
                Named c[kLook];
                for (int i = 0; i < look; ++i) c[i] = {order[(size_t) i], score[(size_t) order[(size_t) i]] - tenth};
                if (d == 1 && n_tokens == 1) named[(size_t) it].assign(c, c + look);
                const std::vector<float>& b = bias[(size_t) it];
                std::sort(c, c + look, [&b](const Named& p, const Named& q) {
                    return p.margin + b[(size_t) p.expert] > q.margin + b[(size_t) q.expert];
                });
                for (int i = 0; i < look && i < top && n < cap; ++i) {
                    if (c[i].margin + b[(size_t) c[i].expert] < margin) break;
                    layers[n] = it;
                    experts[n] = c[i].expert;
                    ++n;
                }
            }
        }
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++calls;
        return n;
    }
};

}   // namespace strata::metal
