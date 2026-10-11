// metal/src/mtp_head.hpp - the model's MTP head: drafts of the next tokens, checked by the model in one pass.
//
// The MTP head is one more layer of the model (attention, 512 experts) that reads the hidden state of the last
// position and the token just chosen, and predicts the token after it; fed its own output it predicts the one after
// that.  `--mtp FILE` loads it (ggml-org's mtp-<model>.gguf) and the engine then writes like this: the model
// chooses a token, the head drafts up to `--spec` more, and ONE pass of the model reads the token and the drafts.
// That pass gives the model's own choice after each of them: as long as a draft is what the model chose, the next
// position's result is already there; at the first draft that is not, the rest of the pass is dropped (the context
// goes back) and the model's choice stands.  Every token that is written is the model's, from its own logits.
//
// The head has a context of its own, so every token the model reads or writes is also shown to the head, with the
// hidden state before it ("catch-up").
//
// STRATA_MTP_PROBE=FILE is a measurement: the head drafts after every token, the drafts are counted and not used,
// and the model writes one token per pass as without it (docs/APPLE_METAL.md says what it measured).
#pragma once

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// llama.cpp src/llama-ext.h (C++ linkage)
void llama_set_embeddings_nextn(struct llama_context* ctx, bool value, bool masked);
float* llama_get_embeddings_nextn(struct llama_context* ctx);
float* llama_get_embeddings_nextn_ith(struct llama_context* ctx, int32_t i);

namespace strata::metal {

struct MtpHead {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* tgt = nullptr;
    llama_batch_ext* batch = nullptr;
    int n_embd = 0, n_vocab = 0, n_draft = 3;
    int n_batch = 256;   // tokens per pass of the head (the model's longer passes are shown to it in pieces)
    bool use = false;    // the drafts are used (--mtp); false: counted only (STRATA_MTP_PROBE)
    std::vector<float> pending_h, chain_h;
    bool have_h = false;

    struct Step {
        std::vector<llama_token> drafts;
        std::vector<float> probs;   // the head's probability of each draft
    };
    std::vector<Step> steps;   // one per token the model wrote
    double draft_ms = 0, catch_ms = 0;
    uint64_t n_catch_tokens = 0, n_catch_passes = 0, n_drafts = 0;

    ~MtpHead() {
        free_context();
        if (model) llama_model_free(model);
    }

    void free_context() {
        if (batch) llama_batch_ext_free(batch);
        if (ctx) llama_free(ctx);
        batch = nullptr;
        ctx = nullptr;
    }

    // the head's weights, all on the GPU (2.0 GiB for the Q4_0 file)
    bool load_model(const std::string& path, ggml_backend_buffer_type_t gpu_alloc, std::string& err) {
        const llama_model_tensor_buft_override overrides[] = {{".*", gpu_alloc}, {nullptr, nullptr}};
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.use_extra_bufts = false;
        mp.load_mtp = true;
        mp.tensor_buft_overrides = overrides;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (model == nullptr) {
            err = "the MTP head could not be loaded from " + path;
            return false;
        }
        return true;
    }

    // the head's context, beside the model's
    bool make_context(llama_context* target, const llama_context_params& like, std::string& err) {
        free_context();
        tgt = target;
        llama_context_params cp = like;
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = target;
        cp.n_rs_seq = 0;
        cp.moe_cache_size = 0;
        cp.n_batch = cp.n_ubatch = (uint32_t) n_batch;
        cp.cb_eval = nullptr;
        ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr) {
            err = "the MTP head's context could not be created";
            return false;
        }
        n_embd = llama_model_n_embd_out(model);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (n_embd != llama_model_n_embd_out(llama_get_model(target))) {
            err = "the MTP head does not fit this model (hidden width)";
            return false;
        }
        llama_set_embeddings_nextn(target, true, false);
        llama_set_embeddings_nextn(ctx, true, true);
        batch = llama_batch_ext_init(ctx);
        pending_h.assign((size_t) n_embd, 0.0f);
        chain_h.assign((size_t) n_embd, 0.0f);
        return batch != nullptr;
    }

    // the context was emptied or went back: the head starts again from position `keep`
    void rewind(llama_pos keep) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (keep <= 0) llama_memory_clear(mem, true);
        else llama_memory_seq_rm(mem, 0, keep, -1);
        std::fill(pending_h.begin(), pending_h.end(), 0.0f);   // the state before `keep` is not kept: the next draft is poorer
        have_h = false;
    }

    // The model has just read tokens at positions pos0... in one pass: the head reads the first `n` of them too, each
    // with the state before it (the rows of the pass's hidden states; of the pass before for the first token).
    bool catch_up(const llama_token* toks, int32_t n, llama_pos pos0) {
        const auto t0 = std::chrono::steady_clock::now();
        const float* h = llama_get_embeddings_nextn(tgt);
        if (h == nullptr || n <= 0) return false;
        int32_t rc = 0;
        for (int32_t c0 = 0; c0 < n && rc == 0; c0 += n_batch) {
            llama_batch_ext_clear(batch);
            for (int32_t k = c0; k < std::min(n, c0 + n_batch); ++k) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[k]);
                if (idx < 0) return false;
                const llama_pos p = pos0 + k;
                llama_batch_ext_set_pos(batch, idx, &p);
                llama_batch_ext_set_embd_state(batch, idx, {k == 0 ? pending_h.data() : h + (size_t) (k - 1) * (size_t) n_embd, 1, (size_t) n_embd});
                llama_batch_ext_set_output_logits(batch, idx, false);
            }
            rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        }
        // (the pointer `h` is the model's own buffer: copy the last row before anything else runs)
        std::memcpy(pending_h.data(), h + (size_t) (n - 1) * (size_t) n_embd, sizeof(float) * (size_t) n_embd);
        have_h = true;
        llama_synchronize(ctx);
        catch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        n_catch_tokens += (uint64_t) n;
        ++n_catch_passes;
        return rc == 0;
    }

    // The model chose `id_last` for position pos0 (and has not read it yet): what does the head expect after it?
    // Up to `most` tokens, and only as long as the head gives its choice at least `p_min`.
    void draft(llama_token id_last, llama_pos pos0, int most, float p_min, std::vector<llama_token>& out, std::vector<float>& probs) {
        out.clear();
        probs.clear();
        if (!have_h || most <= 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        llama_token tok = id_last;
        const float* hrow = pending_h.data();
        for (int i = 0; i < most; ++i) {
            llama_batch_ext_clear(batch);
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tok);
            const llama_pos p = pos0 + i;
            llama_batch_ext_set_pos(batch, idx, &p);
            llama_batch_ext_set_embd_state(batch, idx, {hrow, 1, (size_t) n_embd});
            llama_batch_ext_set_output_logits(batch, idx, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) break;
            const float* logits = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v)
                if (logits[v] > logits[best]) best = v;
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) logits[v] - (double) logits[best]);
            const float prob = (float) (1.0 / sum);
            if (prob < p_min) break;
            out.push_back((llama_token) best);
            probs.push_back(prob);
            const float* h2 = llama_get_embeddings_nextn_ith(ctx, -1);
            if (h2 == nullptr) break;
            std::memcpy(chain_h.data(), h2, sizeof(float) * (size_t) n_embd);
            hrow = chain_h.data();
            tok = (llama_token) best;
        }
        // the drafts leave the head's context: the model's own tokens come in with catch_up
        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos0, -1);
        draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++n_drafts;
    }

    // the probe: remember what the head drafted after a token
    void note(const std::vector<llama_token>& d, const std::vector<float>& pr) {
        Step st;
        st.drafts = d;
        st.probs = pr;
        steps.push_back(std::move(st));
    }

    // `written`: what the model wrote, token j after step j's drafts were made for tokens j+1, j+2, ...
    void report(const std::vector<llama_token>& written) {
        if (steps.empty()) return;
        const size_t n = std::min(steps.size(), written.size());
        for (float p_min : {0.0f, 0.5f, 0.75f, 0.9f}) {
            uint64_t tried[4] = {0, 0, 0, 0}, right[4] = {0, 0, 0, 0};
            double passes = 0, tokens = 0, drafted = 0;
            // a speculative engine: draft while the head is this sure, check them in one pass, go on after the last right one
            for (size_t j = 0; j < n;) {
                size_t ok = 0, k = 0;
                for (; k < steps[j].drafts.size() && steps[j].probs[k] >= p_min; ++k) {
                    if (j + 1 + k >= written.size()) break;
                    ++tried[k];
                    if (ok == k && steps[j].drafts[k] == written[j + 1 + k]) {
                        ++right[k];
                        ++ok;
                    }
                }
                passes += 1;
                tokens += 1 + (double) ok;
                drafted += (double) k;
                j += 1 + ok;
            }
            std::fprintf(stderr, "strata metal: MTP probe, drafting while the head's probability is >= %.2f: %.2f tokens per pass (%.2f "
                                 "drafted per pass); right: first %llu/%llu, second %llu/%llu, third %llu/%llu\n",
                         (double) p_min, tokens / std::max(1.0, passes), drafted / std::max(1.0, passes),
                         (unsigned long long) right[0], (unsigned long long) tried[0], (unsigned long long) right[1],
                         (unsigned long long) tried[1], (unsigned long long) right[2], (unsigned long long) tried[2]);
        }
        std::fprintf(stderr, "strata metal: MTP probe: a draft of %d tokens takes %.1f ms; showing the head a token the model read takes "
                             "%.1f ms per pass (%.2f ms per token over %llu tokens)\n",
                     n_draft, draft_ms / (double) steps.size(), n_catch_passes ? catch_ms / (double) n_catch_passes : 0.0,
                     n_catch_tokens ? catch_ms / (double) n_catch_tokens : 0.0, (unsigned long long) n_catch_tokens);
        steps.clear();
        draft_ms = catch_ms = 0;
        n_catch_tokens = n_catch_passes = 0;
    }
};

}   // namespace strata::metal
