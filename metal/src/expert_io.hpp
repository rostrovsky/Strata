// metal/src/expert_io.hpp - the SSD tier of the Apple engine: where every tensor of the model sits in its GGUF
// shards, and a pool of threads that reads routed experts from there straight into the expert cache.
//
// The files are opened O_RDONLY and nothing here writes: the experts that fit neither the expert cache nor the OS
// file cache are read again when they are routed again, they are never copied to swap or to a pack on the disk.
#pragma once

#include "gpu_keepalive.hpp"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::metal {

inline bool is_expert_tensor(const std::string& name) {
    return name.find(".ffn_gate_exps.") != std::string::npos || name.find(".ffn_up_exps.") != std::string::npos ||
           name.find(".ffn_down_exps.") != std::string::npos || name.find(".ffn_gate_up_exps.") != std::string::npos;
}

// A tensor of the model: its shard, its bytes there, and (for the routed experts) how they divide.
struct TensorAt {
    int file = -1;
    uint64_t offs = 0;      // from the start of the shard
    uint64_t size = 0;
    int64_t n_expert = 0;   // ne[2] of an expert tensor, else 0
    int layer = -1;
    ggml_type type = GGML_TYPE_F32;
};

struct ModelFiles {
    std::vector<std::string> paths;
    std::vector<int> fds;           // O_RDONLY
    std::vector<uint64_t> sizes;
    std::unordered_map<std::string, TensorAt> tensors;
    uint64_t expert_bytes = 0;      // the routed experts: host memory, mapped from the files
    uint64_t lazy_bytes = 0;        // the n-gram table: rows read on demand
    uint64_t dense_bytes = 0;       // everything else
    int64_t n_expert = 0;
    int n_layer = 0;                // layers that have routed experts
    uint64_t n_pairs = 0;           // (layer, expert) pairs

    ~ModelFiles() {
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    }

    double mean_expert_bytes() const { return n_pairs ? (double) expert_bytes / (double) n_pairs : 0.0; }

    // `shard1` is the first (or only) GGUF; the others are found by the split naming llama.cpp writes.
    bool open(const std::string& shard1, bool no_cache, std::string& err) {
        int n_split = 1;
        {
            gguf_init_params ip = {/*no_alloc*/ true, /*ctx*/ nullptr};
            gguf_context* g = gguf_init_from_file(shard1.c_str(), ip);
            if (g == nullptr) {
                err = "cannot read " + shard1 + " (is it a GGUF file?)";
                return false;
            }
            const int64_t k = gguf_find_key(g, "split.count");
            if (k >= 0) n_split = (int) gguf_get_val_u16(g, k);
            gguf_free(g);
        }
        if (n_split < 1) n_split = 1;
        if (n_split == 1) {
            paths.push_back(shard1);
        } else {
            char prefix[4096];
            if (llama_split_prefix(prefix, sizeof prefix, shard1.c_str(), 0, n_split) <= 0) {
                err = shard1 + " has " + std::to_string(n_split) + " shards but is not named <name>-00001-of-0000N.gguf";
                return false;
            }
            for (int i = 0; i < n_split; ++i) {
                char p[4096];
                llama_split_path(p, sizeof p, prefix, i, n_split);
                paths.emplace_back(p);
            }
        }
        std::vector<bool> layer_seen;
        for (size_t f = 0; f < paths.size(); ++f) {
            const int fd = ::open(paths[f].c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                err = "cannot open " + paths[f] + ": " + std::strerror(errno);
                return false;
            }
            fds.push_back(fd);
#if defined(__APPLE__)
            // direct mode: the bytes go to the expert cache only, not through the OS file cache as well
            if (no_cache) (void) ::fcntl(fd, F_NOCACHE, 1);
#else
            (void) no_cache;
#endif
            struct stat st {};
            ::fstat(fd, &st);
            sizes.push_back((uint64_t) st.st_size);

            ggml_context* meta = nullptr;
            gguf_init_params ip = {/*no_alloc*/ true, /*ctx*/ &meta};
            gguf_context* g = gguf_init_from_file(paths[f].c_str(), ip);
            if (g == nullptr) {
                err = "cannot read " + paths[f];
                return false;
            }
            const uint64_t data0 = gguf_get_data_offset(g);
            const int64_t n = gguf_get_n_tensors(g);
            for (int64_t i = 0; i < n; ++i) {
                const std::string name = gguf_get_tensor_name(g, i);
                TensorAt t;
                t.file = (int) f;
                t.offs = data0 + gguf_get_tensor_offset(g, i);
                t.size = gguf_get_tensor_size(g, i);
                t.type = gguf_get_tensor_type(g, i);
                if (name.rfind("blk.", 0) == 0) t.layer = std::atoi(name.c_str() + 4);
                if (is_expert_tensor(name)) {
                    const ggml_tensor* mt = meta ? ggml_get_tensor(meta, name.c_str()) : nullptr;
                    t.n_expert = mt ? mt->ne[2] : 0;
                    if (t.n_expert <= 0 || t.size % (uint64_t) t.n_expert != 0) {
                        err = name + ": cannot tell its experts apart (" + std::to_string(t.size) + " bytes)";
                        gguf_free(g);
                        if (meta) ggml_free(meta);
                        return false;
                    }
                    expert_bytes += t.size;
                    n_expert = std::max(n_expert, t.n_expert);
                    if (t.layer >= 0) {
                        if ((size_t) t.layer >= layer_seen.size()) layer_seen.resize((size_t) t.layer + 1, false);
                        if (!layer_seen[(size_t) t.layer]) {
                            layer_seen[(size_t) t.layer] = true;
                            ++n_layer;
                            n_pairs += (uint64_t) t.n_expert;
                        }
                    }
                } else if (name == "per_layer_token_embd.weight") {
                    lazy_bytes += t.size;
                } else {
                    dense_bytes += t.size;
                }
                tensors.emplace(name, t);
            }
            gguf_free(g);
            if (meta) ggml_free(meta);
        }
        return true;
    }
};

// Reads many byte ranges at once.  One request is the misses of one layer (or a slice of the start-up fill): a few
// dozen ranges of 0.3-1 MB each, so the threads are what turns an NVMe's queue depth into bytes per second.
class ReadPool {
public:
    struct Job {
        int fd;
        uint64_t offs;
        void* dst;
        size_t size;
    };

    explicit ReadPool(int n_threads) {
        for (int i = 0; i < n_threads; ++i) workers_.emplace_back([this] { loop(); });
    }

    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // false: a read failed or came back short (`err` says which)
    bool run(const std::vector<Job>& jobs, std::string& err) {
        if (jobs.empty()) return true;
        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_ = &jobs;
            next_ = 0;
            left_ = jobs.size();
            failed_.clear();
            ++epoch_;
        }
        cv_.notify_all();
        work();   // the caller reads too
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return left_ == 0; });
        jobs_ = nullptr;
        if (!failed_.empty()) {
            err = failed_;
            return false;
        }
        return true;
    }

private:
    void work() {
        for (;;) {
            Job j;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (jobs_ == nullptr || next_ >= jobs_->size()) return;
                j = (*jobs_)[next_++];
            }
            std::string e;
            size_t got = 0;
            while (got < j.size) {
                const ssize_t r = ::pread(j.fd, (char*) j.dst + got, j.size - got, (off_t) (j.offs + got));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    e = r < 0 ? std::string("read failed: ") + std::strerror(errno) : "the file ended inside a tensor";
                    break;
                }
                got += (size_t) r;
            }
            std::lock_guard<std::mutex> lk(mu_);
            if (!e.empty() && failed_.empty()) failed_ = e;
            if (--left_ == 0) done_cv_.notify_all();
        }
    }

    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || epoch_ != seen; });
                if (quit_) return;
                seen = epoch_;
            }
            work();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::vector<Job>* jobs_ = nullptr;
    size_t next_ = 0;
    size_t left_ = 0;
    uint64_t epoch_ = 0;
    bool quit_ = false;
    std::string failed_;
};

// The fetch side of the expert cache: llama.cpp's MoE cache hands over the experts a layer routed to and does not
// hold; they are read from the shards into their cache slots.  On Apple silicon the cache's buffers are shared
// memory, so the slot the GPU reads is the buffer the SSD read lands in: no staging copy, no upload.
struct ExpertFetch {
    const ModelFiles* files = nullptr;
    ReadPool* pool = nullptr;
    std::unordered_map<const ggml_tensor*, const TensorAt*> known;
    std::unordered_map<const ggml_tensor*, bool> checked;
    std::vector<ReadPool::Job> jobs;
    GpuKeepalive* keepalive = nullptr;   // holds the GPU's clock while the reads are under way (gpu_keepalive.hpp)
    bool disabled = false;        // a read did not match the mapped file: llama.cpp's own upload is used from then on
    uint64_t n_runs = 0, n_experts = 0, bytes = 0;
    double ms = 0.0;

    static bool callback(const llama_moe_cache_fill* fills, size_t n, void* ud) {
        return static_cast<ExpertFetch*>(ud)->fetch(fills, n);
    }

    bool fetch(const llama_moe_cache_fill* fills, size_t n) {
        if (disabled) return false;
        const auto t0 = std::chrono::steady_clock::now();
        jobs.clear();
        uint64_t b = 0, ne = 0;
        for (size_t i = 0; i < n; ++i) {
            const llama_moe_cache_fill& f = fills[i];
            auto it = known.find(f.tensor);
            if (it == known.end()) {
                const auto t = files->tensors.find(ggml_get_name(f.tensor));
                if (t == files->tensors.end() || t->second.n_expert <= 0) return false;
                it = known.emplace(f.tensor, &t->second).first;
            }
            const TensorAt& t = *it->second;
            const uint64_t each = t.size / (uint64_t) t.n_expert;
            if (f.dst == nullptr || f.n <= 0 || f.size != each * (uint64_t) f.n || f.expert < 0 ||
                (int64_t) f.expert + f.n > t.n_expert)
                return false;
            jobs.push_back({files->fds[(size_t) t.file], t.offs + each * (uint64_t) f.expert, f.dst, f.size});
            b += f.size;
            ne += (uint64_t) f.n;
        }
        std::string err;
        if (keepalive) keepalive->begin();
        const bool read_ok = pool->run(jobs, err);
        if (keepalive) keepalive->end();
        if (!read_ok) {
            std::fprintf(stderr, "strata metal: reading experts from the model file failed (%s); falling back to "
                                 "the mapped file\n", err.c_str());
            disabled = true;
            return false;
        }
        // once per tensor: what was read is what the mapped file holds (a wrong offset would decode garbage quietly)
        for (size_t i = 0; i < n; ++i) {
            const llama_moe_cache_fill& f = fills[i];
            if (checked.emplace(f.tensor, true).second &&
                std::memcmp(f.dst, f.src, std::min<size_t>(f.size, 4096)) != 0) {
                std::fprintf(stderr, "strata metal: %s: the bytes read do not match the mapped file; falling back to "
                                     "the mapped file\n", ggml_get_name(f.tensor));
                disabled = true;
                return false;
            }
        }
        n_runs += n;
        n_experts += ne;
        bytes += b;
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
};

}   // namespace strata::metal
