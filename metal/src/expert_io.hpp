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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
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
    int kind = -1;          // of an expert tensor: 0 up, 1 gate, 2 down (the order llama.cpp's MoE asks for them)
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
    uint64_t max_expert_tensor = 0; // bytes of the largest expert tensor

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
                    max_expert_tensor = std::max(max_expert_tensor, t.size);
                    t.kind = name.find(".ffn_up_exps.") != std::string::npos ? 0 : name.find(".ffn_gate_exps.") != std::string::npos ? 1
                           : name.find(".ffn_down_exps.") != std::string::npos ? 2 : -1;
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
// A request is a ticket: `run` waits for it (and reads along), `submit` returns at once.  Requests that a layer is
// waiting for are read before the ones submitted ahead of their layer.
class ReadPool {
public:
    struct Job {
        int fd;
        uint64_t offs;
        void* dst;
        size_t size;
        const void* fallback = nullptr;   // the same bytes in the mapped file: copied from there if the read fails
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
        return wait(submit(jobs, false), true, &err);
    }

    // starts the reads; `ahead`: behind every request that is waited for
    uint64_t submit(const std::vector<Job>& jobs, bool ahead) {
        std::lock_guard<std::mutex> lk(mu_);
        const uint64_t id = ++last_id_;
        if (jobs.empty()) return id;
        open_[id] = Ticket{jobs.size(), {}};
        auto& q = ahead ? ahead_ : now_;
        for (const Job& j : jobs) q.push_back({j, id, ahead});
        cv_.notify_all();
        return id;
    }

    // true: the ticket's reads are done (and `err` is empty or says what failed).  block = wait, reading along
    bool wait(uint64_t id, bool block, std::string* err = nullptr) {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            const auto it = open_.find(id);
            if (it == open_.end()) return true;
            if (it->second.left == 0) {
                const bool ok = it->second.failed.empty();
                if (err && !ok) *err = it->second.failed;
                open_.erase(it);
                return ok;
            }
            if (!block) return false;
            Item item;
            if (take(item, id)) {
                lk.unlock();
                read(item);
                lk.lock();
            } else {
                done_cv_.wait(lk);
            }
        }
    }

    // drops the ticket's reads that no thread has started.  true: none of its reads is under way either (the ticket
    // is over); false: `wait` says when.  What the ticket has read so far is of no use to the caller.
    bool cancel(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = open_.find(id);
        if (it == open_.end()) return true;
        for (auto q = ahead_.begin(); q != ahead_.end();) {
            if (q->id == id) {
                q = ahead_.erase(q);
                --it->second.left;
            } else {
                ++q;
            }
        }
        if (it->second.left == 0) {
            open_.erase(it);
            return true;
        }
        return false;
    }

    // of the threads, how many may read ahead at once: the reads a layer waits for always find the SSD with room
    void set_ahead_limit(size_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        ahead_limit_ = std::max<size_t>(1, n);
    }

    // reads submitted ahead that no thread has started
    size_t backlog() {
        std::lock_guard<std::mutex> lk(mu_);
        return ahead_.size();
    }

    // every read that was started or queued is over when this returns
    void drain() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return now_.empty() && ahead_.empty() && active_ == 0; });
    }

private:
    struct Item {
        Job job;
        uint64_t id;
        bool ahead = false;
    };
    struct Ticket {
        size_t left;
        std::string failed;
    };

    // mu_ held.  The waiter of ticket `id` reads what a layer waits for, and its own ticket's reads wherever they are
    bool take(Item& item, uint64_t id) {
        item.ahead = false;
        if (!now_.empty()) {
            item = now_.front();
            now_.pop_front();
        } else {
            auto it = ahead_.begin();
            while (it != ahead_.end() && id != 0 && it->id != id) ++it;
            if (it == ahead_.end() || (id == 0 && active_ahead_ >= ahead_limit_)) return false;
            item = *it;
            item.ahead = true;
            ahead_.erase(it);
            ++active_ahead_;
        }
        ++active_;
        return true;
    }

    void read(const Item& item) {
        const Job& j = item.job;
        std::string e;
        size_t got = 0;
        if (j.fd < 0) {   // no file: the bytes are in memory already (a tensor that was read ahead)
            std::memcpy(j.dst, j.fallback, j.size);
            got = j.size;
        }
        while (got < j.size) {
            const ssize_t r = ::pread(j.fd, (char*) j.dst + got, j.size - got, (off_t) (j.offs + got));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) {
                e = r < 0 ? std::string("read failed: ") + std::strerror(errno) : "the file ended inside a tensor";
                break;
            }
            got += (size_t) r;
        }
        if (!e.empty() && j.fallback != nullptr && j.fd >= 0) {
            std::memcpy(j.dst, j.fallback, j.size);
            e.clear();
        }
        std::lock_guard<std::mutex> lk(mu_);
        --active_;
        if (item.ahead) {
            --active_ahead_;
            cv_.notify_one();   // a thread that the limit kept from reading ahead
        }
        const auto it = open_.find(item.id);
        if (it != open_.end()) {
            if (!e.empty() && it->second.failed.empty()) it->second.failed = e;
            --it->second.left;
        }
        done_cv_.notify_all();
    }

    void loop() {
        for (;;) {
            Item item;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !now_.empty() || (!ahead_.empty() && active_ahead_ < ahead_limit_); });
                if (quit_) return;
                if (!take(item, 0)) continue;
            }
            read(item);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Item> now_, ahead_;
    std::unordered_map<uint64_t, Ticket> open_;
    uint64_t last_id_ = 0;
    size_t active_ = 0, active_ahead_ = 0, ahead_limit_ = 4;
    bool quit_ = false;
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

    // ---- STRATA_VERIFY_CACHE=1 (tests): every slot a layer is about to compute from is compared with the file ----
    std::vector<uint8_t> verify_buf;
    uint64_t n_verified = 0;

    static bool callback_verify(const llama_moe_cache_fill* fills, size_t n, void* ud) {
        return static_cast<ExpertFetch*>(ud)->verify(fills, n);
    }

    bool verify(const llama_moe_cache_fill* fills, size_t n) {
        std::vector<ReadPool::Job> vj;
        size_t total = 0;
        for (size_t i = 0; i < n; ++i) total += fills[i].size;
        verify_buf.resize(total);
        size_t at = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto t = files->tensors.find(ggml_get_name(fills[i].tensor));
            if (t == files->tensors.end() || t->second.n_expert <= 0) return false;
            const uint64_t each = t->second.size / (uint64_t) t->second.n_expert;
            vj.push_back({files->fds[(size_t) t->second.file], t->second.offs + each * (uint64_t) fills[i].expert,
                          verify_buf.data() + at, fills[i].size});
            at += fills[i].size;
        }
        std::string err;
        if (!pool->run(vj, err)) return false;
        at = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(verify_buf.data() + at, fills[i].dst, fills[i].size) != 0) {
                std::fprintf(stderr, "strata metal: VERIFY: %s expert %d: the cache slot differs from the file\n",
                             ggml_get_name(fills[i].tensor), fills[i].expert);
                return false;
            }
            at += fills[i].size;
        }
        n_verified += n;
        return true;
    }

    // ---- reads that are not waited for at once (llama_moe_cache_set_prefetch) ----
    uint64_t n_ahead = 0, bytes_ahead = 0;   // experts and bytes read ahead of their layer
    uint64_t n_cancelled = 0;                // of those, dropped (in part or whole) when their layer routed elsewhere
    size_t max_backlog = 24;                 // reads queued ahead and not started: more than this and a request is declined


    // 0: declined (the cache then reads what it needs now the usual way, and drops what was asked ahead)
    uint64_t fetch_async(const llama_moe_cache_fill* fills, size_t n, bool ahead) {
        if (disabled || (ahead && pool->backlog() > max_backlog)) return 0;
        jobs.clear();
        uint64_t b = 0, ne = 0;
        for (size_t i = 0; i < n; ++i) {
            const llama_moe_cache_fill& f = fills[i];
            const auto it = known.find(f.tensor);
            // only tensors whose place in the file a waited-for read has already proven (`fetch` below)
            if (it == known.end() || checked.find(f.tensor) == checked.end()) return 0;
            const TensorAt& t = *it->second;
            const uint64_t each = t.size / (uint64_t) t.n_expert;
            if (f.dst == nullptr || f.n <= 0 || f.size != each * (uint64_t) f.n || f.expert < 0 ||
                (int64_t) f.expert + f.n > t.n_expert)
                return 0;
            jobs.push_back({files->fds[(size_t) t.file], t.offs + each * (uint64_t) f.expert, f.dst, f.size, f.src});
            b += f.size;
            ne += (uint64_t) f.n;
        }
        n_runs += n;
        n_experts += ne;
        bytes += b;
        if (ahead) {
            n_ahead += ne;
            bytes_ahead += b;
        }
        return pool->submit(jobs, ahead);
    }

    // mode 0: are the ticket's reads over?  1: wait for them.  2: cancel what has not started
    bool fetch_wait(uint64_t ticket, int mode) {
        if (mode == 2) {
            ++n_cancelled;
            return pool->cancel(ticket);
        }
        if (mode == 0) return pool->wait(ticket, false);
        const auto t0 = std::chrono::steady_clock::now();
        if (keepalive) keepalive->begin();
        pool->wait(ticket, true);   // a failed read was filled from the mapped file
        if (keepalive) keepalive->end();
        ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }

    // ---- long prompts: the next expert tensors are read while the GPU computes with this one ----
    //
    // A pass of a long prompt routes to nearly all of a layer's experts, and llama.cpp asks for them one tensor at
    // a time (up, gate, down), the GPU idle while each is read.  With read-ahead on, the tensors that come next
    // are read into a staging buffer meanwhile (up to a layer's three), and copied from there when they are asked
    // for.  The same layer's next tensors need the same experts.  The next layer's are not known before it routes,
    // so they are guessed: the experts the last long pass used in that layer (a pass of 1,024 tokens was measured to
    // use 68-94% of a layer's 512, much the same ones from pass to pass), or, before there was one, the layer's
    // most used experts by Strata's expert profile, as many as this layer used.  What the guess misses is read
    // when it is asked for; what it reads in vain costs the SSD's time and nothing else.
    struct Staged {
        const TensorAt* t = nullptr;                      // the tensor that is (being) read into the slot
        uint64_t ticket = 0;
        std::vector<std::pair<int32_t, int32_t>> runs;    // its experts in the slot: (first, count), each at its own offset
        std::vector<int32_t> pads;                        // experts of which only the first bytes are there
        std::vector<uint8_t> have;                        // per expert: in the slot
    };
    struct Staging {
        uint8_t* buf = nullptr;
        size_t cap = 0;
        size_t offs[3] = {0, 0, 0};                       // a slot per kind of tensor (up, gate, down), or one for all
        int n_slots = 0;
        Staged slot[3];
    } staging;
    llama_context* ctx = nullptr;                         // for the expert cache's contents
    bool wire_staging = false;                            // the staging buffer is wired when it is made: taken at once, never compressed
    bool read_ahead = false;                              // the engine allows it for this prompt
    bool more_passes = false;                             // the engine: another long pass follows the one under way
    double ahead_from = 0.5;                              // the next layer is read ahead if this share of this layer's experts was used
    std::vector<std::vector<uint8_t>> last_used;          // per layer: the experts the last long pass asked for
    std::vector<std::vector<int32_t>> by_profile;         // per layer: its experts, most used first (the engine sets it)
    std::vector<const TensorAt*> bulk_order;              // the expert tensors in the order a pass asks for them
    std::unordered_map<const TensorAt*, const TensorAt*> next_seen;   // ... as seen
    const TensorAt* last_bulk = nullptr;
    uint64_t bytes_staged = 0, bytes_from_stage = 0;
    static constexpr size_t kPiece = 8u << 20;            // reads are cut to this: a tensor is then read by all threads
    static constexpr size_t kPad = 512;                   // llama.cpp pads a run of experts with the start of the next one

    void pieces(std::vector<ReadPool::Job>& out, int fd, uint64_t offs, void* dst, size_t size, const void* src) const {
        for (size_t at = 0; at < size; at += kPiece)
            out.push_back({fd, offs + at, (char*) dst + at, std::min(kPiece, size - at), src ? (const char*) src + at : nullptr});
    }

    int slot_of(const TensorAt& t) const { return staging.n_slots == 3 ? t.kind : 0; }

    void drop(Staged& st) {
        if (st.ticket != 0) {
            pool->cancel(st.ticket);
            pool->wait(st.ticket, true);
        }
        st = Staged{};
    }

    // The engine calls this before a long prompt's passes.  `whole_layer`: room for a layer's three tensors (they
    // are then all read while the GPU is at the layer before), else for one.  false: no buffer
    bool begin_bulk(bool whole_layer) {
        end_bulk();
        if (bulk_order.empty()) {
            std::vector<std::pair<int, const TensorAt*>> v;
            for (const auto& [name, t] : files->tensors)
                if (t.n_expert > 0 && t.layer >= 0 && t.kind >= 0)   // llama.cpp's graph asks for gate, up, down
                    v.push_back({t.layer * 4 + (t.kind == 1 ? 0 : t.kind == 0 ? 1 : 2), &t});
            std::sort(v.begin(), v.end());
            for (const auto& x : v) bulk_order.push_back(x.second);
        }
        size_t most[3] = {0, 0, 0};
        for (const TensorAt* t : bulk_order) most[t->kind] = std::max<size_t>(most[t->kind], t->size);
        staging.n_slots = whole_layer ? 3 : 1;
        if (whole_layer) {
            staging.offs[0] = 0;
            staging.offs[1] = most[0];
            staging.offs[2] = most[0] + most[1];
            staging.cap = most[0] + most[1] + most[2];
        } else {
            staging.cap = std::max({most[0], most[1], most[2]});
        }
        void* p = ::mmap(nullptr, staging.cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) {
            staging.cap = 0;
            staging.n_slots = 0;
            return false;
        }
        staging.buf = (uint8_t*) p;
        // (munmap ends the wiring; if macOS refuses it, the pages are at least written to)
        if (wire_staging && ::mlock(p, staging.cap) != 0) std::memset(p, 0, staging.cap);
        read_ahead = true;
        return true;
    }

    size_t staging_bytes(bool whole_layer) const {
        size_t most[3] = {0, 0, 0};
        for (const auto& [name, t] : files->tensors)
            if (t.n_expert > 0 && t.kind >= 0) most[t.kind] = std::max<size_t>(most[t.kind], t.size);
        return whole_layer ? most[0] + most[1] + most[2] : std::max({most[0], most[1], most[2]});
    }

    void end_bulk() {
        read_ahead = false;
        last_bulk = nullptr;
        for (Staged& st : staging.slot) drop(st);
        if (staging.buf != nullptr) ::munmap(staging.buf, staging.cap);
        staging.buf = nullptr;
        staging.cap = 0;
        staging.n_slots = 0;
    }

    const TensorAt* next_bulk(const TensorAt* t) const {
        const auto seen = next_seen.find(t);
        if (seen != next_seen.end()) return seen->second;
        const auto it = std::find(bulk_order.begin(), bulk_order.end(), t);
        if (it == bulk_order.end()) return nullptr;
        return it + 1 != bulk_order.end() ? *(it + 1) : bulk_order.front();
    }

    // the reads of `runs` and `pads` of tensor t into its staging slot
    void stage(const TensorAt& t, std::vector<std::pair<int32_t, int32_t>> runs, std::vector<int32_t> pads) {
        Staged& st = staging.slot[slot_of(t)];
        const uint64_t each = t.size / (uint64_t) t.n_expert;
        uint8_t* base = staging.buf + staging.offs[slot_of(t)];
        const size_t pad = (size_t) std::min<uint64_t>(each, kPad);
        std::vector<ReadPool::Job> js;
        std::vector<bool> pad_done(pads.size(), false);
        for (const auto& [first, count] : runs) {
            size_t size = (size_t) (each * (uint64_t) count);
            for (size_t i = 0; i < pads.size(); ++i)   // the padding behind a run is read with the run
                if (pads[i] == first + count) {
                    size += pad;
                    pad_done[i] = true;
                }
            pieces(js, files->fds[(size_t) t.file], t.offs + each * (uint64_t) first, base + each * (uint64_t) first, size, nullptr);
            bytes_staged += size;
        }
        for (size_t i = 0; i < pads.size(); ++i)
            if (!pad_done[i])
                js.push_back({files->fds[(size_t) t.file], t.offs + each * (uint64_t) pads[i], base + each * (uint64_t) pads[i], pad});
        st.t = &t;
        st.have.assign((size_t) t.n_expert, 0);
        for (const auto& [first, count] : runs)
            for (int32_t e = first; e < first + count; ++e) st.have[(size_t) e] = 1;
        st.runs = std::move(runs);
        st.pads = std::move(pads);
        st.ticket = pool->submit(js, false);
    }

    bool fetch(const llama_moe_cache_fill* fills, size_t n) {
        if (disabled) return false;
        const auto t0 = std::chrono::steady_clock::now();
        jobs.clear();
        uint64_t b = 0, ne = 0;
        // a long prompt's pass asks for one tensor at a time
        const TensorAt* bulk = nullptr;
        Staged* staged = nullptr;
        const uint8_t* staged_base = nullptr;
        uint64_t asked = 0;
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
            // n = 0: the first bytes of an expert (llama.cpp pads the run before it with them)
            if (f.dst == nullptr || f.n < 0 || (f.n > 0 ? f.size != each * (uint64_t) f.n : f.size > each) || f.expert < 0 ||
                (int64_t) f.expert + std::max(f.n, 1) > t.n_expert)
                return false;
            if (i == 0 && read_ahead && f.il < 0 && t.kind >= 0) {
                bulk = &t;
                // what was read ahead and is not this tensor or one of the next three will not be asked for
                const TensorAt* nx[4] = {bulk, nullptr, nullptr, nullptr};
                for (int k = 1; k < 4; ++k) nx[k] = nx[k - 1] ? next_bulk(nx[k - 1]) : nullptr;
                for (Staged& st : staging.slot)
                    if (st.t != nullptr && std::find(nx, nx + 4, st.t) == nx + 4) drop(st);
                Staged& st = staging.slot[slot_of(t)];
                if (st.t == bulk && checked.find(f.tensor) != checked.end()) {
                    pool->wait(st.ticket, true);
                    st.ticket = 0;
                    staged = &st;
                    staged_base = staging.buf + staging.offs[slot_of(t)];
                }
            }
            if (bulk != nullptr && &t != bulk) return false;   // not what the engine was told to expect
            const uint64_t offs = t.offs + each * (uint64_t) f.expert;
            bool done = false;
            if (staged != nullptr && f.n > 0) {
                // the experts of the run that were read ahead are copied, the others read
                for (int32_t e = f.expert; e < f.expert + f.n;) {
                    const bool have = staged->have[(size_t) e] != 0;
                    int32_t c = 1;
                    while (e + c < f.expert + f.n && (staged->have[(size_t) (e + c)] != 0) == have) ++c;
                    char* dst = (char*) f.dst + each * (uint64_t) (e - f.expert);
                    const size_t size = (size_t) (each * (uint64_t) c);
                    if (have) {
                        pieces(jobs, -1, 0, dst, size, staged_base + each * (uint64_t) e);
                        bytes_from_stage += size;
                    } else {
                        pieces(jobs, files->fds[(size_t) t.file], t.offs + each * (uint64_t) e, dst, size, nullptr);
                    }
                    e += c;
                }
                done = true;
            } else if (staged != nullptr) {
                const bool have = std::find(staged->pads.begin(), staged->pads.end(), f.expert) != staged->pads.end() ||
                                  staged->have[(size_t) f.expert] != 0;
                if (have) {
                    std::memcpy(f.dst, staged_base + each * (uint64_t) f.expert, f.size);
                    done = true;
                }
            }
            if (!done && f.n == 0 && !jobs.empty() && jobs.back().fd >= 0 && (char*) jobs.back().dst + jobs.back().size == (char*) f.dst &&
                jobs.back().offs + jobs.back().size == offs) {
                jobs.back().size += f.size;   // the padding behind a run is read with the run
                done = true;
            }
            if (!done) {
                if (bulk != nullptr) pieces(jobs, files->fds[(size_t) t.file], offs, f.dst, f.size, nullptr);
                else jobs.push_back({files->fds[(size_t) t.file], offs, f.dst, f.size});
            }
            if (bulk != nullptr) asked += (uint64_t) f.n;
            b += f.size;
            ne += (uint64_t) f.n;
        }
        std::string err;
        if (keepalive) keepalive->begin();
        const bool read_ok = pool->run(jobs, err);
        if (keepalive) keepalive->end();
        if (bulk != nullptr && read_ok) {
            if (staged != nullptr) *staged = Staged{};
            else if (staging.slot[slot_of(*bulk)].t == bulk) drop(staging.slot[slot_of(*bulk)]);   // read ahead, not used
            if (last_bulk != nullptr && last_bulk != bulk) next_seen[last_bulk] = bulk;
            last_bulk = bulk;
            // the tensors after this one: their reads go on while the GPU computes
            std::vector<std::pair<int32_t, int32_t>> runs;
            std::vector<int32_t> pads;
            for (size_t i = 0; i < n; ++i) {
                if (fills[i].n > 0) runs.push_back({fills[i].expert, fills[i].n});
                else pads.push_back(fills[i].expert);
            }
            if ((size_t) bulk->layer >= last_used.size()) last_used.resize((size_t) bulk->layer + 1);
            std::vector<uint8_t>& lu = last_used[(size_t) bulk->layer];
            lu.assign((size_t) bulk->n_expert, 0);
            for (const auto& [first, count] : runs)
                for (int32_t e = first; e < first + count; ++e) lu[(size_t) e] = 1;
            // the share of this layer's experts the pass used: those asked for and those the cache holds
            int32_t n_exp = 0;
            const int32_t* slots = ctx ? llama_moe_cache_get_slots(ctx, bulk->layer, &n_exp) : nullptr;
            uint64_t cached = 0;
            for (int32_t e = 0; slots != nullptr && e < n_exp; ++e) cached += slots[e] >= 0;
            const double used = (double) (asked + cached) / (double) bulk->n_expert;
            const TensorAt* cur = bulk;
            for (int k = 0; k < staging.n_slots; ++k) {
                const TensorAt* nx = next_bulk(cur);
                if (nx == nullptr || nx == bulk || nx->n_expert != bulk->n_expert) break;
                const size_t room = staging.n_slots == 3 ? (slot_of(*nx) < 2 ? staging.offs[slot_of(*nx) + 1] - staging.offs[slot_of(*nx)]
                                                                               : staging.cap - staging.offs[2])
                                                         : staging.cap;
                if (nx->size > room) break;
                Staged& st = staging.slot[slot_of(*nx)];
                if (st.t == nx) {   // under way already
                    cur = nx;
                    continue;
                }
                if (st.t != nullptr) break;
                if (nx->layer == bulk->layer) {
                    stage(*nx, runs, pads);
                } else {
                    // a pass that may not come, or a layer whose experts cannot be told: nothing is read for it
                    if (nx->layer < bulk->layer ? !more_passes : false) break;
                    if (used < ahead_from) break;
                    // the guess: what the last pass used there, else the layer's most used by the profile
                    const int32_t n2 = (int32_t) nx->n_expert;
                    std::vector<uint8_t> want((size_t) n2, 0);
                    const size_t l2 = (size_t) nx->layer;
                    if (l2 < last_used.size() && last_used[l2].size() == (size_t) n2) {
                        want = last_used[l2];
                    } else if (l2 < by_profile.size() && by_profile[l2].size() == (size_t) n2) {
                        const size_t take = (size_t) (0.9 * used * (double) n2);
                        for (size_t i = 0; i < take && i < by_profile[l2].size(); ++i) want[(size_t) by_profile[l2][i]] = 1;
                    } else break;
                    int32_t n2c = 0;
                    const int32_t* s2 = ctx ? llama_moe_cache_get_slots(ctx, nx->layer, &n2c) : nullptr;
                    std::vector<std::pair<int32_t, int32_t>> guess;
                    for (int32_t e = 0; e < n2;) {
                        if (!want[(size_t) e] || (s2 != nullptr && n2c == n2 && s2[e] >= 0)) {
                            ++e;
                            continue;
                        }
                        int32_t c = 1;
                        while (e + c < n2 && want[(size_t) (e + c)] && !(s2 != nullptr && n2c == n2 && s2[e + c] >= 0)) ++c;
                        guess.push_back({e, c});
                        e += c;
                    }
                    if (guess.empty()) break;
                    stage(*nx, std::move(guess), {});
                }
                cur = nx;
            }
        }
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
