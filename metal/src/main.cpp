// metal/src/main.cpp - `strata` for Apple silicon (Metal).  EXPERIMENTAL, docs/APPLE_METAL.md.
//
// The CUDA engine in src/ splits the model across VRAM, RAM and the SSD.  A Mac has one pool of memory that the CPU
// and the GPU share, so the tiers are drawn differently here:
//
//     GPU (wired, shared memory)  the dense weights, the KV cache and an EXPERT CACHE that takes the rest of what
//                                 Metal lets one process keep resident (recommendedMaxWorkingSetSize)
//     RAM (the OS file cache)     experts the cache does not hold, as long as macOS has free memory for them
//     SSD (read-only)             every other expert, read into its cache slot when a token routes to it
//
// The model graph and its Metal kernels are llama.cpp's (qwen4exp, ggml-metal), built from a pinned commit with the
// patches in metal/patches/.  This file is the engine around them: the memory plan, the start-up fill of the expert
// cache from Strata's expert profile, the read-only SSD tier (expert_io.hpp), and the `--serve` protocol that
// serve/server.py speaks (READY / GEN / T / PP / RESUME / DONE / STOP), so the server, the web app and the APIs are
// the ones the other backends use.
#include "expert_io.hpp"
#include "mtp_head.hpp"
#include "router_predict.hpp"

#include "ggml-backend.h"
#include "ggml-metal.h"
#include "llama.h"

#include <libproc.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using strata::metal::ExpertFetch;
using strata::metal::ModelFiles;
using strata::metal::ReadPool;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMiB = 1024.0 * 1024.0;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Options {
    bool serve = false;
    bool plan_only = false;
    bool load_only = false;              // --load-only: load, print the plan as it came out, exit
    int io_bench = 0;                    // --io-bench N: time N cold expert reads from the SSD, nothing loaded
    std::string native;                  // shard 1 of the GGUF
    std::string profile;                 // data/expert-profile.bin
    int64_t max_context = 8192;
    std::string expert_cache = "auto";   // auto | <experts> | <N>mib
    double expert_cache_cap_mib = 0.0;   // --expert-cache-cap: auto takes no more than this (0: no cap)
    double reserve_gib = 3.0;            // of the memory that is free at start, this much is left free
    double overcommit_gib = 0.0;         // let the expert cache pass Metal's recommended working set by this much
    std::string expert_io = "direct";    // direct: past the OS file cache (F_NOCACHE); cached: through it
    int io_threads = 8;
    int threads = 0;                     // 0: the performance cores
    int chunk = 0;                       // prompt tokens per pass; 0: the most the expert cache serves (up to 32)
    int prefill = 1024;                  // long prompts: tokens per pass, their experts streamed from the SSD (0: off)
    int prefill_from = 33;               // --prefill-from N: stretches of N or more new tokens go in such passes (33: llama.cpp's own limit)
    bool elastic = false;                // --elastic on: the long passes' work memory and the expert cache take turns in one budget
    int elastic_short = 1024;            // --elastic-short N: a stretch of up to N new tokens is read in one pass beside the cache
    double elastic_refill_gib = 0.0;     // --elastic-refill G|all: a cache that comes back reads G GiB of what it held at once (< 0: all)
    bool elastic_refill_given = false;
    bool prefill_ahead = true;           // --prefill-ahead: read the next expert tensors while the GPU computes with this one
    int prefill_ahead_tensors = 3;       // ... a layer's three (STRATA_PREFILL_AHEAD_TENSORS=1: one)
    bool preload = true;
    std::string mtp;                     // --mtp FILE: the MTP head; its drafts are checked by the model, several tokens a pass
    int spec = 3;                        // --spec N: drafts per pass, at most
    float spec_min_p = 0.75f;            // --spec-min-p P: a draft is added only while the head gives it this probability
    int lookup = 0;                      // --lookup N: draft what followed the last N or more tokens the last time they came (0: off)
    int prefetch = 10;                   // --prefetch N: read up to N experts a layer is predicted to route to ahead of it (0: off)
    int prefetch_depth = 1;              // layers ahead
    float prefetch_margin = 0.0f;        // only experts predicted this far above the predicted 10th score
    float prefetch_learn = 0.1f;         // --prefetch-learn R: the prediction's per-expert correction follows what is seen at this rate
    int prefetch_threads = 4;            // of --io-threads, how many may read ahead at once
    std::string kv = "f16";
    std::vector<int64_t> eos_ids = {248044, 248046};
    // one-shot runs (no --serve)
    std::string tokens, text;
    int64_t max_new = 64;
    float temperature = 0.0f;
    uint32_t seed = 0;
    bool verbose = false;
};

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    while (*s == ' ') ++s;
    while (*s != '\0') {
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || errno != 0) {
            err = "a token id is not a number";
            return false;
        }
        out.push_back(v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0' && *s != '\r' && *s != '\n') {
            err = "token ids are separated by commas";
            return false;
        }
        while (*s == ' ' || *s == '\r' || *s == '\n') ++s;
    }
    if (out.empty()) {
        err = "no token ids";
        return false;
    }
    return true;
}

void usage() {
    std::fprintf(stderr,
        "strata (Apple Metal) " STRATA_VERSION "\n"
        "  strata --serve --native MODEL-00001-of-0000N.gguf [options]     the engine behind serve/server.py\n"
        "  strata --native MODEL.gguf --text \"...\" [--max-new N]           one answer, printed\n"
        "  strata --native MODEL.gguf --plan                               the memory plan, nothing loaded\n"
        "  strata --native MODEL.gguf --io-bench N                         time N cold expert reads from the SSD\n"
        "options:\n"
        "  --max-context N          context in tokens (default 8192)\n"
        "  --expert-cache auto|N|Nmib   the expert cache: all Metal allows (auto), N experts, or N MiB\n"
        "  --expert-cache-cap N     with auto: at most N MiB (benchmarks: the same cache in every run)\n"
        "  --expert-profile FILE    Strata's expert ranking (data/expert-profile.bin): the cache starts filled with it\n"
        "  --no-preload             start with an empty expert cache\n"
        "  --prefetch N             read up to N experts each layer is predicted to route to before it is computed\n"
        "                           (default 10, 0 = off); --prefetch-margin M: only those predicted M above the 10th (0);\n"
        "                           --prefetch-learn R: how fast the prediction corrects itself (0.1, 0 = never)\n"
        "  --mtp FILE               the model's MTP head (mtp-<model>.gguf): it drafts tokens that the model checks, several a pass.\n"
        "                           Takes about 3 GiB of the expert cache's memory; --spec N drafts at most (3), --spec-min-p P (0.75)\n"
        "  --lookup N               draft what followed the last N or more tokens where they stood before in the context\n"
        "                           (text that repeats: quotes, code); checked by the model like the MTP head's drafts (0 = off)\n"
        "  --prefill-ahead on|off   long prompts: read the next expert tensors while the GPU computes (default on)\n"
        "  --expert-io direct|cached    read missed experts past the OS file cache (default) or through it\n"
        "  --io-threads N           threads reading experts from the SSD (default 8)\n"
        "  --reserve-gib G          of the memory that is free when the engine starts, leave G GiB free (default 3)\n"
        "  --overcommit-gib G       let the expert cache pass Metal's recommended working set by G GiB\n"
        "  --threads N              CPU threads (default: the performance cores)\n"
        "  --prefill N              long prompts: tokens per pass, experts streamed from the SSD (default 1024, 0 = off)\n"
        "  --chunk N                short prompts: tokens per pass, 1-32 (default: the most the expert cache serves)\n"
        "  --elastic on|off         (experiment, default off) answer with the memory of the --prefill passes added to the expert\n"
        "                           cache, and give it back to them for a long prompt; up to --elastic-short N new tokens (1024)\n"
        "                           are read in one pass beside the cache, in memory that is free at that moment\n"
        "  --elastic-refill G|all   with --elastic: an expert cache that comes back (after a long prompt, or made smaller) first\n"
        "                           reads G GiB of what it held from the SSD (default 0: it fills as its experts are asked for;\n"
        "                           all: everything it held, about a second per 3 GiB before the next token)\n"
        "  --kv f16|q8_0            KV cache type (default f16)\n");
}

// 0 ok, 1 exit quietly, 2 bad usage
int parse_args(int argc, char** argv, Options& o) {
    // flags of the CUDA engine that a config written for it may carry: accepted, not used here
    static const char* const ignored_with_value[] = {
        "--pack", "--ple-gguf", "--kv-resident", "--resident-budget-gib",
        "--ple-io", "--rope-scaling", "--rope-scale", "--pcie-frac", "--lookup-chain", "--mtp-q4", "--mtp-window",
        "--device", "--vram-reserve-mib", "--batch", "--slots", "--batch-groups", "--layer-split", "--prompt-cache-every",
        "--prompt-cache-root", "--aux-cpus", "--expert-profile-save", "--expert-profile-save-every", "--effort-end"};
    static const char* const ignored_flag[] = {"--mmap-experts", "--resident-experts", "--vram-elastic", "--no-prefill-borrow",
                                               "--split-skip-if-fits", "--session-save-reclaim", "--vision"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--serve") o.serve = true;
        else if (a == "--plan") o.plan_only = true;
        else if (a == "--load-only") o.load_only = true;
        else if (a == "--io-bench") o.io_bench = std::max(1, std::atoi(next("--io-bench")));
        else if (a == "--native" || a == "--model" || a == "-m") o.native = next("--native");
        else if (a == "--expert-profile") o.profile = next("--expert-profile");
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--expert-cache") o.expert_cache = next("--expert-cache");
        else if (a == "--expert-cache-cap") o.expert_cache_cap_mib = std::atof(next("--expert-cache-cap"));
        else if (a == "--reserve-gib") o.reserve_gib = std::atof(next("--reserve-gib"));
        else if (a == "--overcommit-gib") o.overcommit_gib = std::atof(next("--overcommit-gib"));
        else if (a == "--expert-io") o.expert_io = next("--expert-io");
        else if (a == "--io-threads") o.io_threads = std::atoi(next("--io-threads"));
        else if (a == "--threads") o.threads = std::atoi(next("--threads"));
        else if (a == "--chunk") o.chunk = std::atoi(next("--chunk"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill = v == "auto" ? 1024 : std::atoi(v.c_str());
        }
        else if (a == "--prefill-from") o.prefill_from = std::clamp(std::atoi(next("--prefill-from")), 2, 33);
        else if (a == "--prefill-ahead") o.prefill_ahead = std::string(next("--prefill-ahead")) != "off";
        else if (a == "--elastic") o.elastic = std::string(next("--elastic")) != "off";
        else if (a == "--elastic-short") o.elastic_short = std::max(0, std::atoi(next("--elastic-short")));
        else if (a == "--elastic-refill") {
            const std::string v = next("--elastic-refill");
            o.elastic_refill_gib = v == "all" ? -1.0 : std::max(0.0, std::atof(v.c_str()));
            o.elastic_refill_given = true;
        }
        else if (a == "--no-preload") o.preload = false;
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--spec") o.spec = std::clamp(std::atoi(next("--spec")), 0, 8);
        else if (a == "--spec-min-p") o.spec_min_p = (float) std::atof(next("--spec-min-p"));
        else if (a == "--lookup") o.lookup = std::clamp(std::atoi(next("--lookup")), 0, 64);
        else if (a == "--prefetch") o.prefetch = std::clamp(std::atoi(next("--prefetch")), 0, 64);
        else if (a == "--prefetch-margin") o.prefetch_margin = (float) std::atof(next("--prefetch-margin"));
        else if (a == "--prefetch-learn") o.prefetch_learn = (float) std::atof(next("--prefetch-learn"));
        else if (a == "--prefetch-threads") o.prefetch_threads = std::clamp(std::atoi(next("--prefetch-threads")), 1, 64);
        else if (a == "--prefetch-depth") o.prefetch_depth = std::clamp(std::atoi(next("--prefetch-depth")), 1, 4);
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--tokens") o.tokens = next("--tokens");
        else if (a == "--text") o.text = next("--text");
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--seed") o.seed = (uint32_t) std::strtoul(next("--seed"), nullptr, 10);
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) {
                std::fprintf(stderr, "--eos-ids: %s\n", e.c_str());
                return 2;
            }
        } else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h") {
            usage();
            return 1;
        } else {
            bool known = false;
            for (const char* f : ignored_with_value)
                if (a == f) {
                    known = true;
                    std::fprintf(stderr, "strata metal: %s %s is not used by the Metal engine\n", f, next(f));
                }
            for (const char* f : ignored_flag)
                if (a == f) {
                    known = true;
                    std::fprintf(stderr, "strata metal: %s is not used by the Metal engine\n", f);
                }
            if (!known) {
                std::fprintf(stderr, "strata: unknown option %s\n", a.c_str());
                usage();
                return 2;
            }
        }
    }
    if (o.native.empty()) {
        std::fprintf(stderr, "strata: --native MODEL.gguf is required\n");
        return 2;
    }
    if (o.expert_io != "cached" && o.expert_io != "direct") {
        std::fprintf(stderr, "strata: --expert-io is cached or direct\n");
        return 2;
    }
    if (o.kv != "f16" && o.kv != "q8_0" && o.kv != "f32") {
        // the CUDA engine's own spellings (q4_0, k8v4, int8) have no counterpart here
        std::fprintf(stderr, "strata metal: --kv %s is not available on Metal, using f16\n", o.kv.c_str());
        o.kv = "f16";
    }
    o.chunk = std::clamp(o.chunk, 0, 32);
    o.prefill = o.prefill < 33 ? 0 : std::min(o.prefill, 8192);
    o.io_threads = std::clamp(o.io_threads, 1, 64);
    if (o.max_context < 256) o.max_context = 256;
    return 0;
}

uint64_t sysctl_u64(const char* name) {
    uint64_t v64 = 0;
    size_t n = sizeof v64;
    if (sysctlbyname(name, &v64, &n, nullptr, 0) != 0) return 0;
    if (n == sizeof(uint32_t)) {
        uint32_t v32;
        std::memcpy(&v32, &v64, sizeof v32);
        return v32;
    }
    return v64;
}

// What the engine can wire without making macOS compress or swap other programs: the free pages and the file cache
// (clean pages that are dropped, not written).  `apps` is what the programs that are open hold (anonymous memory and
// what the compressor stores of it): memory the engine must not take.
struct HostMemory {
    uint64_t available = 0, apps = 0, wired = 0;
    uint64_t swapped = 0;   // what macOS has in swap already: memory of open programs that will want to come back
    uint64_t swapouts = 0;  // pages macOS has written to swap since it started
    uint64_t compressed = 0;   // bytes of programs' memory macOS has compressed since it started (it does when it runs short)

    // what the engine may take: memory in swap is counted as owed to the programs it belongs to
    double usable() const { return std::max(0.0, (double) available - (double) swapped); }
};

HostMemory host_memory() {
    HostMemory m;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n) != KERN_SUCCESS) return m;
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    m.available = ((uint64_t) vm.free_count + vm.external_page_count + vm.purgeable_count) * page;
    m.apps = ((uint64_t) vm.internal_page_count + vm.compressor_page_count) * page;
    m.wired = (uint64_t) vm.wire_count * page;
    m.swapouts = vm.swapouts;
    m.compressed = vm.compressions * page;
    xsw_usage sw{};
    size_t len = sizeof sw;
    if (sysctlbyname("vm.swapusage", &sw, &len, nullptr, 0) == 0) m.swapped = sw.xsu_used;
    return m;
}

// this process's own memory (what Activity Monitor shows as its Memory)
uint64_t own_footprint() {
    task_vm_info_data_t ti{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &n) != KERN_SUCCESS) return 0;
    return ti.phys_footprint;
}

// how much of this process's memory macOS holds compressed (or in swap) instead of in RAM
uint64_t own_compressed() {
    task_vm_info_data_t ti{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &n) != KERN_SUCCESS) return 0;
    return ti.compressed;
}

// bytes this process has read from disk so far
uint64_t disk_bytes_read() {
    rusage_info_v2 ri{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t*) &ri) != 0) return 0;
    return ri.ri_diskio_bytesread;
}

// Strata's expert profile: every (layer, expert) pair, most used first (tools/make_profile.py)
bool read_profile(const std::string& path, std::vector<int32_t>& layers, std::vector<int32_t>& experts, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    char magic[4];
    uint32_t h[5];
    f.read(magic, 4);
    f.read((char*) h, sizeof h);
    if (!f || std::memcmp(magic, "STRP", 4) != 0) {
        err = path + " is not a Strata expert profile";
        return false;
    }
    const uint32_t n = h[4];
    std::vector<uint16_t> pairs((size_t) n * 2);
    f.read((char*) pairs.data(), (std::streamsize) (pairs.size() * 2));
    if (!f) {
        err = path + " is cut short";
        return false;
    }
    layers.resize(n);
    experts.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        layers[i] = pairs[(size_t) i * 2];
        experts[i] = pairs[(size_t) i * 2 + 1];
    }
    return true;
}

// stdin on its own thread, so a STOP is seen while a request runs
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    void start() {
        std::thread([this] {
            std::string buf, l;
            char chunk[1 << 16];
            for (;;) {
                size_t nl;
                while ((nl = buf.find('\n')) != std::string::npos) {
                    l.assign(buf, 0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP") {
                        stop.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    lines.push_back(l);
                    cv.notify_one();
                }
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
            }
            // nobody is left to read the answer: a request that is under way ends at its next pass
            stop.store(true);
            std::lock_guard<std::mutex> lk(mu);
            if (!buf.empty()) lines.push_back(buf);
            eof = true;
            cv.notify_one();
        }).detach();
    }

    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int penalty_last_n = 0;
    uint32_t seed = 0;
};

llama_sampler* make_sampler(const Sampling& s, int32_t n_vocab) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (s.penalty_last_n != 0 && (s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f))
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, s.penalty_last_n, s.penalty_repeat, s.penalty_freq,
                                                                    s.penalty_present));
    if (s.temperature <= 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return chain;
    }
    if (s.top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.top_k));
    if (s.top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.top_p, 1));
    if (s.min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(s.min_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(s.seed != 0 ? s.seed : LLAMA_DEFAULT_SEED));
    return chain;
}

// A point of the conversation the engine can return to: the recurrent (DeltaNet) state after `tokens`.  The
// attention's KV needs no copy - its cells past the point are dropped when the engine goes back.
struct Checkpoint {
    std::vector<llama_token> tokens;
    std::vector<uint8_t> state;
};

struct Engine {
    Options o;
    ModelFiles files;
    bool tidy_ = true;   // what the model leaves in the OS file cache is given back (STRATA_FILE_CACHE=keep: it stays)
    std::unique_ptr<ReadPool> pool;
    std::unique_ptr<strata::metal::GpuKeepalive> keepalive;
    bool keepalive_always = false;        // for the whole request (the default), not only its reads
    ExpertFetch fetch;
    strata::metal::RouterPredict predict;
    std::unique_ptr<strata::metal::MtpHead> mtp;    // --mtp, or STRATA_MTP_PROBE: a measurement (mtp_head.hpp)
    ggml_backend_buffer_type_t gpu_alloc_ = nullptr;
    llama_batch spec_batch{};                        // the token and its drafts, every position's logits asked for
    std::vector<llama_token> drafts_;
    std::vector<float> draft_probs_;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    ggml_backend_dev_t gpu = nullptr;
    int32_t n_vocab = 0;
    llama_token im_start = LLAMA_TOKEN_NULL;
    uint64_t cache_bytes = 0;
    llama_moe_cache_stats cache0{};       // the cache's figures at READY
    std::FILE* route_trace = nullptr;
    int n_used_ = 10;                     // experts a token routes to in a layer

    // --elastic.  A long prompt's passes need gigabytes of work memory that answering does not, and answering wants
    // the expert cache, which those passes hardly use.  The context has one of three shapes:
    //   Answer  passes of up to 16 tokens, and all of the budget in the cache
    //   Long    --prefill passes, and a token cache that only carries the engine's reads: for a long prompt
    //   Short   passes of up to --elastic-short tokens beside the whole cache: for a stretch that one such pass reads,
    //           in memory that is free at that moment (as the read-ahead buffer is), given back after the pass
    // llama_context_resize changes the shape; the KV cache and the recurrent state stay as they are.
    // (Answer passes stay below the 32 tokens from which llama.cpp computes a layer the cache does not serve on the GPU,
    // from copies: the shape's work memory is then the same with and without a cache, and can be measured without one.)
    enum class Shape { Answer, Short, Long };
    static constexpr int kAnswerBatch = 16;
    Shape shape_ = Shape::Answer;
    bool broken_ = false;                 // a change of shape failed: the context cannot compute
    uint64_t cache_bulk_bytes_ = 0;
    uint64_t cache_plan_bytes_ = 0;       // the answering cache as planned at the start: it never grows past this
    int short_batch_ = 0;                 // the Short shape's passes (0: there is none)
    int long_batch_ = 0;                  // the Long shape's passes: --prefill, or the largest half of it the memory has room for
    double short_work_ = 0.0, short_gpu_ = 0.0;   // what it takes on top of the Answer shape: in all, and on the GPU
    double answer_work_ = 0.0;            // the Answer shape's work memory (GPU and host)
    uint64_t answer_footprint_ = 0;       // this process's memory in the Answer shape, when it was last left for the Long one
    uint64_t answer_cache_ = 0;           // ... and the cache it had then
    double long_work_ = 0.0;              // the Long shape's work memory (GPU and host), without its cache and read-ahead
    // The Long shape and the Answer shape hold the same amount of memory, and a change between them takes what it
    // frees again within the second.  On a desktop whose other programs sit compressed, memory that is free for a
    // few seconds is theirs (a trace showed them take 0.3-0.5 GiB a second), and getting it back squeezes them at
    // once, which is when macOS swaps: 41 MiB over five long prompts while the cache's memory was simply given back,
    // 0.5 MiB when only the idle part of it was held.  So: the long passes' work memory and read-ahead buffer are
    // written to when they are made, the cache is cleared when it is made, and what the long passes do not need of
    // the cache's memory is held in this buffer.
    ggml_backend_buffer_t ballast_ = nullptr;
    // --elastic: the KV cache takes its memory as the conversation grows, kv_step_ cells at a time (llama.cpp:
    // LLAMA_KV_LAZY_CELLS), and the plan gives what it has not taken to the expert cache.  Before the context grows
    // past the cells that are paid for, the cache gives the memory for the next ones back (make_room_for_kv).
    // Measured for this model at 131,072 context: 28.6 KiB a cell, 3.3 GiB that a conversation of up to 16,000
    // tokens leaves to the cache (10.6 GiB instead of 7.1).
    uint32_t kv_step_ = 0;                // 0: the KV cache has all of its memory from the start
    uint64_t kv_cell_bytes_ = 0;
    uint32_t kv_paid_ = 0;                // cells the KV cache has its memory for
    uint64_t kv_slack_ = 0;               // of what it has not taken, what the cache did not use (a capped cache)
    uint32_t kv_due_ = 0;                 // cells that are paid for and not mapped yet (kv_take)
    struct KvNote {
        int64_t tokens;
        uint64_t grow, was;
        bool emptied;         // the cache was there and was made anew, smaller
        int64_t filled;       // ... and so many experts were read back into it
        double ms;
    } kv_note_{};
    uint64_t swapouts_seen_ = 0;          // macOS's count of pages written to swap, at the last look
    uint64_t compressed_seen_ = 0;        // ... and of the bytes it compressed
    llama_moe_cache_stats stats_gone_{};  // what the caches that were given up had counted
    std::vector<int32_t> profile_layers_, profile_experts_;   // Strata's expert profile, most used first
    std::vector<int32_t> kept_layers_, kept_experts_;         // what the answering cache held when it was given up
    bool verify_cache_ = false;
    double reshape_ms_ = 0;

    std::vector<llama_token> live;        // what the context holds
    std::deque<Checkpoint> checkpoints;   // oldest first
    static constexpr size_t kMaxCheckpoints = 6;

    ~Engine() {
        if (ballast_) ggml_backend_buffer_free(ballast_);
        mtp.reset();
        if (spec_batch.token) llama_batch_free(spec_batch);
        if (pool) pool->drain();   // reads under way write into the context's buffers
        if (route_trace) std::fclose(route_trace);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    int used_count() const {
        char buf[32] = {0};
        if (llama_model_meta_val_str(model, "qwen4exp.expert_used_count", buf, sizeof buf) <= 0) return 10;
        return std::max(1, std::atoi(buf));
    }

    bool eos(llama_token t) const {
        return llama_vocab_is_eog(vocab, t) || std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) t) != o.eos_ids.end();
    }

    llama_context_params ctx_params(uint64_t moe_cache, int batch = 0) const {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) o.max_context;
        cp.n_batch = (uint32_t) (batch > 0 ? batch : std::max(32, o.prefill));   // the expert cache serves passes of up to 32 tokens
        cp.n_ubatch = cp.n_batch;
        // Normal passes return only the last token's logits. Speculative verification needs every draft's logits.
        // The llama default reserves vocabulary-sized output for every prefill token, even though we never ask for it.
        cp.n_outputs_max = drafting() ? (uint32_t) o.spec + 1 : 1;
        cp.n_seq_max = 1;
        cp.n_threads = o.threads;
        cp.n_threads_batch = o.threads;
        cp.moe_cache_size = (size_t) moe_cache;
        // a rejected draft takes the recurrent layers' state back: one snapshot per draft of a pass
        cp.n_rs_seq = drafting() ? (uint32_t) o.spec : 0;
        cp.type_k = cp.type_v = o.kv == "q8_0" ? GGML_TYPE_Q8_0 : o.kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.no_perf = true;
        // experiment: STRATA_DUMP_ROUTER=FILE writes every layer's router input (the MoE's input vector) of one-token passes
        if (const char* dp = std::getenv("STRATA_DUMP_ROUTER")) {
            static std::FILE* f = std::fopen(dp, "wb");
            cp.cb_eval_user_data = f;
            cp.cb_eval = [](ggml_tensor* t, bool ask, void* ud) -> bool {
                const bool want = std::strncmp(t->name, "ffn_moe_logits-", 15) == 0 && t->ne[1] == 1;
                if (ask || !want || ud == nullptr) return ask ? want : true;
                const ggml_tensor* x = t->src[1];
                std::vector<float> buf((size_t) ggml_nelements(x));
                ggml_backend_tensor_get(x, buf.data(), 0, ggml_nbytes(x));
                const int32_t h[2] = {std::atoi(t->name + 15), (int32_t) buf.size()};
                std::fwrite(h, sizeof(int32_t), 2, (std::FILE*) ud);
                std::fwrite(buf.data(), sizeof(float), buf.size(), (std::FILE*) ud);
                return true;
            };
        }
        return cp;
    }

    void gpu_memory(uint64_t& used, uint64_t& total) const {
        size_t fr = 0, tot = 0;
        if (gpu) ggml_backend_dev_memory(gpu, &fr, &tot);
        total = tot;
        used = tot > fr ? tot - fr : 0;
    }

    // the cache's figures; with --elastic the counters go on over the caches that were given up
    bool cache_stats(llama_moe_cache_stats& st) const {
        if (!llama_moe_cache_get_stats(ctx, &st)) return false;
        st.hits += stats_gone_.hits;
        st.misses += stats_gone_.misses;
        st.bytes_uploaded += stats_gone_.bytes_uploaded;
        st.t_sync_us += stats_gone_.t_sync_us;
        st.t_fetch_us += stats_gone_.t_fetch_us;
        st.n_lookups += stats_gone_.n_lookups;
        st.prefetched += stats_gone_.prefetched;
        st.prefetch_used += stats_gone_.prefetch_used;
        st.prefetch_waits += stats_gone_.prefetch_waits;
        return true;
    }

    // the engine's side of the expert cache the context has now: its reads, its routing trace, its read-ahead
    void attach_cache() {
        llama_moe_cache_set_fetch_callback(ctx, ExpertFetch::callback, &fetch);
        if (verify_cache_) llama_moe_cache_set_verify_callback(ctx, ExpertFetch::callback_verify, &fetch);
        llama_moe_cache_set_route_callback(
            ctx,
            [](int32_t il, const int32_t* ids, size_t n, void* ud) {
                Engine* e = static_cast<Engine*>(ud);
                if (e->route_trace) {
                    const int32_t h[2] = {il, (int32_t) n};
                    std::fwrite(h, sizeof(int32_t), 2, e->route_trace);
                    std::fwrite(ids, sizeof(int32_t), n, e->route_trace);
                }
                // the prediction learns from what a one-token pass routed to (router_predict.hpp)
                if (e->o.prefetch > 0 && e->o.prefetch_learn > 0.0f && n == (size_t) e->n_used_) e->predict.saw(il, ids, n);
            },
            this);
        if (o.prefetch > 0)
            llama_moe_cache_set_prefetch(
                ctx,
                [](int32_t il, const float* x, int64_t ne, int64_t nt, int32_t* l, int32_t* e, size_t cap, void* ud) {
                    return static_cast<Engine*>(ud)->predict.predict(il, x, ne, nt, l, e, cap);
                },
                [](const llama_moe_cache_fill* f, size_t n, bool ahead, void* ud) {
                    return static_cast<Engine*>(ud)->fetch.fetch_async(f, n, ahead);
                },
                [](uint64_t t, int mode, void* ud) { return static_cast<Engine*>(ud)->fetch.fetch_wait(t, mode); },
                this);
    }

    static const char* shape_name(Shape x) { return x == Shape::Long ? "long-prompt" : x == Shape::Short ? "short-prompt" : "answering"; }

    // --elastic: is the memory the Short shape adds free now (with 1.5 GiB left over, and within Metal's working set)?
    bool can_borrow() const {
        if (short_batch_ <= 0) return false;
        uint64_t used = 0, ws = 0;
        gpu_memory(used, ws);
        return (double) host_memory().available >= short_work_ + 1.5 * kGiB &&
               (double) used + short_gpu_ <= (double) ws + o.overcommit_gib * kGiB;
    }

    // which experts the cache holds now
    void remember_cache() {
        kept_layers_.clear();
        kept_experts_.clear();
        for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
            int32_t ne = 0;
            const int32_t* slots = llama_moe_cache_get_slots(ctx, il, &ne);
            for (int32_t e = 0; slots != nullptr && e < ne; ++e)
                if (slots[e] >= 0) {
                    kept_layers_.push_back(il);
                    kept_experts_.push_back(e);
                }
        }
    }

    // Reads back into a new cache what the last one held, most used first by the profile (a cache that comes back
    // smaller keeps the best of it), then the rest of the profile (one that comes back larger is filled up).
    // --elastic-refill G: the first G GiB of that list, and by default none of it: the cache fills as its experts
    // are asked for.  (At first all of it was read back, at 3 GiB/s, before the next token.)  Measured on the M4 Air,
    // IQ3_S, a 2,000-token prompt, the same 6.99 GiB cache in every run, two runs each: the first token came after
    // 24.2 s with all of it read back and after 22.4-23.1 s with none, the 50th after 31.3 s against 29.8-30.6.  A
    // pass that finds the cache empty reads the experts it needs and no others, and the next tokens need mostly the
    // same ones: the 49 tokens after the first took 0.4 s longer than with the cache read back, which takes 2.2 s.
    // Reading back the first 1 or 2 GiB returned 0.13 and 0.24 s over those tokens for the 0.3 and 0.65 s it took.
    // What is read back and not used does not stay either: at 55 misses a token that cache (3,576 experts) is
    // replaced within 65 tokens, and a second and third turn of the conversation had the same misses whether it
    // had been read back or not (7.6 GiB cache: 6,080 and 6,426 against 6,080 and 6,429).
    int64_t refill_cache() {
        if (!o.preload || o.elastic_refill_gib == 0.0) return 0;
        const size_t ne = (size_t) std::max<int64_t>(1, files.n_expert);
        std::vector<uint8_t> held((size_t) llama_model_n_layer(model) * ne, 0);
        for (size_t i = 0; i < kept_layers_.size(); ++i) held[(size_t) kept_layers_[i] * ne + (size_t) kept_experts_[i]] = 1;
        std::vector<int32_t> l, e;
        const auto take = [&](uint8_t which, uint8_t then) {
            for (size_t i = 0; i < profile_layers_.size(); ++i) {
                const size_t k = (size_t) profile_layers_[i] * ne + (size_t) profile_experts_[i];
                if (profile_layers_[i] < 0 || profile_experts_[i] < 0 || (size_t) profile_experts_[i] >= ne || k >= held.size()) continue;
                if (held[k] != which) continue;
                held[k] = then;
                l.push_back(profile_layers_[i]);
                e.push_back(profile_experts_[i]);
            }
        };
        take(1, 2);   // held and in the profile
        for (size_t i = 0; i < kept_layers_.size(); ++i)   // held, and unknown to the profile
            if (held[(size_t) kept_layers_[i] * ne + (size_t) kept_experts_[i]] == 1) {
                l.push_back(kept_layers_[i]);
                e.push_back(kept_experts_[i]);
            }
        take(0, 2);   // not held
        size_t n = l.size();
        llama_moe_cache_stats st{};
        if (o.elastic_refill_gib >= 0.0 && llama_moe_cache_get_stats(ctx, &st) && st.n_slots > 0)
            n = std::min(n, (size_t) (o.elastic_refill_gib * kGiB * (double) st.n_slots / (double) st.size));
        return n == 0 ? 0 : llama_moe_cache_preload(ctx, l.data(), e.data(), n);
    }

    // --elastic, before a request: has macOS run short of memory since the last look?  Two signs:
    //   it wrote pages to swap: the plan was too large for what the other programs need now.  The cache gives a
    //     quarter of itself back (1 GiB at most) and stays at that, so the swapping ends after a request or two;
    //   before it comes to that, it compressed other programs' memory and less than half of --reserve-gib is free
    //     now.  The cache gives back what is missing (a quarter of itself at most); it may take it again after the
    //     next long prompt if the memory is free then.
    // (STRATA_ELASTIC_TEST_BACKOFF: tests, as if macOS had swapped before every request; =tight: as if it were short.)
    bool back_off(std::string& err) {
        if (!o.elastic || shape_ != Shape::Answer) return true;
        const HostMemory hm = host_memory();
        const char* test = std::getenv("STRATA_ELASTIC_TEST_BACKOFF");
        const bool test_tight = test != nullptr && std::strcmp(test, "tight") == 0;
        const bool swapped = hm.swapouts > swapouts_seen_ || (test != nullptr && !test_tight);
        const uint64_t pages = hm.swapouts > swapouts_seen_ ? hm.swapouts - swapouts_seen_ : 0;
        const uint64_t squeezed = hm.compressed > compressed_seen_ ? hm.compressed - compressed_seen_ : 0;
        swapouts_seen_ = hm.swapouts;
        compressed_seen_ = hm.compressed;
        const double low = 0.5 * o.reserve_gib * kGiB;
        const bool tight = !swapped && (test_tight || ((double) squeezed >= 256 * kMiB && (double) hm.available < low));
        const uint64_t floor = std::min<uint64_t>((uint64_t) kGiB, cache_plan_bytes_ / 4);
        if ((!swapped && !tight) || cache_bytes <= floor) return true;
        uint64_t give = std::min<uint64_t>((uint64_t) kGiB, cache_bytes / 4);
        if (tight) {   // up to half the reserve and 0.5 GiB more, in steps of 256 MiB
            const double missing = test_tight ? 300 * kMiB : low + 0.5 * kGiB - (double) hm.available;
            give = std::min<uint64_t>(cache_bytes / 4, (uint64_t) std::ceil(missing / (256 * kMiB)) * (uint64_t) (256 * kMiB));
        }
        const uint64_t smaller = std::max(floor, cache_bytes - give);
        pool->drain();
        llama_moe_cache_stats st{};
        if (cache_stats(st)) stats_gone_ = st;
        remember_cache();
        if (!llama_context_resize(ctx, kAnswerBatch, smaller, false)) {
            broken_ = true;
            err = "the expert cache could not be made smaller";
            return false;
        }
        if (swapped)
            std::fprintf(stderr, "strata metal: macOS wrote %llu pages to swap since the last request: the expert cache gives memory "
                                 "back, %.2f -> %.2f GiB (a larger --reserve-gib avoids this)\n", (unsigned long long) pages,
                         (double) cache_bytes / kGiB, (double) smaller / kGiB);
        else
            std::fprintf(stderr, "strata metal: macOS compressed %.2f GiB of other programs' memory since the last request and %.2f "
                                 "GiB are free: the expert cache gives memory back, %.2f -> %.2f GiB\n", (double) squeezed / kGiB,
                         (double) hm.available / kGiB, (double) cache_bytes / kGiB, (double) smaller / kGiB);
        cache_bytes = smaller;
        if (swapped) cache_plan_bytes_ = smaller;
        attach_cache();
        refill_cache();
        return true;
    }

    // --elastic: the context is about to hold `tokens` tokens.  If that is past the cells the KV cache has memory
    // for, the expert cache gives the memory for the next ones first (one budget).  Three cases:
    //   the context is about to take its long-prompt shape (`before_long`): the cache is about to be given up
    //     anyway, so only the books change here - the buffer that will hold the cache's place is made that much
    //     smaller, and kv_take() maps the cells once it is.  (Making that buffer first and a smaller one after it
    //     was tried: Metal lets go of a released buffer's memory a second late, and for that second macOS
    //     compressed 1.4 GiB of other programs' memory.)
    //   the cache is there: it is made anew, smaller, and fills again as its experts are asked for (once per
    //     kv_step_ tokens of a conversation that grows in short turns or in the middle of an answer; with
    //     --elastic-refill it is read back first: 4 s for 11 GiB);
    //   the context has its long-prompt shape already (llama.cpp had to map cells by itself): the books only.
    bool make_room_for_kv(int64_t tokens, bool before_long, std::string& err) {
        if (kv_step_ == 0 || tokens + 256 <= (int64_t) kv_paid_) return true;
        const uint32_t n_ctx = (uint32_t) o.max_context;
        uint32_t want = (uint32_t) std::min<int64_t>(n_ctx, (tokens + 256 + kv_step_ - 1) / kv_step_ * kv_step_);
        want = std::max(want, llama_kv_cells_mapped(ctx, nullptr));   // (what llama.cpp mapped by itself, if it had to)
        if (want <= kv_paid_) return true;
        const auto t0 = Clock::now();
        const uint64_t grow = (uint64_t) (want - kv_paid_) * kv_cell_bytes_;
        const uint64_t from_slack = std::min(grow, kv_slack_);
        kv_slack_ -= from_slack;
        const uint64_t floor = (uint64_t) (512 * kMiB);
        const uint64_t was = cache_bytes;
        const uint64_t give = std::min(grow - from_slack, cache_bytes > floor ? cache_bytes - floor : 0);
        int64_t filled = 0;
        const bool there = !before_long && shape_ != Shape::Long;
        if (give > 0 && there) {
            pool->drain();   // reads under way write into the cache that is given up
            llama_moe_cache_stats st{};
            if (cache_stats(st)) stats_gone_ = st;
            remember_cache();
            const uint32_t batch = shape_ == Shape::Short ? (uint32_t) short_batch_ : (uint32_t) kAnswerBatch;
            if (!llama_context_resize(ctx, batch, cache_bytes - give, false)) {
                broken_ = true;
                err = "the expert cache could not be made smaller for the KV cache";
                return false;
            }
            attach_cache();
            filled = refill_cache();
        } else if (give > 0 && shape_ == Shape::Long) {
            answer_cache_ = answer_cache_ > give ? answer_cache_ - give : 0;
        }
        cache_bytes -= give;
        cache_plan_bytes_ = cache_plan_bytes_ > give ? cache_plan_bytes_ - give : 0;
        kv_note_ = {tokens, grow, was, give > 0 && there, filled, ms_since(t0)};
        if (before_long && shape_ != Shape::Long) {
            kv_due_ = want;
            return true;
        }
        kv_due_ = want;
        return kv_take(err);
    }

    // maps the cells make_room_for_kv made room for
    bool kv_take(std::string& err) {
        if (kv_due_ == 0) return true;
        const auto t0 = Clock::now();
        if (!llama_kv_map_cells(ctx, kv_due_)) {
            err = "the KV cache could not take more memory";
            return false;
        }
        kv_paid_ = kv_due_;
        kv_due_ = 0;
        const double ms = kv_note_.ms + ms_since(t0);
        reshape_ms_ += ms;
        const std::string how = shape_ == Shape::Long ? " when it comes back"
                              : !kv_note_.emptied ? ""
                              : kv_note_.filled > 0 ? " (" + std::to_string(kv_note_.filled) + " experts read back)"
                              : " (it fills again as its experts are asked for)";
        std::fprintf(stderr, "strata metal: the conversation reaches %lld tokens: the KV cache takes its memory for %u (%.2f GiB "
                             "more), the expert cache gives it: %.2f -> %.2f GiB%s, %.2f s\n", (long long) kv_note_.tokens, kv_paid_,
                     (double) kv_note_.grow / kGiB, (double) kv_note_.was / kGiB, (double) cache_bytes / kGiB, how.c_str(), ms / 1000.0);
        return true;
    }

    // --elastic: gives the context another shape.  The cache goes when the Long shape comes; with the Answer shape it
    // comes back empty and fills as its experts are asked for (--elastic-refill: the experts it held are read again
    // from the SSD first, a second per 3 GiB).
    bool reshape(Shape to, std::string& err) {
        if (!o.elastic || to == shape_) return true;
        const auto t0 = Clock::now();
        fetch.end_bulk();   // the read-ahead buffer goes before other memory comes
        const bool cache_goes = to == Shape::Long, cache_comes = shape_ == Shape::Long;
        if (cache_goes || cache_comes) {
            pool->drain();   // reads under way write into the cache that is given up
            llama_moe_cache_stats st{};
            if (cache_stats(st)) stats_gone_ = st;
        }
        if (cache_goes) {
            remember_cache();
            answer_footprint_ = own_footprint();
            answer_cache_ = cache_bytes;
        }
        if (cache_comes) {
            // The cache comes back as it was, plus or minus what is free beyond --reserve-gib now and what this shape
            // holds more or less than the answering shape did (other programs may have grown meanwhile); it may come
            // back larger the next time, up to the plan.
            const double fits = (double) answer_cache_ + host_memory().usable() - o.reserve_gib * kGiB +
                                (double) own_footprint() - (double) answer_footprint_;
            const uint64_t floor = (uint64_t) (512 * kMiB);
            const uint64_t now = std::min(cache_plan_bytes_, (uint64_t) std::max(fits, (double) floor));
            if (now + (uint64_t) (64 * kMiB) < cache_plan_bytes_)
                std::fprintf(stderr, "strata metal: less memory is free than at the start: the expert cache comes back with %.2f "
                                     "GiB of its %.2f\n", (double) now / kGiB, (double) cache_plan_bytes_ / kGiB);
            cache_bytes = now;
            if (ballast_) ggml_backend_buffer_free(ballast_);
            ballast_ = nullptr;
        }
        const uint64_t cache_was = cache_bytes;
        const uint32_t batch = to == Shape::Long ? (uint32_t) long_batch_ : to == Shape::Short ? (uint32_t) short_batch_ : (uint32_t) kAnswerBatch;
        // (the Short shape borrows: its work memory is taken as its one pass needs it, and no more)
        if (!llama_context_resize(ctx, batch, to == Shape::Long ? cache_bulk_bytes_ : cache_bytes, cache_goes || cache_comes)) {
            broken_ = true;
            err = std::string("the context could not be given its ") + shape_name(to) +
                  " shape (is the memory it had taken by other programs now?)";
            return false;
        }
        shape_ = to;
        int64_t filled = 0;
        if (cache_goes || cache_comes) attach_cache();
        if (cache_goes) {
            // what the answering shape held, less what this shape has taken and its read-ahead buffer will
            const double staging = o.prefill_ahead ? (double) fetch.staging_bytes(true) : 0.0;
            const double idle = (double) cache_was + answer_work_ - (double) cache_bulk_bytes_ - long_work_ - staging;
            if (idle >= 64 * kMiB && std::getenv("STRATA_ELASTIC_NO_BALLAST") == nullptr) {
                ballast_ = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(gpu), (size_t) idle);
                if (ballast_) ggml_backend_buffer_clear(ballast_, 0);   // touched: the pages are this process's now
            }
        }
        if (cache_comes) filled = refill_cache();
        const double ms = ms_since(t0);
        reshape_ms_ += ms;
        if (o.verbose) {
            uint64_t used = 0, ws = 0;
            gpu_memory(used, ws);
            std::fprintf(stderr, "strata metal: the context has its %s shape after %.2f s (%lld experts read back): %.1f MiB on "
                                 "the GPU, %.1f MiB in all, %.1f MiB of it compressed\n", shape_name(to), ms / 1000.0, (long long) filled,
                         (double) used / kMiB, (double) own_footprint() / kMiB, (double) own_compressed() / kMiB);
        }
        return true;
    }

    bool load(std::string& err) {
        if (!files.open(o.native, o.expert_io == "direct", err)) return false;
        fetch.files = &files;
        const uint64_t ram = sysctl_u64("hw.memsize");
        if (o.threads <= 0) o.threads = (int) std::max<uint64_t>(1, sysctl_u64("hw.perflevel0.physicalcpu"));
        if (o.threads <= 0) o.threads = 4;

        // The GPU buffers (dense weights, KV, expert cache) are anonymous shared memory that Metal keeps wired only
        // while its residency requests go on; ggml stops them after 3 idle minutes, and macOS would then be free to
        // compress the expert cache or write it to swap.  It is re-readable from the model file, so writing it to
        // the SSD is the one thing this engine must not cause: the requests go on for as long as the engine runs
        // (serve/server.py's "idle_unload_s" ends the engine, which gives the memory back).
        setenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "2000000", 0);
        // llama.cpp computes a layer's routed experts on the GPU, from copies, only in passes of 32 tokens or more;
        // --prefill-from lowers that
        if (o.prefill_from < 33) setenv("GGML_OP_OFFLOAD_MIN_BATCH", std::to_string(o.prefill_from).c_str(), 1);
        llama_backend_init();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(d);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu = d;
                break;
            }
        }
        if (gpu == nullptr) {
            err = "no Metal GPU was found";
            return false;
        }
        uint64_t used0 = 0, ws = 0;
        gpu_memory(used0, ws);

        std::fprintf(stderr,
                     "strata metal: %s, %.1f GiB of unified memory, Metal's working set %.1f GiB\n"
                     "strata metal: model: %zu shard(s), %.1f GiB dense, %.1f GiB of routed experts (%d layers x %lld, "
                     "%.2f MiB each on average), %.1f GiB n-gram table\n",
                     ggml_backend_dev_description(gpu), (double) ram / kGiB, (double) ws / kGiB, files.paths.size(),
                     (double) files.dense_bytes / kGiB, (double) files.expert_bytes / kGiB, files.n_layer,
                     (long long) files.n_expert, files.mean_expert_bytes() / kMiB, (double) files.lazy_bytes / kGiB);
        const HostMemory hm0 = host_memory();
        std::fprintf(stderr, "strata metal: memory now: %.1f GiB free or file cache, %.1f GiB held by other programs, %.1f GiB "
                             "wired, %.1f GiB in swap\n", (double) hm0.available / kGiB, (double) hm0.apps / kGiB,
                     (double) hm0.wired / kGiB, (double) hm0.swapped / kGiB);
        if (o.plan_only) {
            const double gpu_fixed = (double) files.dense_bytes / kGiB + 1.0;
            const double by_metal = (double) ws / kGiB + o.overcommit_gib - gpu_fixed;
            const double by_ram = hm0.usable() / kGiB - o.reserve_gib - gpu_fixed;
            const double cache = std::max(0.0, std::min(by_metal, by_ram));
            std::fprintf(stderr,
                         "strata metal: plan: ~%.1f GiB dense weights + KV on the GPU, ~%.1f GiB expert cache (%.0f%% of "
                         "the experts; Metal has room for %.1f, the free memory for %.1f), the other %.1f GiB of experts "
                         "from the OS file cache and the SSD (read-only)\n",
                         gpu_fixed, cache, files.expert_bytes ? 100.0 * cache * kGiB / (double) files.expert_bytes : 0.0,
                         std::max(0.0, by_metal), std::max(0.0, by_ram),
                         std::max(0.0, (double) files.expert_bytes / kGiB - cache));
            return true;
        }

        // ---- the weights.  The routed experts stay in host memory, mapped from the files (never copied: the
        // mapping is read-only and file-backed, so macOS drops its pages under pressure instead of swapping them).
        // Everything else of the layers goes into GPU buffers of its own.  llama.cpp would map those from the file
        // too, as ONE Metal buffer from the first dense tensor to the last - the whole 55 GB shard with the experts
        // in between, which Metal then has to keep resident.  A buffer type that is not the device's default is
        // allocated and filled instead; the "mapped" type is the one this API hands out.
        if (hm0.usable() < (double) files.dense_bytes + (1.5 + o.reserve_gib) * kGiB) {
            err = "not enough free memory to start: the dense weights and the context need about " +
                  std::to_string((int) std::ceil((double) files.dense_bytes / kGiB + 1.5)) + " GiB and " +
                  std::to_string((int) (hm0.usable() / kGiB)) + " GiB are free. Close other programs first "
                  "(starting anyway would make macOS swap them out)";
            return false;
        }
        static char page[1 << 15] __attribute__((aligned(16384)));
        ggml_backend_buffer_t probe = ggml_backend_dev_buffer_from_host_ptr(gpu, page, 16384, 16384);
        ggml_backend_buffer_type_t gpu_alloc = probe ? ggml_backend_buffer_get_type(probe) : nullptr;
        if (gpu_alloc == nullptr || gpu_alloc == ggml_backend_dev_buffer_type(gpu)) {
            err = "this Metal backend has no separate buffer type for allocated weights";
            return false;
        }
        gpu_alloc_ = gpu_alloc;
        const llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(gate_up|gate|up|down)_exps\\.", ggml_backend_cpu_buffer_type()},
            {"^(blk\\.[0-9]+\\.|output)", gpu_alloc},
            {nullptr, nullptr},
        };
        setenv("LLAMA_MMAP_NO_PREFETCH", "1", 1);   // metal/patches: no read-ahead of a file larger than the RAM
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;          // the n-gram table: rows read on demand
        // no repacked copies: ggml's ARM kernels would take 17.5 GiB of this model's experts (the Q2_0, Q4_K and
        // IQ4_NL ones) into a repacked buffer of their own - anonymous memory, on a 32 GB Mac straight into swap -
        // and the expert cache only serves layers whose experts all stay mapped
        mp.use_extra_bufts = false;
        mp.tensor_buft_overrides = overrides;
        mp.progress_callback = [](float p, void*) {
            static int last = -1;
            const int pct = (int) (p * 100.0f);
            if (pct / 10 != last / 10) std::fprintf(stderr, "strata metal: loading the dense weights: %d%%\n", pct);
            last = pct;
            return true;
        };
        // The loader copies the dense weights out of its mapping of the files: page faults one after another when
        // they are not in the OS file cache (5.8 s for this model's 3.9 GiB), and a second copy of them in the file
        // cache for as long as macOS keeps it (ModelFiles::map).  The kernel reads them ahead of the loader, and
        // once they are on the GPU the file cache gives the model's pages back.
        if (const char* fc = std::getenv("STRATA_FILE_CACHE")) tidy_ = std::strcmp(fc, "keep") != 0;
        if (tidy_) {
            std::string merr;
            if (!files.map(merr)) {
                std::fprintf(stderr, "strata metal: %s; the OS file cache keeps what is read through it\n", merr.c_str());
                tidy_ = false;
            }
        }
        std::thread ahead;
        if (tidy_) ahead = std::thread([this] { for (const auto& t : files.gpu_weights()) files.read_ahead(t.file, t.offs, t.size); });
        const auto t0 = Clock::now();
        model = llama_model_load_from_file(o.native.c_str(), mp);
        if (ahead.joinable()) ahead.join();
        if (tidy_) files.drop_all();
        if (probe) ggml_backend_buffer_free(probe);
        if (model == nullptr) {
            err = "the model could not be loaded (see the lines above)";
            return false;
        }
        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        {
            llama_token t[4];
            if (llama_tokenize(vocab, "<|im_start|>", 12, t, 4, false, true) == 1) im_start = t[0];
        }
        // the MTP head's weights go on the GPU before the expert cache is sized: its memory comes out of the cache's
        uint64_t mtp_bytes = 0;
        {
            uint64_t u0 = 0, u1 = 0;
            gpu_memory(u0, ws);
            const char* probe = std::getenv("STRATA_MTP_PROBE");
            const std::string path = !o.mtp.empty() ? o.mtp : probe ? probe : "";
            if (!path.empty()) {
                mtp = std::make_unique<strata::metal::MtpHead>();
                mtp->use = !o.mtp.empty() && o.spec > 0;
                std::string merr;
                if (!mtp->load_model(path, gpu_alloc, merr)) {
                    if (!o.mtp.empty()) {
                        err = merr;
                        return false;
                    }
                    std::fprintf(stderr, "strata metal: STRATA_MTP_PROBE: %s\n", merr.c_str());
                    mtp.reset();
                }
                gpu_memory(u1, ws);
                mtp_bytes = u1 > u0 ? u1 - u0 : 0;
                if (mtp) std::fprintf(stderr, "strata metal: MTP head: %.2f GiB on the GPU\n", (double) mtp_bytes / kGiB);
            }
        }
        uint64_t used1 = 0;
        gpu_memory(used1, ws);
        // only the dense weights may have become this process's memory; the experts are mapped, not copied
        if ((double) own_footprint() > (double) files.dense_bytes + (double) mtp_bytes + 2.0 * kGiB) {
            err = "the model took " + std::to_string(own_footprint() >> 20) + " MiB of memory while loading, where " +
                  std::to_string(files.dense_bytes >> 20) + " MiB of dense weights were expected: its experts were "
                  "copied instead of mapped. Stopping before the Mac starts to swap";
            return false;
        }
        std::fprintf(stderr, "strata metal: weights loaded in %.1f s, %.2f GiB on the GPU\n", ms_since(t0) / 1000.0,
                     (double) (used1 - used0 - mtp_bytes) / kGiB);

        // (STRATA_ELASTIC_REFILL=all|G: --elastic-refill from the environment where the option is not given; tests)
        if (const char* rf = std::getenv("STRATA_ELASTIC_REFILL"); rf != nullptr && !o.elastic_refill_given)
            o.elastic_refill_gib = std::strcmp(rf, "all") == 0 ? -1.0 : std::max(0.0, std::atof(rf));

        // --elastic: the KV cache takes its memory 16,384 cells at a time, as the conversation reaches them
        // (STRATA_KV_STEP=N: another step; 0: all of it at the start, as without --elastic)
        {
            const char* ks = std::getenv("STRATA_KV_STEP");
            const int step = ks ? std::atoi(ks) : 16384;
            if (o.elastic && o.prefill > 32 && step > 0 && o.max_context > step) setenv("LLAMA_KV_LAZY_CELLS", std::to_string(step).c_str(), 1);
            else unsetenv("LLAMA_KV_LAZY_CELLS");
        }
        // ---- the expert cache: what is left of Metal's working set once the context (KV cache, work buffers) has
        // its memory.  The context is made twice: once without a cache to see what it takes, then with the cache.
        uint64_t want = 0;
        bool fixed = false;
        double elastic_total = 0.0, elastic_metal = 0.0;   // --elastic: what the passes' work memory and the cache share
        double elastic_work = 0.0;                         // ... and the long passes' work memory of it
        double kv_rest = 0.0;                              // memory the KV cache has not taken yet (kv_step_)
        double elastic_answer_gpu = 0.0;                   // the answering shape's work memory on the GPU
        struct LongShape {
            int batch;
            double gpu, host;
        };
        std::vector<LongShape> smaller_long;               // long passes smaller than --prefill, and what they take
        if (o.expert_cache != "auto") {
            const bool mib = o.expert_cache.size() > 3 && o.expert_cache.compare(o.expert_cache.size() - 3, 3, "mib") == 0;
            const double v = std::atof(o.expert_cache.c_str());
            want = (uint64_t) (mib ? v * kMiB : v * files.mean_expert_bytes());
            fixed = true;
        }
        if (files.expert_bytes > 0 && (!fixed || want > 0)) {
            const HostMemory before_context = host_memory();
            ctx = llama_init_from_model(model, ctx_params(0));
            if (ctx == nullptr) {
                err = "the context could not be created (is --max-context too large for this Mac?)";
                return false;
            }
            if (mtp) {   // its context is measured with the model's
                std::string merr;
                if (!mtp->make_context(ctx, ctx_params(0), merr)) {
                    err = merr;
                    return false;
                }
            }
            uint64_t used2 = 0;
            gpu_memory(used2, ws);
            // Sample with the context still alive: Metal may release its physical pages later than llama_free.
            const HostMemory hm = host_memory();
            const double ctx_bytes = (double) (used2 - used1);
            // (with a KV cache that takes its memory as it fills, the context is measured with its first cells only:
            // what the others will take is taken out of every figure here, and --elastic gives it to the cache below)
            size_t kv_per = 0;
            const uint32_t kv_cells = llama_kv_cells_mapped(ctx, &kv_per);
            kv_rest = kv_cells > 0 && (uint32_t) o.max_context > kv_cells ? (double) ((uint32_t) o.max_context - kv_cells) * (double) kv_per : 0.0;
            if (kv_rest > 0.0) {
                kv_step_ = kv_cells;
                kv_cell_bytes_ = kv_per;
                kv_paid_ = kv_cells;
            }
            // The context's compute buffer in host memory (the inputs of a pass: 0.45 GiB for passes of 4,096 tokens)
            // has no pages yet; the passes of a long prompt touch all of it.
            size_t ctx_device = 0, ctx_host = 0;
            llama_get_compute_buffer_sizes(ctx, &ctx_device, &ctx_host);
            const double by_ram = std::min(before_context.usable() - ctx_bytes, hm.usable()) - (double) ctx_host - o.reserve_gib * kGiB - kv_rest;
            if (o.elastic && !mtp && o.prefill > 32) {
                // the other shapes' work memory, measured on this context too: the cache is then allocated once
                size_t dev = 0, host = 0, sdev = 0, shost = 0;
                // (and that of long passes of half and a quarter of --prefill, should the whole not fit)
                for (int b = o.prefill / 2; b >= 1024 && (int) smaller_long.size() < 3; b /= 2) {
                    size_t d2 = 0, h2 = 0;
                    if (!llama_context_resize(ctx, (uint32_t) b, 0, false)) break;
                    llama_get_compute_buffer_sizes(ctx, &d2, &h2);
                    smaller_long.push_back({b, (double) d2, (double) h2});
                }
                short_batch_ = std::min(o.elastic_short, o.prefill);
                if (short_batch_ <= 32 || !llama_context_resize(ctx, (uint32_t) short_batch_, 0, false)) short_batch_ = 0;
                else llama_get_compute_buffer_sizes(ctx, &sdev, &shost);
                if (llama_context_resize(ctx, kAnswerBatch, 0, false)) {
                    llama_get_compute_buffer_sizes(ctx, &dev, &host);
                    answer_work_ = (double) dev + (double) host;
                    short_gpu_ = (double) sdev - (double) dev;
                    short_work_ = short_gpu_ + (double) shost - (double) host;
                    elastic_answer_gpu = (double) dev;
                }
            }
            if (mtp) mtp->free_context();
            llama_free(ctx);
            ctx = nullptr;
            const double margin = 0.25 * kGiB;
            const double by_metal = (double) ws + o.overcommit_gib * kGiB - (double) used2 - margin - kv_rest;
            const double room = std::max(0.0, std::min(by_metal, by_ram));
            if (!fixed) want = (uint64_t) room;
            // benchmarks: the same cache in every run, and still never more than the free memory has room for
            if (!fixed && o.expert_cache_cap_mib > 0.0) want = std::min<uint64_t>(want, (uint64_t) (o.expert_cache_cap_mib * kMiB));
            else if ((double) want > room)
                std::fprintf(stderr, "strata metal: --expert-cache %s is %.1f GiB, more than the %.1f GiB the plan has "
                                     "room for; kept as asked (macOS may compress or swap other programs, or the GPU "
                                     "may refuse it)\n",
                             o.expert_cache.c_str(), (double) want / kGiB, room / kGiB);
            want = std::min<uint64_t>(want, files.expert_bytes + (uint64_t) (64 * kMiB));
            if (o.elastic) {
                // The long-prompt shape: the passes' work memory, a token cache, the read-ahead buffer.  The answering
                // shape: its own (small) work memory and the cache, which gets what is left of the same total.
                cache_bulk_bytes_ = (uint64_t) (256 * kMiB);
                const double staging = o.prefill_ahead ? (double) fetch.staging_bytes(true) : 0.0;
                elastic_work = (double) ctx_device + (double) ctx_host;
                elastic_total = room + elastic_work;
                elastic_metal = by_metal + (double) ctx_device;
                // the long passes are --prefill tokens, or the largest half of that the total has room for
                long_batch_ = o.prefill;
                long_work_ = elastic_work;
                for (size_t i = 0; long_work_ + (double) cache_bulk_bytes_ + staging > elastic_total && i < smaller_long.size(); ++i) {
                    long_batch_ = smaller_long[i].batch;
                    long_work_ = smaller_long[i].gpu + smaller_long[i].host;
                }
                const double bulk_need = long_work_ + (double) cache_bulk_bytes_ + staging;
                if (long_batch_ != o.prefill && elastic_total >= bulk_need)
                    std::fprintf(stderr, "strata metal: --elastic: no room for passes of %d tokens (%.2f GiB with the read-ahead "
                                         "buffer, of %.2f): long prompts are read in passes of %d\n", o.prefill,
                                 (elastic_work + (double) cache_bulk_bytes_ + staging) / kGiB, elastic_total / kGiB, long_batch_);
                short_batch_ = std::min(short_batch_, long_batch_);
                if (mtp || fixed || o.prefill <= 32 || answer_work_ <= 0.0 || elastic_total < bulk_need) {   // (no room: hardly any memory is free)
                    std::fprintf(stderr, "strata metal: --elastic is off: %s\n",
                                 mtp ? "not with the MTP head" : fixed ? "the expert cache has a set size" : o.prefill <= 32
                                     ? "there are no --prefill passes" : answer_work_ <= 0.0 ? "the context could not change its shape"
                                     : "the free memory has no room for the long passes");
                    o.elastic = false;
                }
            }
            std::fprintf(stderr, "strata metal: context %.2f GiB on the GPU (%.2f of it for the passes) and %.2f GiB of host "
                                 "memory; expert cache %.2f GiB (%s; Metal has room for %.1f, the free memory for %.1f)\n",
                         (ctx_bytes + kv_rest) / kGiB, (double) ctx_device / kGiB, (double) ctx_host / kGiB, (double) want / kGiB,
                         fixed ? "as asked" : "auto",
                         std::max(0.0, by_metal) / kGiB, std::max(0.0, by_ram) / kGiB);
            if (!fixed && by_ram < by_metal - 0.5 * kGiB && (double) files.expert_bytes > room)
                std::fprintf(stderr, "strata metal: other programs hold %.1f GiB of memory. The expert cache takes only "
                                     "what is free, so that macOS does not have to swap them out; with programs closed "
                                     "before the start it can be up to %.1f GiB larger (every GiB holds ~%.0f more "
                                     "experts)\n",
                             (double) (hm.apps > own_footprint() ? hm.apps - own_footprint() : 0) / kGiB, (std::min(std::max(0.0, by_metal), (double) files.expert_bytes) - std::max(0.0, by_ram)) / kGiB,
                             files.mean_expert_bytes() > 0 ? kGiB / files.mean_expert_bytes() : 0.0);
        }
        if (elastic_total <= 0.0) o.elastic = false;
        cache_bytes = want;
        if (o.elastic) {
            fetch.wire_staging = true;
            // the answering shape: its work memory (64 MiB more than measured, for the cache's own bookkeeping in the
            // graphs) and the cache, which gets the rest of the total
            const double margin = 64 * kMiB;
            double all = std::min(elastic_total - answer_work_ - margin, elastic_metal - elastic_answer_gpu - margin);
            all = std::min(all, (double) files.expert_bytes + 64 * kMiB);
            if (o.expert_cache_cap_mib > 0.0) all = std::min(all, o.expert_cache_cap_mib * kMiB);
            cache_bytes = (uint64_t) std::max(all, (double) want);
            // what the KV cache has not taken yet goes to the cache (as far as the cache may grow: the rest is kept
            // as slack, used up first when the KV cache takes its memory)
            if (kv_rest > 0.0) {
                double most = (double) files.expert_bytes + 64 * kMiB;
                if (o.expert_cache_cap_mib > 0.0) most = std::min(most, o.expert_cache_cap_mib * kMiB);
                const double more = std::clamp(most - (double) cache_bytes, 0.0, kv_rest);
                cache_bytes += (uint64_t) more;
                kv_slack_ = (uint64_t) (kv_rest - more);
            }
            cache_plan_bytes_ = cache_bytes;
            ctx = llama_init_from_model(model, ctx_params(cache_bytes, kAnswerBatch));
            std::fprintf(stderr, "strata metal: --elastic: answers are written with a %.2f GiB expert cache (%.2f GiB of work "
                                 "memory). A long prompt is read in passes of %d with %.2f GiB of work memory in the cache's "
                                 "place", (double) cache_bytes / kGiB, answer_work_ / kGiB, long_batch_, long_work_ / kGiB);
            if (short_batch_ > 0)
                std::fprintf(stderr, "; up to %d new tokens in one pass beside the cache, where %.2f GiB are free for it",
                             short_batch_, short_work_ / kGiB);
            std::fprintf(stderr, "\n");
            if (kv_step_ > 0)
                std::fprintf(stderr, "strata metal: --elastic: the KV cache has its memory for the first %u tokens; every %u more "
                                     "take %.2f GiB from the expert cache (%.2f GiB of it at %lld tokens)\n", kv_paid_, kv_step_,
                             (double) kv_step_ * (double) kv_cell_bytes_ / kGiB, ((double) cache_bytes - kv_rest + (double) kv_slack_) / kGiB,
                             (long long) o.max_context);
        } else {
            kv_step_ = 0;
            ctx = llama_init_from_model(model, ctx_params(cache_bytes));
            // (without --elastic the cache cannot give memory back: the KV cache takes all of its own now)
            if (ctx != nullptr && llama_kv_cells_mapped(ctx, nullptr) > 0 && !llama_kv_map_cells(ctx, (uint32_t) o.max_context)) {
                err = "the KV cache could not take its memory";
                return false;
            }
        }
        if (ctx == nullptr && cache_bytes > 0) {
            err = "the expert cache could not be allocated (" + std::to_string(cache_bytes >> 20) +
                  " MiB); a smaller --expert-cache Nmib or a larger --reserve-gib may fit";
            return false;
        }
        if (ctx == nullptr) {
            err = "the context could not be created";
            return false;
        }

        llama_moe_cache_stats st{};
        if (llama_moe_cache_get_stats(ctx, &st)) {
            pool = std::make_unique<ReadPool>(o.io_threads - 1);
            // direct reads are block-aligned, or they are not direct (expert_io.hpp); STRATA_ALIGNED_READS=0: as before
            const char* ar = std::getenv("STRATA_ALIGNED_READS");
            pool->set_aligned(o.expert_io == "direct" && (ar == nullptr || std::atoi(ar) != 0));
            fetch.pool = pool.get();
            // The GPU is kept at its working clock for the whole of a request (gpu_keepalive.hpp).  Measured on the
            // M4 Air, IQ3_S, 75% cache hits, 120 tokens: 2.93 tok/s without it, 4.19 with it during the SSD reads
            // only (STRATA_KEEPALIVE=fetch), 5.94 for the whole request (the default: the waits between the
            // layers' passes and for the sampler count too); with every expert cached it changes nothing
            // (104 ms per token either way).  STRATA_KEEPALIVE=off|fetch|always.
            const std::string ka = std::getenv("STRATA_KEEPALIVE") ? std::getenv("STRATA_KEEPALIVE") : "always";
            if (ka != "off") {
                keepalive = std::make_unique<strata::metal::GpuKeepalive>();
                if (!keepalive->ok()) keepalive.reset();
                keepalive_always = keepalive && ka == "always";
                if (keepalive && !keepalive_always) fetch.keepalive = keepalive.get();
            }
            verify_cache_ = std::getenv("STRATA_VERIFY_CACHE") != nullptr;
            if (verify_cache_)
                std::fprintf(stderr, "strata metal: STRATA_VERIFY_CACHE: every expert a layer computes from is compared with "
                                     "the model file (slow)\n");
            // STRATA_ROUTE_TRACE=FILE: every cached pass's routing, as int32 records (layer, n, n expert ids), for
            // metal/tools/cache_sim.py (a few kB per token; nothing of the model is written)
            if (const char* tp = std::getenv("STRATA_ROUTE_TRACE")) route_trace = std::fopen(tp, "wb");
            // Prefetch (router_predict.hpp): experts the next layer will probably route to are read while the GPU computes
            if (o.prefetch > 0) {
                std::string perr;
                if (!predict.load(files, perr)) {
                    std::fprintf(stderr, "strata metal: --prefetch is off: %s\n", perr.c_str());
                    o.prefetch = 0;
                } else {
                    predict.top = o.prefetch;
                    predict.depth = o.prefetch_depth;
                    predict.margin = o.prefetch_margin;
                    predict.learn = o.prefetch_learn;
                    pool->set_ahead_limit((size_t) std::max(1, o.prefetch_threads));
                    if (const char* b = std::getenv("STRATA_PREFETCH_BACKLOG")) fetch.max_backlog = (size_t) std::max(0, std::atoi(b));
                }
            }
            attach_cache();
            std::fprintf(stderr, "strata metal: expert cache: %llu of %llu experts (%.0f%%), %.2f GiB; missed experts "
                                 "are read from the SSD by %d threads (%s)\n",
                         (unsigned long long) st.n_slots, (unsigned long long) files.n_pairs,
                         files.n_pairs ? 100.0 * (double) st.n_slots / (double) files.n_pairs : 0.0, (double) st.size / kGiB,
                         o.io_threads, o.expert_io == "direct" ? "past the OS file cache" : "through the OS file cache");
            if (o.preload && !o.profile.empty()) {
                std::vector<int32_t>& pl = profile_layers_;
                std::vector<int32_t>& pe = profile_experts_;
                std::string perr;
                if (!read_profile(o.profile, pl, pe, perr)) {
                    std::fprintf(stderr, "strata metal: expert profile: %s; the cache starts empty\n", perr.c_str());
                } else {
                    const auto tp = Clock::now();
                    const uint64_t d0 = disk_bytes_read();
                    const int64_t n = llama_moe_cache_preload(ctx, pl.data(), pe.data(), pl.size());
                    // the profile also says which of a layer's experts a long prompt will most likely need (expert_io.hpp)
                    for (size_t i = 0; i < pl.size(); ++i) {
                        if (pl[i] < 0 || pe[i] < 0 || pe[i] >= files.n_expert) continue;
                        if ((size_t) pl[i] >= fetch.by_profile.size()) fetch.by_profile.resize((size_t) pl[i] + 1);
                        fetch.by_profile[(size_t) pl[i]].push_back(pe[i]);
                    }
                    const double s = ms_since(tp) / 1000.0;
                    std::fprintf(stderr, "strata metal: expert cache filled with the profile's %lld most used experts "
                                         "in %.1f s (%.2f GiB, %.2f GiB/s; %.2f GiB came from the SSD)\n",
                                 (long long) n, s, (double) fetch.bytes / kGiB, s > 0 ? (double) fetch.bytes / kGiB / s : 0.0,
                                 (double) (disk_bytes_read() - d0) / kGiB);
                }
            }
            if (o.prefetch > 0)
                std::fprintf(stderr, "strata metal: prefetch: the %d experts each layer is predicted to route to are read "
                                     "%d layer(s) ahead\n", o.prefetch, o.prefetch_depth);
            llama_moe_cache_get_stats(ctx, &cache0);
            // A pass of T tokens can route a layer to 10 T different experts, and a layer whose share of the cache
            // has fewer slots than that is computed without the cache for that pass (on the CPU, over the mapped
            // file).  The prompt is read in the largest passes every layer's share serves.
            const int n_used = used_count();
            n_used_ = n_used;
            const int fit = (int) std::clamp<uint64_t>(st.n_slots_min / (uint64_t) n_used, 1, 32);
            if (st.n_layers < (uint64_t) files.n_layer)
                std::fprintf(stderr, "strata metal: the expert cache is too small for %d of the %d layers: they are computed "
                                     "on the CPU from the mapped file (slow)\n", files.n_layer - (int) st.n_layers, files.n_layer);
            if (o.elastic) o.chunk = std::min(o.chunk == 0 ? fit : o.chunk, (int) kAnswerBatch);
            if (o.chunk == 0) o.chunk = fit;
            else if (o.chunk > fit)
                std::fprintf(stderr, "strata metal: --chunk %d is more than the expert cache serves for every layer (%d): "
                                     "some layers of a prompt pass are computed without it\n", o.chunk, fit);
            std::fprintf(stderr, "strata metal: prompts are read %d tokens per pass (more than 32 new tokens: %d per pass, "
                                 "their experts streamed from the SSD)\n", o.chunk, o.prefill > 32 ? o.prefill : o.chunk);
        } else if (files.expert_bytes > 0) {
            std::fprintf(stderr, "strata metal: no expert cache: every routed expert is computed from the mapped file\n");
            o.elastic = false;
        }
        if (o.chunk == 0) o.chunk = 32;
        swapouts_seen_ = host_memory().swapouts;
        compressed_seen_ = host_memory().compressed;
        if (drafting()) spec_batch = llama_batch_init(o.spec + 1, 0, 1);
        if (o.lookup > 0 && o.spec > 0)
            std::fprintf(stderr, "strata metal: lookup: where the last %d tokens stood before in the context, what followed is drafted "
                                 "(up to %d tokens a pass), each checked by the model\n", o.lookup, o.spec);
        if (mtp) {
            std::string merr;
            if (!mtp->make_context(ctx, ctx_params(0), merr)) {
                err = merr;
                return false;
            }
            if (mtp->use) {
                std::fprintf(stderr, "strata metal: MTP head: up to %d drafted tokens per pass (while the head gives them %.2f or more), "
                                     "each checked by the model\n", o.spec, (double) o.spec_min_p);
            } else {
                std::fprintf(stderr, "strata metal: STRATA_MTP_PROBE: the MTP head drafts after every token; the drafts are counted, "
                                     "not used\n");
            }
        }
        return true;
    }

    // ---- a request ------------------------------------------------------------------------------------------
    // Where the time of a phase (reading the prompt, writing the answer) went, as the CPU sees it: waiting for the
    // GPU, bringing missed experts in from the SSD, and the rest of the passes (building and encoding the graphs).
    struct Phase {
        double wall_ms = 0, pass_ms = 0, gpu_wait_ms = 0, fetch_ms = 0, sample_ms = 0;
        double encode_ms = 0, gpu_busy_ms = 0;   // the graphs' encoding; the GPU's own time in their command buffers
        uint64_t hits = 0, misses = 0, lookups = 0, passes = 0, waits = 0;
        uint64_t ahead = 0, ahead_used = 0, ahead_waits = 0;   // experts read ahead of their layer; routed to; not yet in place then
        double ahead_mb = 0, predict_ms = 0;
        double staged_mb = 0, staged_used_mb = 0;   // long prompts: read ahead of the tensor they are for; used
        double fetch_mb = 0, disk_mb = 0;
    };
    struct Result {
        int64_t produced = 0, prompt_tokens = 0, prompt_read = 0, resume = 0;
        double prompt_ms = 0, decode_ms = 0, first_token_ms = 0;
        const char* finish = "length";
        int64_t drafts_offered = 0, drafts_accepted = 0;
        uint64_t hits = 0, lookups = 0, file_blobs = 0;
        double file_mb = 0, fetch_ms = 0, fetch_mb = 0;
        Phase prompt, answer;
    };
    // what a benchmark or a test may ask of a request (GEN keys probe=1, force=id.id.id); the server sends neither
    struct Extras {
        bool probe = false;                  // report the three largest logits behind every token
        std::vector<llama_token> force;      // teacher forcing: token i of the answer is fed as force[i], whatever was chosen
    };

    struct Mark {
        Clock::time_point t;
        llama_moe_cache_stats st{};
        ggml_backend_metal_perf gp{};
        uint64_t disk = 0, fetch_bytes = 0, ahead_bytes = 0, staged = 0, staged_used = 0;
        double pass_ms = 0, sample_ms = 0, fetch_ms = 0, predict_ms = 0;
        uint64_t passes = 0;
    };
    double pass_ms_ = 0, sample_ms_ = 0;
    uint64_t passes_ = 0;

    Mark mark_now() {
        Mark m;
        m.t = Clock::now();
        cache_stats(m.st);
        ggml_backend_metal_get_perf(&m.gp);
        m.disk = disk_bytes_read();
        m.fetch_bytes = fetch.bytes;
        m.fetch_ms = fetch.ms;
        m.ahead_bytes = fetch.bytes_ahead;
        m.staged = fetch.bytes_staged;
        m.staged_used = fetch.bytes_from_stage;
        m.predict_ms = predict.ms;
        m.pass_ms = pass_ms_;
        m.sample_ms = sample_ms_;
        m.passes = passes_;
        return m;
    }

    static Phase phase_between(const Mark& a, const Mark& b) {
        Phase p;
        p.wall_ms = std::chrono::duration<double, std::milli>(b.t - a.t).count();
        p.pass_ms = b.pass_ms - a.pass_ms;
        p.gpu_wait_ms = (double) (b.gp.t_wait_us - a.gp.t_wait_us) / 1000.0;
        p.encode_ms = (double) (b.gp.t_encode_us - a.gp.t_encode_us) / 1000.0;
        p.gpu_busy_ms = (double) (b.gp.t_gpu_us - a.gp.t_gpu_us) / 1000.0;
        p.waits = b.gp.n_wait - a.gp.n_wait;
        p.fetch_ms = b.fetch_ms - a.fetch_ms;
        p.ahead = b.st.prefetched - a.st.prefetched;
        p.ahead_used = b.st.prefetch_used - a.st.prefetch_used;
        p.ahead_waits = b.st.prefetch_waits - a.st.prefetch_waits;
        p.ahead_mb = (double) (b.ahead_bytes - a.ahead_bytes) / 1e6;
        p.predict_ms = b.predict_ms - a.predict_ms;
        p.staged_mb = (double) (b.staged - a.staged) / 1e6;
        p.staged_used_mb = (double) (b.staged_used - a.staged_used) / 1e6;
        p.sample_ms = b.sample_ms - a.sample_ms;
        p.hits = b.st.hits - a.st.hits;
        p.misses = b.st.misses - a.st.misses;
        p.lookups = b.st.n_lookups - a.st.n_lookups;
        p.passes = b.passes - a.passes;
        p.fetch_mb = (double) (b.fetch_bytes - a.fetch_bytes) / 1e6;
        p.disk_mb = (double) (b.disk - a.disk) / 1e6;
        return p;
    }

    bool decode(const llama_token* toks, int32_t n, std::string& err) {
        llama_batch b = llama_batch_get_one(const_cast<llama_token*>(toks), n);
        const auto t0 = Clock::now();
        const int32_t r = llama_decode(ctx, b);
        pass_ms_ += ms_since(t0);
        ++passes_;
        if (mtp && r == 0) mtp->catch_up(toks, n, (llama_pos) live.size());   // the head reads what the model read
        if (r != 0) {
            err = r == 1 ? "the context is full" : "the model pass failed (llama_decode " + std::to_string(r) + ")";
            return false;
        }
        return true;
    }

    // drafts are made and checked: by the MTP head, by lookup, or both
    bool drafting() const { return o.spec > 0 && ((mtp && mtp->use) || o.lookup > 0); }

    // Lookup: where the context last ended as it does now (the chosen token `t` and at least o.lookup - 1 tokens before
    // it), what came next is the draft.  Text that repeats - a quoted passage, code that is being edited - is written
    // several tokens a pass this way; where nothing repeats, nothing is drafted.
    void lookup_draft(llama_token t, int most, std::vector<llama_token>& out) const {
        out.clear();
        const size_t n = live.size();
        if (most <= 0 || o.lookup <= 0 || n < (size_t) o.lookup) return;
        const size_t longest = 32;
        size_t best_len = 0, best_at = 0;
        for (size_t j = n; j-- > 0;) {
            if (live[j] != t) continue;
            size_t len = 1;   // t itself; then the tokens before it against the context's last ones
            while (len < longest && j >= len && live[j - len] == live[n - len]) ++len;
            if (len > best_len) {
                best_len = len;
                best_at = j;
                if (len >= longest) break;
            }
        }
        if (best_len < (size_t) o.lookup) return;
        for (size_t i = best_at + 1; i < n && (int) out.size() < most; ++i) out.push_back(live[i]);
    }

    // one pass over the token `t` and the drafts after it; every position's logits are computed
    bool decode_spec(llama_token t, const std::vector<llama_token>& d, std::string& err) {
        const int32_t n = 1 + (int32_t) d.size();
        spec_batch.n_tokens = n;
        for (int32_t j = 0; j < n; ++j) {
            spec_batch.token[j] = j == 0 ? t : d[(size_t) j - 1];
            spec_batch.pos[j] = (llama_pos) live.size() + j;
            spec_batch.n_seq_id[j] = 1;
            spec_batch.seq_id[j][0] = 0;
            spec_batch.logits[j] = 1;
        }
        const auto t0 = Clock::now();
        const int32_t r = llama_decode(ctx, spec_batch);
        pass_ms_ += ms_since(t0);
        ++passes_;
        if (r != 0) {
            err = r == 1 ? "the context is full" : "the model pass failed (llama_decode " + std::to_string(r) + ")";
            return false;
        }
        return true;
    }

    void take_checkpoint() {
        const size_t size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0) return;
        for (auto it = checkpoints.begin(); it != checkpoints.end();)   // the same point again: the newer one stays
            it = it->tokens.size() == live.size() ? checkpoints.erase(it) : it + 1;
        Checkpoint c;
        c.state.resize(size);
        if (llama_state_seq_get_data_ext(ctx, c.state.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) return;
        c.tokens = live;
        if (o.verbose) std::fprintf(stderr, "strata metal: conversation checkpoint at %zu tokens, %.1f MiB\n", live.size(), (double) size / kMiB);
        checkpoints.push_back(std::move(c));
        while (checkpoints.size() > kMaxCheckpoints) checkpoints.pop_front();
    }

    // Brings the context to the longest start of `ids` it can: what it already holds, or a checkpoint.  Returns the
    // number of prompt tokens that need no reading.  At least one token is always left to read (its logits).
    int64_t resume(const std::vector<llama_token>& ids) {
        llama_memory_t mem = llama_get_memory(ctx);
        size_t lcp = 0;
        while (lcp < live.size() && lcp < ids.size() && live[lcp] == ids[lcp]) ++lcp;
        if (lcp == live.size() && lcp < ids.size() && lcp > 0) return (int64_t) lcp;   // the prompt continues the context
        const Checkpoint* best = nullptr;
        for (const Checkpoint& c : checkpoints)
            if (c.tokens.size() <= lcp && c.tokens.size() < ids.size() && (!best || c.tokens.size() > best->tokens.size()) &&
                std::equal(c.tokens.begin(), c.tokens.end(), ids.begin()))
                best = &c;
        if (best != nullptr) {
            const size_t n = best->tokens.size();
            const bool ok =
                llama_state_seq_set_data_ext(ctx, best->state.data(), best->state.size(), 0,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == best->state.size() &&
                llama_memory_seq_rm(mem, 0, (llama_pos) n, -1);
            if (ok) {
                live.resize(n);
                for (auto it = checkpoints.begin(); it != checkpoints.end();)   // points past it are gone with its KV
                    it = it->tokens.size() > n ? checkpoints.erase(it) : it + 1;
                return (int64_t) n;
            }
            std::fprintf(stderr, "strata metal: a conversation checkpoint could not be restored; reading the prompt again\n");
        }
        llama_memory_clear(mem, true);
        live.clear();
        checkpoints.clear();
        return 0;
    }

    // The rows of the n-gram table that the last passes looked up leave the OS file cache (ModelFiles::map): a
    // long prompt reads 110 KiB of them per token.
    void drop_rows() {
        if (!tidy_) return;
        if (const auto* t = files.lazy_table()) files.drop(t->file, t->offs, t->size);
    }

    template <class OnToken, class OnProgress, class OnProbe>
    bool generate(const std::vector<llama_token>& ids, int64_t max_new, const Sampling& sp, const Extras& ex,
                  std::atomic<bool>& stop, OnToken on_token, OnProgress on_progress, OnProbe on_probe, Result& r,
                  std::string& err) {
        const int64_t n = (int64_t) ids.size();
        const Mark m0 = mark_now();
        r.prompt_tokens = n;
        llama_moe_cache_stats s0{}, s1{};
        const bool cached = cache_stats(s0);
        const uint64_t disk0 = disk_bytes_read();
        const uint64_t fetched0 = fetch.n_experts, fetch_bytes0 = fetch.bytes;
        const double fetch_ms0 = fetch.ms;

        struct Busy {
            strata::metal::GpuKeepalive* k;
            explicit Busy(strata::metal::GpuKeepalive* k_) : k(k_) { if (k) k->begin(); }
            ~Busy() { if (k) k->end(); }
        } busy(keepalive_always ? keepalive.get() : nullptr);
        const auto tp = Clock::now();
        if (!back_off(err)) return false;
        const size_t live0 = live.size();
        int64_t at = resume(ids);
        r.resume = at;
        if (mtp && (at == 0 || (size_t) at < live0)) mtp->rewind((llama_pos) at);
        std::vector<llama_token> written;
        on_progress(-1, n, 0.0);   // RESUME
        // where the prompt is read up to a checkpoint: before its last message opens (the next turn of the chat
        // shares everything up to there) and before its last token (the same prompt asked again)
        std::vector<int64_t> marks;
        if (im_start != LLAMA_TOKEN_NULL)
            for (int64_t i = n - 1; i > at; --i)
                if (ids[(size_t) i] == im_start) {
                    marks.push_back(i);
                    break;
                }
        if (n - 1 > at) marks.push_back(n - 1);
        marks.push_back(n);
        bool cancelled = false;
        auto last_pp = Clock::now();
        // Long stretches: the next expert tensors are read into a staging buffer while the GPU computes (expert_io.hpp).
        // The buffer is a layer's three tensors large (1.1 GiB for IQ3_S; one tensor, 0.45 GiB, if that is all there is
        // room for), taken for the time of the prompt and only from memory that is free on top of what the engine
        // left for other programs.
        struct Bulk {
            ExpertFetch& f;
            bool on = false;
            void finish() { if (on) { f.end_bulk(); on = false; } }
            ~Bulk() { finish(); }
        } bulk{fetch};
        if (const char* w = std::getenv("STRATA_PREFILL_AHEAD_TENSORS")) o.prefill_ahead_tensors = std::atoi(w);
        const auto stage = [&](double spoken_for) {   // spoken_for: memory that is free now and about to be used
            if (bulk.on || !o.prefill_ahead || !pool) return;
            const double free_now = (double) host_memory().available - spoken_for;
            if (const char* w = std::getenv("STRATA_PREFILL_AHEAD_FROM")) fetch.ahead_from = std::atof(w);
            fetch.ctx = ctx;
            const bool layer = o.prefill_ahead_tensors >= 3 && free_now >= (double) fetch.staging_bytes(true) + 1.5 * kGiB;
            if (layer || free_now >= (double) fetch.staging_bytes(false) + 1.5 * kGiB) bulk.on = fetch.begin_bulk(layer);
            else std::fprintf(stderr, "strata metal: %.1f GiB of memory are free: this prompt is read without read-ahead\n", free_now / kGiB);
        };
        // (--elastic: the buffer is taken once the context has the shape for long passes)
        if (!o.elastic && o.prefill > 32 && n - at >= o.prefill_from) stage(0.0);
        for (int64_t mark : marks) {
            while (at < mark) {
                if (stop.load()) {
                    cancelled = true;
                    break;
                }
                // A long stretch goes in passes of up to --prefill tokens: llama.cpp computes those without the
                // expert cache, on copies of the layer's routed experts that the fetch reads from the SSD (most of
                // a layer's 512 experts are routed to by then, so the cache would only be flushed by them).  A
                // short stretch goes in passes the cache serves.  Nothing in between: a pass of 12-32 tokens that
                // the cache does not serve would be computed on the CPU.
                const int64_t rem = mark - at;
                const bool long_pass = o.prefill > 32 && rem >= o.prefill_from && rem > o.chunk;
                int64_t most = o.elastic ? long_batch_ : o.prefill;
                if (o.elastic) {
                    // What one short pass reads is read beside the cache, if the memory for that is free; more takes
                    // the cache's memory.  The long-prompt shape is kept for as long as long passes follow.
                    const Shape want = !long_pass ? Shape::Answer
                                     : shape_ == Shape::Long ? Shape::Long
                                     : rem <= short_batch_ && (shape_ == Shape::Short || can_borrow()) ? Shape::Short : Shape::Long;
                    // The KV cache's memory for the whole prompt, once, at its first pass: from the cache while the
                    // context still has its answering shape, or (below) from what holds the cache's place once it
                    // has the long-prompt one - the cache is then not read back just to be given up.
                    // (before a long prompt for 2,048 tokens beyond it: the answer and the next turns then do not
                    // cross a step, which costs nothing here and a cache made anew while it is there)
                    if (shape_ != Shape::Long && !make_room_for_kv(want == Shape::Long ? n + 2048 : n, want == Shape::Long, err)) return false;
                    if (want != shape_) {
                        bulk.finish();
                        if (!reshape(want, err)) return false;
                        if (want != Shape::Answer) stage(want == Shape::Short ? short_work_ : 0.0);
                    }
                    if (!kv_take(err)) return false;
                    if (shape_ == Shape::Short) most = short_batch_;
                }
                const int32_t take = (int32_t) (long_pass ? std::min<int64_t>(most, rem) : std::min<int64_t>(o.chunk, rem));
                fetch.more_passes = bulk.on && mark - (at + take) >= o.prefill_from;
                if (!decode(ids.data() + at, take, err)) return false;
                live.insert(live.end(), ids.begin() + at, ids.begin() + at + take);
                at += take;
                if (long_pass) drop_rows();
                if (o.verbose) {   // where the context's memory grows as it fills (the plan measures an empty one)
                    uint64_t used = 0, ws = 0;
                    gpu_memory(used, ws);
                    std::fprintf(stderr, "strata metal: pass of %d tokens to %lld: %.1f MiB on the GPU, %.1f MiB in all, %.1f MiB of it "
                                         "compressed\n", (int) take, (long long) at, (double) used / kMiB, (double) own_footprint() / kMiB,
                                 (double) own_compressed() / kMiB);
                }
                if (ms_since(last_pp) >= 500.0 && at < n) {
                    on_progress(at, n, ms_since(tp));
                    last_pp = Clock::now();
                }
            }
            if (cancelled) break;
            if (mark < n) take_checkpoint();
        }
        r.prompt_read = at - r.resume;
        if (!cancelled) {   // the prompt's last pass is over when the GPU has finished it
            llama_synchronize(ctx);
        }
        bulk.finish();   // release prompt staging before decoding an answer or waiting for the next request
        if (!reshape(Shape::Answer, err)) return false;
        r.prompt_ms = ms_since(tp);
        const Mark m1 = mark_now();

        const auto td = Clock::now();
        if (cancelled) {
            r.finish = "cancel";
        } else {
            llama_sampler* smpl = make_sampler(sp, n_vocab);
            // the model's choice from the logits of output `idx` of the last pass; false: the answer ends with it
            auto emit = [&](int32_t idx, llama_token& fed) {
                llama_synchronize(ctx);
                const auto ts = Clock::now();
                const llama_token t = llama_sampler_sample(smpl, ctx, idx);
                sample_ms_ += ms_since(ts);
                const size_t i = (size_t) r.produced;
                fed = i < ex.force.size() ? ex.force[i] : t;
                ++r.produced;
                if (ex.probe) on_probe(t, fed, llama_get_logits_ith(ctx, idx));
                on_token(t);
                if (r.produced == 1) r.first_token_ms = ms_since(tp);
                if (eos(fed)) r.finish = "stop";
                else if (r.produced >= max_new) r.finish = "length";
                else if (stop.load()) r.finish = "cancel";
                else return true;
                return false;
            };
            // (teacher forcing keeps the drafts: a forced token is checked against them as a chosen one would be, so
            // every position's logits come from the same kind of pass as in a free run)
            const bool spec = drafting();
            llama_memory_t mem = llama_get_memory(ctx);
            llama_token t = LLAMA_TOKEN_NULL;
            bool go = emit(-1, t);
            while (go) {
                // `t` is chosen and not read yet; it goes to position `at_pos`
                const llama_pos at_pos = (llama_pos) live.size();
                drafts_.clear();
                if (spec) {
                    const int most = (int) std::min<int64_t>(o.spec, max_new - r.produced - 1);
                    lookup_draft(t, most, drafts_);
                    if (drafts_.empty() && mtp && mtp->use) mtp->draft(t, at_pos, most, o.spec_min_p, drafts_, draft_probs_);
                } else if (mtp) {   // the probe
                    written.push_back(t);
                    mtp->draft(t, at_pos, mtp->n_draft, 0.0f, drafts_, draft_probs_);
                    mtp->note(drafts_, draft_probs_);
                    drafts_.clear();
                }
                if (!make_room_for_kv((int64_t) live.size() + 1 + (int64_t) drafts_.size(), false, err)) {
                    llama_sampler_free(smpl);
                    return false;
                }
                if (drafts_.empty()) {
                    if (!decode(&t, 1, err)) {
                        llama_sampler_free(smpl);
                        return false;
                    }
                    live.push_back(t);
                    go = emit(-1, t);
                    continue;
                }
                // One pass over the token and its drafts.  Output j of it is the model's choice after position
                // at_pos + j: while that is the draft the pass read at the next position, the next output holds too.
                const int32_t k = (int32_t) drafts_.size();
                if (!decode_spec(t, drafts_, err)) {
                    llama_sampler_free(smpl);
                    return false;
                }
                r.drafts_offered += k;
                live.push_back(t);
                int32_t m = 0;
                llama_token next = LLAMA_TOKEN_NULL;
                for (;;) {
                    go = emit(m, next);
                    if (m < k && next == drafts_[(size_t) m]) {
                        live.push_back(next);   // the draft was right: the model has read it already
                        ++m;
                        if (go) continue;
                    }
                    break;
                }
                r.drafts_accepted += m;
                // the drafts after the first wrong one leave the context (attention cells, and the recurrent layers'
                // state through the snapshots of the pass), and the head is shown what stayed
                if (m < k && !llama_memory_seq_rm(mem, 0, at_pos + m + 1, -1)) {
                    err = "the context could not go back after a rejected draft";
                    llama_sampler_free(smpl);
                    return false;
                }
                if (mtp) mtp->catch_up(spec_batch.token, m + 1, at_pos);
                t = next;
            }
            llama_sampler_free(smpl);
        }
        r.decode_ms = ms_since(td);
        drop_rows();
        if (mtp) mtp->report(written);
        const Mark m2 = mark_now();
        r.prompt = phase_between(m0, m1);
        r.answer = phase_between(m1, m2);
        if (cached && cache_stats(s1)) {
            r.hits = s1.hits - s0.hits;
            r.lookups = r.hits + (s1.misses - s0.misses);
        }
        r.file_blobs = r.lookups - r.hits;
        (void) fetched0;
        r.fetch_ms = fetch.ms - fetch_ms0;
        r.fetch_mb = (double) (fetch.bytes - fetch_bytes0) / 1e6;
        r.file_mb = (double) (disk_bytes_read() - disk0) / 1e6;
        return true;
    }
};

// --io-bench N: how fast this Mac's SSD hands over routed experts.  Reads N experts (gate, up and down of each) from
// the model file past the OS file cache, in the groups a token causes (the misses of one layer at a time), with 1 to
// 16 threads.  Nothing is loaded and nothing is written; a shard that is still downloading is read as far as it goes.
int io_bench(const Options& o) {
    ModelFiles files;
    std::string err;
    if (!files.open(o.native, true, err)) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    struct Blob {
        int fd;
        uint64_t offs, size;
    };
    std::vector<std::vector<Blob>> layers;   // per layer: its expert tensors
    for (const auto& [name, t] : files.tensors) {
        if (t.n_expert <= 0 || t.layer < 0 || t.offs + t.size > files.sizes[(size_t) t.file]) continue;
        if ((size_t) t.layer >= layers.size()) layers.resize((size_t) t.layer + 1);
        layers[(size_t) t.layer].push_back({files.fds[(size_t) t.file], t.offs, t.size / (uint64_t) t.n_expert});
    }
    std::vector<int> usable;
    for (size_t l = 0; l < layers.size(); ++l)
        if (layers[l].size() >= 2) usable.push_back((int) l);
    if (usable.empty()) {
        std::fprintf(stderr, "strata: no complete expert tensors in the file yet\n");
        return 1;
    }
    size_t max_bytes = 0;
    for (int l : usable) {
        size_t b = 0;
        for (const Blob& x : layers[(size_t) l]) b += x.size;
        max_bytes = std::max(max_bytes, b);
    }
    // experts read together: the misses of one layer in one pass (about 2 while answering; STRATA_IO_BENCH_GROUP sets it).
    // One read per tensor: reads cut into smaller pieces were measured slower at every thread count (M4 Air)
    const int per_group = std::getenv("STRATA_IO_BENCH_GROUP") ? std::max(1, std::atoi(std::getenv("STRATA_IO_BENCH_GROUP"))) : 2;
    std::vector<uint8_t> buf(max_bytes * per_group);
    std::fprintf(stderr, "strata metal: io-bench: %zu layers with experts on disk, %d experts per run, groups of %d\n",
                 usable.size(), o.io_bench, per_group);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&seed] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (int threads : {1, 2, 4, 8, 12, 16}) {
        ReadPool pool(threads - 1);
        std::vector<ReadPool::Job> jobs;
        uint64_t bytes = 0;
        const auto t0 = Clock::now();
        for (int done = 0; done < o.io_bench; done += per_group) {
            const int l = usable[rnd() % usable.size()];
            jobs.clear();
            size_t at = 0;
            for (int k = 0; k < per_group; ++k) {
                const uint64_t e = rnd() % (uint64_t) files.n_expert;
                for (const Blob& x : layers[(size_t) l]) {
                    jobs.push_back({x.fd, x.offs + e * x.size, buf.data() + at, (size_t) x.size});
                    at += x.size;
                    bytes += x.size;
                }
            }
            if (!pool.run(jobs, err)) {
                std::fprintf(stderr, "strata: %s\n", err.c_str());
                return 1;
            }
        }
        const double s = ms_since(t0) / 1000.0;
        std::printf("io-bench threads=%d  %.0f MB in %.2f s  %.0f MB/s  %.2f ms per expert\n", threads, (double) bytes / 1e6, s,
                    (double) bytes / 1e6 / s, 1000.0 * s / o.io_bench);
    }
    return 0;
}

std::atomic<bool> g_never_stop{false};

int run_once(Engine& e) {
    std::vector<llama_token> ids;
    if (!e.o.tokens.empty()) {
        std::vector<int64_t> v;
        std::string err;
        if (!parse_i64_list(e.o.tokens.c_str(), v, err)) {
            std::fprintf(stderr, "--tokens: %s\n", err.c_str());
            return 2;
        }
        ids.assign(v.begin(), v.end());
    } else {
        const std::string text = e.o.text.empty() ? "Hello" : e.o.text;
        ids.resize(text.size() + 16);
        const int32_t n = llama_tokenize(e.vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), true, true);
        if (n <= 0) {
            std::fprintf(stderr, "strata: the text could not be tokenized\n");
            return 2;
        }
        ids.resize((size_t) n);
    }
    Sampling sp;
    sp.temperature = e.o.temperature;
    sp.seed = e.o.seed;
    Engine::Result r;
    std::string err;
    const bool ok = e.generate(
        ids, e.o.max_new, sp, Engine::Extras{}, g_never_stop,
        [&](llama_token t) {
            char buf[256];
            const int32_t k = llama_token_to_piece(e.vocab, t, buf, sizeof buf, 0, true);
            if (k > 0) std::fwrite(buf, 1, (size_t) k, stdout);
            std::fflush(stdout);
        },
        [&](int64_t done, int64_t total, double ms) {
            if (done > 0 && e.o.verbose) std::fprintf(stderr, "strata metal: prompt %lld / %lld (%.1f s)\n", (long long) done, (long long) total, ms / 1000.0);
        },
        [](llama_token, llama_token, const float*) {}, r, err);
    std::fputc('\n', stdout);
    if (!ok) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    const int64_t dec = std::max<int64_t>(0, r.produced - 1);
    std::fprintf(stderr,
                 "strata metal: prompt %lld tokens in %.2f s (%.1f tok/s), output %lld tokens in %.2f s (%.2f tok/s), "
                 "finish %s\nstrata metal: expert cache hit rate %.1f%% (%llu of %llu), %llu experts fetched (%.0f MB in "
                 "%.2f s, %.0f MB/s), %.1f MB read from the SSD\n",
                 (long long) r.prompt_read, r.prompt_ms / 1000.0, r.prompt_ms > 0 ? 1000.0 * (double) r.prompt_read / r.prompt_ms : 0.0,
                 (long long) r.produced, r.decode_ms / 1000.0, r.decode_ms > 0 ? 1000.0 * (double) dec / r.decode_ms : 0.0, r.finish,
                 r.lookups ? 100.0 * (double) r.hits / (double) r.lookups : 0.0, (unsigned long long) r.hits,
                 (unsigned long long) r.lookups, (unsigned long long) r.file_blobs, r.fetch_mb, r.fetch_ms / 1000.0,
                 r.fetch_ms > 0 ? r.fetch_mb / (r.fetch_ms / 1000.0) : 0.0, r.file_mb);
    return 0;
}

int serve(Engine& e) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Input in;
    in.start();
    uint64_t used = 0, ws = 0;
    e.gpu_memory(used, ws);
    std::printf("INFO context=%lld kv=%s kv_resident=0 expert_slots=%llu expert_cache_mib=%llu expert_slots_primary=%llu "
                "expert_cache_primary_mib=%llu spec=%d mtp_max=%d lookup=%d prefetch=%d vram_free_mib=%lld arena_mib=0 pool_workers=%d "
                "backend=metal expert_io=%s engine=" STRATA_VERSION "\n",
                (long long) e.o.max_context, e.o.kv.c_str(), (unsigned long long) e.cache0.n_slots,
                (unsigned long long) (e.cache0.size >> 20), (unsigned long long) e.cache0.n_slots,
                (unsigned long long) (e.cache0.size >> 20), e.drafting() ? 1 : 0, e.mtp && e.mtp->use ? e.o.spec : 0, e.o.lookup > 0 && e.o.spec > 0 ? 1 : 0, e.o.prefetch,
                (long long) ((ws > used ? ws - used : 0) >> 20), e.o.io_threads,
                e.o.expert_io.c_str());
    std::printf("READY %lld stop\n", (long long) e.o.max_context);
    std::fflush(stdout);

    std::string line;
    while (in.next(line)) {
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line.rfind("GENI ", 0) == 0) {
            std::printf("ERR this engine was started without --vision\n");
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR %s is not available in the Metal engine (GEN <max_new> <id,id,...>)\n",
                        line.substr(0, line.find(' ')).c_str());
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        Sampling sp;
        Engine::Extras ex;
        bool has_bias = false, want_stats = false;
        for (;;) {   // sampling keys between max_new and the ids
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) {
                endp = const_cast<char*>(start);
                break;
            }
            const std::string key = tok.substr(0, eq);
            const char* val = tok.c_str() + eq + 1;
            if (key == "temperature") sp.temperature = std::strtof(val, nullptr);
            else if (key == "top_p") sp.top_p = std::strtof(val, nullptr);
            else if (key == "top_k") sp.top_k = std::atoi(val);
            else if (key == "min_p") sp.min_p = std::strtof(val, nullptr);
            else if (key == "penalty_last_n") sp.penalty_last_n = std::atoi(val);
            else if (key == "penalty_repeat") sp.penalty_repeat = std::strtof(val, nullptr);
            else if (key == "penalty_freq") sp.penalty_freq = std::strtof(val, nullptr);
            else if (key == "penalty_present") sp.penalty_present = std::strtof(val, nullptr);
            else if (key == "seed") sp.seed = (uint32_t) std::strtoull(val, nullptr, 10);
            else if (key == "logit_bias") has_bias = true;
            // benchmarks and tests (metal/tools/bench.py): P and STAT lines, teacher forcing
            else if (key == "probe") ex.probe = std::atoi(val) != 0;
            else if (key == "stats") want_stats = std::atoi(val) != 0;
            else if (key == "force") {
                for (const char* q = val; *q != '\0';) {
                    char* e2 = nullptr;
                    const long v = std::strtol(q, &e2, 10);
                    if (e2 == q) break;
                    ex.force.push_back((llama_token) v);
                    q = *e2 == '.' ? e2 + 1 : e2;
                }
            }
            // other keys (ckpt, pin, cvec, the CUDA engine's tuning keys) change nothing here
        }
        if (has_bias) {
            std::printf("ERR logit_bias: not available in the Metal engine\n");
            continue;
        }
        std::vector<int64_t> raw;
        std::string pe;
        if (max_new < 1 || !parse_i64_list(endp, raw, pe)) {
            std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
            continue;
        }
        const int64_t n = (int64_t) raw.size();
        if (n + max_new + 8 > e.o.max_context) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        (long long) max_new, (long long) e.o.max_context);
            continue;
        }
        std::vector<llama_token> ids((size_t) n);
        bool bad = false;
        // tests only: a tiny model under the real tokenizer (serve/ runs end to end without the 55 GB file)
        static const bool fold = std::getenv("STRATA_METAL_TEST_FOLD_VOCAB") != nullptr;
        for (int64_t i = 0; fold && i < n; ++i) raw[(size_t) i] = ((raw[(size_t) i] % e.n_vocab) + e.n_vocab) % e.n_vocab;
        for (int64_t i = 0; i < n; ++i) {
            if (raw[(size_t) i] < 0 || raw[(size_t) i] >= e.n_vocab) bad = true;
            ids[(size_t) i] = (llama_token) raw[(size_t) i];
        }
        if (bad) {
            std::printf("ERR a token id is outside the vocabulary\n");
            continue;
        }
        in.stop.store(false);
        Engine::Result r;
        std::string err;
        for (llama_token f : ex.force)
            if (f < 0 || f >= e.n_vocab) bad = true;
        if (bad) {
            std::printf("ERR a forced token id is outside the vocabulary\n");
            continue;
        }
        const int32_t n_vocab = e.n_vocab;
        const bool ok = e.generate(
            ids, max_new, sp, ex, in.stop,
            [](llama_token t) {
                std::printf("T %d\n", (int) t);
                std::fflush(stdout);
            },
            [&](int64_t done, int64_t total, double ms) {
                if (done < 0) std::printf("RESUME %lld\n", (long long) r.resume);
                else std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) total, ms,
                                 ms > 0 ? 1000.0 * (double) (done - r.resume) / ms : 0.0);
                std::fflush(stdout);
            },
            // P <chosen> <fed> <logit of fed> <id> <logit> x3: the three largest logits, largest first
            [n_vocab](llama_token t, llama_token fed, const float* logits) {
                int best[3] = {-1, -1, -1};
                for (int i = 0; i < n_vocab; ++i) {
                    const float v = logits[i];
                    if (best[2] >= 0 && v <= logits[best[2]]) continue;
                    int k = 2;
                    while (k > 0 && (best[k - 1] < 0 || v > logits[best[k - 1]])) {
                        best[k] = best[k - 1];
                        --k;
                    }
                    best[k] = i;
                }
                std::printf("P %d %d %.6f", (int) t, (int) fed, (double) logits[fed]);
                for (int k = 0; k < 3; ++k) std::printf(" %d %.6f", best[k], best[k] >= 0 ? (double) logits[best[k]] : 0.0);
                std::printf("\n");
            },
            r, err);
        if (!ok) {
            // the context is in an unknown state after a failed pass: the next request reads its prompt from the start
            if (!e.broken_) llama_memory_clear(llama_get_memory(e.ctx), true);
            e.live.clear();
            e.checkpoints.clear();
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            if (e.broken_) return 1;   // serve/server.py starts the engine again for the next request
            continue;
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <offered> <reused> <cache hits>
        //      <lookups> <RAM blobs> <file blobs> <file MB> <prompt read> <offloaded>
        if (want_stats) {
            // STAT <phase> then key=value: the phase's wall time and where it went (ms), its passes and cache figures
            const Engine::Phase* ph[2] = {&r.prompt, &r.answer};
            for (int k = 0; k < 2; ++k)
                std::printf("STAT %s wall=%.1f pass=%.1f gpu_wait=%.1f fetch=%.1f encode=%.1f gpu_busy=%.1f sample=%.1f "
                            "passes=%llu waits=%llu lookups=%llu hits=%llu misses=%llu fetch_mb=%.1f disk_mb=%.1f "
                            "first_token=%.1f ahead=%llu ahead_used=%llu ahead_waits=%llu ahead_mb=%.1f predict=%.1f "
                            "staged_mb=%.1f staged_used_mb=%.1f\n",
                            k == 0 ? "prompt" : "answer", ph[k]->wall_ms, ph[k]->pass_ms, ph[k]->gpu_wait_ms, ph[k]->fetch_ms,
                            ph[k]->encode_ms, ph[k]->gpu_busy_ms, ph[k]->sample_ms, (unsigned long long) ph[k]->passes,
                            (unsigned long long) ph[k]->waits, (unsigned long long) ph[k]->lookups,
                            (unsigned long long) ph[k]->hits,
                            (unsigned long long) ph[k]->misses, ph[k]->fetch_mb, ph[k]->disk_mb, r.first_token_ms,
                            (unsigned long long) ph[k]->ahead, (unsigned long long) ph[k]->ahead_used,
                            (unsigned long long) ph[k]->ahead_waits, ph[k]->ahead_mb, ph[k]->predict_ms, ph[k]->staged_mb,
                            ph[k]->staged_used_mb);
        }
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %llu %llu 0 %llu %.1f %lld 0\n", (long long) r.produced,
                    (long long) n, r.prompt_ms, r.decode_ms, r.finish, (long long) r.drafts_accepted, (long long) r.drafts_offered,
                    (long long) r.resume, (unsigned long long) r.hits,
                    (unsigned long long) r.lookups, (unsigned long long) r.file_blobs, r.file_mb, (long long) r.prompt_read);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "strata serve: prompt %lld (%lld reused, %lld read in %.2f s), %lld tokens in %.2f s (%.2f tok/s), %s; "
                     "expert cache %.1f%% hits, %llu fetched in %.2f s, %.1f MB from the SSD\n",
                     (long long) n, (long long) r.resume, (long long) r.prompt_read, r.prompt_ms / 1000.0, (long long) r.produced,
                     r.decode_ms / 1000.0, r.decode_ms > 0 ? 1000.0 * (double) std::max<int64_t>(0, r.produced - 1) / r.decode_ms : 0.0,
                     r.finish, r.lookups ? 100.0 * (double) r.hits / (double) r.lookups : 0.0, (unsigned long long) r.file_blobs,
                     r.fetch_ms / 1000.0, r.file_mb);
    }
    return 0;
}

}   // namespace

int main(int argc, char** argv) {
    Engine e;
    const int pa = parse_args(argc, argv, e.o);
    if (pa != 0) return pa == 1 ? 0 : 2;
    if (!e.o.verbose)
        llama_log_set(
            [](ggml_log_level level, const char* text, void*) {
                if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
            },
            nullptr);
    if (e.o.io_bench > 0) return io_bench(e.o);
    std::string err;
    if (!e.load(err)) {
        std::fprintf(stderr, "strata: %s\n", err.c_str());
        return 1;
    }
    if (e.o.plan_only || e.o.load_only) return 0;
    return e.o.serve ? serve(e) : run_once(e);
}
