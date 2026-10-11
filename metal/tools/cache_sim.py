"""metal/tools/cache_sim.py - which experts the expert cache should hold: eviction rules replayed on a routing trace.

    STRATA_ROUTE_TRACE=trace.bin  build-metal/bin/strata --serve ...      (or metal/tools/bench.py; one file per run)
    .venv/bin/python metal/tools/cache_sim.py --model MODEL-00001-of-0000N.gguf --trace trace.bin [--trace ...]
                                              --profile data/expert-profile.bin --cache-gib 3,5,7.5,10,15.8

The trace is what the engine's cached passes routed to (layer, expert ids), in order.  Every rule starts from the
cache the engine starts with (the profile's top entries) and is charged the bytes of every expert it has to read:
the SSD's time is by the byte, and so is the cache's room, so megabytes read per token is the figure that counts.
Nothing here touches the model's weights; the GGUF is only read for the sizes of its tensors.

  lru          the engine's rule: least recently used, each group of layers (same tensor layout) a share of the
               room by its size
  lru-split    the same rule, the room divided between the groups by the profile (its top entries decide)
  lfu          least frequently used with ageing: counts halve every --half-life passes, the profile's rank is the
               count an expert starts with
  s3fifo       a small first-in-first-out queue in front of the main one: an expert stays only if routed to twice
  belady       evicts the expert whose next use is furthest away (needs the future: the bound for any rule)
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def model_layout(gguf_path, gguf_py):
    """layer -> (bytes of one expert, group key)."""
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader
    per = defaultdict(list)
    for t in GGUFReader(gguf_path).tensors:
        name = t.name
        if "_exps." not in name or not name.startswith("blk."):
            continue
        il = int(name.split(".")[1])
        n_expert = int(t.shape[-1])
        per[il].append((name.split(".")[2], t.tensor_type.name, int(t.n_bytes) // n_expert))
    out = {}
    for il, ts in per.items():
        ts.sort()
        out[il] = (sum(x[2] for x in ts), tuple((x[0], x[1], x[2]) for x in ts))
    return out


def read_trace(path):
    """-> list of passes; a pass is a list of (layer, [expert ids])."""
    blob = Path(path).read_bytes()
    off, passes, cur, last = 0, [], [], 1 << 30
    while off + 8 <= len(blob):
        il, n = struct.unpack_from("<ii", blob, off)
        off += 8
        ids = struct.unpack_from("<%di" % n, blob, off)
        off += 4 * n
        if il <= last and cur:
            passes.append(cur)
            cur = []
        cur.append((il, sorted(set(ids))))
        last = il
    if cur:
        passes.append(cur)
    return passes


def read_profile(path):
    blob = Path(path).read_bytes()
    n = struct.unpack_from("<5I", blob, 4)[4]
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


class Groups:
    def __init__(self, layout, cache_bytes, profile, split):
        self.layout = layout
        self.of = {}
        keys = []
        for il in sorted(layout):
            k = layout[il][1]
            if k not in keys:
                keys.append(k)
            self.of[il] = keys.index(k)
        self.n = len(keys)
        self.size = [0] * self.n          # bytes of one expert
        self.host = [0] * self.n          # bytes of all experts of the group
        self.layers = [[] for _ in range(self.n)]
        for il in sorted(layout):
            g = self.of[il]
            self.size[g] = layout[il][0]
            self.host[g] += layout[il][0] * 512
            self.layers[g].append(il)
        total = sum(self.host)
        if split == "size":               # the engine: every group caches the same fraction of its experts
            self.slots = [int(cache_bytes * self.host[g] / total / self.size[g]) for g in range(self.n)]
        else:                             # by the profile: its entries in order until the room is used
            self.slots = [0] * self.n
            left = cache_bytes
            for il, e in profile:
                g = self.of[il]
                if self.size[g] <= left:
                    self.slots[g] += 1
                    left -= self.size[g]
            floor = 40                    # a pass of up to 4 tokens must fit: 10 experts each
            for g in range(self.n):
                self.slots[g] = max(self.slots[g], floor)
        for g in range(self.n):
            self.slots[g] = min(self.slots[g], 512 * len(self.layers[g]))


def preload(groups, profile):
    """per group: the profile's entries that fit, most wanted first"""
    out = [[] for _ in range(groups.n)]
    for il, e in profile:
        g = groups.of.get(il)
        if g is not None and len(out[g]) < groups.slots[g]:
            out[g].append((il, e))
    return out


def sim_lru(groups, pre, passes):
    caches = []
    for g in range(groups.n):
        c = OrderedDict()
        for key in reversed(pre[g]):      # least wanted first: the most wanted are the most recently used
            c[key] = True
        caches.append(c)
    hits = misses = miss_bytes = 0
    for p in passes:
        for il, ids in p:
            g = groups.of[il]
            c = caches[g]
            need = []
            for e in ids:
                if (il, e) in c:
                    c.move_to_end((il, e))
                    hits += 1
                else:
                    need.append(e)
            for e in need:
                if len(c) >= groups.slots[g]:
                    c.popitem(last=False)
                c[(il, e)] = True
                misses += 1
                miss_bytes += groups.size[g]
    return hits, misses, miss_bytes


def sim_lfu(groups, pre, passes, half_life, prior):
    """evict the smallest aged count; `prior` is the count the profile's best entry starts with"""
    count = [dict() for _ in range(groups.n)]
    cached = [set() for _ in range(groups.n)]
    last_use = [dict() for _ in range(groups.n)]
    for g in range(groups.n):
        n = max(1, len(pre[g]))
        for r, key in enumerate(pre[g]):
            cached[g].add(key)
            count[g][key] = prior * (1.0 - r / n)
            last_use[g][key] = -1
    hits = misses = miss_bytes = 0
    for t, p in enumerate(passes):
        if half_life and t and t % half_life == 0:
            for g in range(groups.n):
                for k in count[g]:
                    count[g][k] *= 0.5
        for il, ids in p:
            g = groups.of[il]
            c, cnt, lu = cached[g], count[g], last_use[g]
            need = []
            for e in ids:
                key = (il, e)
                cnt[key] = cnt.get(key, 0.0) + 1.0
                lu[key] = t
                if key in c:
                    hits += 1
                else:
                    need.append(key)
            for key in need:
                if len(c) >= groups.slots[g]:
                    # not one of this pass's own experts
                    victim = min((k for k in c if lu.get(k, -1) != t or k[0] != il), key=lambda k: (cnt.get(k, 0.0), lu.get(k, -1)))
                    c.discard(victim)
                c.add(key)
                misses += 1
                miss_bytes += groups.size[g]
    return hits, misses, miss_bytes


def sim_s3fifo(groups, pre, passes, small_share=0.1):
    hits = misses = miss_bytes = 0
    state = []
    for g in range(groups.n):
        cap = groups.slots[g]
        s_cap = max(20, int(cap * small_share))
        main = deque(reversed(pre[g][:cap - s_cap]))
        st = {"small": deque(), "main": main, "freq": {k: 1 for k in main}, "in": set(main), "ghost": OrderedDict(),
              "s_cap": s_cap, "m_cap": cap - s_cap}
        state.append(st)
    for t, p in enumerate(passes):
        for il, ids in p:
            g = groups.of[il]
            st = state[g]
            for e in ids:
                key = (il, e)
                if key in st["in"]:
                    st["freq"][key] = min(3, st["freq"].get(key, 0) + 1)
                    hits += 1
                    continue
                misses += 1
                miss_bytes += groups.size[g]
                if key in st["ghost"]:
                    del st["ghost"][key]
                    st["main"].append(key)
                else:
                    st["small"].append(key)
                st["in"].add(key)
                st["freq"][key] = 0
                while len(st["small"]) > st["s_cap"]:
                    k = st["small"].popleft()
                    if st["freq"].get(k, 0) > 0:
                        st["main"].append(k)
                        st["freq"][k] = 0
                    else:
                        st["in"].discard(k)
                        st["ghost"][k] = True
                        if len(st["ghost"]) > st["m_cap"]:
                            st["ghost"].popitem(last=False)
                while len(st["main"]) > st["m_cap"]:
                    k = st["main"].popleft()
                    if st["freq"].get(k, 0) > 0:
                        st["freq"][k] -= 1
                        st["main"].append(k)
                    else:
                        st["in"].discard(k)
    return hits, misses, miss_bytes


def sim_belady(groups, pre, passes):
    # next use of every (layer, expert) after each pass
    uses = defaultdict(list)
    for t, p in enumerate(passes):
        for il, ids in p:
            for e in ids:
                uses[(il, e)].append(t)
    ptr = defaultdict(int)
    INF = 1 << 60

    def next_use(key, t):
        u = uses.get(key)
        if not u:
            return INF
        i = ptr[key]
        while i < len(u) and u[i] <= t:
            i += 1
        ptr[key] = i
        return u[i] if i < len(u) else INF

    cached = [set(pre[g]) for g in range(groups.n)]
    hits = misses = miss_bytes = 0
    for t, p in enumerate(passes):
        for il, ids in p:
            g = groups.of[il]
            c = cached[g]
            need = []
            for e in ids:
                if (il, e) in c:
                    hits += 1
                else:
                    need.append((il, e))
            now = set((il, e) for e in ids)
            for key in need:
                if len(c) >= groups.slots[g]:
                    victim = max((k for k in c if k not in now), key=lambda k: next_use(k, t))
                    c.discard(victim)
                c.add(key)
                misses += 1
                miss_bytes += groups.size[g]
    return hits, misses, miss_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf-py", default=str(ROOT / "build-metal" / "llama.cpp" / "gguf-py"))
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--profile", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--cache-gib", default="3,5,7.5,10,15.8")
    ap.add_argument("--half-life", type=int, default=200)
    ap.add_argument("--rules", default="lru,lru-split,lfu,lfu-split,s3fifo,belady,belady-split")
    ap.add_argument("--max-pass", type=int, default=4, help="passes of more tokens than this are left out (prompt reading)")
    a = ap.parse_args()
    layout = model_layout(a.model, a.gguf_py)
    profile = read_profile(a.profile)
    passes = []
    for t in a.trace:
        passes += read_trace(t)
    n_before = len(passes)
    passes = [p for p in passes if max(len(ids) for _, ids in p) <= 10 * a.max_pass]
    n_tok = len(passes)
    lookups = sum(len(ids) for p in passes for _, ids in p)
    full_mb = sum(layout[il][0] * len(ids) for p in passes for il, ids in p) / 1e6 / n_tok
    print(f"{n_tok} passes ({n_before - n_tok} larger ones left out), {lookups} lookups, {full_mb:.0f} MB of experts per pass")
    g0 = Groups(layout, 1, profile, "size")
    for g in range(g0.n):
        print(f"  group {g}: {len(g0.layers[g])} layers, {g0.size[g] / 2**20:.2f} MiB per expert, {g0.host[g] / 2**30:.1f} GiB")
    for gib in [float(x) for x in a.cache_gib.split(",")]:
        print(f"cache {gib} GiB")
        for rule in a.rules.split(","):
            split = "profile" if rule.endswith("-split") else "size"
            base = rule.replace("-split", "")
            groups = Groups(layout, gib * 2**30, profile, split)
            pre = preload(groups, profile)
            if base == "lru":
                h, m, b = sim_lru(groups, pre, passes)
            elif base == "lfu":
                h, m, b = sim_lfu(groups, pre, passes, a.half_life, 4.0)
            elif base == "s3fifo":
                h, m, b = sim_s3fifo(groups, pre, passes)
            elif base == "belady":
                h, m, b = sim_belady(groups, pre, passes)
            else:
                continue
            print(f"  {rule:13s} hits {100 * h / (h + m):5.1f}%  {m / n_tok:6.1f} misses and {b / 1e6 / n_tok:6.1f} MB per pass"
                  f"   slots {groups.slots}", flush=True)


if __name__ == "__main__":
    main()
