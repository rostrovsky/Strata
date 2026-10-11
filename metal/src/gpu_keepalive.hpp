// metal/src/gpu_keepalive.hpp - keeps the GPU at its working clock while the engine waits for the SSD.
//
// A token is 48 layers, and at a layer whose routed experts are not all in the cache the GPU has nothing to do until
// they are read: about a millisecond.  An Apple GPU that idles that often is clocked down, and then runs the work
// between the waits several times slower.  Measured on an M4 MacBook Air (IQ3_S, every expert in the cache, a wait
// put in at each layer): no wait 101 ms per token; 0.5 ms 347 ms; 1.5 ms 409 ms - where the waits themselves are
// 24 and 72 ms.  So the waits, not the reads, were what a missed expert cost.
//
// Between begin() and end() this runs a small compute kernel in a loop on a command queue of its own, so the GPU
// never sees the gaps.  The engine holds it for the whole of a request; the kernel is small enough that the real
// work loses nothing to it (every expert cached: 104 ms per token with and without).
#pragma once

#include <memory>

namespace strata::metal {

class GpuKeepalive {
public:
    GpuKeepalive();
    ~GpuKeepalive();
    bool ok() const;
    void begin();   // a wait starts: keep the GPU busy
    void end();     // the wait is over
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}   // namespace strata::metal
