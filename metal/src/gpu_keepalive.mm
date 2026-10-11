// metal/src/gpu_keepalive.mm - see gpu_keepalive.hpp
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_keepalive.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace strata::metal {

// each thread does a short chain of multiply-adds: the kernel is sized by its grid, not by memory traffic
static NSString* const kSource = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void strata_keepalive(device float* buf [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    float x = buf[id];
    for (int i = 0; i < 48; ++i) {
        x = x * 1.0001f + 0.25f;
    }
    buf[id] = x - floor(x);
}
)";

struct GpuKeepalive::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> buf = nil;
    NSUInteger n = 0;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> active{false};
    bool quit = false;

    void loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return quit || active.load(); });
                if (quit) return;
            }
            while (active.load(std::memory_order_relaxed)) {
                @autoreleasepool {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
                    [enc setBuffer:buf offset:0 atIndex:0];
                    const NSUInteger group = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger) 256);
                    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
            }
        }
    }
};

GpuKeepalive::GpuKeepalive() : p_(new Impl) {
    @autoreleasepool {
        p_->dev = MTLCreateSystemDefaultDevice();
        if (p_->dev == nil) return;
        NSError* error = nil;
        id<MTLLibrary> lib = [p_->dev newLibraryWithSource:kSource options:nil error:&error];
        if (lib == nil) return;
        id<MTLFunction> fn = [lib newFunctionWithName:@"strata_keepalive"];
        if (fn == nil) return;
        p_->pso = [p_->dev newComputePipelineStateWithFunction:fn error:&error];
        if (p_->pso == nil) return;
        // STRATA_KEEPALIVE_THREADS: the kernel's grid (its length; grids of 16K-64K were the fastest of 16K-4M on an M4)
        const char* env = std::getenv("STRATA_KEEPALIVE_THREADS");
        p_->n = env ? (NSUInteger) std::max(1024, std::atoi(env)) : (NSUInteger) (1 << 15);
        p_->buf = [p_->dev newBufferWithLength:p_->n * sizeof(float) options:MTLResourceStorageModePrivate];
        p_->queue = [p_->dev newCommandQueue];
        if (p_->buf == nil || p_->queue == nil) {
            p_->pso = nil;
            return;
        }
    }
    p_->th = std::thread([this] { p_->loop(); });
}

GpuKeepalive::~GpuKeepalive() {
    if (p_->th.joinable()) {
        {
            std::lock_guard<std::mutex> lk(p_->mu);
            p_->quit = true;
            p_->active.store(false);
        }
        p_->cv.notify_all();
        p_->th.join();
    }
}

bool GpuKeepalive::ok() const { return p_->pso != nil; }

void GpuKeepalive::begin() {
    if (p_->pso == nil) return;
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->active.store(true);
    }
    p_->cv.notify_one();
}

void GpuKeepalive::end() { p_->active.store(false); }

}   // namespace strata::metal
