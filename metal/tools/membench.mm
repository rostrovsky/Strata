// metal/tools/membench.mm - how fast this Mac's GPU and CPU read memory: the bound of a pass that reads every
// weight once (docs/APPLE_METAL.md, "How close to the hardware").
//
//     clang++ -O2 -std=c++17 -fobjc-arc -framework Metal -framework Foundation metal/tools/membench.mm -o membench && ./membench [GiB]
//
// The GPU kernel adds up a buffer of 32-bit words, every thread a stretch of its own (what a matrix-vector kernel
// does with its weights, minus the arithmetic); the CPU figure is the same sum on all performance cores.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sys/sysctl.h>

static NSString* const kSource = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void sum_words(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                      constant uint& span [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    const uint base = id * span;
    uint4 acc = uint4(0);
    for (uint i = 0; i < span; i += 4) {
        acc += uint4(src[base + i], src[base + i + 1], src[base + i + 2], src[base + i + 3]);
    }
    dst[id] = acc.x + acc.y + acc.z + acc.w;
}
)";

int main(int argc, char** argv) {
    @autoreleasepool {
        const double gib = argc > 1 ? atof(argv[1]) : 2.0;
        const size_t bytes = (size_t) (gib * 1024 * 1024 * 1024) & ~((size_t) (1 << 22) - 1);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&error];
        if (!lib) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum_words"] error:&error];
        id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memset(src.contents, 1, bytes);
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("%s, %.1f GiB buffer\n", dev.name.UTF8String, bytes / 1073741824.0);
        for (uint32_t span : {256u, 1024u, 4096u, 16384u, 65536u}) {   // words per thread
            const NSUInteger threads = bytes / 4 / span;
            id<MTLBuffer> dst = [dev newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
            double best = 1e9;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                [enc setBytes:&span length:4 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN(pso.maxTotalThreadsPerThreadgroup, 256), 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (rep > 0) best = MIN(best, cb.GPUEndTime - cb.GPUStartTime);
            }
            printf("GPU, %6u words per thread: %.1f GiB/s\n", span, bytes / 1073741824.0 / best);
        }
        int ncpu = 4;
        size_t len = sizeof ncpu;
        sysctlbyname("hw.perflevel0.physicalcpu", &ncpu, &len, nullptr, 0);
        const uint32_t* w = (const uint32_t*) src.contents;
        double best = 1e9;
        for (int rep = 0; rep < 4; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            std::vector<uint64_t> sums((size_t) ncpu);
            for (int t = 0; t < ncpu; ++t)
                th.emplace_back([&, t] {
                    const size_t n = bytes / 4 / (size_t) ncpu;
                    uint64_t s = 0;
                    for (size_t i = 0; i < n; ++i) s += w[(size_t) t * n + i];
                    sums[(size_t) t] = s;
                });
            for (auto& x : th) x.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("CPU, %d performance cores: %.1f GiB/s\n", ncpu, bytes / 1073741824.0 / best);
    }
    return 0;
}
