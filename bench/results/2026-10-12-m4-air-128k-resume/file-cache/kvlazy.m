// kvlazy MIB: is a Metal buffer over untouched vm_allocate memory resident before the GPU writes to it, and after
// it wrote to a part of it?  Mirrors ggml-metal: newBufferWithBytesNoCopy, a residency set, requestResidency.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
static char* g_p; static size_t g_len;
static double resident(void) {
    const size_t n = g_len / 16384; static char* vec; if (!vec) vec = malloc(n);
    if (!g_p || mincore(g_p, g_len, vec) != 0) return -1;
    size_t in = 0; for (size_t i = 0; i < n; ++i) in += vec[i] & 1; return in * 16384 / 1048576.0;
}
static double footprint(void) {
    task_vm_info_data_t v; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &v, &n);
    return v.phys_footprint / 1048576.0;
}
static double wired(void) {
    vm_statistics64_data_t s; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &s, &n);
    return s.wire_count * 16384.0 / 1048576.0;
}
static void say(const char* what, double f0, double w0) {
    printf("%-58s footprint %+8.1f MiB, system wired %+8.1f MiB, buffer resident %8.1f MiB\n", what, footprint() - f0, wired() - w0, resident());
}
int main(int argc, char** argv) {
    @autoreleasepool {
        const size_t mib = argc > 1 ? (size_t) atol(argv[1]) : 512, len = mib << 20;
        const int use_rset = argc > 2 ? atoi(argv[2]) : 1;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void fill(device float* b [[buffer(0)]], constant uint& at [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[at + i] = 1.0f + (float) i; }\n"
                                                options:nil error:&e];
        if (!lib) { NSLog(@"%@", e); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill"] error:&e];
        // warm the pipeline on another small buffer so that its own memory is not counted below
        id<MTLBuffer> small = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        void (^run)(id<MTLBuffer>, uint32_t, uint32_t) = ^(id<MTLBuffer> b, uint32_t at, uint32_t n) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:b offset:0 atIndex:0];
            [enc setBytes:&at length:4 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };
        run(small, 0, 1024);
        const double f0 = footprint(), w0 = wired();
        vm_address_t p = 0;
        vm_allocate(mach_task_self(), &p, len, VM_FLAGS_ANYWHERE);
        g_p = (char*) p; g_len = len;
        say("vm_allocate", f0, w0);
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:(void*) p length:len options:MTLResourceStorageModeShared deallocator:nil];
        say("newBufferWithBytesNoCopy", f0, w0);
        id rset = nil;
        if (use_rset) {
            if (@available(macOS 15.0, *)) {
                MTLResidencySetDescriptor* d = [[MTLResidencySetDescriptor alloc] init];
                d.initialCapacity = 1;
                rset = [dev newResidencySetWithDescriptor:d error:&e];
                [rset addAllocation:buf];
                [rset commit];
                [rset requestResidency];
                say("residency set: requestResidency", f0, w0);
            }
        }
        const uint32_t mibf = (1u << 20) / 4;   // floats in a MiB
        run(buf, 0, 16 * mibf);
        say("the GPU wrote the first 16 MiB", f0, w0);
        run(buf, (uint32_t) (mib / 2) * mibf, 16 * mibf);
        say("the GPU wrote 16 MiB in the middle", f0, w0);
        if (use_rset) {
            if (@available(macOS 15.0, *)) { [rset requestResidency]; say("requestResidency again", f0, w0); }
        }
        const float* fp = (const float*) p;
        printf("read back: %.1f %.1f, untouched float at the 3/4 mark: %.1f\n", fp[1], fp[(mib / 2) * mibf + 2], fp[(mib * 3 / 4) * mibf]);
        say("the CPU read one float at the 3/4 mark", f0, w0);
        if (mlock((void*) p, 64u << 20) != 0) perror("mlock");
        say("mlock of the first 64 MiB", f0, w0);
        run(buf, 0, 16 * mibf);
        say("the GPU wrote the first 16 MiB again", f0, w0);
    }
    return 0;
}
