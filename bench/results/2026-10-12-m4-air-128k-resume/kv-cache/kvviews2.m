// kvviews2 MIB ORDER: ggml_metal_buffer_set_views' sequence without ARC, with GPU work afterwards.
//   ORDER 0: new view made resident first, then the whole buffer released (as written)
//   ORDER 1: the whole buffer released and its memory remapped first, then the view made
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
static void say(const char* what) {
    vm_statistics64_data_t s; mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &s, &n);
    static double w_first = -1, f_first = -1;
    const double w = s.wire_count * 16384.0 / 1048576.0, f = (s.free_count + s.external_page_count) * 16384.0 / 1048576.0;
    if (w_first < 0) { w_first = w; f_first = f; }
    printf("  %-56s system wired %+8.1f MiB, free+file %+8.1f MiB\n", what, w - w_first, f - f_first);
}
int main(int argc, char** argv) {
    const size_t mib = (size_t) atol(argv[1]), len = mib << 20, part = len / 8;
    const int order = atoi(argv[2]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void fill(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = 1.0f + (float) i; }\n" options:nil error:&e];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill"] error:&e];
    printf("order %d\n", order);
    say("start");
    vm_address_t p = 0;
    vm_allocate(mach_task_self(), &p, len, VM_FLAGS_ANYWHERE);
    id<MTLBuffer> whole = [dev newBufferWithBytesNoCopy:(void*) p length:len options:MTLResourceStorageModeShared deallocator:nil];
    MTLResidencySetDescriptor* d = [[MTLResidencySetDescriptor alloc] init];
    id<MTLResidencySet> rset_old = [dev newResidencySetWithDescriptor:d error:nil];
    [rset_old addAllocation:whole]; [rset_old commit]; [rset_old requestResidency];
    say("whole buffer in a residency set");
    id<MTLBuffer> view = nil; id<MTLResidencySet> rset_new = nil;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer]; [cb commit]; [cb waitUntilCompleted];
        if (order == 0) {
            view = [dev newBufferWithBytesNoCopy:(void*) p length:part options:MTLResourceStorageModeShared deallocator:nil];
            rset_new = [dev newResidencySetWithDescriptor:d error:nil];
            [rset_new addAllocation:view]; [rset_new commit]; [rset_new requestResidency];
        }
        [whole release];
        [rset_old endResidency]; [rset_old removeAllAllocations]; [rset_old commit]; [rset_old release];
    }
    say("whole buffer and its set released");
    mmap((void*) (p + part), len - part, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
    say("the rest remapped");
    if (order == 1) {
        usleep(200000); say("0.2 s later");
        view = [dev newBufferWithBytesNoCopy:(void*) p length:part options:MTLResourceStorageModeShared deallocator:nil];
        rset_new = [dev newResidencySetWithDescriptor:d error:nil];
        [rset_new addAllocation:view]; [rset_new commit]; [rset_new requestResidency];
        say("the view made, in its own set");
    }
    for (int round = 0; round < 4; ++round) {
        @autoreleasepool {
            for (int i = 0; i < 20; ++i) {
                id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps]; [enc setBuffer:view offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(1 << 20, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
        }
        char b[64]; snprintf(b, sizeof b, "after %d command buffers on the view", 20 * (round + 1)); say(b);
        usleep(500000);
    }
    sleep(2); say("2 s later");
    return 0;
}
