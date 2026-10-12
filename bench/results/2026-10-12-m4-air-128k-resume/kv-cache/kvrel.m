// kvrel MIB MODE: when does the memory of a released Metal buffer (over vm_allocate memory, in a residency set) come back?
//   0: release the buffer and its set, wait      1: ... then vm_deallocate      2: ... then mmap(MAP_FIXED) over it
//   3: as 2, and only then a view of the first eighth with its own set
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
    printf("  %-50s system wired %+8.1f MiB, free+file %+8.1f MiB\n", what, w - w_first, f - f_first);
}
int main(int argc, char** argv) {
    const size_t mib = (size_t) atol(argv[1]), len = mib << 20, part = len / 8;
    const int mode = atoi(argv[2]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("mode %d\n", mode);
    say("start");
    vm_address_t p = 0;
    vm_allocate(mach_task_self(), &p, len, VM_FLAGS_ANYWHERE);
    @autoreleasepool {
        id<MTLBuffer> whole = [dev newBufferWithBytesNoCopy:(void*) p length:len options:MTLResourceStorageModeShared deallocator:nil];
        MTLResidencySetDescriptor* d = [[MTLResidencySetDescriptor alloc] init];
        id<MTLResidencySet> rset = [dev newResidencySetWithDescriptor:d error:nil];
        [rset addAllocation:whole]; [rset commit]; [rset requestResidency];
        say("whole buffer in a residency set");
        [rset endResidency];
        say("endResidency");
        [rset removeAllAllocations]; [rset commit];
        say("removeAllAllocations, commit");
        rset = nil; whole = nil;
    }
    say("released, pool drained");
    if (mode == 1) { vm_deallocate(mach_task_self(), p, len); say("vm_deallocate"); }
    if (mode >= 2) { mmap((void*) (p + part), len - part, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0); say("mmap(MAP_FIXED) over 7/8"); }
    usleep(500000); say("0.5 s later");
    if (mode == 4 || mode == 5) {
        if (mode == 5) { @autoreleasepool { id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer]; [cb commit]; [cb waitUntilCompleted]; } say("an empty command buffer ran"); }
        for (int i = 0; i < 16; ++i) { usleep(500000); char b[64]; snprintf(b, sizeof b, "%.1f s later", 1.0 + 0.5 * i); say(b); }
    }
    if (mode == 3) {
        @autoreleasepool {
            id<MTLBuffer> view = [dev newBufferWithBytesNoCopy:(void*) p length:part options:MTLResourceStorageModeShared deallocator:nil];
            MTLResidencySetDescriptor* d = [[MTLResidencySetDescriptor alloc] init];
            id<MTLResidencySet> rset = [dev newResidencySetWithDescriptor:d error:nil];
            [rset addAllocation:view]; [rset commit]; [rset requestResidency];
            say("a view of the first eighth in its own set");
            sleep(1); say("1 s later");
        }
    }
    return 0;
}
