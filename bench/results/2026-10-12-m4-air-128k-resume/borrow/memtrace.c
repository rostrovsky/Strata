// memtrace: once a second, the figures the engine's borrow decision reads (HostMemory in metal/src/main.cpp), in GiB:
// free pages (with the speculative ones), the file cache, the floor the kernel keeps of it (vm.vm_page_filecache_min),
// what is above the floor, wired, and the MiB macOS compressed and pages it wrote to swap in that second.
//   clang -O2 -o memtrace memtrace.c && ./memtrace > trace.txt
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

static uint64_t sysctl_u(const char* name) {
    uint64_t v = 0;
    size_t n = sizeof v;
    if (sysctlbyname(name, &v, &n, NULL, 0) != 0) return 0;
    if (n == 4) {
        uint32_t w;
        memcpy(&w, &v, 4);
        return w;
    }
    return v;
}

int main(void) {
    const double page = (double) sysconf(_SC_PAGESIZE), g = 1024.0 * 1024.0 * 1024.0;
    uint64_t c0 = 0, s0 = 0;
    int first = 1;
    printf("time      free  file  floor above  wired  compressed_MiB swapouts\n");
    for (;;) {
        vm_statistics64_data_t vm;
        mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n);
        const double floor = (double) sysctl_u("vm.vm_page_filecache_min") * page / g;
        const double file = (double) vm.external_page_count * page / g;
        char t[16];
        const time_t now = time(NULL);
        strftime(t, sizeof t, "%H:%M:%S", localtime(&now));
        if (!first)
            printf("%s %5.2f %5.2f %5.2f %5.2f %6.2f %8.0f %8llu\n", t, (double) vm.free_count * page / g, file, floor,
                   file > floor ? file - floor : 0.0, (double) vm.wire_count * page / g,
                   (double) (vm.compressions - c0) * page / (1024.0 * 1024.0), (unsigned long long) (vm.swapouts - s0));
        fflush(stdout);
        first = 0;
        c0 = vm.compressions;
        s0 = vm.swapouts;
        sleep(1);
    }
}
