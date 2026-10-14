// givable_probe: how much memory macOS hands over before it compresses other programs' memory.
//
// Wires memory in steps (mlock: as Metal's resident buffers are), STEP_MIB every MS milliseconds, and stops at the
// first step after which macOS has compressed STOP_MIB or more, or wrote anything to swap, or at MAX_MIB.  Prints
// per step: what it holds, the free pages, the file cache, the floor the kernel keeps of the file cache
// (vm.vm_page_filecache_min), and what was compressed since the start.  Then it lets go of everything.
//
//   clang -O2 -o givable_probe givable_probe.c
//   givable_probe [STEP_MIB=64] [MS=50] [STOP_MIB=16] [MAX_MIB=8192]
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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

struct state {
    double free, spec, file, floor, compressed, swapouts;
};

static struct state now(void) {
    struct state s = {0};
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n);
    const double page = (double) sysconf(_SC_PAGESIZE), g = 1024.0 * 1024.0 * 1024.0;
    s.free = (double) (vm.free_count - vm.speculative_count) * page / g;
    s.spec = (double) vm.speculative_count * page / g;
    s.file = (double) vm.external_page_count * page / g;
    s.floor = (double) sysctl_u("vm.vm_page_filecache_min") * page / g;
    s.compressed = (double) vm.compressions * page / g;
    s.swapouts = (double) vm.swapouts;
    return s;
}

static double secs(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

int main(int argc, char** argv) {
    const size_t step = (size_t) (argc > 1 ? atof(argv[1]) : 64) << 20;
    const int ms = argc > 2 ? atoi(argv[2]) : 50;
    const double stop = (argc > 3 ? atof(argv[3]) : 16) / 1024.0;
    const size_t most = (size_t) (argc > 4 ? atof(argv[4]) : 8192) << 20;
    const struct state s0 = now();
    const double t0 = secs();
    printf("start: free %.2f GiB, speculative %.2f, file cache %.2f, its floor %.2f (vm.vm_page_filecache_min)\n", s0.free,
           s0.spec, s0.file, s0.floor);
    printf("     s  held GiB  free  spec  file  floor  compressed MiB\n");
    void* parts[4096];
    size_t n = 0, held = 0;
    int why = 0;
    while (held + step <= most && n < 4096) {
        void* p = mmap(NULL, step, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) break;
        if (mlock(p, step) != 0) memset(p, 1, step);
        parts[n++] = p;
        held += step;
        usleep((useconds_t) ms * 1000);
        const struct state s = now();
        printf("%6.2f  %8.2f  %4.2f  %4.2f  %4.2f  %5.2f  %8.0f\n", secs() - t0, (double) held / (1 << 30), s.free, s.spec,
               s.file, s.floor, (s.compressed - s0.compressed) * 1024.0);
        fflush(stdout);
        if (s.swapouts > s0.swapouts) {
            why = 2;
            break;
        }
        if (s.compressed - s0.compressed >= stop) {
            why = 1;
            break;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        munlock(parts[i], step);
        munmap(parts[i], step);
    }
    sleep(1);
    const struct state s1 = now();
    printf("%s at %.2f GiB held; started with %.2f GiB free (and %.2f speculative), file cache %.2f -> floor %.2f; "
           "compressed %.0f MiB in all, swap-outs %.0f\n",
           why == 1 ? "compressed" : why == 2 ? "SWAPPED" : "stopped without compressing", (double) held / (1 << 30), s0.free,
           s0.spec, s0.file, s0.floor, (s1.compressed - s0.compressed) * 1024.0, s1.swapouts - s0.swapouts);
    return why == 2 ? 2 : 0;
}
