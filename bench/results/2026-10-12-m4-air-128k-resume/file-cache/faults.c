// faults FILE N MODE [THREADS]: time reading N random 16 KiB pages of FILE through a mapping (after dropping them
// from the file cache).  MODE: serial | willneed (madvise each page, then touch) | threads (THREADS touch in parallel)
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
static double now(void) { struct timeval t; gettimeofday(&t, 0); return t.tv_sec + t.tv_usec * 1e-6; }
static char* base; static size_t* at; static size_t n; static int n_threads = 1;
static volatile unsigned sink;
static void* touch(void* arg) {
    const size_t k = (size_t) arg; unsigned s = 0;
    for (size_t i = k; i < n; i += (size_t) n_threads) s += (unsigned char) base[at[i]];
    sink += s; return NULL;
}
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    n = (size_t) atol(argv[2]);
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    const size_t page = 16384, pages = (size_t) st.st_size / page;
    base = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(base, (size_t) st.st_size, MADV_RANDOM);
    at = malloc(n * sizeof *at);
    uint64_t x = 88172645463325252ull ^ (uint64_t) getpid();
    for (size_t i = 0; i < n; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; at[i] = (size_t) (x % pages) * page; }
    for (size_t i = 0; i < n; ++i) msync(base + at[i], page, MS_INVALIDATE);
    const double t0 = now();
    if (!strcmp(argv[3], "willneed")) {
        for (size_t i = 0; i < n; ++i) madvise(base + at[i], page, MADV_WILLNEED);
        printf("  madvise calls: %.3f s\n", now() - t0);
        touch((void*) 0);
    } else if (!strcmp(argv[3], "threads")) {
        n_threads = argc > 4 ? atoi(argv[4]) : 8;
        pthread_t th[64];
        for (int k = 0; k < n_threads; ++k) pthread_create(&th[k], NULL, touch, (void*) (size_t) k);
        for (int k = 0; k < n_threads; ++k) pthread_join(th[k], NULL);
    } else {
        touch((void*) 0);
    }
    const double dt = now() - t0;
    printf("%s%s%s: %zu pages in %.3f s, %.1f us a page, %.0f MiB/s\n", argv[3], argc > 4 ? " x" : "", argc > 4 ? argv[4] : "", n, dt,
           dt / (double) n * 1e6, (double) n * page / 1048576.0 / dt);
    for (size_t i = 0; i < n; ++i) msync(base + at[i], page, MS_INVALIDATE);   // leave the file cache as it was found
    return 0;
}
