// dropcost FILE: time mincore and msync(MS_INVALIDATE) over the whole file, twice (the second time nothing is cached)
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
static double now(void) { struct timeval t; gettimeofday(&t, 0); return t.tv_sec + t.tv_usec * 1e-6; }
int main(int argc, char** argv) {
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    const size_t size = (size_t) st.st_size, page = 16384, n = (size + page - 1) / page;
    char* p = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    char* vec = malloc(n);
    for (int round = 0; round < 2; ++round) {
        double t0 = now();
        mincore(p, size, vec);
        double t1 = now();
        size_t in = 0;
        for (size_t i = 0; i < n; ++i) in += vec[i] & 1;
        double t2 = now();
        msync(p, size, MS_INVALIDATE);
        double t3 = now();
        printf("%.2f GiB cached: mincore %.1f ms, msync(MS_INVALIDATE) %.1f ms\n", in * page / 1073741824.0, (t1 - t0) * 1e3, (t3 - t2) * 1e3);
    }
    return 0;
}
