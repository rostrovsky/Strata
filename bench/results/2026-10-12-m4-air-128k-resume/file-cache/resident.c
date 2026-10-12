// resident FILE...: how much of each file is in the OS file cache, per GiB of file offset (mincore on a fresh mapping)
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
int main(int argc, char** argv) {
    const size_t page = (size_t) getpagesize();
    double all = 0;
    int quiet = 0, evict = 0;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "-q")) { quiet = 1; continue; }
        if (!strcmp(argv[a], "--evict")) { evict = 1; continue; }
        int fd = open(argv[a], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) { perror(argv[a]); return 1; }
        const size_t size = (size_t) st.st_size;
        char* p = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { perror("mmap"); return 1; }
        const size_t gib = (size_t) 1 << 30, per = gib / page;
        char* vec = malloc(per);
        size_t total = 0;
        if (evict && msync(p, size, MS_INVALIDATE) != 0) perror("msync");
        if (!quiet) printf("%s (%.1f GiB)\n", argv[a], size / (double) gib);
        for (size_t at = 0; at < size; at += gib) {
            const size_t len = size - at < gib ? size - at : gib, n = (len + page - 1) / page;
            if (mincore(p + at, len, vec) != 0) { perror("mincore"); return 1; }
            size_t in = 0;
            for (size_t i = 0; i < n; ++i) in += (vec[i] & MINCORE_INCORE) != 0;
            total += in;
            if (!quiet && in * page >= 8u << 20) printf("  GiB %3zu: %7.1f MiB\n", at / gib, in * page / 1048576.0);
        }
        printf("  %s: %.3f GiB in the file cache%s\n", strrchr(argv[a], '-') - 8, total * page / (double) gib, evict ? " after evicting" : "");
        all += total * page / (double) gib;
        munmap(p, size); close(fd); free(vec);
    }
    printf("all files: %.2f GiB in the file cache\n", all);
    return 0;
}
