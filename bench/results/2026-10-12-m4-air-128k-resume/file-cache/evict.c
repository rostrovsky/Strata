// evict FILE GIB METHOD: try to drop one GiB of FILE from the OS file cache; mincore before and after
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
static double now(void) { struct timeval t; gettimeofday(&t, 0); return t.tv_sec + t.tv_usec * 1e-6; }
static double resident(char* p, size_t len, char* vec) {
    const size_t page = (size_t) getpagesize(), n = len / page;
    size_t in = 0;
    if (mincore(p, len, vec) != 0) { perror("mincore"); exit(1); }
    for (size_t i = 0; i < n; ++i) in += (vec[i] & MINCORE_INCORE) != 0;
    return in * page / 1048576.0;
}
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    const size_t gib = (size_t) 1 << 30, at = (size_t) atol(argv[2]) * gib;
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    char* p = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    char* vec = malloc(gib / (size_t) getpagesize());
    const double before = resident(p + at, gib, vec), t0 = now();
    int rc = -1;
    if (!strcmp(argv[3], "inval")) rc = msync(p + at, gib, MS_INVALIDATE);
    else if (!strcmp(argv[3], "inval-sync")) rc = msync(p + at, gib, MS_INVALIDATE | MS_SYNC);
    else if (!strcmp(argv[3], "inval-async")) rc = msync(p + at, gib, MS_INVALIDATE | MS_ASYNC);
    else if (!strcmp(argv[3], "kill")) rc = msync(p + at, gib, MS_KILLPAGES);
    else if (!strcmp(argv[3], "deact")) rc = msync(p + at, gib, MS_DEACTIVATE);
    else if (!strcmp(argv[3], "dontneed")) rc = madvise(p + at, gib, MADV_DONTNEED);
    else if (!strcmp(argv[3], "pageout")) rc = madvise(p + at, gib, MADV_PAGEOUT);
    else if (!strcmp(argv[3], "free")) rc = madvise(p + at, gib, MADV_FREE);
    const double t1 = now();
    if (rc != 0) perror(argv[3]);
    printf("%s GiB %s: %.1f MiB resident, %s (rc %d, %.1f ms), %.1f MiB after\n", argv[3], argv[2], before, argv[3], rc, (t1 - t0) * 1e3,
           resident(p + at, gib, vec));
    return 0;
}
