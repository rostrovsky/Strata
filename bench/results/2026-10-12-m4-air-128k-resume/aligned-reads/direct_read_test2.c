// Do F_NOCACHE preads leave pages in the file cache?  For each alignment variant: pick a 256 MiB region of the file
// that is not cached (mincore on a mapping that is never touched), read it in 8 MiB pieces, and count how much of
// the region is cached afterwards, and how many bytes the process read from disk.  Read-only.
#include <fcntl.h>
#include <libproc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static unsigned long long my_diskread(void) {
    struct rusage_info_v2 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t *) &ri) != 0) return 0;
    return ri.ri_diskio_bytesread;
}
static double now(void) { struct timeval t; gettimeofday(&t, 0); return t.tv_sec + t.tv_usec / 1e6; }
static size_t resident(char *map, off_t at, size_t len) {
    static char vec[1 << 16];
    size_t pages = len / 16384, n = 0;
    if (mincore(map + at, len, vec) != 0) { perror("mincore"); return 0; }
    for (size_t i = 0; i < pages; ++i) n += vec[i] & 1;
    return n;
}

struct variant { const char *name; off_t file_shift; size_t buf_shift; size_t len_cut; int piece_mib; };

int main(int argc, char **argv) {
    const size_t region = 256u << 20;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    char *map = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    int rfd = open(argv[1], O_RDONLY);
    fcntl(rfd, F_NOCACHE, 1);
    char *base;
    posix_memalign((void **) &base, 16384, (64u << 20) + 65536);
    memset(base, 1, (64u << 20) + 65536);
    struct variant v[] = {
        {"offset +4096 (a block, not a page), buffer page-aligned", 4096, 0, 0, 8},
        {"offset aligned, buffer +4096", 0, 4096, 0, 8},
        {"offset aligned, buffer +512", 0, 512, 0, 8},
        {"offset aligned, buffer +8", 0, 8, 0, 8},
        {"offset +512, buffer +512", 512, 512, 0, 8},
        {"offset +4096, buffer +4096, length -1", 4096, 4096, 1, 8},
        {"offset aligned, buffer aligned, 64 MiB pieces", 0, 0, 0, 64},
    };
    off_t at = (off_t) 2 << 30;
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; ++i) {
        while (at + (off_t) region + 65536 < st.st_size && resident(map, at, region) > 16) at += region;   // an uncached region
        if (at + (off_t) region + 65536 >= st.st_size) { printf("no uncached region left\n"); break; }
        const size_t piece = (size_t) v[i].piece_mib << 20;
        unsigned long long d0 = my_diskread(); double t0 = now(); size_t got = 0;
        for (size_t done = 0; done < region; done += piece) {
            ssize_t r = pread(rfd, base + v[i].buf_shift, piece - v[i].len_cut, at + (off_t) done + v[i].file_shift);
            if (r <= 0) { perror("pread"); return 1; }
            got += (size_t) r;
        }
        double s = now() - t0;
        usleep(200000);
        size_t res = resident(map, at, region);
        printf("%-76s %5.0f MiB/s, read from disk %4.0f MiB, left in the file cache %4.0f MiB of %zu\n", v[i].name, got / 1048576.0 / s,
               (my_diskread() - d0) / 1048576.0, res * 16384.0 / 1048576.0, region >> 20);
        at += region + (64 << 20);
    }
    return 0;
}
