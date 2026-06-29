// Real multi-process all-reduce for the lightweight libvgre_nn collective
// (cluster_collective.cpp). Forks a worker (rank 1) and runs rank 0 in the
// parent, both over loopback TCP, and asserts the buffer is genuinely
// sum-reduced across the two processes — i.e. the lightweight library can do
// real distributed training, not a single-node no-op.
//
// POSIX-only (fork); on Windows the same collective is exercised via the Python
// distributed tests. Skipped gracefully where fork is unavailable.

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
int main() { printf("SKIP (no fork on Windows)\n"); return 0; }
#else
#include <sys/wait.h>
#include <unistd.h>

extern "C" int vgre_cluster_all_reduce(void*, size_t, int);
extern "C" int vgre_cluster_world_size(void);

static int run(int rank) {
    setenv("VGRE_NN_WORLD_SIZE", "2", 1);
    char r[8];
    snprintf(r, sizeof r, "%d", rank);
    setenv("VGRE_NN_RANK", r, 1);
    setenv("VGRE_NN_MASTER_PORT", "29581", 1);

    if (vgre_cluster_world_size() != 2) {
        fprintf(stderr, "rank %d: world_size != 2\n", rank);
        return 1;
    }
    // rank 0 -> [1,2,3,4], rank 1 -> [10,20,30,40]; expected sum -> [11,22,33,44]
    float buf[4];
    for (int i = 0; i < 4; ++i) buf[i] = (rank == 0) ? float(i + 1) : float(10 * (i + 1));
    if (int rc = vgre_cluster_all_reduce(buf, 4, /*FLOAT32*/ 3)) {
        fprintf(stderr, "rank %d: all_reduce rc=%d\n", rank, rc);
        return 1;
    }
    const float expected[4] = {11, 22, 33, 44};
    for (int i = 0; i < 4; ++i) {
        if (buf[i] != expected[i]) {
            fprintf(stderr, "rank %d: buf[%d]=%g, expected %g\n", rank, i, buf[i], expected[i]);
            return 1;
        }
    }
    printf("rank %d OK: [%g %g %g %g]\n", rank, buf[0], buf[1], buf[2], buf[3]);
    return 0;
}

int main() {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) _exit(run(1));   // child = worker rank 1

    int rc0 = run(0);              // parent = reducer rank 0
    int status = 0;
    waitpid(pid, &status, 0);
    int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

    bool pass = (rc0 == 0 && rc1 == 0);
    printf("%s\n", pass ? "CLUSTER ALL-REDUCE PASS" : "CLUSTER ALL-REDUCE FAIL");
    return pass ? 0 : 1;
}
#endif
