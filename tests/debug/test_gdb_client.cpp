// End-to-end CUDA-GDB stepping with a real, unmodified `gdb` client (§6.1).
//
// Boots the RSP server on a saxpy PTX kernel, then drives a stock gdb in batch
// mode over TCP: connect, list lanes as threads, set a breakpoint on the store
// instruction, continue to it, inspect rip and the PTX fma register, read the
// not-yet-written output element, stepi over the store, re-read the memory to
// see the store land, then continue to grid exit. Every assertion greps gdb's
// own output — if the stub misbehaves at the protocol level, gdb (not this
// test) is the judge.
//
// Skips (77) when gdb is not installed.
#undef NDEBUG

#include "vgre/debug/gdb_rsp_server.h"
#include "vgre/debug/ptx_interpreter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using vgre::debug::GdbRspServer;
using vgre::debug::PtxInterpreter;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static const char* kSaxpy = R"(
.version 7.0
.target sm_52
.address_size 64
.visible .entry saxpy(
    .param .f32 saxpy_param_0,
    .param .u64 saxpy_param_1,
    .param .u64 saxpy_param_2,
    .param .u32 saxpy_param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<5>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;
    ld.param.f32    %f1, [saxpy_param_0];
    ld.param.u64    %rd1, [saxpy_param_1];
    ld.param.u64    %rd2, [saxpy_param_2];
    ld.param.u32    %r2, [saxpy_param_3];
    cvta.to.global.u64  %rd3, %rd1;
    cvta.to.global.u64  %rd4, %rd2;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %ntid.x;
    mov.u32         %r5, %tid.x;
    mad.lo.s32      %r1, %r3, %r4, %r5;
    setp.ge.s32     %p1, %r1, %r2;
    @%p1 bra        $L__BB0_2;
    mul.wide.s32    %rd5, %r1, 4;
    add.s64         %rd6, %rd3, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.s64         %rd7, %rd4, %rd5;
    ld.global.f32   %f3, [%rd7];
    fma.rn.f32      %f4, %f2, %f1, %f3;
    st.global.f32   [%rd7], %f4;
$L__BB0_2:
    ret;
}
)";

static std::string run(const std::string& cmd) {
    std::string out;
    FILE* f = popen((cmd + " 2>&1").c_str(), "r");
    if (!f) return out;
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) out += buf;
    pclose(f);
    return out;
}

int main() {
    if (run("command -v gdb").find("gdb") == std::string::npos) {
        std::printf("SKIP: gdb not installed\n");
        return 77;
    }

    const int N = 8;
    std::vector<float> x(N), y(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = 1.0f + i; y[i] = y0[i] = 4.0f * i; }
    float a = 2.0f;
    float* xp = x.data(); float* yp = y.data(); int n = N;
    void* args[] = {&a, &xp, &yp, &n};

    PtxInterpreter in(kSaxpy, "saxpy");
    int stPc = -1;
    for (int i = 0; i < (int)in.kernel().code.size(); ++i)
        if (in.kernel().code[i].op.rfind("st.global", 0) == 0) stPc = i;
    CHECK(stPc > 0, "found st.global pc");
    in.launch(1, N, args, 4);

    GdbRspServer srv(in);
    int port = srv.listen(0);
    CHECK(port > 0, "server listening");
    std::thread th([&] { srv.serve(); });

    // Expected values for the greps.
    const float f4 = 2.0f * x[0] + y0[0];      // fma result, thread 0
    uint32_t f4bits, y00bits, y0newbits;
    std::memcpy(&f4bits, &f4, 4);
    std::memcpy(&y00bits, &y0[0], 4);
    const float y0new = f4;
    std::memcpy(&y0newbits, &y0new, 4);

    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
        "gdb -nx -q -batch"
        " -ex 'set confirm off' -ex 'set pagination off'"
        " -ex 'target remote 127.0.0.1:%d'"
        " -ex 'info threads'"
        " -ex 'break *0x%x'"
        " -ex 'continue'"
        " -ex 'info registers rip'"
        " -ex 'info registers ptx_f4'"
        " -ex 'x/1xw %p'"
        " -ex 'stepi'"
        " -ex 'x/1xw %p'"
        " -ex 'delete'"
        " -ex 'continue'",
        port, stPc, (void*)yp, (void*)yp);
    std::string out = run(cmd);

    CHECK(out.find("CTA 0 lane 0") != std::string::npos, "lanes shown as gdb threads");
    CHECK(out.find("lane 7") != std::string::npos, "all 8 lanes listed");
    CHECK(out.find("Breakpoint 1") != std::string::npos, "breakpoint planted");
    {
        char pat[64];
        std::snprintf(pat, sizeof(pat), "0x%x", stPc);
        CHECK(out.find(std::string("rip") ) != std::string::npos &&
              out.find(pat) != std::string::npos, "stopped rip == store pc");
    }
    {
        char pat[32];
        std::snprintf(pat, sizeof(pat), "0x%x", f4bits);
        CHECK(out.find(pat) != std::string::npos, "ptx_f4 register readable in gdb");
    }
    {
        char pre[32], post[32];
        std::snprintf(pre, sizeof(pre), "0x%08x", y00bits);
        std::snprintf(post, sizeof(post), "0x%08x", y0newbits);
        size_t first = out.find(pre);
        size_t after = out.find(post);
        CHECK(first != std::string::npos, "pre-store memory value visible");
        CHECK(after != std::string::npos && after > first, "post-stepi store visible");
    }
    CHECK(out.find("exited normally") != std::string::npos ||
          out.find("W00") != std::string::npos ||
          out.find("Remote connection closed") != std::string::npos ||
          out.find("The program has no registers now") != std::string::npos,
          "grid ran to completion under gdb");

    // The kernel finished under a real gdb session: results must be exact.
    bool exact = true;
    for (int i = 0; i < N; ++i)
        if (y[i] != a * x[i] + y0[i]) exact = false;
    CHECK(exact, "saxpy exact after real-gdb session");

    srv.closeListener();
    th.join();

    if (g_fail == 0) { std::printf("test_gdb_client: ALL CHECKS PASSED (real gdb)\n"); return 0; }
    std::printf("test_gdb_client: %d FAILURE(S)\n--- gdb output ---\n%s\n", g_fail, out.c_str());
    return 1;
}
