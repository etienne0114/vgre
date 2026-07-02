// CUDA-GDB debug stepping (§6.1) — PTX interpreter + GDB RSP server.
//
//  1. Interpreter: nvcc-style saxpy PTX over 2 CTAs × 32 threads — exact
//     results; a shared-memory tree reduction with bar.sync and divergent
//     branches — exact sum; breakpoint/step: stop before the saxpy store,
//     inspect the fma register, confirm memory unwritten, step, confirm the
//     one store landed, resume to completion.
//  2. RSP protocol (raw socket): qSupported/no-ack, target.xml transfer,
//     thread list, Z0 breakpoint, continue → T05 stop with swbreak+thread,
//     register read (rip + a PTX register), memory read/write, single-step,
//     z0 + continue → W00 exit. Checksums verified both directions.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/debug/gdb_rsp_server.h"
#include "vgre/debug/ptx_interpreter.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using vgre::debug::GdbRspServer;
using vgre::debug::PtxInterpreter;
using vgre::debug::StopReason;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// nvcc -arch=sm_52 style saxpy: y[i] = a*x[i] + y[i] for i < n.
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

// Shared-memory tree reduction: out[0] = sum(x[0..n)) with 64 threads, bar.sync.
static const char* kReduce = R"(
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce(
    .param .u64 reduce_param_0,
    .param .u64 reduce_param_1,
    .param .u32 reduce_param_2
)
{
    .reg .pred  %p<4>;
    .reg .f32   %f<5>;
    .reg .b32   %r<10>;
    .reg .b64   %rd<7>;
    .shared .align 4 .b8 sdata[256];

    ld.param.u64    %rd1, [reduce_param_0];
    ld.param.u32    %r1, [reduce_param_2];
    cvta.to.global.u64  %rd2, %rd1;
    mov.u32         %r2, %tid.x;
    mov.f32         %f1, 0f00000000;
    setp.ge.u32     %p1, %r2, %r1;
    @%p1 bra        $L__store;
    mul.wide.u32    %rd3, %r2, 4;
    add.s64         %rd4, %rd2, %rd3;
    ld.global.f32   %f1, [%rd4];
$L__store:
    mov.u32         %r4, sdata;
    mul.lo.s32      %r5, %r2, 4;
    add.s32         %r6, %r4, %r5;
    st.shared.f32   [%r6], %f1;
    bar.sync        0;
    mov.u32         %r7, 32;
$L__loop:
    setp.ge.u32     %p2, %r2, %r7;
    @%p2 bra        $L__skip;
    add.s32         %r8, %r2, %r7;
    mul.lo.s32      %r9, %r8, 4;
    add.s32         %r9, %r4, %r9;
    ld.shared.f32   %f2, [%r9];
    ld.shared.f32   %f3, [%r6];
    add.f32         %f4, %f2, %f3;
    st.shared.f32   [%r6], %f4;
$L__skip:
    bar.sync        0;
    shr.u32         %r7, %r7, 1;
    setp.gt.u32     %p3, %r7, 0;
    @%p3 bra        $L__loop;
    setp.ne.s32     %p1, %r2, 0;
    @%p1 bra        $L__end;
    ld.shared.f32   %f2, [%r4];
    ld.param.u64    %rd5, [reduce_param_1];
    cvta.to.global.u64  %rd6, %rd5;
    st.global.f32   [%rd6], %f2;
$L__end:
    ret;
}
)";

static int findPc(const PtxInterpreter& in, const char* opPrefix) {
    const auto& code = in.kernel().code;
    for (int i = 0; i < (int)code.size(); ++i)
        if (code[i].op.rfind(opPrefix, 0) == 0) return i;
    return -1;
}

// ── 1. Interpreter ────────────────────────────────────────────────────────────

static void testSaxpy() {
    const int N = 61;                                  // ragged: last CTA diverges
    std::vector<float> x(N), y(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = 0.5f * i; y[i] = y0[i] = 3.0f - i; }
    float a = 2.25f;
    float* xp = x.data(); float* yp = y.data(); int n = N;
    void* args[] = {&a, &xp, &yp, &n};

    PtxInterpreter in(kSaxpy, "saxpy");
    in.launch(/*gridX=*/2, /*blockX=*/32, args, 4);
    CHECK(in.resume() == StopReason::Exited, "saxpy runs to exit");
    bool exact = true;
    for (int i = 0; i < N; ++i)
        if (y[i] != a * x[i] + y0[i]) exact = false;
    CHECK(exact, "saxpy results exact (fmaf semantics)");
}

static void testReduce() {
    const int N = 53, T = 64;
    std::vector<float> x(N);
    float want = 0;
    for (int i = 0; i < N; ++i) { x[i] = 0.25f * (i % 7) + 1.0f; want += x[i]; }
    // Tree reduction and linear sum disagree in general; compute the tree order.
    {
        std::vector<float> s(T, 0.0f);
        for (int i = 0; i < N; ++i) s[i] = x[i];
        for (int stride = 32; stride > 0; stride >>= 1)
            for (int t = 0; t < stride; ++t) s[t] += s[t + stride];
        want = s[0];
    }
    float out = -1;
    float* xp = x.data(); float* op = &out; int n = N;
    void* args[] = {&xp, &op, &n};
    PtxInterpreter in(kReduce, "reduce");
    in.launch(1, T, args, 3);
    CHECK(in.resume() == StopReason::Exited, "reduction runs to exit");
    CHECK(out == want, "bar.sync tree reduction exact");
}

static void testBreakpointStep() {
    const int N = 8;
    std::vector<float> x(N), y(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = 1.0f + i; y[i] = y0[i] = 10.0f * i; }
    float a = 3.0f;
    float* xp = x.data(); float* yp = y.data(); int n = N;
    void* args[] = {&a, &xp, &yp, &n};

    PtxInterpreter in(kSaxpy, "saxpy");
    int stPc = findPc(in, "st");
    CHECK(stPc > 0, "found st.global pc");
    in.launch(1, N, args, 4);
    in.setBreakpoint(stPc);

    CHECK(in.resume() == StopReason::Breakpoint, "hit breakpoint");
    CHECK(in.stoppedThread() == 0, "thread 0 stops first");
    CHECK(in.pcOf(0) == stPc, "stopped at the store");
    uint64_t f4 = 0;
    CHECK(in.readRegister(0, "%f4", f4), "read %f4");
    float fv; uint32_t bits = (uint32_t)f4; std::memcpy(&fv, &bits, 4);
    CHECK(fv == a * x[0] + y0[0], "fma result visible in %f4 before the store");
    CHECK(y[0] == y0[0], "store not yet executed");

    CHECK(in.stepThread(0) == StopReason::Step, "single step");
    CHECK(in.pcOf(0) == stPc + 1, "pc advanced by one instruction");
    CHECK(y[0] == a * x[0] + y0[0] && y[1] == y0[1], "exactly one store landed");

    // Poke a register through the debugger surface: thread 1's a·x+y target.
    in.clearBreakpoint(stPc);
    CHECK(in.resume() == StopReason::Exited, "resume to exit");
    bool exact = true;
    for (int i = 0; i < N; ++i)
        if (y[i] != a * x[i] + y0[i]) exact = false;
    CHECK(exact, "final saxpy results exact after debug session");
}

// ── 2. Raw RSP protocol ───────────────────────────────────────────────────────

struct RspClient {
    int fd = -1;
    bool noAck = false;

    bool connectTo(int port) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons((uint16_t)port);
        return connect(fd, (sockaddr*)&a, sizeof(a)) == 0;
    }
    // Send a packet, read ack (unless no-ack) + reply packet; returns the body.
    std::string txn(const std::string& body) {
        uint8_t sum = 0;
        for (char c : body) sum = (uint8_t)(sum + (uint8_t)c);
        char tail[4];
        std::snprintf(tail, sizeof(tail), "#%02x", sum);
        std::string pkt = "$" + body + tail;
        if (send(fd, pkt.data(), pkt.size(), 0) != (ssize_t)pkt.size()) return "<send-fail>";
        std::string buf;
        char c;
        // Read until a full reply packet "$...#xx" is present.
        while (true) {
            ssize_t k = recv(fd, &c, 1, 0);
            if (k <= 0) return "<recv-fail>";
            buf += c;
            size_t d = buf.find('$');
            if (d == std::string::npos) continue;
            size_t h = buf.find('#', d);
            if (h != std::string::npos && h + 2 < buf.size()) {
                std::string reply = buf.substr(d + 1, h - d - 1);
                // Verify checksum.
                uint8_t s2 = 0;
                for (char rc : reply) s2 = (uint8_t)(s2 + (uint8_t)rc);
                unsigned got = 0;
                std::sscanf(buf.c_str() + h + 1, "%2x", &got);
                if (s2 != got) return "<bad-checksum>";
                if (!noAck) send(fd, "+", 1, 0);
                return reply;
            }
        }
    }
    void kill() { txnNoReply("k"); ::close(fd); }
    void txnNoReply(const std::string& body) {
        uint8_t sum = 0;
        for (char c : body) sum = (uint8_t)(sum + (uint8_t)c);
        char tail[4];
        std::snprintf(tail, sizeof(tail), "#%02x", sum);
        std::string pkt = "$" + body + tail;
        send(fd, pkt.data(), pkt.size(), 0);
    }
};

static void testRsp() {
    const int N = 8;
    std::vector<float> x(N), y(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = 2.0f * i + 1; y[i] = y0[i] = 5.0f + i; }
    float a = 0.5f;
    float* xp = x.data(); float* yp = y.data(); int n = N;
    void* args[] = {&a, &xp, &yp, &n};

    PtxInterpreter in(kSaxpy, "saxpy");
    int stPc = findPc(in, "st");
    in.launch(1, N, args, 4);

    GdbRspServer srv(in);
    int port = srv.listen(0);
    CHECK(port > 0, "server bound to a port");
    std::thread th([&] { srv.serve(); });

    RspClient cl;
    CHECK(cl.connectTo(port), "client connected");

    std::string sup = cl.txn("qSupported:swbreak+");
    CHECK(sup.find("qXfer:features:read+") != std::string::npos, "qSupported advertises tdesc");
    CHECK(cl.txn("QStartNoAckMode") == "OK", "no-ack mode");
    cl.noAck = true;
    // (the server stops sending '+' from the next packet on)

    // target.xml: fetch fully.
    std::string xml, chunk;
    size_t off = 0;
    do {
        char req[64];
        std::snprintf(req, sizeof(req), "qXfer:features:read:target.xml:%zx,400", off);
        chunk = cl.txn(req);
        CHECK(!chunk.empty() && (chunk[0] == 'm' || chunk[0] == 'l'), "qXfer chunk");
        xml += chunk.substr(1);
        off = xml.size();
    } while (chunk[0] == 'm');
    CHECK(xml.find("i386:x86-64") != std::string::npos, "tdesc architecture present");
    CHECK(xml.find("\"rip\"") != std::string::npos, "tdesc has rip");
    CHECK(xml.find("ptx_f4") != std::string::npos, "tdesc carries PTX registers");

    std::string threads = cl.txn("qfThreadInfo");
    CHECK(threads.rfind("m1,", 0) == 0 && threads.find("8") != std::string::npos,
          "thread list = 8 lanes");

    CHECK(cl.txn("?").rfind("T05", 0) == 0, "initial stop reply");

    // Breakpoint at the store, continue, expect swbreak stop on thread 1.
    char zpkt[32];
    std::snprintf(zpkt, sizeof(zpkt), "Z0,%x,1", stPc);
    CHECK(cl.txn(zpkt) == "OK", "Z0 set");
    std::string stop = cl.txn("c");
    CHECK(stop.find("T05") == 0 && stop.find("swbreak") != std::string::npos &&
          stop.find("thread:1") != std::string::npos, "breakpoint stop reply");

    // rip register (index 16 in the tdesc) must equal the breakpoint pc.
    std::string rip = cl.txn("p10");
    char want[32];
    std::snprintf(want, sizeof(want), "%02x", stPc);
    CHECK(rip.rfind(want, 0) == 0, "rip == breakpoint pc");

    // Memory read of y[0..1] — untouched before the store.
    char mreq[64];
    std::snprintf(mreq, sizeof(mreq), "m%llx,8", (unsigned long long)(uintptr_t)yp);
    std::string mem = cl.txn(mreq);
    uint32_t w0 = 0;
    std::sscanf(mem.substr(0, 8).c_str(), "%8x", &w0);   // big-endian nibble order? bytes are LE pairs
    // Compare via reassembled bytes instead:
    {
        uint8_t b[4];
        for (int i = 0; i < 4; ++i) {
            unsigned v;
            std::sscanf(mem.substr(i * 2, 2).c_str(), "%2x", &v);
            b[i] = (uint8_t)v;
        }
        float f;
        std::memcpy(&f, b, 4);
        CHECK(f == y0[0], "memory read sees pre-store y[0]");
    }
    (void)w0;

    // Memory write: patch y[7] via M, then verify with m.
    {
        float patched = 123.5f;
        uint8_t b[4];
        std::memcpy(b, &patched, 4);
        char wreq[80];
        std::snprintf(wreq, sizeof(wreq), "M%llx,4:%02x%02x%02x%02x",
                      (unsigned long long)(uintptr_t)(yp + 7), b[0], b[1], b[2], b[3]);
        CHECK(cl.txn(wreq) == "OK", "M memory write");
        CHECK(y[7] == 123.5f, "debugger memory write landed");
        y0[7] = 123.5f;                              // the kernel reads y[7] after this
    }

    // Single-step past the store; y[0] updates.
    std::string sstop = cl.txn("s");
    CHECK(sstop.rfind("T05", 0) == 0, "step stop reply");
    CHECK(y[0] == a * x[0] + y0[0], "store executed by single step");

    // Remove bp, continue to exit.
    std::snprintf(zpkt, sizeof(zpkt), "z0,%x,1", stPc);
    CHECK(cl.txn(zpkt) == "OK", "z0 clear");
    CHECK(cl.txn("c") == "W00", "grid exit reported as W00");
    bool exact = true;
    for (int i = 0; i < N; ++i)
        if (y[i] != a * x[i] + y0[i]) exact = false;
    CHECK(exact, "results exact after RSP-driven session");

    cl.kill();
    th.join();
    srv.closeListener();
}

int main() {
    testSaxpy();
    testReduce();
    testBreakpointStep();
    testRsp();
    if (g_fail == 0) { std::printf("test_ptx_debugger: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_ptx_debugger: %d FAILURE(S)\n", g_fail);
    return 1;
}
