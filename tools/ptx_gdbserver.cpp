// vgre_ptx_gdbserver — debug a PTX kernel with any stock gdb (CUDA-GDB-style
// stepping, missingFeatures §6.1).
//
//   vgre_ptx_gdbserver <file.ptx> <kernel> <gridX> <blockX> <port> <arg>...
//
// Each <arg> binds one kernel .param, in order:
//   buf:f32:<count>[=v0,v1,...]   float buffer (pointer param); prints after exit
//   buf:u32:<count>[=v0,...]      int buffer
//   f32:<value>                   float scalar
//   u32:<value> | u64:<value>     integer scalar
//
// Then from another terminal:
//   gdb -ex 'target remote 127.0.0.1:<port>'
//   (gdb) info threads              # CUDA lanes of the running CTA
//   (gdb) break *0x12               # instruction-index breakpoints
//   (gdb) continue / stepi
//   (gdb) info registers ptx_f4 rip # the kernel's own virtual registers
//   (gdb) x/8fw <buffer address>    # global memory (printed at startup)
//
// The instruction listing (PC → PTX source line) is printed at startup so
// breakpoint addresses are easy to pick. Port 0 chooses an ephemeral port.

#include "vgre/debug/gdb_rsp_server.h"
#include "vgre/debug/ptx_interpreter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using vgre::debug::GdbRspServer;
using vgre::debug::PtxInterpreter;

namespace {

struct Arg {
    std::vector<float> f32buf;
    std::vector<uint32_t> u32buf;
    float f32 = 0;
    uint64_t u64 = 0;
    void* ptr = nullptr;   // what gets passed (points at buf data or the scalar)
    bool isFloatBuf = false, isIntBuf = false;
};

bool parseArg(const std::string& s, Arg& a) {
    auto vals = [](const std::string& v) {
        std::vector<std::string> out;
        std::stringstream ss(v);
        std::string t;
        while (std::getline(ss, t, ',')) out.push_back(t);
        return out;
    };
    if (s.rfind("buf:f32:", 0) == 0) {
        std::string rest = s.substr(8);
        size_t eq = rest.find('=');
        int count = std::atoi(rest.c_str());
        if (count <= 0) return false;
        a.f32buf.assign((size_t)count, 0.0f);
        if (eq != std::string::npos) {
            auto v = vals(rest.substr(eq + 1));
            for (size_t i = 0; i < v.size() && i < a.f32buf.size(); ++i)
                a.f32buf[i] = std::strtof(v[i].c_str(), nullptr);
        }
        a.isFloatBuf = true;
        return true;
    }
    if (s.rfind("buf:u32:", 0) == 0) {
        std::string rest = s.substr(8);
        size_t eq = rest.find('=');
        int count = std::atoi(rest.c_str());
        if (count <= 0) return false;
        a.u32buf.assign((size_t)count, 0);
        if (eq != std::string::npos) {
            auto v = vals(rest.substr(eq + 1));
            for (size_t i = 0; i < v.size() && i < a.u32buf.size(); ++i)
                a.u32buf[i] = (uint32_t)std::strtoul(v[i].c_str(), nullptr, 0);
        }
        a.isIntBuf = true;
        return true;
    }
    if (s.rfind("f32:", 0) == 0) { a.f32 = std::strtof(s.c_str() + 4, nullptr); return true; }
    if (s.rfind("u32:", 0) == 0 || s.rfind("u64:", 0) == 0) {
        a.u64 = std::strtoull(s.c_str() + 4, nullptr, 0);
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <file.ptx> <kernel> <gridX> <blockX> <port> "
            "[buf:f32:N[=v,..] | buf:u32:N[=v,..] | f32:V | u32:V | u64:V]...\n",
            argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::stringstream ss;
    ss << f.rdbuf();

    try {
        PtxInterpreter interp(ss.str(), argv[2]);
        int gridX = std::atoi(argv[3]), blockX = std::atoi(argv[4]);
        int port = std::atoi(argv[5]);

        std::vector<Arg> owned((size_t)(argc - 6));
        std::vector<void*> args;
        // Stable pointer slots for buffer-pointer params.
        std::vector<void*> bufPtrs(owned.size(), nullptr);
        for (int i = 6; i < argc; ++i) {
            Arg& a = owned[(size_t)(i - 6)];
            if (!parseArg(argv[i], a)) {
                std::fprintf(stderr, "bad arg spec: %s\n", argv[i]);
                return 2;
            }
            if (a.isFloatBuf) { bufPtrs[(size_t)(i - 6)] = a.f32buf.data(); a.ptr = &bufPtrs[(size_t)(i - 6)]; }
            else if (a.isIntBuf) { bufPtrs[(size_t)(i - 6)] = a.u32buf.data(); a.ptr = &bufPtrs[(size_t)(i - 6)]; }
            else if (argv[i][0] == 'f') a.ptr = &a.f32;
            else a.ptr = &a.u64;
            args.push_back(a.ptr);
        }
        interp.launch(gridX, blockX, args.data(), (int)args.size());

        // Print the instruction listing (breakpoint addresses) and buffers.
        std::printf("kernel %s: %zu instructions, grid=%d block=%d\n",
                    argv[2], interp.kernel().code.size(), gridX, blockX);
        for (size_t i = 0; i < interp.kernel().code.size(); ++i)
            std::printf("  0x%02zx  %s\n", i, interp.kernel().code[i].text.c_str());
        for (size_t i = 0; i < owned.size(); ++i)
            if (owned[i].isFloatBuf || owned[i].isIntBuf)
                std::printf("param %zu buffer @ %p (%zu elems)\n", i, bufPtrs[i],
                            owned[i].isFloatBuf ? owned[i].f32buf.size()
                                                : owned[i].u32buf.size());

        GdbRspServer srv(interp);
        int bound = srv.listen(port);
        if (bound < 0) { std::fprintf(stderr, "cannot bind port %d\n", port); return 2; }
        std::printf("gdb remote stub listening on 127.0.0.1:%d\n", bound);
        std::printf("  gdb -ex 'target remote 127.0.0.1:%d'\n", bound);
        std::fflush(stdout);
        srv.serve();
        srv.closeListener();

        // Show final buffer contents so a headless session still yields results.
        for (size_t i = 0; i < owned.size(); ++i) {
            if (owned[i].isFloatBuf) {
                std::printf("param %zu (f32):", i);
                for (float v : owned[i].f32buf) std::printf(" %g", v);
                std::printf("\n");
            } else if (owned[i].isIntBuf) {
                std::printf("param %zu (u32):", i);
                for (uint32_t v : owned[i].u32buf) std::printf(" %u", v);
                std::printf("\n");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
