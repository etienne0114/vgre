// GDB remote-serial-protocol (RSP) server over a PtxInterpreter — VGRE's
// CUDA-GDB-compatible debug stepping (missingFeatures §6.1).
//
// Speaks the stock RSP that every gdb client implements: packet framing with
// checksums and ack/no-ack mode, qSupported, target.xml register description
// via qXfer:features:read, stop replies, g/G/p/P registers, m/M memory, Z0/z0
// software breakpoints, s/c step & continue, and per-lane thread control
// (Hg/qfThreadInfo/qsThreadInfo/T — each CUDA thread in the running CTA is a
// GDB thread, like cuda-gdb's lane model).
//
// The register file presented to the client is the kernel's own virtual PTX
// register set (declaration order) plus a 64-bit `pc` register that holds the
// current instruction index; `info registers`, `x/…`, breakpoints on
// instruction addresses, stepi, and thread switching all behave normally in an
// unmodified gdb.
//
// One client at a time; serve() blocks until the client detaches/kills or the
// kernel grid exits.
#ifndef VGRE_DEBUG_GDB_RSP_SERVER_H
#define VGRE_DEBUG_GDB_RSP_SERVER_H

#include <string>

#include "vgre/debug/ptx_interpreter.h"

namespace vgre {
namespace debug {

class GdbRspServer {
public:
    // The interpreter must already be launched (threads exist, PC = 0).
    explicit GdbRspServer(PtxInterpreter& interp) : interp_(interp) {}

    // Bind + listen on 127.0.0.1:port (port 0 = ephemeral). Returns the bound
    // port, or -1 on error. Call before serve().
    int listen(int port);
    int boundPort() const { return port_; }

    // Accept one client and run the RSP session to completion. Returns true if
    // the session ended cleanly (detach/kill/exit), false on socket error.
    bool serve();

    void closeListener();

private:
    std::string handlePacket(const std::string& p);

    std::string readRegistersHex(int thread) const;
    std::string stopReply(StopReason r) const;

    PtxInterpreter& interp_;
    int listenFd_ = -1;
    int clientFd_ = -1;
    int port_ = -1;
    bool noAck_ = false;
    int focusThread_ = 0;   // Hg selection (0-based lane; GDB tids are 1-based)
    bool killed_ = false;
};

}  // namespace debug
}  // namespace vgre

#endif  // VGRE_DEBUG_GDB_RSP_SERVER_H
