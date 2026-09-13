// GDB remote-serial-protocol server — see include/vgre/debug/gdb_rsp_server.h.
//
// The target description claims i386:x86-64 (so any stock gdb accepts it and
// drives the session) and carries the kernel's virtual PTX registers as an
// additional feature. `rip` is the PTX PC (instruction index); the x86 GP/FPU
// registers exist only to satisfy the architecture's required register file
// and read as zero. Everything the debugger actually manipulates — PC,
// breakpoints, stepping, PTX registers, global memory — is real.

#include "vgre/debug/gdb_rsp_server.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSESOCK ::close
#endif

namespace vgre {
namespace debug {

namespace {

const char kHex[] = "0123456789abcdef";

std::string toHex(const uint8_t* p, size_t n) {
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s += kHex[p[i] >> 4]; s += kHex[p[i] & 15]; }
    return s;
}

int unhexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool fromHex(const std::string& s, std::vector<uint8_t>& out) {
    if (s.size() % 2) return false;
    out.clear();
    out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) {
        int hi = unhexDigit(s[i]), lo = unhexDigit(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return true;
}

// Little-endian hex of an integer value, `bytes` wide (RSP register encoding).
std::string leHex(uint64_t v, int bytes) {
    std::string s;
    for (int i = 0; i < bytes; ++i) { uint8_t b = (uint8_t)(v >> (8 * i));
        s += kHex[b >> 4]; s += kHex[b & 15]; }
    return s;
}

uint64_t leUnhex(const std::string& s) {
    uint64_t v = 0;
    for (size_t i = 0; i + 1 < s.size() && i < 16; i += 2) {
        int hi = unhexDigit(s[i]), lo = unhexDigit(s[i + 1]);
        if (hi < 0 || lo < 0) break;
        v |= (uint64_t)((hi << 4) | lo) << (4 * i);
    }
    return v;
}

// Sanitize a PTX register name for target.xml ("%rd1" → "ptx_rd1").
std::string xmlName(const std::string& ptxReg) {
    std::string s = "ptx_";
    for (char c : ptxReg)
        if (std::isalnum((unsigned char)c) || c == '_') s += c;
    return s;
}

struct RegDesc {
    enum Kind { X86Zero, Rip, Ptx } kind;
    std::string ptxName;   // for Kind::Ptx
    int bytes;             // register width in the g packet
};

}  // namespace

// ── Socket plumbing ───────────────────────────────────────────────────────────

int GdbRspServer::listen(int port) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    listenFd_ = (int)::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) return -1;
    int one = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(listenFd_, (sockaddr*)&addr, sizeof(addr)) != 0 ||
        ::listen(listenFd_, 1) != 0) {
        CLOSESOCK(listenFd_);
        listenFd_ = -1;
        return -1;
    }
    socklen_t len = sizeof(addr);
    getsockname(listenFd_, (sockaddr*)&addr, &len);
    port_ = ntohs(addr.sin_port);
    return port_;
}

void GdbRspServer::closeListener() {
    if (listenFd_ >= 0) { CLOSESOCK(listenFd_); listenFd_ = -1; }
}

// ── Register file description (see file header) ───────────────────────────────

static void buildRegTable(const PtxInterpreter& in, std::vector<RegDesc>& table,
                          std::string& xml) {
    static const char* kGp[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
                                "rsp", "r8", "r9", "r10", "r11", "r12", "r13",
                                "r14", "r15"};
    static const char* kSeg[] = {"cs", "ss", "ds", "es", "fs", "gs"};
    static const char* kFpc[] = {"fctrl", "fstat", "ftag", "fiseg", "fioff",
                                 "foseg", "fooff", "fop"};
    std::ostringstream x;
    x << "<?xml version=\"1.0\"?>\n"
      << "<!DOCTYPE target SYSTEM \"gdb-target.dtd\">\n"
      << "<target version=\"1.0\">\n"
      << "<architecture>i386:x86-64</architecture>\n"
      << "<feature name=\"org.gnu.gdb.i386.core\">\n";
    table.clear();
    for (const char* r : kGp) {
        x << "  <reg name=\"" << r << "\" bitsize=\"64\" type=\"int64\"/>\n";
        table.push_back({RegDesc::X86Zero, "", 8});
    }
    x << "  <reg name=\"rip\" bitsize=\"64\" type=\"code_ptr\"/>\n";
    table.push_back({RegDesc::Rip, "", 8});
    x << "  <reg name=\"eflags\" bitsize=\"32\" type=\"int32\"/>\n";
    table.push_back({RegDesc::X86Zero, "", 4});
    for (const char* r : kSeg) {
        x << "  <reg name=\"" << r << "\" bitsize=\"32\" type=\"int32\"/>\n";
        table.push_back({RegDesc::X86Zero, "", 4});
    }
    for (int i = 0; i < 8; ++i) {
        x << "  <reg name=\"st" << i << "\" bitsize=\"80\" type=\"i387_ext\"/>\n";
        table.push_back({RegDesc::X86Zero, "", 10});
    }
    for (const char* r : kFpc) {
        x << "  <reg name=\"" << r << "\" bitsize=\"32\" type=\"int32\"/>\n";
        table.push_back({RegDesc::X86Zero, "", 4});
    }
    x << "</feature>\n"
      << "<feature name=\"org.vgre.ptx\">\n";
    for (const std::string& r : in.registerNames()) {
        x << "  <reg name=\"" << xmlName(r) << "\" bitsize=\"64\" type=\"int64\"/>\n";
        table.push_back({RegDesc::Ptx, r, 8});
    }
    x << "</feature>\n</target>\n";
    xml = x.str();
}

std::string GdbRspServer::readRegistersHex(int thread) const {
    std::vector<RegDesc> table;
    std::string xml;
    buildRegTable(interp_, table, xml);
    std::string out;
    for (const auto& r : table) {
        uint64_t v = 0;
        if (r.kind == RegDesc::Rip) v = (uint64_t)interp_.pcOf(thread);
        else if (r.kind == RegDesc::Ptx) interp_.readRegister(thread, r.ptxName, v);
        out += leHex(v, r.bytes);
    }
    return out;
}

std::string GdbRspServer::stopReply(StopReason r) const {
    if (r == StopReason::Exited || interp_.exited()) return "W00";
    char buf[64];
    // 'swbreak' tells gdb the PC needs no decrement adjustment (we advertise
    // swbreak+ in qSupported, so gdb expects the reason on breakpoint stops).
    std::snprintf(buf, sizeof(buf), "T05%sthread:%x;",
                  r == StopReason::Breakpoint ? "swbreak:;" : "",
                  interp_.stoppedThread() + 1);
    return buf;
}

// ── Packet dispatch ───────────────────────────────────────────────────────────

std::string GdbRspServer::handlePacket(const std::string& p) {
    if (p.empty()) return "";

    if (p == "?") return interp_.exited() ? "W00" : stopReply(StopReason::Breakpoint);
    if (p == "!") return "OK";
    if (p == "D") { killed_ = true; return "OK"; }
    if (p[0] == 'k') { killed_ = true; return ""; }

    if (p.rfind("qSupported", 0) == 0)
        return "PacketSize=4000;qXfer:features:read+;QStartNoAckMode+;swbreak+";
    if (p == "QStartNoAckMode") { noAck_ = true; return "OK"; }
    if (p == "qC") {
        char b[16];
        std::snprintf(b, sizeof(b), "QC%x", focusThread_ + 1);
        return b;
    }
    if (p == "qAttached") return "1";
    if (p == "vMustReplyEmpty") return "";
    if (p == "qfThreadInfo") {
        std::string r = "m";
        for (int i = 0; i < interp_.numThreads(); ++i) {
            if (i) r += ",";
            char b[16];
            std::snprintf(b, sizeof(b), "%x", i + 1);
            r += b;
        }
        return r;
    }
    if (p == "qsThreadInfo") return "l";
    if (p.rfind("qThreadExtraInfo,", 0) == 0) {
        int tid = (int)std::strtol(p.c_str() + 17, nullptr, 16) - 1;
        char b[64];
        std::snprintf(b, sizeof(b), "CTA %d lane %d", interp_.currentCta(), tid);
        return toHex((const uint8_t*)b, std::strlen(b));
    }
    if (p.rfind("qXfer:features:read:target.xml:", 0) == 0) {
        std::vector<RegDesc> table;
        std::string xml;
        buildRegTable(interp_, table, xml);
        size_t comma = p.find(',', 31);
        size_t off = std::strtoul(p.substr(31, comma - 31).c_str(), nullptr, 16);
        size_t len = std::strtoul(p.substr(comma + 1).c_str(), nullptr, 16);
        if (off >= xml.size()) return "l";
        std::string chunk = xml.substr(off, len);
        return (off + chunk.size() >= xml.size() ? "l" : "m") + chunk;
    }
    if (p[0] == 'q' || p[0] == 'Q' || p[0] == 'v') return "";  // politely unsupported

    if (p[0] == 'H') {                       // Hg<tid> / Hc<tid>
        long tid = std::strtol(p.c_str() + 2, nullptr, 16);
        if (p[1] == 'g' && tid > 0 && tid <= interp_.numThreads()) focusThread_ = (int)tid - 1;
        return "OK";
    }
    if (p[0] == 'T') {                       // thread alive?
        int tid = (int)std::strtol(p.c_str() + 1, nullptr, 16) - 1;
        return (tid >= 0 && tid < interp_.numThreads() && !interp_.threadExited(tid))
                   ? "OK" : "E01";
    }

    if (p[0] == 'g') return readRegistersHex(focusThread_);
    if (p[0] == 'G') {                       // write all registers
        std::vector<RegDesc> table;
        std::string xml;
        buildRegTable(interp_, table, xml);
        size_t at = 1;
        for (const auto& r : table) {
            if (at + r.bytes * 2 > p.size()) break;
            uint64_t v = leUnhex(p.substr(at, r.bytes * 2));
            if (r.kind == RegDesc::Ptx) interp_.writeRegister(focusThread_, r.ptxName, v);
            at += r.bytes * 2;
        }
        return "OK";
    }
    if (p[0] == 'p') {                       // read one register
        std::vector<RegDesc> table;
        std::string xml;
        buildRegTable(interp_, table, xml);
        size_t idx = std::strtoul(p.c_str() + 1, nullptr, 16);
        if (idx >= table.size()) return "E01";
        uint64_t v = 0;
        if (table[idx].kind == RegDesc::Rip) v = (uint64_t)interp_.pcOf(focusThread_);
        else if (table[idx].kind == RegDesc::Ptx)
            interp_.readRegister(focusThread_, table[idx].ptxName, v);
        return leHex(v, table[idx].bytes);
    }
    if (p[0] == 'P') {                       // write one register
        std::vector<RegDesc> table;
        std::string xml;
        buildRegTable(interp_, table, xml);
        size_t eq = p.find('=');
        size_t idx = std::strtoul(p.substr(1, eq - 1).c_str(), nullptr, 16);
        if (idx >= table.size() || eq == std::string::npos) return "E01";
        uint64_t v = leUnhex(p.substr(eq + 1));
        if (table[idx].kind == RegDesc::Ptx)
            interp_.writeRegister(focusThread_, table[idx].ptxName, v);
        return "OK";
    }

    if (p[0] == 'm') {                       // read memory
        size_t comma = p.find(',');
        uint64_t addr = std::strtoull(p.substr(1, comma - 1).c_str(), nullptr, 16);
        size_t len = std::strtoul(p.substr(comma + 1).c_str(), nullptr, 16);
        std::vector<uint8_t> buf(len);
        if (!len || !PtxInterpreter::readGlobal(addr, buf.data(), len)) return "E14";
        return toHex(buf.data(), len);
    }
    if (p[0] == 'M') {                       // write memory
        size_t comma = p.find(','), colon = p.find(':');
        uint64_t addr = std::strtoull(p.substr(1, comma - 1).c_str(), nullptr, 16);
        size_t len = std::strtoul(p.substr(comma + 1, colon - comma - 1).c_str(), nullptr, 16);
        std::vector<uint8_t> data;
        if (!fromHex(p.substr(colon + 1), data) || data.size() != len) return "E01";
        if (!PtxInterpreter::writeGlobal(addr, data.data(), len)) return "E14";
        return "OK";
    }

    if (p[0] == 'Z' || p[0] == 'z') {        // software breakpoints only
        if (p.size() < 3 || p[1] != '0') return "";
        uint64_t addr = std::strtoull(p.substr(3, p.find(',', 3) - 3).c_str(), nullptr, 16);
        if (p[0] == 'Z') interp_.setBreakpoint((int)addr);
        else interp_.clearBreakpoint((int)addr);
        return "OK";
    }

    if (p[0] == 'c') return stopReply(interp_.resume());
    if (p[0] == 's') return stopReply(interp_.stepThread(focusThread_));

    return "";
}

// ── RSP framing ───────────────────────────────────────────────────────────────

bool GdbRspServer::serve() {
    if (listenFd_ < 0) return false;
    clientFd_ = (int)accept(listenFd_, nullptr, nullptr);
    if (clientFd_ < 0) return false;
    int one = 1;
    setsockopt(clientFd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));

    std::string inbuf, lastSent;
    char chunk[4096];
    killed_ = false;

    auto sendRaw = [&](const std::string& s) {
        size_t at = 0;
        while (at < s.size()) {
            auto n = send(clientFd_, s.data() + at, (int)(s.size() - at), 0);
            if (n <= 0) return false;
            at += (size_t)n;
        }
        return true;
    };
    auto sendPacket = [&](const std::string& body) {
        uint8_t sum = 0;
        for (char c : body) sum = (uint8_t)(sum + (uint8_t)c);
        std::string pkt = "$" + body + "#";
        pkt += kHex[sum >> 4];
        pkt += kHex[sum & 15];
        lastSent = pkt;
        return sendRaw(pkt);
    };

    bool ok = true;
    while (!killed_) {
        auto n = recv(clientFd_, chunk, sizeof(chunk), 0);
        if (n <= 0) { ok = false; break; }
        inbuf.append(chunk, (size_t)n);

        for (;;) {
            // Discard acks / interrupts before the next '$'.
            size_t start = inbuf.find('$');
            bool nak = false;
            for (size_t i = 0; i < (start == std::string::npos ? inbuf.size() : start); ++i)
                if (inbuf[i] == '-') nak = true;
            if (nak && !lastSent.empty() && !noAck_) sendRaw(lastSent);
            if (start == std::string::npos) { inbuf.clear(); break; }
            size_t hash = inbuf.find('#', start);
            if (hash == std::string::npos || hash + 2 >= inbuf.size()) {
                inbuf.erase(0, start);
                break;                        // incomplete — need more bytes
            }
            std::string body = inbuf.substr(start + 1, hash - start - 1);
            int cks = (unhexDigit(inbuf[hash + 1]) << 4) | unhexDigit(inbuf[hash + 2]);
            inbuf.erase(0, hash + 3);
            uint8_t sum = 0;
            for (char c : body) sum = (uint8_t)(sum + (uint8_t)c);
            if (!noAck_) {
                if (sum != (uint8_t)cks) { sendRaw("-"); continue; }
                sendRaw("+");
            }
            std::string reply = handlePacket(body);
            if (body[0] == 'k') break;        // no reply to kill
            if (!sendPacket(reply)) { ok = false; killed_ = true; break; }
        }
    }
    CLOSESOCK(clientFd_);
    clientFd_ = -1;
    return ok || killed_;
}

}  // namespace debug
}  // namespace vgre
