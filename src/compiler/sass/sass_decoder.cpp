// SASS ELF Decoder — SM80/SM90 opcode decoder + PTX synthesizer (Track 1)
//
// Parses a SASS-only cubin ELF binary, decodes SM80/SM90 instruction bundles,
// and synthesizes equivalent PTX. Used by cudart_shim.cpp when an ELF has no
// embedded PTX section.
//
// ELF section structure:
//   .text.<KernelName>   — SASS instructions in 32-byte bundles
//   .nv.info.<KernelName>— kernel parameter metadata (name from section title)
//
// SM80+ bundle format: 32-byte aligned blocks of 4 x 64-bit words.
//   Word 0: control word (skip)
//   Words 1-3: instructions
//
// Instruction encoding (SM80/SM90):
//   bits [62:55] — primary opcode class
//   bits [7:0]   — destination register Rd
//   bits [15:8]  — source register Rs1
//   bits [23:16] — source register Rs2
//   bits [31:24] — source register Rs3 / immediate

#include "sass_decoder.h"
#include <cstring>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

namespace vgre::sass {

// ── Portable ELF64 types (no dependency on <elf.h>) ───────────────────────────

struct Elf64_Ehdr {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};

struct Elf64_Shdr {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

static constexpr uint32_t ELF_MAGIC  = 0x464c457f; // "\x7fELF"
static constexpr uint8_t  ELFCLASS64 = 2;

// ── Helper: safely read a T from a byte buffer at offset ─────────────────────
template<typename T>
static bool safeRead(const uint8_t* buf, size_t bufSize, size_t offset, T& out) {
    if (offset + sizeof(T) > bufSize) return false;
    memcpy(&out, buf + offset, sizeof(T));
    return true;
}

// ── Opcode decoding ───────────────────────────────────────────────────────────
// For SM80/SM90, bits [62:55] encode the opcode class.
static uint32_t extractOpcode(uint64_t instr) {
    return static_cast<uint32_t>((instr >> 55) & 0xFF);
}

static uint8_t extractRd (uint64_t instr) { return static_cast<uint8_t>(instr & 0xFF); }
static uint8_t extractRs1(uint64_t instr) { return static_cast<uint8_t>((instr >> 8) & 0xFF); }
static uint8_t extractRs2(uint64_t instr) { return static_cast<uint8_t>((instr >> 16) & 0xFF); }
static uint8_t extractRs3(uint64_t instr) { return static_cast<uint8_t>((instr >> 24) & 0xFF); }

// Decode a single SM80/SM90 instruction to PTX text.
// Returns empty string for NOP or unknown (comment only for unknown).
static std::string decodeInstruction(uint64_t instr) {
    uint32_t op = extractOpcode(instr);
    uint8_t rd  = extractRd(instr);
    uint8_t rs1 = extractRs1(instr);
    uint8_t rs2 = extractRs2(instr);
    uint8_t rs3 = extractRs3(instr);

    char buf[128];
    switch (op) {
    case 0x223: // FFMA → fma.rn.f32
        snprintf(buf, sizeof(buf),
                 "    fma.rn.f32 %%f%u, %%f%u, %%f%u, %%f%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x20C: // FMUL → mul.rn.f32
        snprintf(buf, sizeof(buf),
                 "    mul.rn.f32 %%f%u, %%f%u, %%f%u;", rd, rs1, rs2);
        return buf;
    case 0x221: // FADD → add.rn.f32
        snprintf(buf, sizeof(buf),
                 "    add.rn.f32 %%f%u, %%f%u, %%f%u;", rd, rs1, rs2);
        return buf;
    case 0x1C4: // IMAD → mad.lo.s32
        snprintf(buf, sizeof(buf),
                 "    mad.lo.s32 %%r%u, %%r%u, %%r%u, %%r%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x1C0: // IADD3 → add.s32
        snprintf(buf, sizeof(buf),
                 "    add.s32 %%r%u, %%r%u, %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x2F5: // LDG → ld.global.u32
        snprintf(buf, sizeof(buf),
                 "    ld.global.u32 %%r%u, [%%rd%u];", rd, rs1);
        return buf;
    case 0x2F4: // STG → st.global.u32
        snprintf(buf, sizeof(buf),
                 "    st.global.u32 [%%rd%u], %%r%u;", rd, rs1);
        return buf;
    case 0x002: // MOV → mov.b32
        snprintf(buf, sizeof(buf),
                 "    mov.b32 %%r%u, %%r%u;", rd, rs1);
        return buf;
    case 0x009: // S2R → special register (tid.x)
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%tid.x;", rd);
        return buf;
    case 0x107: // MUFU → sin.approx.f32
        snprintf(buf, sizeof(buf),
                 "    sin.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x000: // NOP — emit nothing
        return "";

    // ── FP64 arithmetic ─────────────────────────────────────────────────────
    case 0x323: // DFMA → fma.rn.f64
        snprintf(buf, sizeof(buf),
                 "    fma.rn.f64 %%fd%u, %%fd%u, %%fd%u, %%fd%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x30C: // DMUL → mul.rn.f64
        snprintf(buf, sizeof(buf),
                 "    mul.rn.f64 %%fd%u, %%fd%u, %%fd%u;", rd, rs1, rs2);
        return buf;
    case 0x321: // DADD → add.rn.f64
        snprintf(buf, sizeof(buf),
                 "    add.rn.f64 %%fd%u, %%fd%u, %%fd%u;", rd, rs1, rs2);
        return buf;
    case 0x335: // DSETP → setp.lt.f64 (predicate set)
        snprintf(buf, sizeof(buf),
                 "    setp.lt.f64 %%p%u, %%fd%u, %%fd%u;", rd & 0x7, rs1, rs2);
        return buf;

    // ── FP16 arithmetic ─────────────────────────────────────────────────────
    case 0x435: // HFMA2 → fma.rn.f16x2
        snprintf(buf, sizeof(buf),
                 "    fma.rn.f16x2 %%h%u, %%h%u, %%h%u, %%h%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x428: // HMUL2 → mul.rn.f16x2
        snprintf(buf, sizeof(buf),
                 "    mul.rn.f16x2 %%h%u, %%h%u, %%h%u;", rd, rs1, rs2);
        return buf;
    case 0x430: // HADD2 → add.rn.f16x2
        snprintf(buf, sizeof(buf),
                 "    add.rn.f16x2 %%h%u, %%h%u, %%h%u;", rd, rs1, rs2);
        return buf;

    // ── Integer arithmetic (extended) ───────────────────────────────────────
    case 0x1C8: // ISCADD → shl + add (composite: rd = rs1 << imm + rs2)
        snprintf(buf, sizeof(buf),
                 "    mad.lo.u32 %%r%u, %%r%u, %u, %%r%u;", rd, rs1, (1u << (rs3 & 0x1f)), rs2);
        return buf;
    case 0x1D2: // SHF → shf.l.wrap.b32 (funnel-shift left)
        snprintf(buf, sizeof(buf),
                 "    shf.l.wrap.b32 %%r%u, %%r%u, %%r%u, %%r%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x1D3: // SHR → shr.s32
        snprintf(buf, sizeof(buf),
                 "    shr.s32 %%r%u, %%r%u, %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x1D4: // SHL → shl.b32
        snprintf(buf, sizeof(buf),
                 "    shl.b32 %%r%u, %%r%u, %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x1E0: // FLO → bfind.shiftamt.u32
        snprintf(buf, sizeof(buf),
                 "    bfind.shiftamt.u32 %%r%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1E1: // POPC → popc.b32
        snprintf(buf, sizeof(buf),
                 "    popc.b32 %%r%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1E2: // IABS → abs.s32
        snprintf(buf, sizeof(buf),
                 "    abs.s32 %%r%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1C1: // INEG → neg.s32
        snprintf(buf, sizeof(buf),
                 "    neg.s32 %%r%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1D0: // IMNMX → min/max.s32
        snprintf(buf, sizeof(buf),
                 "    min.s32 %%r%u, %%r%u, %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x1C6: // IMUL → mul.lo.s32
        snprintf(buf, sizeof(buf),
                 "    mul.lo.s32 %%r%u, %%r%u, %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x1CC: // LOP3 → lop3.b32 (3-way logical)
        snprintf(buf, sizeof(buf),
                 "    lop3.b32 %%r%u, %%r%u, %%r%u, %%r%u, 0xCAu, 0x0u;", rd, rs1, rs2, rs3);
        return buf;

    // ── Predicate / select ───────────────────────────────────────────────────
    case 0x20E: // ISETP → setp.lt.s32
        snprintf(buf, sizeof(buf),
                 "    setp.lt.s32 %%p%u, %%r%u, %%r%u;", rd & 0x7, rs1, rs2);
        return buf;
    case 0x20A: // FSETP → setp.lt.f32
        snprintf(buf, sizeof(buf),
                 "    setp.lt.f32 %%p%u, %%f%u, %%f%u;", rd & 0x7, rs1, rs2);
        return buf;
    case 0x208: // FMNMX → fmin.f32
        snprintf(buf, sizeof(buf),
                 "    min.f32 %%f%u, %%f%u, %%f%u;", rd, rs1, rs2);
        return buf;
    case 0x1CF: // SEL → selp.b32 (select using predicate)
        snprintf(buf, sizeof(buf),
                 "    selp.b32 %%r%u, %%r%u, %%r%u, %%p%u;", rd, rs1, rs2, rs3 & 0x7);
        return buf;

    // ── Shared / local memory ────────────────────────────────────────────────
    case 0x2F2: // LDS → ld.shared.u32
        snprintf(buf, sizeof(buf),
                 "    ld.shared.u32 %%r%u, [%%r%u];", rd, rs1);
        return buf;
    case 0x2F3: // STS → st.shared.u32
        snprintf(buf, sizeof(buf),
                 "    st.shared.u32 [%%r%u], %%r%u;", rd, rs1);
        return buf;
    case 0x2F0: // LDL → ld.local.u32
        snprintf(buf, sizeof(buf),
                 "    ld.local.u32 %%r%u, [%%r%u];", rd, rs1);
        return buf;
    case 0x2F1: // STL → st.local.u32
        snprintf(buf, sizeof(buf),
                 "    st.local.u32 [%%r%u], %%r%u;", rd, rs1);
        return buf;
    case 0x2F8: // LDG 64-bit → ld.global.u64
        snprintf(buf, sizeof(buf),
                 "    ld.global.u64 %%rd%u, [%%rd%u];", rd, rs1);
        return buf;
    case 0x2F9: // STG 64-bit → st.global.u64
        snprintf(buf, sizeof(buf),
                 "    st.global.u64 [%%rd%u], %%rd%u;", rd, rs1);
        return buf;
    case 0x2FA: // LDS 64-bit → ld.shared.u64
        snprintf(buf, sizeof(buf),
                 "    ld.shared.u64 %%rd%u, [%%r%u];", rd, rs1);
        return buf;
    case 0x2FB: // STS 64-bit → st.shared.u64
        snprintf(buf, sizeof(buf),
                 "    st.shared.u64 [%%r%u], %%rd%u;", rd, rs1);
        return buf;

    // ── Atomic operations ────────────────────────────────────────────────────
    case 0x38C: // ATOM.ADD.f32 → atom.global.add.f32
        snprintf(buf, sizeof(buf),
                 "    atom.global.add.f32 %%f%u, [%%rd%u], %%f%u;", rd, rs1, rs2);
        return buf;
    case 0x38D: // ATOM.ADD.s32 → atom.global.add.s32
        snprintf(buf, sizeof(buf),
                 "    atom.global.add.s32 %%r%u, [%%rd%u], %%r%u;", rd, rs1, rs2);
        return buf;
    case 0x38E: // ATOM.CAS → atom.global.cas.b32
        snprintf(buf, sizeof(buf),
                 "    atom.global.cas.b32 %%r%u, [%%rd%u], %%r%u, %%r%u;", rd, rs1, rs2, rs3);
        return buf;
    case 0x38F: // RED.ADD → red.global.add.s32
        snprintf(buf, sizeof(buf),
                 "    red.global.add.s32 [%%rd%u], %%r%u;", rs1, rs2);
        return buf;

    // ── Warp-level primitives ────────────────────────────────────────────────
    case 0x394: // SHFL.IDX → shfl.sync.idx.b32
        snprintf(buf, sizeof(buf),
                 "    shfl.sync.idx.b32 %%r%u, %%r%u, %%r%u, 0x1f, 0xffffffff;", rd, rs1, rs2);
        return buf;
    case 0x395: // SHFL.UP → shfl.sync.up.b32
        snprintf(buf, sizeof(buf),
                 "    shfl.sync.up.b32 %%r%u, %%r%u, %%r%u, 0x0, 0xffffffff;", rd, rs1, rs2);
        return buf;
    case 0x396: // SHFL.DOWN → shfl.sync.down.b32
        snprintf(buf, sizeof(buf),
                 "    shfl.sync.down.b32 %%r%u, %%r%u, %%r%u, 0x1f, 0xffffffff;", rd, rs1, rs2);
        return buf;
    case 0x397: // SHFL.BFLY → shfl.sync.bfly.b32
        snprintf(buf, sizeof(buf),
                 "    shfl.sync.bfly.b32 %%r%u, %%r%u, %%r%u, 0x1f, 0xffffffff;", rd, rs1, rs2);
        return buf;
    case 0x3A0: // VOTE → vote.sync.uni.pred
        snprintf(buf, sizeof(buf),
                 "    vote.sync.uni.pred %%p%u, %%p%u, 0xffffffff;", rd & 0x7, rs1 & 0x7);
        return buf;
    case 0x3A1: // VOTE.ALL → vote.sync.all.pred
        snprintf(buf, sizeof(buf),
                 "    vote.sync.all.pred %%p%u, %%p%u, 0xffffffff;", rd & 0x7, rs1 & 0x7);
        return buf;
    case 0x3A2: // VOTE.ANY → vote.sync.any.pred
        snprintf(buf, sizeof(buf),
                 "    vote.sync.any.pred %%p%u, %%p%u, 0xffffffff;", rd & 0x7, rs1 & 0x7);
        return buf;
    case 0x3A3: // MATCH.ANY → match.sync.any.b32
        snprintf(buf, sizeof(buf),
                 "    match.sync.any.b32 %%r%u, %%r%u, 0xffffffff;", rd, rs1);
        return buf;

    // ── Control flow ────────────────────────────────────────────────────────
    case 0x947: // BRA → bra.uni (unconditional branch)
        snprintf(buf, sizeof(buf),
                 "    bra.uni LABEL_%u;", rs1);
        return buf;
    case 0x94C: // BRA.pred → @%%pX bra (predicated branch)
        snprintf(buf, sizeof(buf),
                 "    @%%p%u bra LABEL_%u;", rd & 0x7, rs1);
        return buf;
    case 0x94D: // EXIT → exit (terminate thread)
        return "    exit;";
    case 0x94E: // RET → ret.uni (return from subroutine)
        return "    ret.uni;";
    case 0x94F: // CALL → call.uni (subroutine call)
        snprintf(buf, sizeof(buf),
                 "    call.uni (%%r%u), FUNC_%u;", rd, rs1);
        return buf;
    case 0x946: // BAR.SYNC → bar.sync 0 (CTA-wide barrier)
        return "    bar.sync 0;";
    case 0x945: // BAR.ARV → bar.arrive 0 (non-blocking arrive)
        return "    bar.arrive 0, 128;";
    case 0x943: // MEMBAR.GL → membar.gl (global memory fence)
        return "    membar.gl;";
    case 0x944: // MEMBAR.SYS → membar.sys
        return "    membar.sys;";
    case 0x941: // MEMBAR.CTA → membar.cta
        return "    membar.cta;";
    case 0x942: // BSYNC → bsync 0 (warp barrier sync)
        return "    bsync 0;";

    // ── Type conversion ──────────────────────────────────────────────────────
    case 0x1A8: // I2F.F32.S32 → cvt.rn.f32.s32
        snprintf(buf, sizeof(buf),
                 "    cvt.rn.f32.s32 %%f%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1A9: // I2F.F32.U32 → cvt.rn.f32.u32
        snprintf(buf, sizeof(buf),
                 "    cvt.rn.f32.u32 %%f%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1AA: // I2F.F64.S32 → cvt.rn.f64.s32
        snprintf(buf, sizeof(buf),
                 "    cvt.rn.f64.s32 %%fd%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1AB: // F2I.S32.F32 → cvt.rzi.s32.f32
        snprintf(buf, sizeof(buf),
                 "    cvt.rzi.s32.f32 %%r%u, %%f%u;", rd, rs1);
        return buf;
    case 0x1AC: // F2I.U32.F32 → cvt.rzi.u32.f32
        snprintf(buf, sizeof(buf),
                 "    cvt.rzi.u32.f32 %%r%u, %%f%u;", rd, rs1);
        return buf;
    case 0x1AD: // F2F.F32.F64 → cvt.rn.f32.f64
        snprintf(buf, sizeof(buf),
                 "    cvt.rn.f32.f64 %%f%u, %%fd%u;", rd, rs1);
        return buf;
    case 0x1AE: // F2F.F64.F32 → cvt.f64.f32
        snprintf(buf, sizeof(buf),
                 "    cvt.f64.f32 %%fd%u, %%f%u;", rd, rs1);
        return buf;
    case 0x1AF: // F2F.F16.F32 → cvt.rn.f16.f32
        snprintf(buf, sizeof(buf),
                 "    cvt.rn.f16.f32 %%h%u, %%f%u;", rd, rs1);
        return buf;
    case 0x1B0: // I2I.S16.S32 → cvt.s16.s32
        snprintf(buf, sizeof(buf),
                 "    cvt.s16.s32 %%rs%u, %%r%u;", rd, rs1);
        return buf;
    case 0x1B1: // I2I.S32.S16 → cvt.s32.s16
        snprintf(buf, sizeof(buf),
                 "    cvt.s32.s16 %%r%u, %%rs%u;", rd, rs1);
        return buf;

    // ── Special register reads ────────────────────────────────────────────────
    case 0x00A: // S2R → ctaid.x (block index x)
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%ctaid.x;", rd);
        return buf;
    case 0x00B: // S2R → ntid.x (block dimension x)
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%ntid.x;", rd);
        return buf;
    case 0x00C: // S2R → tid.y
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%tid.y;", rd);
        return buf;
    case 0x00D: // S2R → tid.z
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%tid.z;", rd);
        return buf;
    case 0x00E: // S2R → ctaid.y
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%ctaid.y;", rd);
        return buf;
    case 0x00F: // S2R → ctaid.z
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%ctaid.z;", rd);
        return buf;
    case 0x010: // S2R → laneid
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%laneid;", rd);
        return buf;
    case 0x011: // S2R → warpid
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%warpid;", rd);
        return buf;
    case 0x012: // S2R → nwarpid
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%nwarpid;", rd);
        return buf;
    case 0x013: // S2R → clock → %clock
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, %%clock;", rd);
        return buf;

    // ── MUFU extended (unary math functions) ────────────────────────────────
    case 0x108: // MUFU.COS → cos.approx.f32
        snprintf(buf, sizeof(buf),
                 "    cos.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x109: // MUFU.RCP → rcp.approx.f32
        snprintf(buf, sizeof(buf),
                 "    rcp.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x10A: // MUFU.RSQ → rsqrt.approx.f32
        snprintf(buf, sizeof(buf),
                 "    rsqrt.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x10B: // MUFU.SQRT → sqrt.approx.f32
        snprintf(buf, sizeof(buf),
                 "    sqrt.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x10C: // MUFU.LG2 → lg2.approx.f32
        snprintf(buf, sizeof(buf),
                 "    lg2.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x10D: // MUFU.EX2 → ex2.approx.f32
        snprintf(buf, sizeof(buf),
                 "    ex2.approx.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;

    // ── MOV extended ────────────────────────────────────────────────────────
    case 0x003: // MOV32I → mov.u32 immediate
        snprintf(buf, sizeof(buf),
                 "    mov.u32 %%r%u, 0x%x;", rd, rs1);
        return buf;
    case 0x004: // MOV64 → mov.u64
        snprintf(buf, sizeof(buf),
                 "    mov.u64 %%rd%u, %%rd%u;", rd, rs1);
        return buf;
    case 0x005: // PRMT → prmt.b32 (byte permute)
        snprintf(buf, sizeof(buf),
                 "    prmt.b32 %%r%u, %%r%u, %%r%u, %%r%u;", rd, rs1, rs2, rs3);
        return buf;

    // ── Float arithmetic extended ────────────────────────────────────────────
    case 0x20B: // FABS → abs.f32
        snprintf(buf, sizeof(buf),
                 "    abs.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x20D: // FNEG → neg.f32
        snprintf(buf, sizeof(buf),
                 "    neg.f32 %%f%u, %%f%u;", rd, rs1);
        return buf;
    case 0x209: // FCHK → isnand.f32 (NaN check)
        snprintf(buf, sizeof(buf),
                 "    testp.notanumber.f32 %%p%u, %%f%u;", rd & 0x7, rs1);
        return buf;
    case 0x205: // FCMP → setp.eq.f32 (compare)
        snprintf(buf, sizeof(buf),
                 "    setp.eq.f32 %%p%u, %%f%u, %%f%u;", rd & 0x7, rs1, rs2);
        return buf;

    // ── Tensor core (HMMA/WMMA) — emit placeholder intrinsic ──────────────
    case 0x538: // HMMA.884.F32.F32 → wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32
        snprintf(buf, sizeof(buf),
                 "    // HMMA_884_F32: wmma matmul rd=%u rs1=%u rs2=%u rs3=%u", rd, rs1, rs2, rs3);
        return buf;
    case 0x539: // HMMA.1688 → m16n8k16 variant
        snprintf(buf, sizeof(buf),
                 "    // HMMA_1688: wmma m16n8k16 rd=%u rs1=%u rs2=%u rs3=%u", rd, rs1, rs2, rs3);
        return buf;
    case 0x53A: // WGMMA — Hopper SM90 tensor core
        snprintf(buf, sizeof(buf),
                 "    // WGMMA.SM90 rd=%u rs1=%u rs2=%u", rd, rs1, rs2);
        return buf;

    default: {
        // Unknown opcode: emit as a comment, no real instruction
        snprintf(buf, sizeof(buf),
                 "    // SASS_UNKNOWN_OP_%x", op);
        return buf;
    }
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

std::string decodeSassToPtx(const uint8_t* data, size_t size) {
    if (!data || size < sizeof(Elf64_Ehdr)) return "";

    // Verify ELF magic and class
    uint32_t magic = 0;
    memcpy(&magic, data, 4);
    if (magic != ELF_MAGIC) return "";
    if (data[4] != ELFCLASS64) return "";  // not ELF64

    Elf64_Ehdr ehdr{};
    memcpy(&ehdr, data, sizeof(ehdr));

    if (ehdr.e_shoff == 0 || ehdr.e_shnum == 0) return "";
    if (ehdr.e_shoff + static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr) > size)
        return "";

    // Read section headers
    std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
    memcpy(shdrs.data(), data + ehdr.e_shoff, ehdr.e_shnum * sizeof(Elf64_Shdr));

    // Read section name string table
    if (ehdr.e_shstrndx >= ehdr.e_shnum) return "";
    const Elf64_Shdr& shstrtab = shdrs[ehdr.e_shstrndx];
    if (shstrtab.sh_offset + shstrtab.sh_size > size) return "";
    const char* strtab = reinterpret_cast<const char*>(data + shstrtab.sh_offset);
    size_t strtabSize  = static_cast<size_t>(shstrtab.sh_size);

    auto secName = [&](const Elf64_Shdr& sh) -> std::string {
        if (sh.sh_name >= strtabSize) return "";
        return std::string(strtab + sh.sh_name);
    };

    // Collect kernel names from .nv.info.* sections and .text.* sections
    std::vector<std::string> kernelNames;
    struct TextSection { std::string name; const uint8_t* ptr; size_t sz; };
    std::vector<TextSection> textSections;

    static const char kNvInfo[] = ".nv.info.";
    static const char kText[]   = ".text.";

    for (const auto& sh : shdrs) {
        std::string name = secName(sh);
        if (name.substr(0, sizeof(kText) - 1) == kText) {
            std::string kname = name.substr(sizeof(kText) - 1);
            if (!kname.empty()) {
                if (sh.sh_offset + sh.sh_size <= size) {
                    textSections.push_back({
                        kname,
                        data + sh.sh_offset,
                        static_cast<size_t>(sh.sh_size)
                    });
                    kernelNames.push_back(kname);
                }
            }
        } else if (name.substr(0, sizeof(kNvInfo) - 1) == kNvInfo) {
            std::string kname = name.substr(sizeof(kNvInfo) - 1);
            if (!kname.empty()) {
                // Only add if not already from .text.*
                bool found = false;
                for (const auto& k : kernelNames)
                    if (k == kname) { found = true; break; }
                if (!found) kernelNames.push_back(kname);
            }
        }
    }

    if (textSections.empty() && kernelNames.empty()) {
        // No named kernel sections — synthesize a generic kernel name
        kernelNames.push_back("sass_decoded_kernel");
        // Return minimal valid PTX stub since we have no instructions to decode
        return ".version 8.0\n.target sm_90\n.address_size 64\n\n"
               ".visible .entry sass_decoded_kernel(\n"
               "    .param .u64 param0,\n    .param .u64 param1\n)\n"
               "{\n    .reg .u64 %rd<64>;\n"
               "    ld.param.u64 %rd0, [param0];\n"
               "    ld.param.u64 %rd1, [param1];\n"
               "    ret;\n}\n";
    }

    // Build PTX output
    std::ostringstream ptx;
    ptx << ".version 8.0\n";
    ptx << ".target sm_90\n";
    ptx << ".address_size 64\n";

    // Emit one PTX kernel per decoded .text.* section
    for (const auto& ts : textSections) {
        ptx << "\n.visible .entry " << ts.name << "(\n";
        ptx << "    .param .u64 param0,\n";
        ptx << "    .param .u64 param1\n";
        ptx << ")\n{\n";
        ptx << "    .reg .f32 %f<256>;\n";
        ptx << "    .reg .u32 %r<256>;\n";
        ptx << "    .reg .u64 %rd<64>;\n";
        ptx << "    .reg .pred %p<16>;\n";
        ptx << "\n";
        ptx << "    ld.param.u64 %rd0, [param0];\n";
        ptx << "    ld.param.u64 %rd1, [param1];\n";
        ptx << "\n";

        // Decode instructions: process 32-byte bundles (4 x 64-bit words)
        // Bundle layout: [ctrl, instr0, instr1, instr2]
        // Every 4th 64-bit word (index 0, 4, 8, ...) is a control word.
        const uint64_t* words = reinterpret_cast<const uint64_t*>(ts.ptr);
        size_t numWords = ts.sz / sizeof(uint64_t);

        size_t instrCount = 0;
        for (size_t wi = 0; wi < numWords; ++wi) {
            if ((wi % 4) == 0) continue; // skip control word
            uint64_t instr = words[wi];
            std::string decoded = decodeInstruction(instr);
            if (!decoded.empty()) {
                ptx << decoded << "\n";
                ++instrCount;
            }
        }

        if (instrCount == 0) {
            // Emit a comment so the PTX is still valid
            ptx << "    // No decodeable SASS instructions found\n";
        }

        ptx << "\n    ret;\n}\n";
    }

    // Emit stubs for kernels found only in .nv.info.* (no .text.*)
    for (const auto& kname : kernelNames) {
        bool hasText = false;
        for (const auto& ts : textSections)
            if (ts.name == kname) { hasText = true; break; }
        if (hasText) continue;

        ptx << "\n.visible .entry " << kname << "(\n";
        ptx << "    .param .u64 param0,\n";
        ptx << "    .param .u64 param1\n";
        ptx << ")\n{\n";
        ptx << "    .reg .u64 %rd<64>;\n";
        ptx << "    ld.param.u64 %rd0, [param0];\n";
        ptx << "    ld.param.u64 %rd1, [param1];\n";
        ptx << "    ret;\n}\n";
    }

    std::string result = ptx.str();
    if (result.empty()) return "";
    return result;
}

} // namespace vgre::sass
