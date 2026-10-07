// PTX conversion, precision, and cooperative-group instruction translation map.

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

// ── Blackwell tcgen05 emitters (P3-7) ────────────────────────────────────────
// tcgen05.mma [d-tmem], a-desc, b-desc [, idesc, enable-input-d] accumulates
// D = A·B (+D when enable-input-d) into the TMEM address d-tmem. After
// splitOperands: o[0]=d-tmem addr, o[1]=descA, o[2]=descB, o[4]=enable (if any).
static std::string tcgen05_mma_emit(const std::vector<std::string>& o,
                                    int M, int N, int K, int kind) {
    const std::string acc = o.size() > 4 ? o[4] : std::string("1");  // enable_input_d
    return "vgre_jit_tcgen05_mma((uint32_t)(" + (o.size() > 0 ? o[0] : std::string("0")) +
           "),(uint64_t)(" + (o.size() > 1 ? o[1] : std::string("0")) +
           "),(uint64_t)(" + (o.size() > 2 ? o[2] : std::string("0")) + ")," +
           std::to_string(M) + "," + std::to_string(N) + "," + std::to_string(K) + "," +
           std::to_string(kind) + ",(int)(" + acc + "));";
}

// tcgen05.ld.sync.aligned.<shape>.xN.b32 {r0..r_{N-1}}, [taddr]  — load N words
// from TMEM into the register list. o[0]={regs}, o[1]=taddr (brackets stripped).
static std::string tcgen05_ld_emit(const std::vector<std::string>& o, int n) {
    if (o.size() < 2)
        throw std::runtime_error("tcgen05.ld requires destination and address operands");
    auto regs = splitOperands(o[0].size() >= 2 && o[0].front() == '{'
                              ? o[0].substr(1, o[0].size() - 2) : o[0]);
    if (static_cast<int>(regs.size()) != n)
        throw std::runtime_error("tcgen05.ld register count does not match its xN modifier");
    std::string out = "{ uint32_t _tl[" + std::to_string(n) + "]; "
                      "vgre_jit_tcgen05_ld(_tl,(uint32_t)(" + o[1] + "),1," + std::to_string(n) + ");";
    for (int i = 0; i < n && i < static_cast<int>(regs.size()); ++i)
        out += " " + regs[i] + " = _tl[" + std::to_string(i) + "];";
    return out + " }";
}

// tcgen05.st.sync.aligned.<shape>.xN.b32 [taddr], {r0..r_{N-1}}  — store N words
// from the register list into TMEM. o[0]=taddr, o[1]={regs}.
static std::string tcgen05_st_emit(const std::vector<std::string>& o, int n) {
    if (o.size() < 2)
        throw std::runtime_error("tcgen05.st requires address and source operands");
    auto regs = splitOperands(o[1].size() >= 2 && o[1].front() == '{'
                              ? o[1].substr(1, o[1].size() - 2) : o[1]);
    if (static_cast<int>(regs.size()) != n)
        throw std::runtime_error("tcgen05.st register count does not match its xN modifier");
    std::string out = "{ uint32_t _ts[" + std::to_string(n) + "] = {";
    for (int i = 0; i < n; ++i) { out += (i < (int)regs.size() ? regs[i] : "0"); if (i + 1 < n) out += ","; }
    out += "}; vgre_jit_tcgen05_st((uint32_t)(" + o[0] + "),_ts,1," + std::to_string(n) + "); }";
    return out;
}

static std::string tf32_convert(const std::vector<std::string>& o,
                                const char* rounding,
                                bool satfinite = false,
                                bool relu = false,
                                bool positiveZeroOnly = false) {
    if (o.size() < 2) throw std::runtime_error("cvt.tf32.f32 needs destination and source");
    const char* mode = "NearestEven";
    if (std::string(rounding) == "rna") mode = "NearestAway";
    else if (std::string(rounding) == "rz") mode = "TowardZero";
    return o[0] + " = vgre_mma_detail::cvt_f32_to_tf32_bits((float)(" + o[1] + "),"
           "vgre_mma_detail::Tf32Rounding::" + mode + "," +
           (satfinite ? "true" : "false") + "," +
           (relu ? "true" : "false") + "," +
           (positiveZeroOnly ? "true" : "false") + ");";
}

static std::string float_to_integer_convert(const std::vector<std::string>& o,
                                            const char* destination,
                                            const char* source,
                                            const char* rounding) {
    if (o.size() < 2)
        throw std::runtime_error("float-to-integer cvt needs destination and source");
    return o[0] + " = vgre_ptx_conversion::cvt_float_to_integer<" + destination + ">((" +
           source + ")(" + o[1] + "),vgre_ptx_conversion::IntegerRounding::" + rounding + ");";
}

static std::string float_to_half_convert(const std::vector<std::string>& o,
                                         const char* rounding) {
    if (o.size() < 2)
        throw std::runtime_error("f32-to-f16 cvt needs destination and source");
    return o[0] + " = vgre_ptx_conversion::cvt_f32_to_f16((float)(" + o[1] +
           "),vgre_cuda::FloatRounding::" + rounding + ");";
}

static std::string float_to_float_convert(const std::vector<std::string>& o,
                                          const char* source,
                                          const char* rounding) {
    if (o.size() < 2)
        throw std::runtime_error("floating-point cvt needs destination and source");
    return o[0] + " = vgre_ptx_conversion::cvt_f64_to_f32((" + source + ")(" +
           o[1] + "),vgre_cuda::FloatRounding::" + rounding + ");";
}

static std::string integer_to_float_convert(const std::vector<std::string>& o,
                                            const char* destination,
                                            const char* source,
                                            const char* rounding) {
    if (o.size() < 2)
        throw std::runtime_error("integer-to-floating-point cvt needs destination and source");
    const char* converter = std::string(destination) == "float"
        ? "cvt_integer_to_f32" : "cvt_integer_to_f64";
    return o[0] + " = vgre_ptx_conversion::" + converter + "((" + source + ")(" +
           o[1] + "),vgre_cuda::FloatRounding::" + rounding + ");";
}

static std::string integer_to_integer_convert(const std::vector<std::string>& o,
                                              const char* destination,
                                              unsigned sourceBits,
                                              bool sourceSigned) {
    if (o.size() < 2)
        throw std::runtime_error("integer-to-integer cvt needs destination and source");
    return o[0] + " = vgre_ptx_conversion::cvt_integer_to_integer<" + destination + "," +
           std::to_string(sourceBits) + "," + (sourceSigned ? "true" : "false") +
           ">(" + o[1] + ");";
}

#define VGRE_CVT_F2I(KEY, DST, SRC, MODE) \
    {KEY, [](auto& o){ return float_to_integer_convert(o, DST, SRC, MODE); }}
#define VGRE_CVT_F2I_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE) \
    VGRE_CVT_F2I("cvt.rni." PTX_TYPE "." SOURCE_TYPE, CPP_TYPE, CPP_SOURCE_TYPE, "NearestEven"), \
    VGRE_CVT_F2I("cvt.rzi." PTX_TYPE "." SOURCE_TYPE, CPP_TYPE, CPP_SOURCE_TYPE, "TowardZero"), \
    VGRE_CVT_F2I("cvt.rmi." PTX_TYPE "." SOURCE_TYPE, CPP_TYPE, CPP_SOURCE_TYPE, "Downward"), \
    VGRE_CVT_F2I("cvt.rpi." PTX_TYPE "." SOURCE_TYPE, CPP_TYPE, CPP_SOURCE_TYPE, "Upward")
#define VGRE_CVT_F2I_TYPE(PTX_TYPE, CPP_TYPE) \
    VGRE_CVT_F2I_ROUNDED(PTX_TYPE, CPP_TYPE, "f16", "float"), \
    VGRE_CVT_F2I_ROUNDED(PTX_TYPE, CPP_TYPE, "bf16", "float"), \
    VGRE_CVT_F2I_ROUNDED(PTX_TYPE, CPP_TYPE, "f32", "float"), \
    VGRE_CVT_F2I_ROUNDED(PTX_TYPE, CPP_TYPE, "f64", "double")
#define VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, ROUNDING, MODE) \
    {"cvt." ROUNDING "." PTX_TYPE "." SOURCE_TYPE, [](auto& o){ \
        return integer_to_float_convert(o, CPP_TYPE, CPP_SOURCE_TYPE, MODE); }}
#define VGRE_CVT_I2F_TYPE(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE) \
    VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, "rn", "NearestEven"), \
    VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, "rna", "NearestAway"), \
    VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, "rz", "TowardZero"), \
    VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, "rm", "Downward"), \
    VGRE_CVT_I2F_ROUNDED(PTX_TYPE, CPP_TYPE, SOURCE_TYPE, CPP_SOURCE_TYPE, "rp", "Upward")
#define VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, SOURCE_PTX, SOURCE_BITS, SOURCE_SIGNED) \
    {"cvt." DEST_PTX "." SOURCE_PTX, [](auto& o){ \
        return integer_to_integer_convert(o, DEST_CPP, SOURCE_BITS, SOURCE_SIGNED); }}
#define VGRE_CVT_I2I_DEST(DEST_PTX, DEST_CPP) \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "s8", 8, true), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "u8", 8, false), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "s16", 16, true), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "u16", 16, false), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "s32", 32, true), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "u32", 32, false), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "s64", 64, true), \
    VGRE_CVT_I2I_ENTRY(DEST_PTX, DEST_CPP, "u64", 64, false)

static std::string fp8x2_convert_emit(const std::vector<std::string>& o,
                                      bool e4m3, bool roundTowardZero = false,
                                      bool relu = false) {
    if (o.size() < 3)
        throw std::runtime_error("packed FP8 conversion needs destination and two sources");
    const char* encoder = e4m3 ? "vgre_f32_to_fp8e4m3_satfinite"
                               : "vgre_f32_to_fp8e5m2_satfinite";
    const std::string flags = std::string(",") +
                              (roundTowardZero ? "true" : "false") + "," +
                              (relu ? "true" : "false") + ")";
    // PTX spells the operands d,b,a. It stores b in the low byte and a in the
    // high byte, so keep the textual operand order while building the result.
    return o[0] + " = (unsigned)" + encoder + "((float)(" + o[1] + ")" + flags +
           " | ((unsigned)" + encoder + "((float)(" + o[2] + ")" + flags + "<<8;";
}

const TranslateMap& getConversionMap() {
    static const TranslateMap kMap = {
        // Integer-to-integer conversions use PTX source-width interpretation,
        // then truncate or extend to each signed/unsigned destination width.
        VGRE_CVT_I2I_DEST("s8", "std::int8_t"),
        VGRE_CVT_I2I_DEST("u8", "std::uint8_t"),
        VGRE_CVT_I2I_DEST("s16", "std::int16_t"),
        VGRE_CVT_I2I_DEST("u16", "std::uint16_t"),
        VGRE_CVT_I2I_DEST("s32", "int"),
        VGRE_CVT_I2I_DEST("u32", "unsigned"),
        VGRE_CVT_I2I_DEST("s64", "long long"),
        VGRE_CVT_I2I_DEST("u64", "unsigned long long"),
        // FP32 → TF32 bit-format conversions. The output remains a b32
        // register containing the rounded TF32 value used by mma.sync.
        {"cvt.rn.tf32.f32", [](auto& o){ return tf32_convert(o, "rn"); }},
        {"cvt.rz.tf32.f32", [](auto& o){ return tf32_convert(o, "rz"); }},
        {"cvt.rna.tf32.f32", [](auto& o){ return tf32_convert(o, "rna"); }},
        {"cvt.rn.satfinite.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", true); }},
        {"cvt.rz.satfinite.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", true); }},
        {"cvt.rna.satfinite.tf32.f32", [](auto& o){ return tf32_convert(o, "rna", true); }},
        {"cvt.rn.relu.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", false, true); }},
        {"cvt.rz.relu.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", false, true); }},
        {"cvt.rn.satfinite.relu.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", true, true); }},
        {"cvt.rz.satfinite.relu.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", true, true); }},
        {"cvt.rn.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", false, false, true); }},
        {"cvt.rz.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", false, false, true); }},
        {"cvt.rn.satfinite.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", true, false, true); }},
        {"cvt.rz.satfinite.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", true, false, true); }},
        {"cvt.rn.relu.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", false, true, true); }},
        {"cvt.rz.relu.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", false, true, true); }},
        {"cvt.rn.satfinite.relu.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rn", true, true, true); }},
        {"cvt.rz.satfinite.relu.pzo.tf32.f32", [](auto& o){ return tf32_convert(o, "rz", true, true, true); }},
        // Float-to-integer conversion rounds according to the PTX integer
        // rounding modifier, then clamps to the destination range. f16/bf16
        // sources widen exactly to f32 before conversion; all forms use the
        // shared helper for defined NaN and range behavior.
        VGRE_CVT_F2I_TYPE("s8", "std::int8_t"),
        VGRE_CVT_F2I_TYPE("u8", "std::uint8_t"),
        VGRE_CVT_F2I_TYPE("s16", "std::int16_t"),
        VGRE_CVT_F2I_TYPE("u16", "std::uint16_t"),
        VGRE_CVT_F2I_TYPE("s32", "int"),
        VGRE_CVT_F2I_TYPE("u32", "unsigned"),
        VGRE_CVT_F2I_TYPE("s64", "long long"),
        VGRE_CVT_F2I_TYPE("u64", "unsigned long long"),
        // cvt.sat limits the rounded result to the integer destination range.
        VGRE_CVT_F2I("cvt.rzi.sat.u8.f32", "std::uint8_t", "float", "TowardZero"),
        // Integer-to-float uses exact significand rounding for every PTX
        // integer width; host casts would inherit the process rounding mode.
        VGRE_CVT_I2F_TYPE("f32", "float", "s8", "std::int8_t"),
        VGRE_CVT_I2F_TYPE("f32", "float", "u8", "std::uint8_t"),
        VGRE_CVT_I2F_TYPE("f32", "float", "s16", "std::int16_t"),
        VGRE_CVT_I2F_TYPE("f32", "float", "u16", "std::uint16_t"),
        VGRE_CVT_I2F_TYPE("f32", "float", "s32", "int"),
        VGRE_CVT_I2F_TYPE("f32", "float", "u32", "unsigned"),
        VGRE_CVT_I2F_TYPE("f32", "float", "s64", "long long"),
        VGRE_CVT_I2F_TYPE("f32", "float", "u64", "unsigned long long"),
        VGRE_CVT_I2F_TYPE("f64", "double", "s8", "std::int8_t"),
        VGRE_CVT_I2F_TYPE("f64", "double", "u8", "std::uint8_t"),
        VGRE_CVT_I2F_TYPE("f64", "double", "s16", "std::int16_t"),
        VGRE_CVT_I2F_TYPE("f64", "double", "u16", "std::uint16_t"),
        VGRE_CVT_I2F_TYPE("f64", "double", "s32", "int"),
        VGRE_CVT_I2F_TYPE("f64", "double", "u32", "unsigned"),
        VGRE_CVT_I2F_TYPE("f64", "double", "s64", "long long"),
        VGRE_CVT_I2F_TYPE("f64", "double", "u64", "unsigned long long"),
        // f32 → f16. Keep the PTX rounding mode explicit.
        {"cvt.rn.f16.f32", [](auto& o){ return float_to_half_convert(o, "NearestEven"); }},
        {"cvt.rna.f16.f32", [](auto& o){ return float_to_half_convert(o, "NearestAway"); }},
        {"cvt.rz.f16.f32", [](auto& o){ return float_to_half_convert(o, "TowardZero"); }},
        {"cvt.rm.f16.f32", [](auto& o){ return float_to_half_convert(o, "Downward"); }},
        {"cvt.rp.f16.f32", [](auto& o){ return float_to_half_convert(o, "Upward"); }},
        // Widening f16 -> f32 is exact and takes no rounding modifier.
        {"cvt.f32.f16", [](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        // f64 → f32 uses explicit IEEE rounding; f32 → f64 is exact for every
        // finite source, so its directed rounding forms are ordinary widening.
        {"cvt.rn.f32.f64", [](auto& o){ return float_to_float_convert(o, "double", "NearestEven"); }},
        {"cvt.rna.f32.f64", [](auto& o){ return float_to_float_convert(o, "double", "NearestAway"); }},
        {"cvt.rz.f32.f64", [](auto& o){ return float_to_float_convert(o, "double", "TowardZero"); }},
        {"cvt.rm.f32.f64", [](auto& o){ return float_to_float_convert(o, "double", "Downward"); }},
        {"cvt.rp.f32.f64", [](auto& o){ return float_to_float_convert(o, "double", "Upward"); }},
        // Widening f32 -> f64 is exact; PTX forbids a rounding modifier here.
        {"cvt.f64.f32", [](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        // saturating conversions
        {"cvt.sat.f32.f32",[](auto& o){
            return o[0]+" = vgre_ptx_conversion::cvt_sat_f32((float)("+o[1]+"));";
        }},
        {"cvt.rn.sat.f16.f32",[](auto& o){
            return o[0]+" = vgre_ptx_conversion::cvt_sat_f32_to_f16((float)("+o[1]+"));";
        }},
        // ── sqrt.rn.f32 (missing from core map) ────────────────────────────
        {"sqrt.rn.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rz.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rm.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rp.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        // ── FP16 vector loads ────────────────────────────────────────────────
        {"ld.global.v2.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[2]+"); "+o[0]+"=_hp[0]; "+o[1]+"=_hp[1]; }";
        }},
        {"ld.global.v4.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[4]+"); "+o[0]+"=_hp[0]; "+o[1]+"=_hp[1]; "+
                   o[2]+"=_hp[2]; "+o[3]+"=_hp[3]; }";
        }},
        {"st.global.v2.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[0]+"); _hp[0]="+o[1]+"; _hp[1]="+o[2]+"; }";
        }},
        {"st.global.v4.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[0]+"); _hp[0]="+o[1]+"; _hp[1]="+o[2]+
                   "; _hp[2]="+o[3]+"; _hp[3]="+o[4]+"; }";
        }},
        // ── Cooperative group primitives (Hopper / Ampere) ─────────────────
        // match.sync: return a bitmask of threads in the warp with the same value.
        // In serial CPU model all threads appear identical; return full mask.
        {"match.sync.eq.b32", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.eq.b64", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.lt.b32", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.lt.b64", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        // elect.sync: elect one thread as leader. Serial model → thread 0 is elected.
        {"elect.sync", [](auto& o){ return o[0]+" = 0; /* elect.sync serial → tid 0 */"; }},
        // grid.sync / griddepcontrol ────────────────────────────────────────
        {"grid.sync", [](auto&){ return "vgre_jit_syncgrid();"; }},
        {"griddepcontrol.launch_dependents", [](auto&){ return "/* griddepcontrol.launch_dependents */"; }},
        {"griddepcontrol.wait", [](auto&){ return "/* griddepcontrol.wait */"; }},
        // ── TMA 3D / 4D / 5D bulk copy ─────────────────────────────────────
        // Tile dimensions come from desc->boxDim (set by cuTensorMapEncodeTiled),
        // NOT from PTX instruction operands.  Use the _b variants.
        {"cp.async.bulk.tensor.3d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 3); }},
        {"cp.async.bulk.tensor.4d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 4); }},
        {"cp.async.bulk.tensor.5d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 5); }},
        // ── cp.reduce.async (shared → global atomic reduction) ─────────────
        {"cp.reduce.async.add.f32", [](auto& o){
            return "vgre_cp_reduce_async_add_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.add.f64", [](auto& o){
            return "vgre_cp_reduce_async_add_f64((double*)("+o[0]+"),(const double*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.min.f32", [](auto& o){
            return "vgre_cp_reduce_async_min_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.max.f32", [](auto& o){
            return "vgre_cp_reduce_async_max_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        // ── tcgen05.mma (Blackwell SM100) — accumulate into a TMEM address ─────
        // o[0] is the TMEM accumulator address (from tcgen05.alloc), not a raw
        // pointer; the result is read back with tcgen05.ld. kind: 0=bf16 1=f16
        // 2=tf32 3=e4m3 4=e5m2 5=e4m3e5m2.
        {"tcgen05.mma.cta_group::1.m64n256k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 16, 0); }},
        {"tcgen05.mma.cta_group::1.m64n128k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 16, 0); }},
        {"tcgen05.mma.cta_group::1.m64n256k16.f32.f16.f16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 16, 1); }},
        {"tcgen05.mma.cta_group::1.m64n128k16.f32.f16.f16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 16, 1); }},
        {"tcgen05.mma.cta_group::1.m64n256k8.f32.tf32.tf32",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 8, 2); }},
        {"tcgen05.mma.cta_group::1.m128n256k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 128, 256, 16, 0); }},
        // FP8 (E4M3 / E5M2, K=32) tcgen05.mma into TMEM.
        {"tcgen05.mma.cta_group::1.m64n256k32.f32.e4m3.e4m3",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 32, 3); }},
        {"tcgen05.mma.cta_group::1.m64n128k32.f32.e4m3.e4m3",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 32, 3); }},
        {"tcgen05.mma.cta_group::1.m64n256k32.f32.e5m2.e5m2",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 32, 4); }},
        // ── tcgen05 TMEM data path: alloc / dealloc / ld / st / cp / sync ──────
        // tcgen05.alloc writes the allocated TMEM address into shared memory.
        {"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", [](auto& o){
            return "*(uint32_t*)(" + (o.size()>0?o[0]:std::string("0")) +
                   ") = vgre_jit_tmem_alloc((int)(" + (o.size()>1?o[1]:std::string("0")) + "));";
        }},
        {"tcgen05.dealloc.cta_group::1.sync.aligned.b32", [](auto& o){
            return "vgre_jit_tmem_dealloc((uint32_t)(" + (o.size()>0?o[0]:std::string("0")) +
                   "),(int)(" + (o.size()>1?o[1]:std::string("0")) + "));";
        }},
        {"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
            [](auto&){ return "vgre_jit_tmem_relinquish();"; }},
        {"tcgen05.ld.sync.aligned.32x32b.x1.b32",  [](auto& o){ return tcgen05_ld_emit(o, 1); }},
        {"tcgen05.ld.sync.aligned.32x32b.x2.b32",  [](auto& o){ return tcgen05_ld_emit(o, 2); }},
        {"tcgen05.ld.sync.aligned.32x32b.x4.b32",  [](auto& o){ return tcgen05_ld_emit(o, 4); }},
        {"tcgen05.ld.sync.aligned.32x32b.x8.b32",  [](auto& o){ return tcgen05_ld_emit(o, 8); }},
        {"tcgen05.st.sync.aligned.32x32b.x1.b32",  [](auto& o){ return tcgen05_st_emit(o, 1); }},
        {"tcgen05.st.sync.aligned.32x32b.x2.b32",  [](auto& o){ return tcgen05_st_emit(o, 2); }},
        {"tcgen05.st.sync.aligned.32x32b.x4.b32",  [](auto& o){ return tcgen05_st_emit(o, 4); }},
        {"tcgen05.st.sync.aligned.32x32b.x8.b32",  [](auto& o){ return tcgen05_st_emit(o, 8); }},
        // Commit / wait / fence — synchronous on CPU (the MMA already completed).
        {"tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64", [](auto&){ return "/* tcgen05.commit serial */"; }},
        {"tcgen05.wait::ld.sync.aligned",  [](auto&){ return "/* tcgen05.wait::ld serial */"; }},
        {"tcgen05.wait::st.sync.aligned",  [](auto&){ return "/* tcgen05.wait::st serial */"; }},
        {"tcgen05.fence::before_thread_sync", [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"tcgen05.fence::after_thread_sync",  [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        // ── mbarrier (Hopper SM90 async pipeline synchronization) ──────────────
        // CPU serial emulation: all async copies complete synchronously, so
        // barriers are trivially satisfied.  init=noop, arrive=complete token,
        // test_wait/try_wait=1 (always done), wait=noop, inval=noop.
        {"mbarrier.init.shared.b64",       [](auto&){ return "/* mbarrier.init serial */"; }},
        {"mbarrier.init.shared::cta.b64",  [](auto&){ return "/* mbarrier.init serial */"; }},
        {"mbarrier.arrive.shared.b64", [](auto& o){
            // token = arrive(); in serial, token=0 (phase 0 immediately complete)
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive serial */";
        }},
        {"mbarrier.arrive.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive serial */";
        }},
        {"mbarrier.arrive.noComplete.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive.noComplete serial */";
        }},
        {"mbarrier.arrive.noComplete.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive.noComplete serial */";
        }},
        {"mbarrier.test_wait.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.test_wait always done in serial */";
        }},
        {"mbarrier.test_wait.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.test_wait always done */";
        }},
        {"mbarrier.try_wait.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait always done in serial */";
        }},
        {"mbarrier.try_wait.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait always done */";
        }},
        {"mbarrier.try_wait.parity.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait.parity always done */";
        }},
        {"mbarrier.try_wait.parity.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait.parity always done */";
        }},
        {"mbarrier.wait.shared.b64",        [](auto&){ return "/* mbarrier.wait serial noop */"; }},
        {"mbarrier.wait.shared::cta.b64",   [](auto&){ return "/* mbarrier.wait serial noop */"; }},
        {"mbarrier.wait.parity.shared.b64", [](auto&){ return "/* mbarrier.wait.parity serial noop */"; }},
        {"mbarrier.wait.parity.shared::cta.b64",[](auto&){ return "/* mbarrier.wait.parity noop */"; }},
        {"mbarrier.inval.shared.b64",       [](auto&){ return "/* mbarrier.inval serial */"; }},
        {"mbarrier.inval.shared::cta.b64",  [](auto&){ return "/* mbarrier.inval serial */"; }},
        {"mbarrier.complete_tx.shared.b64", [](auto&){
            // In serial CPU model all async copies complete synchronously: noop.
            return "/* mbarrier.complete_tx serial (copy already synchronous) */";
        }},
        {"mbarrier.complete_tx.shared::cta.b64", [](auto&){
            return "/* mbarrier.complete_tx serial */";
        }},
        // ── fence.proxy (Hopper async-copy memory ordering) ────────────────────
        // On CPU, all operations are already sequentially consistent.
        {"fence.proxy.async",                          [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.async.shared::cta",              [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.async.global",                   [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.tensormap::generic.acquire.gpu", [](auto&){ return "/* fence.proxy.tensormap acquire */"; }},
        {"fence.proxy.tensormap::generic.release.cta", [](auto&){ return "/* fence.proxy.tensormap release */"; }},
        {"fence.proxy.tensormap::generic.acquire.cta", [](auto&){ return "/* fence.proxy.tensormap acquire.cta */"; }},
        {"fence.proxy.tensormap::generic.release.gpu", [](auto&){ return "/* fence.proxy.tensormap release.gpu */"; }},
        // ── tensormap.replace (update TMA descriptor address at runtime) ───────
        // The TMA descriptor is laid out as VgreTMADescriptor; baseAddr is at offset 0.
        {"tensormap.replace.tile.global_address.shared::cta.b1024.b64", [](auto& o){
            return "{VgreTMADescriptor* _td=(VgreTMADescriptor*)(uintptr_t)("+
                   (o.size()>0?o[0]:std::string("0"))+
                   "); if(_td) _td->baseAddr=(void*)(uintptr_t)("+
                   (o.size()>1?o[1]:std::string("0"))+"); } /* tensormap.replace.tile.global_address */";
        }},
        // NOTE: the 1d/2d cp.async.bulk.tensor STORE opcodes are handled in
        // getMap() (via tma_store_emit) — getMap is searched first, so duplicating
        // them here would be dead code. Only the 3d/4d/5d store forms live above.
        // ── Prefetch / cache hint variants (no-ops on CPU) ─────────────────────
        {"cp.async.bulk.prefetch.tensor.2d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 2D */"; }},
        {"cp.async.bulk.prefetch.tensor.3d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 3D */"; }},
        {"cp.async.bulk.prefetch.tensor.4d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 4D */"; }},
        {"cp.async.bulk.prefetch.tensor.5d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 5D */"; }},

        // ── elect.sync — cooperative group leader election (SM90+) ─────────────
        // In serial CPU model there is exactly one thread per warp, so it is
        // always "elected".  Output predicate is always true; membermask ignored.
        {"elect.sync", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1;" : "")
                   + " /* elect.sync: always elected in serial model */";
        }},

        // ── griddepcontrol — CDP2 grid dependency (CUDA 12.0+) ────────────────
        // All streams execute serially in VGRE; dependencies are always satisfied.
        {"griddepcontrol.launch_dependents", [](auto&){
            return "/* griddepcontrol.launch_dependents: serial noop */";
        }},
        {"griddepcontrol.wait", [](auto&){
            return "/* griddepcontrol.wait: serial noop */";
        }},
        {"griddepcontrol.wait_ifnot_lbi", [](auto&){
            return "/* griddepcontrol.wait_ifnot_lbi: serial noop */";
        }},

        // ── setmaxnreg — register-file reconfiguration (Ada SM89+) ───────────
        // No register file on CPU; ignored.
        {"setmaxnreg.inc.sync.aligned.u32", [](auto&){
            return "/* setmaxnreg.inc: no-op on CPU */";
        }},
        {"setmaxnreg.dec.sync.aligned.u32", [](auto&){
            return "/* setmaxnreg.dec: no-op on CPU */";
        }},

        // ── cp.reduce.async.bulk — write-combining reduction to global ─────────
        // On CPU emulator all memory is directly accessible; perform the add
        // reduction immediately (no async needed).
        {"cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.f32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.1d */");
            return "{ float* _dst=(float*)(uintptr_t)("+o[0]+");"
                   " const float* _src=(const float*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},
        {"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.f32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.2d */");
            return "{ float* _dst=(float*)(uintptr_t)("+o[0]+");"
                   " const float* _src=(const float*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},
        // Integer accumulation variant
        {"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.u32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.u32 */");
            return "{ uint32_t* _dst=(uint32_t*)(uintptr_t)("+o[0]+");"
                   " const uint32_t* _src=(const uint32_t*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},

        // ── FP8 cvt — scalar and packed E4M3/E5M2 conversions ─────────────────
        // Scalar: FP8 → f32 (byte in low 8 bits of source register)
        {"cvt.rn.f32.e4m3", [](auto& o){
            return o[0]+" = vgre_fp8e4m3_to_f32((uint8_t)((unsigned)("+o[1]+")&0xFFu));";
        }},
        {"cvt.rn.f32.e5m2", [](auto& o){
            return o[0]+" = vgre_fp8e5m2_to_f32((uint8_t)((unsigned)("+o[1]+")&0xFFu));";
        }},
        // Scalar: f32 → FP8 (result in low 8 bits of destination register)
        {"cvt.rn.e4m3.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e4m3((float)("+o[1]+"));";
        }},
        {"cvt.rn.e5m2.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e5m2((float)("+o[1]+"));";
        }},
        // Saturating variants clamp both formats to their largest finite value.
        {"cvt.rn.satfinite.e4m3.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e4m3_satfinite((float)("+o[1]+"));";
        }},
        {"cvt.rn.satfinite.e5m2.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e5m2_satfinite((float)("+o[1]+"));";
        }},
        // PTX 9.4 adds round-toward-zero conversions to packed FP8, and both
        // packed formats support ReLU on conversion.
        {"cvt.rz.satfinite.e4m3x2.f32", [](auto& o){ return fp8x2_convert_emit(o, true, true); }},
        {"cvt.rz.satfinite.e5m2x2.f32", [](auto& o){ return fp8x2_convert_emit(o, false, true); }},
        {"cvt.rn.relu.satfinite.e4m3x2.f32", [](auto& o){ return fp8x2_convert_emit(o, true, false, true); }},
        {"cvt.rn.relu.satfinite.e5m2x2.f32", [](auto& o){ return fp8x2_convert_emit(o, false, false, true); }},
        {"cvt.rz.relu.satfinite.e4m3x2.f32", [](auto& o){ return fp8x2_convert_emit(o, true, true, true); }},
        {"cvt.rz.relu.satfinite.e5m2x2.f32", [](auto& o){ return fp8x2_convert_emit(o, false, true, true); }},
        // Packed: f32×2 → e4m3x2. PTX orders sources as b, a; source a goes
        // in the upper byte and source b in the lower byte.
        // PTX syntax: cvt.rn.satfinite.e4m3x2.f32 d, b, a
        //   d[7:0]  = f32_to_e4m3(b)
        //   d[15:8] = f32_to_e4m3(a)
        {"cvt.rn.satfinite.e4m3x2.f32", [](auto& o){
            return fp8x2_convert_emit(o, true);
        }},
        {"cvt.rn.satfinite.e5m2x2.f32", [](auto& o){
            return fp8x2_convert_emit(o, false);
        }},
        // Packed: e4m3x2 → f32×2. The first destination corresponds to the
        // upper source byte, matching PTX's two-value packing order.
        {"cvt.rn.f32x2.e4m3x2", [](auto& o){
            if (o.size() < 3) return std::string("/* cvt.rn.f32x2.e4m3x2 */");
            return o[0]+" = vgre_fp8e4m3_to_f32((uint8_t)(((unsigned)("+o[2]+")>>8)&0xFFu));"
                   " "+o[1]+" = vgre_fp8e4m3_to_f32((uint8_t)((unsigned)("+o[2]+")&0xFFu));";
        }},
        {"cvt.rn.f32x2.e5m2x2", [](auto& o){
            if (o.size() < 3) return std::string("/* cvt.rn.f32x2.e5m2x2 */");
            return o[0]+" = vgre_fp8e5m2_to_f32((uint8_t)(((unsigned)("+o[2]+")>>8)&0xFFu));"
                   " "+o[1]+" = vgre_fp8e5m2_to_f32((uint8_t)((unsigned)("+o[2]+")&0xFFu));";
        }},
    };
    return kMap;
}

#undef VGRE_CVT_F2I_TYPE
#undef VGRE_CVT_F2I_ROUNDED
#undef VGRE_CVT_F2I
#undef VGRE_CVT_I2F_TYPE
#undef VGRE_CVT_I2F_ROUNDED
#undef VGRE_CVT_I2I_DEST
#undef VGRE_CVT_I2I_ENTRY

} // namespace compiler
} // namespace vgre
