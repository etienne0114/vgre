// PTX shared-memory atomic instruction translation map.
// Shared atomics map to the same __atomic_* builtins as global atomics,
// but the address space is __shared__ (local memory in CPU model).

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

const TranslateMap& getSharedAtomicMap() {
    static const TranslateMap kMap = {
        // ── Shared-memory add ──────────────────────────────────────────────
        {"atom.shared.add.s32", [](auto& o){
            return o[0]+" = __atomic_fetch_add((int*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.add.u32", [](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.add.u64", [](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned long long*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.add.f32", [](auto& o){
            return o[0]+" = __atomic_fetch_add((float*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.add.f64", [](auto& o){
            return o[0]+" = __atomic_fetch_add((double*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        // ── Shared-memory CAS / exchange ───────────────────────────────────
        {"atom.shared.cas.b32", [](auto& o){
            return "{unsigned _exp=(unsigned)"+o[2]+"; __atomic_compare_exchange_n((unsigned*)("+
                   o[1]+"), &_exp, "+o[3]+", 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); "+o[0]+"=_exp; }";
        }},
        {"atom.shared.cas.b64", [](auto& o){
            return "{unsigned long long _exp=(unsigned long long)"+o[2]+
                   "; __atomic_compare_exchange_n((unsigned long long*)("+o[1]+
                   "), &_exp, "+o[3]+", 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); "+o[0]+"=_exp; }";
        }},
        {"atom.shared.exch.b32", [](auto& o){
            return o[0]+" = __atomic_exchange_n((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.exch.b64", [](auto& o){
            return o[0]+" = __atomic_exchange_n((unsigned long long*)("+o[1]+"), "+
                   o[2]+", __ATOMIC_SEQ_CST);";
        }},
        // ── Shared-memory max / min ────────────────────────────────────────
        {"atom.shared.max.s32", [](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+
                   "); while(_vv>_cur && !__atomic_compare_exchange_n((int*)("+o[1]+
                   "), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur; }";
        }},
        {"atom.shared.max.u32", [](auto& o){
            return "{unsigned _vv="+o[2]+"; unsigned _cur=*(unsigned*)("+o[1]+
                   "); while(_vv>_cur && !__atomic_compare_exchange_n((unsigned*)("+o[1]+
                   "), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur; }";
        }},
        {"atom.shared.min.s32", [](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+
                   "); while(_vv<_cur && !__atomic_compare_exchange_n((int*)("+o[1]+
                   "), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur; }";
        }},
        {"atom.shared.min.u32", [](auto& o){
            return "{unsigned _vv="+o[2]+"; unsigned _cur=*(unsigned*)("+o[1]+
                   "); while(_vv<_cur && !__atomic_compare_exchange_n((unsigned*)("+o[1]+
                   "), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur; }";
        }},
        // ── Shared-memory bitwise AND / OR / XOR ───────────────────────────
        {"atom.shared.and.b32", [](auto& o){
            return o[0]+" = __atomic_fetch_and((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.or.b32", [](auto& o){
            return o[0]+" = __atomic_fetch_or((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.xor.b32", [](auto& o){
            return o[0]+" = __atomic_fetch_xor((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        // ── Shared-memory increment / decrement ────────────────────────────
        {"atom.shared.inc.u32", [](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned*)("+o[1]+"), 1u, __ATOMIC_SEQ_CST);";
        }},
        {"atom.shared.dec.u32", [](auto& o){
            return o[0]+" = __atomic_fetch_sub((unsigned*)("+o[1]+"), 1u, __ATOMIC_SEQ_CST);";
        }},
    };
    return kMap;
}

} // namespace compiler
} // namespace vgre
