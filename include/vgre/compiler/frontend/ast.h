// AST for VGRE's CUDA-C front-end (Track Z). Compact tagged nodes (no virtuals)
// so the PTX code generator can walk them with a simple switch. Owned by
// unique_ptr; produced by the parser, consumed by codegen.
#ifndef VGRE_COMPILER_FRONTEND_AST_H
#define VGRE_COMPILER_FRONTEND_AST_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

// A C scalar type plus pointer depth and const-ness (the subset we lower).
// A struct-valued type carries base==Struct and the struct's name; its size and
// member layout are resolved against the module's struct table.
struct Type {
    enum Base { Void, Bool, Char, Short, Int, Long, Float, Double, Struct };
    Base base = Int;
    bool isUnsigned = false;
    int  ptr = 0;           // pointer depth: float* -> 1
    bool isConst = false;
    std::string structName; // when base == Struct

    bool isPointer() const { return ptr > 0; }
    bool isStruct()  const { return base == Struct && ptr == 0; }
    bool isFloating() const { return ptr == 0 && (base == Float || base == Double); }
    // Size in bytes of one element (the pointee if a pointer, else the scalar).
    // Struct sizes are not known here — the codegen resolves them via the table.
    int elemBytes() const {
        switch (base) {
            case Void:   return 1;
            case Bool: case Char: return 1;
            case Short:  return 2;
            case Int: case Float: return 4;
            case Long: case Double: return 8;
            case Struct: return 0;  // resolved from the struct table
        }
        return 4;
    }
};

// A user struct definition (scalar members, natural alignment).
struct StructMember { Type type; std::string name; int offset = 0; };
struct StructDef {
    std::string name;
    std::vector<StructMember> members;
    int size = 0;
    const StructMember* find(const std::string& m) const {
        for (const auto& mm : members) if (mm.name == m) return &mm;
        return nullptr;
    }
};

// ── Expressions ───────────────────────────────────────────────────────────────
struct Expr {
    enum Kind { IntLit, FloatLit, Ident, Member, Index, Unary, Binary, Assign, Call, Cast, Ternary };
    Kind kind;
    int line = 0, col = 0;

    int64_t     ival = 0;   // IntLit
    double      fval = 0;   // FloatLit
    bool        wide = false;  // IntLit: 64-bit (long); FloatLit: 64-bit (double)
    std::string str;        // Ident name / Member field / Call callee / operator spelling
    Type        castType;   // Cast: the target type
    std::vector<std::unique_ptr<Expr>> args;
    // Member: args[0]=object, str=field.  Index: args[0]=base, args[1]=index.
    // Unary: str=op, args[0].  Binary: str=op, args[0],args[1].
    // Assign: str=op ("="/"+="/…), args[0]=lhs, args[1]=rhs.  Call: str=callee, args=params.
    // Cast: args[0]=operand, castType=target.  Ternary: args[0]=cond,args[1]=then,args[2]=else.
};
using ExprPtr = std::unique_ptr<Expr>;

// ── Statements ────────────────────────────────────────────────────────────────
struct Stmt {
    enum Kind { VarDecl, ExprStmt, If, For, While, Block, Return, Empty };
    Kind kind;
    int line = 0, col = 0;

    Type        type;       // VarDecl: declared type (element type when arraySize > 0)
    std::string name;       // VarDecl: variable name
    bool        isShared = false;  // VarDecl: had __shared__
    int         arraySize = 0;     // VarDecl: 0 = scalar; >0 = total element count
    std::vector<int> arrayDims;    // VarDecl: per-dimension sizes ([N] or [N][M]…);
                                   // empty for a scalar. arraySize == product(dims).
    ExprPtr     expr;       // VarDecl init / ExprStmt / Return value / If & While condition

    std::vector<std::unique_ptr<Stmt>> body;      // Block stmts / loop body / If then-branch
    std::vector<std::unique_ptr<Stmt>> elseBody;  // If else-branch

    // For: init statement + condition + increment expression (+ body above).
    std::unique_ptr<Stmt> forInit;
    ExprPtr forCond;
    ExprPtr forIncr;
};
using StmtPtr = std::unique_ptr<Stmt>;

// ── Kernel / module ───────────────────────────────────────────────────────────
struct Param {
    Type type;
    std::string name;
};

struct Kernel {
    std::string name;
    Type returnType;              // void for __global__; the real type for __device__
    std::vector<Param> params;
    std::vector<StmtPtr> body;
    bool isGlobal = false;        // had __global__ (vs __device__ helper function)
};

struct Module {
    std::vector<std::unique_ptr<Kernel>> kernels;
    std::vector<StructDef> structs;
    const StructDef* findStruct(const std::string& n) const {
        for (const auto& s : structs) if (s.name == n) return &s;
        return nullptr;
    }
};

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_AST_H
