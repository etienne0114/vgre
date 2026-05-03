// NCCL emulation header for VGRE (NCCL 2.x compatible API subset)
// Frameworks that include <nccl.h> will get this shim instead of real NCCL.

#ifndef NCCL_H_
#define NCCL_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Result codes ─────────────────────────────────────────────────────────────
typedef enum {
    ncclSuccess         = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError     = 2,
    ncclInternalError   = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage    = 5,
    ncclRemoteError     = 6,
    ncclInProgress      = 7,
    ncclNumResults      = 8
} ncclResult_t;

// ── Data types ────────────────────────────────────────────────────────────────
typedef enum {
    ncclInt8    = 0, ncclChar   = 0,
    ncclUint8   = 1,
    ncclInt32   = 2, ncclInt    = 2,
    ncclUint32  = 3,
    ncclInt64   = 4,
    ncclUint64  = 5,
    ncclFloat16 = 6, ncclHalf   = 6,
    ncclFloat32 = 7, ncclFloat  = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16= 9,
    ncclNumTypes= 10
} ncclDataType_t;

// ── Reduction operations ──────────────────────────────────────────────────────
typedef enum {
    ncclSum  = 0,
    ncclProd = 1,
    ncclMax  = 2,
    ncclMin  = 3,
    ncclAvg  = 4,
    ncclNumOps = 5
} ncclRedOp_t;

// ── Communicator handle ───────────────────────────────────────────────────────
typedef struct ncclComm* ncclComm_t;

// ── Unique ID (128 bytes — same layout as real NCCL) ─────────────────────────
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

// ── Version ───────────────────────────────────────────────────────────────────
#define NCCL_MAJOR 2
#define NCCL_MINOR 21
#define NCCL_PATCH 5
#define NCCL_VERSION_CODE ((NCCL_MAJOR)*1000 + (NCCL_MINOR)*100 + (NCCL_PATCH))

// ── Core API ──────────────────────────────────────────────────────────────────
ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                               ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devList);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

// ── Collective operations ─────────────────────────────────────────────────────
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, void* stream);

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, void* stream);

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, void* stream);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                            size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, void* stream);

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, void* stream);

// ── Group API (batch multiple collectives) ────────────────────────────────────
ncclResult_t ncclGroupStart();
ncclResult_t ncclGroupEnd();

// ── Utilities ─────────────────────────────────────────────────────────────────
const char* ncclGetErrorString(ncclResult_t result);
ncclResult_t ncclGetLastError(ncclComm_t comm, const char** errstr);

#ifdef __cplusplus
}
#endif

#endif // NCCL_H_
