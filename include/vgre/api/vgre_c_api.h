/**
 * VGRE C API — Flat extern "C" interface for FFI (Python ctypes, etc.)
 *
 * All functions return int status codes:
 *   0 = success, negative = error
 */

#ifndef VGRE_C_API_H
#define VGRE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VGRE_EXPORT
#if defined(_WIN32)
#define VGRE_EXPORT __declspec(dllexport)
#else
#define VGRE_EXPORT __attribute__((visibility("default")))
#endif
#endif

/* ── Status codes ───────────────────────────────────────────────────────────
 */
#define VGRE_SUCCESS 0
#define VGRE_ERROR_NOT_INIT -1
#define VGRE_ERROR_INVALID_VALUE -2
#define VGRE_ERROR_OUT_OF_MEMORY -3
#define VGRE_ERROR_INVALID_KERNEL -4
#define VGRE_ERROR_LAUNCH_FAILURE -5
#define VGRE_ERROR_IO -6
#define VGRE_ERROR_AUTH_FAILED -7
#define VGRE_ERROR_CRYPTO -8
#define VGRE_ERROR_GENERIC -99

/* ── Memcpy direction ───────────────────────────────────────────────────────
 */
#define VGRE_MEMCPY_HOST_TO_DEVICE 0
#define VGRE_MEMCPY_DEVICE_TO_HOST 1
#define VGRE_MEMCPY_DEVICE_TO_DEVICE 2

/* ── Kernel argument types ─────────────────────────────────────────────────
 */
#define VGRE_ARG_POINTER 0
#define VGRE_ARG_INT32 1
#define VGRE_ARG_INT64 2
#define VGRE_ARG_FLOAT32 3
#define VGRE_ARG_FLOAT64 4
#define VGRE_ARG_UINT32 5
#define VGRE_ARG_UINT64 6
#define VGRE_ARG_STRUCT 7

/* ── Device properties struct (C-compatible) ────────────────────────────────
 */
typedef struct {
  char name[256];
  uint64_t total_global_mem;
  uint64_t shared_mem_per_block;
  int max_threads_per_block;
  int max_threads_dim[3];
  int max_grid_size[3];
  int warp_size;
  int multi_processor_count;
  int major;
  int minor;
  int clock_rate;
  uint64_t total_const_mem;
} vgre_device_properties_t;

typedef struct {
  char address[64];
  int port;
  int cpu_cores;
  uint64_t memory_bytes;
  double latency_ms;
  int available;
  char igpu_name[64];
} vgre_cluster_node_t;

#pragma pack(push, 8)
typedef struct {
  uint64_t timestamp;
  uint64_t version_major;
  uint64_t version_minor;
  uint64_t version_patch;

  // GFLOPS Performance
  double gflops;
  double max_gflops;
  double compute_utilization;

  // Memory Bandwidth
  double memory_bandwidth_gbps;
  double max_memory_bandwidth_gbps;
  double memory_bus_utilization;
  uint64_t memory_used_bytes;
  uint64_t memory_total_bytes;

  // UVM Stats
  uint64_t total_pages;
  uint64_t resident_pages;
  uint64_t evicted_pages;
  double page_faults_per_sec;

  // Device Stats
  int64_t active_kernels;
  int64_t active_threads;
  double device_clock_mhz;
  double avg_kernel_latency_ms;
  double device_temperature;
  int64_t ecc_enabled;
  int64_t background_compute_active;

  // Byte array at the end to prevent alignment shifts
  uint8_t uvm_map[1024];
} vgre_telemetry_t;
#pragma pack(pop)

/* ── Configuration (thread-safe, replaces setenv/getenv race) ─────────────
 */
VGRE_EXPORT int vgre_set_config(const char *key, const char *value);
VGRE_EXPORT const char *vgre_get_config(const char *key);

/* ── Initialization ─────────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_init(void);
VGRE_EXPORT int vgre_shutdown(void);

/* ── Device Management ──────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_get_device_count(int *count);
VGRE_EXPORT int vgre_set_device(int device_id);
VGRE_EXPORT int vgre_get_device(int *device_id);
VGRE_EXPORT int vgre_get_device_properties(int device_id, vgre_device_properties_t *props);
VGRE_EXPORT int vgre_synchronize(void);

/* ── Memory Management ──────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_malloc(void **ptr, size_t size);
VGRE_EXPORT int vgre_malloc_managed(void **ptr, size_t size);
VGRE_EXPORT int vgre_free(void *ptr);
VGRE_EXPORT int vgre_memcpy(void *dst, const void *src, size_t count, int direction);
VGRE_EXPORT int vgre_memset(void *ptr, int value, size_t count);

/* ── Peer-to-Peer (P2P) ─────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_device_can_access_peer(int *can_access, int device, int peer_device);
VGRE_EXPORT int vgre_device_enable_peer_access(int peer_device);
VGRE_EXPORT int vgre_device_disable_peer_access(int peer_device);

/* ── Kernel Registration & Launch ───────────────────────────────────────────
 */
/**
 * Register a kernel by name and source code.
 * Returns kernel ID via out_kernel_id.
 */
VGRE_EXPORT int vgre_register_kernel(const char *name, const char *source,
                         uint64_t *out_kernel_id);

/**
 * Launch a registered kernel.
 * grid/block dims are passed as [x, y, z] arrays.
 * args is a void** array of pointers-to-arguments (CUDA convention).
 * num_args is the count.
 */
VGRE_EXPORT int vgre_launch_kernel(uint64_t kernel_id, const uint32_t grid_dim[3],
                       const uint32_t block_dim[3], void **args, int num_args,
                       size_t shared_mem, uint64_t stream_id);

/* ── Stream Management ──────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_stream_create(uint64_t *out_stream_id);
VGRE_EXPORT int vgre_stream_create_with_priority(uint64_t *out_stream_id, int priority);
VGRE_EXPORT int vgre_stream_synchronize(uint64_t stream_id);
VGRE_EXPORT int vgre_stream_destroy(uint64_t stream_id);

/* ── CUDA Graphs (DAG) ──────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_graphCreate(uint64_t *out_graph);
VGRE_EXPORT int vgre_graphClone(uint64_t src_graph, uint64_t *out_clone);
VGRE_EXPORT int vgre_graphDestroy(uint64_t graph);
VGRE_EXPORT int vgre_graphInstantiate(uint64_t graph, uint64_t *out_exec);
VGRE_EXPORT int vgre_graphExecDestroy(uint64_t exec);
VGRE_EXPORT int vgre_graphLaunch(uint64_t exec, uint64_t stream);

VGRE_EXPORT int vgre_graphAddKernelNodeEx(uint64_t graph, uint64_t kernel_id,
                              const char *name, const uint32_t grid_dim[3],
                              const uint32_t block_dim[3], void **args,
                              const uint8_t *arg_types, int num_args,
                              const uint64_t *deps, int num_deps,
                              uint64_t *out_node_id);

VGRE_EXPORT int vgre_graphAddMemcpyNodeEx(uint64_t graph, void *dst, void *src,
                              size_t count, int kind,
                              const uint64_t *deps, int num_deps,
                              uint64_t *out_node_id);

/* Add a conditional (IF/WHILE) node.  cond_type: 0=IF, 1=WHILE.
 * max_iterations: WHILE loop safety cap (0 → default 65536). */
VGRE_EXPORT int vgre_graphAddConditionalNodeEx(uint64_t graph,
                                    int (*cond_fn)(void *), void *cond_ctx,
                                    uint64_t body_graph, int cond_type,
                                    unsigned int max_iterations,
                                    const uint64_t *deps, int num_deps,
                                    uint64_t *out_node_id);

VGRE_EXPORT int vgre_graphAddDependency(uint64_t graph, uint64_t node_id,
                            uint64_t depends_on);

VGRE_EXPORT int vgre_graphUpdateKernelNode(uint64_t graph, uint64_t node_id, void **args,
                               const uint8_t *arg_types, int num_args);

VGRE_EXPORT int vgre_graphUpdateMemcpyNode(uint64_t graph, uint64_t node_id, void *dst,
                               void *src, size_t count, int kind);

/* ── Version Info ───────────────────────────────────────────────────────────
 */
VGRE_EXPORT const char *vgre_get_version(void);

/* ── Telemetry ──────────────────────────────────────────────────────────────
 */
VGRE_EXPORT int vgre_get_telemetry(vgre_telemetry_t *telemetry);
/**
 * @brief Returns profiler JSON with top-N kernels (by total time).
 * @param out_json Allocated C string. Must be freed with vgre_free_string().
 * @param top_n If <= 0, returns all kernels.
 */
VGRE_EXPORT int vgre_get_profiler_json(char **out_json, int top_n);
VGRE_EXPORT void vgre_free_string(char *str);
VGRE_EXPORT int vgre_set_profiler_enabled(int enabled);
VGRE_EXPORT int vgre_get_memory_info_json(char **out_json);
VGRE_EXPORT int vgre_get_kernel_history_json(const char *kernel_name, char **out_json);

/**
 * @brief Exports the VGRE telemetry timeline as a Perfetto binary trace.
 *
 * Serializes all recorded kernel and memory events into Perfetto's
 * TracePacket protobuf format.  Each kernel execution becomes a
 * TYPE_SLICE_BEGIN / TYPE_SLICE_END pair with grid/block dimensions as
 * debug annotations.  Memory operations become counter events on the
 * MemoryBandwidth track.
 *
 * Open the resulting file at https://ui.perfetto.dev
 *
 * @param output_path  Filesystem path for the .perfetto-trace file.
 * @return 0 on success, non-zero on error.
 */
VGRE_EXPORT int vgre_export_perfetto_trace(const char *output_path);

/**
 * @brief Retrieves recent log lines from the engine.
 *
 * @param buffer Double pointer to a string buffer that will be allocated by the
 * API. Must be freed by the caller using vgre_free_logs().
 * @param count Pointer to int receiving the number of log lines.
 */
VGRE_EXPORT int vgre_get_logs(char ***buffer, int *count);
VGRE_EXPORT void vgre_free_logs(char **buffer, int count);

/**
 * @brief Retrieves information about all connected cluster nodes.
 * @param nodes Array of vgre_cluster_node_t to be filled.
 * @param count On entry, size of nodes array. On exit, number of nodes filled.
 */
VGRE_EXPORT int vgre_get_cluster_nodes(vgre_cluster_node_t *nodes, int *count);

/**
 * @brief Enables or disables the internal background compute engine.
 * Useful for maintaining realistic load without external apps.
 */
VGRE_EXPORT int vgre_set_background_compute(int enabled);

/* ── GPU Cache Statistics ────────────────────────────────────────────────────
 * Software-modeled L1/L2 GPU cache hit/miss counters.
 * Accumulated across all blocks since the last vgre_reset_cache_stats() call.
 */
typedef struct {
    uint64_t l2_hits;
    uint64_t l2_misses;
    uint64_t l2_evictions;
    double   l2_hit_rate;   /* hits / (hits + misses), in [0.0, 1.0] */
    uint64_t l1_config_kb;  /* configured L1 size (VGRE_L1_CACHE_KB) */
    uint64_t l2_config_mb;  /* configured L2 size (VGRE_L2_CACHE_MB) */
} vgre_cache_stats_t;

/**
 * @brief Returns accumulated GPU cache model statistics (L2 counters).
 * @param stats Pointer to vgre_cache_stats_t struct to fill.
 * @return 0 on success.
 */
VGRE_EXPORT int vgre_get_cache_stats(vgre_cache_stats_t *stats);

/**
 * @brief Resets all GPU cache model counters and invalidates L2 cache state.
 * @return 0 on success.
 */
VGRE_EXPORT int vgre_reset_cache_stats(void);

/**
 * @brief Sets the IPC service mode.
 * @param is_master If 1, initializes as the master session (Dashboard).
 *                  If 0, initializes as a client.
 */
VGRE_EXPORT int vgre_set_service_mode(int is_master);

/**
 * @brief Enables or disables per-block OS thread execution for __syncthreads correctness.
 * This toggles the VGRE_BLOCK_THREADS environment flag at runtime.
 */
VGRE_EXPORT int vgre_set_block_threads(int enabled);

/* ── Phase 5: Global Compute Network ──────────────────────────────────────
 */

/**
 * @brief Security session information.
 */
typedef struct {
  char cipher_name[64];
  char key_fingerprint[65];
  double session_seconds;
  int is_encrypted;
  uint64_t packets_sent;
  uint64_t packets_received;
  uint64_t bytes_sent;
  uint64_t bytes_received;
} vgre_security_info_t;

/**
 * @brief Credit/billing information for a compute node.
 */
typedef struct {
  char address[64];
  double total_credits;
  double total_debits;
  double balance;
  uint64_t last_activity;
  int transaction_count;
} vgre_credit_info_t;

/**
 * @brief Enables or disables encrypted cluster communication.
 * Requires VGRE_TCP_AUTH_TOKEN environment variable to be set.
 */
VGRE_EXPORT int vgre_cluster_set_security(int enabled);

/**
 * @brief Retrieves the current security session information.
 */
VGRE_EXPORT int vgre_cluster_get_security_info(vgre_security_info_t *info);

/**
 * @brief Wait for a specific kernel to complete on the cluster.
 * @param kernel_id The unique ID of the kernel to wait for.
 * @param timeout_ms Maximum time to wait in milliseconds.
 * @return VGRE_SUCCESS if completed, VGRE_ERROR_TIMEOUT if timed out.
 */
VGRE_EXPORT int vgre_cluster_wait(uint64_t kernel_id, int timeout_ms);

/**
 * @brief Retrieves the credit balance for a specific node.
 */
VGRE_EXPORT int vgre_credits_get_balance(const char *address, vgre_credit_info_t *info);

/**
 * @brief Retrieves credit balances for all nodes.
 * @param nodes Array to fill. On entry, *count is array capacity.
 *              On exit, *count is number of nodes filled.
 */
VGRE_EXPORT int vgre_credits_get_all(vgre_credit_info_t *nodes, int *count);

/**
 * @brief Resets the credit ledger globally.
 */
VGRE_EXPORT int vgre_credits_reset(void);

/**
 * @brief Performs a distributed all_reduce across the cluster.
 * @param ptr Pointer to local buffer (will be reduced in-place).
 * @param count Number of elements in the buffer.
 * @param datatype Argument type (VGRE_ARG_FLOAT32, etc.)
 */
VGRE_EXPORT int vgre_cluster_all_reduce(void* ptr, size_t count, int datatype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VGRE_C_API_H */
