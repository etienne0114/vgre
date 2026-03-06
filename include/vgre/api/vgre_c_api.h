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

/* ── Status codes ───────────────────────────────────────────────────────────
 */
#define VGRE_SUCCESS 0
#define VGRE_ERROR_NOT_INIT -1
#define VGRE_ERROR_INVALID_VALUE -2
#define VGRE_ERROR_OUT_OF_MEMORY -3
#define VGRE_ERROR_INVALID_KERNEL -4
#define VGRE_ERROR_LAUNCH_FAILURE -5
#define VGRE_ERROR_IO -6
#define VGRE_ERROR_GENERIC -99

/* ── Memcpy direction ───────────────────────────────────────────────────────
 */
#define VGRE_MEMCPY_HOST_TO_DEVICE 0
#define VGRE_MEMCPY_DEVICE_TO_HOST 1
#define VGRE_MEMCPY_DEVICE_TO_DEVICE 2

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
  uint64_t timestamp;

  // GFLOPS Performance
  float gflops;
  float max_gflops; // Device peak
  float compute_utilization;

  // Memory Bandwidth
  float memory_bandwidth_gbps;
  float max_memory_bandwidth_gbps;
  float memory_bus_utilization;
  uint64_t memory_used_bytes;
  uint64_t memory_total_bytes;

  // UVM Stats
  int total_pages;
  int resident_pages;
  int evicted_pages;
  float page_faults_per_sec;
  uint8_t uvm_map[1024]; // Residency bitmask/byte-array for 32x32 grid

  // Device Stats
  int active_kernels;
  int active_threads;
  int device_clock_mhz;
  float avg_kernel_latency_ms;
  float device_temperature;
  int ecc_enabled;
  int simulation_enabled;
} vgre_telemetry_t;

/* ── Initialization ─────────────────────────────────────────────────────────
 */
int vgre_init(void);
int vgre_shutdown(void);

/* ── Device Management ──────────────────────────────────────────────────────
 */
int vgre_get_device_count(int *count);
int vgre_set_device(int device_id);
int vgre_get_device(int *device_id);
int vgre_get_device_properties(int device_id, vgre_device_properties_t *props);
int vgre_synchronize(void);

/* ── Memory Management ──────────────────────────────────────────────────────
 */
int vgre_malloc(void **ptr, size_t size);
int vgre_free(void *ptr);
int vgre_memcpy(void *dst, const void *src, size_t count, int direction);
int vgre_memset(void *ptr, int value, size_t count);

/* ── Kernel Registration & Launch ───────────────────────────────────────────
 */
/**
 * Register a kernel by name and source code.
 * Returns kernel ID via out_kernel_id.
 */
int vgre_register_kernel(const char *name, const char *source,
                         uint64_t *out_kernel_id);

/**
 * Launch a registered kernel.
 * grid/block dims are passed as [x, y, z] arrays.
 * args is a void** array of pointers-to-arguments (CUDA convention).
 * num_args is the count.
 */
int vgre_launch_kernel(uint64_t kernel_id, const uint32_t grid_dim[3],
                       const uint32_t block_dim[3], void **args, int num_args,
                       size_t shared_mem, uint64_t stream_id);

/* ── Stream Management ──────────────────────────────────────────────────────
 */
int vgre_stream_create(uint64_t *out_stream_id);
int vgre_stream_synchronize(uint64_t stream_id);
int vgre_stream_destroy(uint64_t stream_id);

/* ── Version Info ───────────────────────────────────────────────────────────
 */
const char *vgre_get_version(void);

/* ── Telemetry ──────────────────────────────────────────────────────────────
 */
int vgre_get_telemetry(vgre_telemetry_t *telemetry);

/**
 * @brief Retrieves recent log lines from the engine.
 *
 * @param buffer Double pointer to a string buffer that will be allocated by the
 * API. Must be freed by the caller using vgre_free_logs().
 * @param count Pointer to int receiving the number of log lines.
 */
int vgre_get_logs(char ***buffer, int *count);
void vgre_free_logs(char **buffer, int count);

/**
 * @brief Enables or disables the internal background simulation engine.
 * Useful for automatic dashboard functioning without external apps.
 */
int vgre_set_simulation_mode(int enabled);

/**
 * @brief Sets the IPC service mode.
 * @param is_master If 1, initializes as the master session (Dashboard).
 *                  If 0, initializes as a client.
 */
int vgre_set_service_mode(int is_master);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VGRE_C_API_H */
