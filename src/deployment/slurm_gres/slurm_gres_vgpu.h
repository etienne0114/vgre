// VGRE SLURM GRES Plugin — exposes virtual GPUs to SLURM workload manager.
//
// This plugin implements SLURM's Generic Resource (GRES) interface so that
// VGRE-emulated devices appear as schedulable GPU resources in SLURM.
//
// Build:
//   gcc -shared -fPIC -o vgre_slurm_gres.so slurm_gres_vgpu.cpp \
//       -I/usr/include/slurm -I${PROJECT_SOURCE_DIR}/include
//
// Install:
//   cp vgre_slurm_gres.so /usr/lib64/slurm/
//   # Add to slurm.conf: GresTypes=gpu
//   # Node config: Gres=gpu:4

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdint>
#include <cstddef>

// ── Plugin metadata ──────────────────────────────────────────────────────────
#define PLUGIN_NAME      "VGRE GPU GRES"
#define PLUGIN_TYPE      "gres/gpu"
#define PLUGIN_VERSION   "0.1.0"

// ── Minimal SLURM GRES plugin callbacks ──────────────────────────────────────
// SLURM loads this shared library and resolves these symbols via dlsym.

// init() — called once when the plugin is loaded by slurmctld/slurmd.
int vgre_gres_init(void);

// fini() — called once at plugin unload.
int vgre_gres_fini(void);

// reconfig() — called when `scontrol reconfig` is issued.
int vgre_gres_reconfig(void);

// node_config_load() — called on each node to discover available VGRE GPUs.
int vgre_gres_node_config_load(
    char *node_name,
    char **gres_data,    // OUT: allocated string with "gpu:<count>"
    size_t *gres_data_size
);

// alloc() — allocate N GPUs to a job.
int vgre_gres_alloc(
    int gpu_count,
    char **env,          // OUT: env vars to set (e.g., VGRE_VISIBLE_DEVICES)
    size_t *env_count
);

// dealloc() — release GPUs from a job.
int vgre_gres_dealloc(int gpu_count);

// step_set_env() — set environment for a job step.
int vgre_gres_step_set_env(
    int gpu_count,
    const char *gpu_list,
    char **env,
    size_t *env_count
);

#ifdef __cplusplus
} // extern "C"
#endif
