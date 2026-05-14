// VGRE SLURM GRES Plugin — exposes virtual GPUs to SLURM workload manager.
//
// Build:
//   gcc -shared -fPIC -o vgre_slurm_gres.so slurm_gres_vgpu.cpp
// Install:
//   cp vgre_slurm_gres.so /usr/lib64/slurm/

#include "slurm_gres_vgpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Configuration ──────────────────────────────────────────────────────────
static int g_vgpu_count = 0;

static void read_vgpu_count(void) {
    const char *env = std::getenv("VGRE_SLURM_VGPU_COUNT");
    if (env) {
        g_vgpu_count = std::atoi(env);
    } else {
        // Default: expose 1 virtual GPU per node if not configured
        g_vgpu_count = 1;
    }
}

// ── Plugin callbacks ─────────────────────────────────────────────────────────

extern "C" {

int vgre_gres_init(void) {
    read_vgpu_count();
    std::fprintf(stderr, "[VGRE GRES] init: exposing %d virtual GPU(s)\n", g_vgpu_count);
    return 0; // success
}

int vgre_gres_fini(void) {
    std::fprintf(stderr, "[VGRE GRES] fini\n");
    return 0;
}

int vgre_gres_reconfig(void) {
    read_vgpu_count();
    std::fprintf(stderr, "[VGRE GRES] reconfig: %d virtual GPU(s)\n", g_vgpu_count);
    return 0;
}

int vgre_gres_node_config_load(char * /*node_name*/, char **gres_data, size_t *gres_data_size) {
    // Format: "gpu:<count>"
    char buf[64];
    int n = std::snprintf(buf, sizeof(buf), "gpu:%d", g_vgpu_count);
    if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) return -1;

    char *out = static_cast<char*>(std::malloc(n + 1));
    if (!out) return -1;
    std::memcpy(out, buf, n + 1);
    *gres_data = out;
    *gres_data_size = n + 1;
    return 0;
}

int vgre_gres_alloc(int gpu_count, char **env, size_t *env_count) {
    if (gpu_count <= 0 || gpu_count > g_vgpu_count) return -1;

    // Build environment variable VGRE_VISIBLE_DEVICES=0,1,...,gpu_count-1
    size_t needed = 32 + static_cast<size_t>(gpu_count) * 4;
    char *e = static_cast<char*>(std::malloc(needed));
    if (!e) return -1;

    int off = std::snprintf(e, needed, "VGRE_VISIBLE_DEVICES=");
    for (int i = 0; i < gpu_count; ++i) {
        if (i > 0) off += std::snprintf(e + off, needed - off, ",");
        off += std::snprintf(e + off, needed - off, "%d", i);
    }
    env[0] = e;
    *env_count = 1;
    return 0;
}

int vgre_gres_dealloc(int /*gpu_count*/) {
    // No persistent state to clean up in this simple implementation.
    return 0;
}

int vgre_gres_step_set_env(int gpu_count, const char *gpu_list, char **env, size_t *env_count) {
    (void)gpu_count;
    (void)gpu_list;
    // Reuse alloc to set environment.
    return vgre_gres_alloc(gpu_count, env, env_count);
}

// ── SLURM plugin symbol exports (slurmstepd looks for these) ────────────────
// SLURM uses a generic plugin loader that calls plugin_init / plugin_fini.
// These wrap our internal init/fini.

struct slurm_plugin_ops {
    const char *plugin_name;
    const char *plugin_type;
    const char *plugin_version;
    int (*plugin_init)(void);
    int (*plugin_fini)(void);
};

static slurm_plugin_ops g_ops = {
    PLUGIN_NAME,
    PLUGIN_TYPE,
    PLUGIN_VERSION,
    vgre_gres_init,
    vgre_gres_fini,
};

// Symbol exported for SLURM dynamic loader.
slurm_plugin_ops *slurm_plugin_ops_get(void) {
    return &g_ops;
}

} // extern "C"
