#include "vgre/common/config_registry.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <cstdlib>
extern "C" char **_environ;
#define VGRE_ENVIRON _environ
#else
extern "C" char **environ;
#define VGRE_ENVIRON environ
#endif

namespace vgre {
namespace common {

ConfigRegistry &ConfigRegistry::instance() {
    static ConfigRegistry *r = new ConfigRegistry();  // leaky (no teardown races)
    return *r;
}

ConfigRegistry::ConfigRegistry() {
    // ── Built-in knobs (name, range/values, default, description) ────────────
    registerInt("VGRE_VIRTUAL_DEVICE_COUNT", 1, 8, 1, "Number of virtual GPU devices.");
    registerInt("VGRE_DEVICE_COUNT", 1, 8, 1, "Legacy alias of VGRE_VIRTUAL_DEVICE_COUNT.");
    registerInt("VGRE_DEFAULT_THREAD_COUNT", 1, 256, 0, "Scheduler worker threads (0 = host cores).");
    registerInt("VGRE_PBKDF2_ITERATIONS", 10000, 10000000, 600000, "Token KDF stretch rounds.");
    registerInt("VGRE_METRICS_PORT", 1, 65535, 0, "Prometheus /metrics port (unset = off).");
    registerInt("VGRE_MAX_CONCURRENT_COMPILES", 1, 4096, 0, "Host-wide concurrent clang JIT cap.");
    registerInt("VGRE_TP_DEGREE", 0, 8, 0, "Tensor-parallel same-process rank count.");
    registerInt("VGRE_MAX_MEMORY_GB", 1, 1048576, 0, "Per-device virtual memory cap (GB).");
    registerInt("VGRE_GRAPH_FUSION_THRESHOLD", 1, 1000000, 3, "Replays before dynamic graph fusion.");
    registerInt("VGRE_GRPC_PORT", 1, 65535, 0, "gRPC transport port.");
    registerInt("VGRE_SHM_RESULT_OFFSET", 0, 1099511627776LL, 0, "Cluster SHM result region offset (bytes).");
    registerEnum("VGRE_BENCHMARK", {"ON", "OFF"}, "ON", "Background hardware benchmarks.");
    registerEnum("VGRE_IPC_MODE", {"ON", "OFF"}, "ON", "Optional IPC service.");
    registerEnum("VGRE_CLUSTER_STRICT_AUTH", {"0", "1"}, "0", "Require cluster auth (fail-closed).");
    registerString("VGRE_CLUSTER_MASTER_ADDRESS", "", "Explicit cluster master host:port.");
    registerString("VGRE_CLUSTER_ADVERTISED_ADDRESS", "", "Advertised master address for WAN/NAT.");
    registerString("VGRE_TCP_AUTH_TOKEN", "", "Cluster auth token (use *_FILE in production).", /*secret*/ true);
    registerString("VGRE_TCP_AUTH_TOKEN_FILE", "", "File containing the cluster auth token.");
    registerString("VGRE_LOG_LEVEL", "INFO", "Logger verbosity (DEBUG/INFO/WARN/ERROR).");
}

void ConfigRegistry::registerInt(const std::string &name, long lo, long hi, long def,
                                 const std::string &desc) {
    specs_.push_back(Spec{name, desc, std::to_string(def), Kind::Int, lo, hi, {}, false});
}
void ConfigRegistry::registerEnum(const std::string &name, std::vector<std::string> values,
                                  const std::string &def, const std::string &desc) {
    specs_.push_back(Spec{name, desc, def, Kind::Enum, 0, 0, std::move(values), false});
}
void ConfigRegistry::registerString(const std::string &name, const std::string &def,
                                    const std::string &desc, bool secret) {
    specs_.push_back(Spec{name, desc, def, Kind::String, 0, 0, {}, secret});
}

const ConfigRegistry::Spec *ConfigRegistry::find(const std::string &name) const {
    for (const auto &s : specs_)
        if (s.name == name) return &s;
    return nullptr;
}

int ConfigRegistry::validateAndLog() {
    if (validated_) return 0;
    validated_ = true;

    int problems = 0;
    VGRE_LOG_INFO("Config", "Effective VGRE_* configuration:");

    for (char **e = VGRE_ENVIRON; e && *e; ++e) {
        const char *eq = std::strchr(*e, '=');
        if (!eq) continue;
        std::string name(*e, eq - *e);
        if (name.rfind("VGRE_", 0) != 0) continue;  // only VGRE_* vars
        std::string value(eq + 1);

        const Spec *s = find(name);
        if (!s) {
            VGRE_LOG_WARN("Config", "Unknown config var " + name +
                                        " (typo or deprecated?) — ignored.");
            ++problems;
            continue;
        }

        bool ok = true;
        if (s->kind == Kind::Int) {
            try {
                long v = std::stol(value);
                if (v < s->lo || v > s->hi) ok = false;
            } catch (...) { ok = false; }
            if (!ok) {
                VGRE_LOG_ERROR("Config", name + "='" + value +
                    "' is invalid (expected integer in [" + std::to_string(s->lo) +
                    ".." + std::to_string(s->hi) + "]); using default " + s->def + ".");
                ++problems;
            }
        } else if (s->kind == Kind::Enum) {
            if (std::find(s->values.begin(), s->values.end(), value) == s->values.end()) {
                std::string allowed;
                for (size_t i = 0; i < s->values.size(); ++i)
                    allowed += (i ? "|" : "") + s->values[i];
                VGRE_LOG_ERROR("Config", name + "='" + value +
                    "' is invalid (expected one of " + allowed + "); using default " +
                    s->def + ".");
                ok = false;
                ++problems;
            }
        }

        std::string shown = s->secret ? "***" : value;
        if (ok)
            VGRE_LOG_INFO("Config", "  " + name + " = " + shown);
    }
    return problems;
}

} // namespace common
} // namespace vgre
