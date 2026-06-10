#ifndef VGRE_COMMON_CONFIG_REGISTRY_H
#define VGRE_COMMON_CONFIG_REGISTRY_H

#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace common {

// Single source of truth for VGRE_* environment configuration (Track 5).
//
// Subsystems still read values via vgre_get_config(); this registry adds a
// validation + visibility layer run once at startup:
//   - registered int/enum vars get range/membership-checked; an invalid value is
//     reported with a clear ERROR naming the allowed range (and the caller falls
//     back to the documented default rather than silently using garbage);
//   - any VGRE_* environment variable that is NOT registered is flagged with a
//     WARN (catches typos / deprecated vars);
//   - the effective config is logged once (secret values are masked).
class ConfigRegistry {
public:
    static ConfigRegistry &instance();

    // Register the known knobs. `secret` masks the value in logs.
    void registerInt(const std::string &name, long lo, long hi, long def,
                     const std::string &desc);
    void registerEnum(const std::string &name, std::vector<std::string> values,
                      const std::string &def, const std::string &desc);
    void registerString(const std::string &name, const std::string &def,
                        const std::string &desc, bool secret = false);

    // Validate every VGRE_* environment variable and log the effective config.
    // Idempotent (runs at most once). Returns the number of invalid/unknown
    // variables found (0 == clean).
    int validateAndLog();

private:
    ConfigRegistry();  // registers the built-in knobs

    enum class Kind { Int, Enum, String };
    struct Spec {
        std::string name, desc, def;
        Kind kind;
        long lo = 0, hi = 0;
        std::vector<std::string> values;  // for Enum
        bool secret = false;
    };
    const Spec *find(const std::string &name) const;

    std::vector<Spec> specs_;
    bool validated_ = false;
};

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_CONFIG_REGISTRY_H
