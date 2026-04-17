#ifndef VGRE_COMMON_ERROR_CODES_H
#define VGRE_COMMON_ERROR_CODES_H

#include <stdexcept>
#include <string>

namespace vgre {

// ── Result / error code enum ───────────────────────────────────────────────
enum class VGREResult : int {
    SUCCESS                 = 0,
    ERR_NOT_INITIALIZED     = 1,
    ERR_INVALID_DEVICE      = 2,
    ERR_OUT_OF_MEMORY       = 3,
    ERR_INVALID_VALUE       = 4,
    ERR_INVALID_KERNEL      = 5,
    ERR_LAUNCH_FAILURE      = 6,
    ERR_COMPILATION         = 7,
    ERR_NOT_SUPPORTED       = 8,
    ERR_STREAM              = 9,
    ERR_TIMEOUT             = 10,
    ERR_ALREADY_EXISTS      = 11,
    ERR_IO                  = 12,
    ERR_COMPRESSION         = 13,
    ERR_NETWORK             = 14,
    ERR_AUTH_FAILED         = 15,
    ERR_CRYPTO              = 16,
    ERR_BUSY                = 17,
    ERR_NOT_FOUND           = 18,
    ERR_AUTH_RETRY          = 19,
    ERR_UNKNOWN             = 99,
    // NOTE: No ERROR_* aliases are defined here.
    // On Windows, <winerror.h> (pulled in via <winsock2.h>) defines many
    // ERROR_* names as preprocessor macros (e.g. ERROR_TIMEOUT = 1460).
    // Defining enum values with those same names would cause the macro to
    // expand inside the enum body, breaking the parse on MSVC.
    // All code should use the ERR_* spelling instead.
};

// ── Human-readable error string ────────────────────────────────────────────
inline const char* resultToString(VGREResult r) {
    switch (r) {
        case VGREResult::SUCCESS:               return "Success";
        case VGREResult::ERR_NOT_INITIALIZED: return "Not initialized";
        case VGREResult::ERR_INVALID_DEVICE:  return "Invalid device";
        case VGREResult::ERR_OUT_OF_MEMORY:   return "Out of memory";
        case VGREResult::ERR_INVALID_VALUE:   return "Invalid value";
        case VGREResult::ERR_INVALID_KERNEL:  return "Invalid kernel";
        case VGREResult::ERR_LAUNCH_FAILURE:  return "Kernel launch failure";
        case VGREResult::ERR_COMPILATION:     return "Compilation error";
        case VGREResult::ERR_NOT_SUPPORTED:   return "Not supported";
        case VGREResult::ERR_STREAM:          return "Stream error";
        case VGREResult::ERR_TIMEOUT:         return "Timeout";
        case VGREResult::ERR_ALREADY_EXISTS:  return "Already exists";
        case VGREResult::ERR_IO:              return "I/O error";
        case VGREResult::ERR_COMPRESSION:     return "Compression error";
        case VGREResult::ERR_NETWORK:         return "Network error";
        case VGREResult::ERR_AUTH_FAILED:     return "Authentication failed";
        case VGREResult::ERR_CRYPTO:          return "Cryptographic error";
        case VGREResult::ERR_BUSY:            return "Device or resource busy";
        case VGREResult::ERR_NOT_FOUND:       return "Not found";
        case VGREResult::ERR_AUTH_RETRY:      return "Authentication retry (fallback mode)";
        case VGREResult::ERR_UNKNOWN:         return "Unknown error";
        default:                                return "Unrecognized error code";
    }
}

// ── Exception helper ───────────────────────────────────────────────────────
class VGREException : public std::runtime_error {
public:
    explicit VGREException(VGREResult code, const std::string& detail = "")
        : std::runtime_error(
              std::string("VGRE Error [") + resultToString(code) + "]: " + detail),
          code_(code) {}

    VGREResult code() const noexcept { return code_; }

private:
    VGREResult code_;
};

// ── Check macro ────────────────────────────────────────────────────────────
#define VGRE_CHECK(expr)                                                     \
    do {                                                                      \
        ::vgre::VGREResult _r = (expr);                                       \
        if (_r != ::vgre::VGREResult::SUCCESS) {                              \
            throw ::vgre::VGREException(_r, #expr);                           \
        }                                                                     \
    } while (0)

#define VGRE_ASSERT(cond, code, msg)                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            throw ::vgre::VGREException(code, msg);                           \
        }                                                                     \
    } while (0)

} // namespace vgre

#endif // VGRE_COMMON_ERROR_CODES_H
