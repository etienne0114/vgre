#ifndef VGRE_COMMON_ERROR_CODES_H
#define VGRE_COMMON_ERROR_CODES_H

#include <stdexcept>
#include <string>

namespace vgre {

// ── Result / error code enum ───────────────────────────────────────────────
enum class VGREResult : int {
    SUCCESS                 = 0,
    ERROR_NOT_INITIALIZED   = 1,
    ERROR_INVALID_DEVICE    = 2,
    ERROR_OUT_OF_MEMORY     = 3,
    ERROR_INVALID_VALUE     = 4,
    ERROR_INVALID_KERNEL    = 5,
    ERROR_LAUNCH_FAILURE    = 6,
    ERROR_COMPILATION       = 7,
    ERROR_NOT_SUPPORTED     = 8,
    ERROR_STREAM            = 9,
    ERROR_TIMEOUT           = 10,
    ERROR_ALREADY_EXISTS    = 11,
    ERROR_IO                = 12,
    ERROR_COMPRESSION       = 13,
    ERROR_NETWORK           = 14,
    ERROR_AUTH_FAILED       = 15,
    ERROR_CRYPTO            = 16,
    ERROR_UNKNOWN           = 99
};

// ── Human-readable error string ────────────────────────────────────────────
inline const char* resultToString(VGREResult r) {
    switch (r) {
        case VGREResult::SUCCESS:               return "Success";
        case VGREResult::ERROR_NOT_INITIALIZED: return "Not initialized";
        case VGREResult::ERROR_INVALID_DEVICE:  return "Invalid device";
        case VGREResult::ERROR_OUT_OF_MEMORY:   return "Out of memory";
        case VGREResult::ERROR_INVALID_VALUE:   return "Invalid value";
        case VGREResult::ERROR_INVALID_KERNEL:  return "Invalid kernel";
        case VGREResult::ERROR_LAUNCH_FAILURE:  return "Kernel launch failure";
        case VGREResult::ERROR_COMPILATION:     return "Compilation error";
        case VGREResult::ERROR_NOT_SUPPORTED:   return "Not supported";
        case VGREResult::ERROR_STREAM:          return "Stream error";
        case VGREResult::ERROR_TIMEOUT:         return "Timeout";
        case VGREResult::ERROR_ALREADY_EXISTS:  return "Already exists";
        case VGREResult::ERROR_IO:              return "I/O error";
        case VGREResult::ERROR_COMPRESSION:     return "Compression error";
        case VGREResult::ERROR_NETWORK:         return "Network error";
        case VGREResult::ERROR_AUTH_FAILED:     return "Authentication failed";
        case VGREResult::ERROR_CRYPTO:          return "Cryptographic error";
        case VGREResult::ERROR_UNKNOWN:         return "Unknown error";
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
