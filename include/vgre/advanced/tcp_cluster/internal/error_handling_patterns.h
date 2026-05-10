#pragma once

#include "vgre/common/error_codes.h"
#include <string>

namespace vgre {
namespace advanced {

class ErrorHandlingPatterns {
public:
    static std::string translateVGREResult(VGREResult result);
    static std::string generateRecoverySuggestion(VGREResult error, const std::string& context);
    static bool isRetryableError(VGREResult error);
    static bool isCriticalError(VGREResult error);
    static void logErrorWithContext(VGREResult error, const std::string& operation, const std::string& component, const std::string& additional_context = "");
    static bool isSocketError(int error_code);
    static std::string getSocketErrorString(int error_code);
    static bool isRecoverableSocketError(int error_code);
};

} // namespace advanced
} // namespace vgre
