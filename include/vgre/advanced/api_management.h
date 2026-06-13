// VGRE API Management — webhooks + OpenAPI.
// docs/missingFeatures.md §11.2.  Real, self-contained: an event webhook system
// that POSTs HMAC-SHA256-signed payloads (GitHub/Stripe-style "sha256=<hex>"
// signature) to subscriber URLs with retry/backoff, and an OpenAPI 3.0 spec
// generator for the VGRE control/metrics surface.
#ifndef VGRE_ADVANCED_API_MANAGEMENT_H
#define VGRE_ADVANCED_API_MANAGEMENT_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

struct DeliveryResult {
    std::string subscriptionId;
    bool        delivered = false;
    int         httpStatus = 0;
    int         attempts = 0;
};

class WebhookManager {
public:
    // Register a subscriber.  `events` are event-name globs; "*" matches all.
    // Returns the subscription id.
    std::string subscribe(const std::string& url, const std::string& secret,
                          const std::vector<std::string>& events);
    bool unsubscribe(const std::string& id);
    size_t subscriptionCount() const;

    // Deliver `payload` for `event` to every matching subscriber, signing with
    // the subscriber's secret. Retries up to `maxAttempts` with backoff.
    std::vector<DeliveryResult> emit(const std::string& event, const std::string& payload,
                                     int maxAttempts = 3);

    // Compute the webhook signature header value: "sha256=" + hex(HMAC-SHA256).
    static std::string signPayload(const std::string& secret, const std::string& payload);

    // Process-wide instance.
    static WebhookManager& instance();

private:
    struct Sub {
        std::string id, url, secret;
        std::vector<std::string> events;
    };
    bool matches(const Sub& s, const std::string& event) const;
    mutable std::mutex mu_;
    std::vector<Sub> subs_;
    uint64_t counter_ = 0;
};

class OpenApiSpec {
public:
    // OpenAPI 3.0 JSON describing the VGRE HTTP surface (metrics/health + webhooks).
    static std::string generate();
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_API_MANAGEMENT_H
