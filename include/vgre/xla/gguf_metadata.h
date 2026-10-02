#ifndef VGRE_XLA_GGUF_METADATA_H
#define VGRE_XLA_GGUF_METADATA_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "vgre/xla/gguf.h"

namespace vgre {
namespace xla {

/**
 * GGUFMetadataReader - Extracts model configuration parameters from GGUF files.
 *
 * This class extends the existing GGUF infrastructure to provide convenient access
 * to model metadata required for automatic configuration. It builds upon the
 * existing vgre::xla::GGUF class for efficient memory-mapped file access.
 *
 * The class handles the complexities of GGUF metadata format variations and
 * provides standardized parameter names that align with LanguageModel expectations.
 */
class GGUFMetadataReader {
public:
    /**
     * Factory method to create a GGUFMetadataReader instance.
     *
     * @param path Path to the GGUF file
     * @return Unique pointer to GGUFMetadataReader, or nullptr on error
     */
    static std::unique_ptr<GGUFMetadataReader> create(const std::string& path);

    /**
     * Destructor - cleans up resources
     */
    ~GGUFMetadataReader() = default;

    // Non-copyable, movable
    GGUFMetadataReader(const GGUFMetadataReader&) = delete;
    GGUFMetadataReader& operator=(const GGUFMetadataReader&) = delete;
    GGUFMetadataReader(GGUFMetadataReader&&) = default;
    GGUFMetadataReader& operator=(GGUFMetadataReader&&) = default;

    /**
     * Extract the number of layers (blocks) in the model.
     * Maps to LanguageModel's n_layer parameter.
     *
     * @return Number of layers, or std::nullopt if not found or invalid
     */
    std::optional<int32_t> getBlockCount() const;

    /**
     * Extract the embedding/hidden dimension size.
     * Maps to LanguageModel's d_model parameter.
     *
     * @return Embedding length, or std::nullopt if not found or invalid
     */
    std::optional<int32_t> getEmbeddingLength() const;

    /**
     * Extract the number of attention heads.
     * Maps to LanguageModel's n_head parameter.
     *
     * @return Number of attention heads, or std::nullopt if not found or invalid
     */
    std::optional<int32_t> getHeadCount() const;

    /**
     * Extract the number of key-value attention heads (for grouped-query attention).
     * Maps to LanguageModel's n_kv_head parameter.
     *
     * @return Number of KV heads, or std::nullopt if not found or invalid
     */
    std::optional<int32_t> getHeadCountKV() const;

    /**
     * Extract the feed-forward network intermediate dimension.
     * Maps to LanguageModel's d_ff parameter.
     *
     * @return Feed-forward length, or std::nullopt if not found or invalid
     */
    std::optional<int32_t> getFeedForwardLength() const;

    std::optional<int32_t> getVocabularySize() const;

    std::optional<int32_t> getContextLength() const;

    std::string getArchitecture() const;

    std::string getChatTemplate() const;

    bool hasAttentionBias() const;

    bool hasOutputWeight() const;

    /**
     * Extract the RoPE (Rotary Position Embedding) frequency base.
     * Maps to LanguageModel's rope_theta parameter.
     *
     * @return RoPE frequency base, or std::nullopt if not found or invalid
     */
    std::optional<float> getRopeFreqBase() const;

    /**
     * Extract the layer normalization epsilon parameter.
     * Maps to LanguageModel's norm_eps parameter.
     *
     * @return Normalization epsilon, or std::nullopt if not found or invalid
     */
    std::optional<float> getNormEps() const;

    /**
     * Check if the GGUF file was loaded successfully and is valid for metadata extraction.
     *
     * @return True if valid, false otherwise
     */
    bool isValid() const;

    /**
     * Get the last error message if any operation failed.
     *
     * @return Error message, or empty string if no error
     */
    std::string getLastError() const;

private:
    /**
     * Private constructor - use create() factory method instead.
     */
    explicit GGUFMetadataReader(std::unique_ptr<GGUF> gguf_loader);

    /**
     * Extract a numeric metadata value with specified type and key variations.
     *
     * @param keys List of possible metadata keys to try
     * @return Optional value of requested type, or nullopt if not found/invalid
     */
    template<typename T>
    std::optional<T> getNumericMetadata(const std::vector<std::string>& keys) const;

    /**
     * Parse a string metadata value to a numeric type.
     *
     * @param value String value to parse
     * @return Parsed numeric value, or nullopt if parsing fails
     */
    template<typename T>
    std::optional<T> parseNumericString(const std::string& value) const;

    /**
     * Set the last error message.
     *
     * @param error Error message to store
     */
    void setError(const std::string& error) const;

private:
    std::unique_ptr<GGUF> gguf_loader_;  ///< Underlying GGUF loader
    mutable std::string last_error_;     ///< Last error message (mutable for const methods)

    /**
     * Standard GGUF metadata key mappings.
     * Maps parameter names to lists of possible GGUF metadata keys.
     * Multiple keys are tried in order to handle format variations.
     */
    static const std::unordered_map<std::string, std::vector<std::string>> METADATA_KEYS;
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_GGUF_METADATA_H
