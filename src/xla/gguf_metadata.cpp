#include "vgre/xla/gguf_metadata.h"

#include <charconv>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace vgre {
namespace xla {

// Static metadata key mappings for different GGUF format variations
const std::unordered_map<std::string, std::vector<std::string>> GGUFMetadataReader::METADATA_KEYS = {
    // Number of transformer layers/blocks
    {"block_count", {
        "llama.block_count",
        "qwen2.block_count",
        "phi3.block_count",
        "gpt_neox.n_layers",
        "gpt2.n_layer",
        "bert.encoder.layers",
        "transformer.n_layer"
    }},

    // Embedding/hidden dimension size
    {"embedding_length", {
        "llama.embedding_length",
        "qwen2.embedding_length",
        "phi3.embedding_length",
        "gpt_neox.hidden_size",
        "gpt2.n_embd",
        "bert.hidden_size",
        "transformer.n_embd"
    }},

    // Number of attention heads
    {"head_count", {
        "llama.attention.head_count",
        "qwen2.attention.head_count",
        "phi3.attention.head_count",
        "gpt_neox.num_attention_heads",
        "gpt2.n_head",
        "bert.num_attention_heads",
        "transformer.n_head"
    }},

    // Number of key-value heads (for grouped-query attention)
    {"head_count_kv", {
        "llama.attention.head_count_kv",
        "qwen2.attention.head_count_kv",
        "phi3.attention.head_count_kv",
        "gpt_neox.num_key_value_heads",
        "transformer.n_kv_head"
    }},

    // Feed-forward network intermediate dimension
    {"feed_forward_length", {
        "llama.feed_forward_length",
        "qwen2.feed_forward_length",
        "phi3.feed_forward_length",
        "gpt_neox.intermediate_size",
        "transformer.d_ff"
    }},

    // RoPE frequency base
    {"rope_freq_base", {
        "llama.rope.freq_base",
        "qwen2.rope.freq_base",
        "phi3.rope.freq_base",
        "gpt_neox.rope_freq_base",
        "transformer.rope_theta"
    }},

    // Layer normalization epsilon
    {"norm_eps", {
        "llama.attention.layer_norm_epsilon",
        "llama.attention.layer_norm_rms_epsilon",
        "qwen2.attention.layer_norm_rms_epsilon",
        "phi3.attention.layer_norm_rms_epsilon",
        "gpt_neox.layer_norm_epsilon",
        "bert.layer_norm_eps",
        "transformer.norm_eps"
    }}
};

namespace {

const std::vector<std::string> CONTEXT_LENGTH_KEYS = {
    "llama.context_length", "qwen2.context_length", "phi3.context_length",
    "gpt_neox.context_length", "gpt2.context_length", "transformer.context_length"
};

}  // namespace

std::unique_ptr<GGUFMetadataReader> GGUFMetadataReader::create(const std::string& path) {
    auto gguf_loader = GGUF::open(path);
    if (!gguf_loader) {
        return nullptr;
    }

    // Use private constructor via unique_ptr with custom deleter
    return std::unique_ptr<GGUFMetadataReader>(
        new GGUFMetadataReader(std::move(gguf_loader))
    );
}

GGUFMetadataReader::GGUFMetadataReader(std::unique_ptr<GGUF> gguf_loader)
    : gguf_loader_(std::move(gguf_loader)) {
}

std::optional<int32_t> GGUFMetadataReader::getBlockCount() const {
    return getNumericMetadata<int32_t>(METADATA_KEYS.at("block_count"));
}

std::optional<int32_t> GGUFMetadataReader::getEmbeddingLength() const {
    auto value = getNumericMetadata<int32_t>(METADATA_KEYS.at("embedding_length"));
    if (value) return value;
    if (!gguf_loader_) return std::nullopt;
    const auto* embedding = gguf_loader_->info("token_embd.weight");
    if (embedding && embedding->shape.size() == 2 && embedding->shape[1] > 0 &&
        embedding->shape[1] <= std::numeric_limits<int32_t>::max())
        return static_cast<int32_t>(embedding->shape[1]);
    return std::nullopt;
}

std::optional<int32_t> GGUFMetadataReader::getHeadCount() const {
    return getNumericMetadata<int32_t>(METADATA_KEYS.at("head_count"));
}

std::optional<int32_t> GGUFMetadataReader::getHeadCountKV() const {
    return getNumericMetadata<int32_t>(METADATA_KEYS.at("head_count_kv"));
}

std::optional<int32_t> GGUFMetadataReader::getFeedForwardLength() const {
    auto value = getNumericMetadata<int32_t>(METADATA_KEYS.at("feed_forward_length"));
    if (value) return value;
    if (!gguf_loader_) return std::nullopt;
    const auto* down = gguf_loader_->info("blk.0.ffn_down.weight");
    if (down && down->shape.size() == 2 && down->shape[1] > 0 &&
        down->shape[1] <= std::numeric_limits<int32_t>::max())
        return static_cast<int32_t>(down->shape[1]);
    return std::nullopt;
}

std::optional<int32_t> GGUFMetadataReader::getVocabularySize() const {
    if (!gguf_loader_) return std::nullopt;
    const auto* embedding = gguf_loader_->info("token_embd.weight");
    if (embedding && embedding->shape.size() == 2 && embedding->shape[0] > 0 &&
        embedding->shape[0] <= std::numeric_limits<int32_t>::max())
        return static_cast<int32_t>(embedding->shape[0]);

    const auto count = gguf_loader_->metadataArrayLength("tokenizer.ggml.tokens");
    if (count && *count > 0 && *count <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        return static_cast<int32_t>(*count);

    static const std::vector<std::string> VOCABULARY_SIZE_KEYS = {
        "llama.vocab_size", "qwen2.vocab_size", "phi3.vocab_size",
        "gpt_neox.vocab_size", "gpt2.vocab_size", "transformer.vocab_size"
    };
    const auto metadata_vocab = getNumericMetadata<int32_t>(VOCABULARY_SIZE_KEYS);
    if (metadata_vocab && *metadata_vocab > 0) return metadata_vocab;
    return std::nullopt;
}

std::optional<int32_t> GGUFMetadataReader::getContextLength() const {
    return getNumericMetadata<int32_t>(CONTEXT_LENGTH_KEYS);
}

std::string GGUFMetadataReader::getArchitecture() const {
    return gguf_loader_ ? gguf_loader_->metadataString("general.architecture") : std::string{};
}

std::string GGUFMetadataReader::getChatTemplate() const {
    return gguf_loader_ ? gguf_loader_->metadataString("tokenizer.chat_template") : std::string{};
}

bool GGUFMetadataReader::hasAttentionBias() const {
    return gguf_loader_ && gguf_loader_->info("blk.0.attn_q.bias") != nullptr;
}

bool GGUFMetadataReader::hasOutputWeight() const {
    return gguf_loader_ && gguf_loader_->info("output.weight") != nullptr;
}

std::optional<float> GGUFMetadataReader::getRopeFreqBase() const {
    return getNumericMetadata<float>(METADATA_KEYS.at("rope_freq_base"));
}

std::optional<float> GGUFMetadataReader::getNormEps() const {
    return getNumericMetadata<float>(METADATA_KEYS.at("norm_eps"));
}

bool GGUFMetadataReader::isValid() const {
    return gguf_loader_ != nullptr;
}

std::string GGUFMetadataReader::getLastError() const {
    return last_error_;
}

template<typename T>
std::optional<T> GGUFMetadataReader::getNumericMetadata(const std::vector<std::string>& keys) const {
    if (!gguf_loader_) {
        setError("GGUF loader is not initialized");
        return std::nullopt;
    }

    for (const auto& key : keys) {
        const auto numeric = gguf_loader_->metadataNumber(key);
        if (numeric.has_value()) {
            if constexpr (std::is_same_v<T, int32_t>) {
                if (std::isfinite(*numeric) && std::floor(*numeric) == *numeric &&
                    *numeric >= std::numeric_limits<int32_t>::min() &&
                    *numeric <= std::numeric_limits<int32_t>::max())
                    return static_cast<int32_t>(*numeric);
            } else if constexpr (std::is_same_v<T, float>) {
                if (std::isfinite(*numeric) &&
                    std::abs(*numeric) <= std::numeric_limits<float>::max())
                    return static_cast<float>(*numeric);
            }
            setError("Invalid numeric value for key '" + key + "'");
            return std::nullopt;
        }
        std::string value = gguf_loader_->metadataString(key);
        if (!value.empty()) {
            auto parsed = parseNumericString<T>(value);
            if (parsed.has_value()) {
                return parsed;
            } else {
                setError("Invalid numeric value '" + value + "' for key '" + key + "'");
                return std::nullopt;
            }
        }
    }

    // If no keys found, try to construct a helpful error message
    std::ostringstream oss;
    oss << "Metadata not found for keys: ";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << keys[i];
    }
    setError(oss.str());
    return std::nullopt;
}

template<typename T>
std::optional<T> GGUFMetadataReader::parseNumericString(const std::string& value) const {
    if (value.empty()) {
        return std::nullopt;
    }

    // Handle different numeric types
    if constexpr (std::is_same_v<T, int32_t>) {
        try {
            // Try direct integer parsing first
            size_t pos;
            long long result = std::stoll(value, &pos);

            // Check if entire string was consumed and value is in valid range
            if (pos == value.length() &&
                result >= std::numeric_limits<int32_t>::min() &&
                result <= std::numeric_limits<int32_t>::max()) {
                return static_cast<int32_t>(result);
            }
        } catch (...) {
            // Fall through to try other parsing methods
        }

        // Try parsing as float and converting to int (for cases like "32.0")
        try {
            size_t pos;
            double f_result = std::stof(value, &pos);
            if (f_result == std::floor(f_result) &&
                pos == value.length() && std::isfinite(f_result) &&
                f_result >= static_cast<double>(std::numeric_limits<int32_t>::min()) &&
                f_result <= static_cast<double>(std::numeric_limits<int32_t>::max())) {
                return static_cast<int32_t>(f_result);
            }
        } catch (...) {
            // Parsing failed
        }
    } else if constexpr (std::is_same_v<T, float>) {
        try {
            size_t pos;
            float result = std::stof(value, &pos);

            // Check if entire string was consumed and result is valid
            if (pos == value.length() && std::isfinite(result)) {
                return result;
            }
        } catch (...) {
            // Parsing failed
        }
    }

    return std::nullopt;
}

void GGUFMetadataReader::setError(const std::string& error) const {
    last_error_ = error;
}

// Explicit template instantiation for the types we use
template std::optional<int32_t> GGUFMetadataReader::getNumericMetadata<int32_t>(const std::vector<std::string>&) const;
template std::optional<float> GGUFMetadataReader::getNumericMetadata<float>(const std::vector<std::string>&) const;
template std::optional<int32_t> GGUFMetadataReader::parseNumericString<int32_t>(const std::string&) const;
template std::optional<float> GGUFMetadataReader::parseNumericString<float>(const std::string&) const;

}  // namespace xla
}  // namespace vgre
