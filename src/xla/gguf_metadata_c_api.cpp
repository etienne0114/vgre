#include "vgre/xla/gguf_metadata_c_api.h"
#include "vgre/xla/gguf_metadata.h"

#include <memory>
#include <string>

namespace {

/**
 * Internal wrapper struct to hold the C++ GGUFMetadataReader.
 * This provides type safety and proper C++ object management.
 */
struct vgre_gguf_metadata_impl {
    std::unique_ptr<vgre::xla::GGUFMetadataReader> reader;
    std::string last_error;  // Cache the last error for C API access
    std::string architecture;
    std::string chat_template;

    explicit vgre_gguf_metadata_impl(std::unique_ptr<vgre::xla::GGUFMetadataReader> r)
        : reader(std::move(r)), architecture(reader->getArchitecture()),
          chat_template(reader->getChatTemplate()) {}
};

/**
 * Helper function to validate reader handle and set error if invalid.
 */
bool validate_reader(vgre_gguf_metadata* reader) {
    if (!reader) {
        return false;
    }
    auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
    if (!impl->reader || !impl->reader->isValid()) {
        return false;
    }
    return true;
}

/**
 * Helper function to update cached error message from C++ reader.
 */
void update_cached_error(vgre_gguf_metadata* reader) {
    if (reader) {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        if (impl->reader) {
            impl->last_error = impl->reader->getLastError();
        }
    }
}

}  // anonymous namespace

extern "C" {

vgre_gguf_metadata* vgre_gguf_metadata_open(const char* path) {
    if (!path) {
        return nullptr;
    }

    try {
        auto cpp_reader = vgre::xla::GGUFMetadataReader::create(std::string(path));
        if (!cpp_reader) {
            return nullptr;
        }

        return reinterpret_cast<vgre_gguf_metadata*>(new vgre_gguf_metadata_impl(std::move(cpp_reader)));
    } catch (...) {
        // Never let C++ exceptions escape to C code
        return nullptr;
    }
}

void vgre_gguf_metadata_free(vgre_gguf_metadata* reader) {
    delete reinterpret_cast<vgre_gguf_metadata_impl*>(reader);  // Safe to call on nullptr
}

int vgre_gguf_metadata_get_block_count(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getBlockCount();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_embedding_length(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getEmbeddingLength();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_head_count(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getHeadCount();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_head_count_kv(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getHeadCountKV();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_feed_forward_length(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getFeedForwardLength();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_vocabulary_size(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) return VGRE_GGUF_ERROR_INVALID_READER;
    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getVocabularySize();
        if (result) { *out = *result; return VGRE_GGUF_SUCCESS; }
        update_cached_error(reader);
        return VGRE_GGUF_ERROR_NOT_FOUND;
    } catch (...) { return VGRE_GGUF_ERROR_PARSE_ERROR; }
}

int vgre_gguf_metadata_get_context_length(vgre_gguf_metadata* reader, int32_t* out) {
    if (!validate_reader(reader) || !out) return VGRE_GGUF_ERROR_INVALID_READER;
    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getContextLength();
        if (result) { *out = *result; return VGRE_GGUF_SUCCESS; }
        update_cached_error(reader);
        return VGRE_GGUF_ERROR_NOT_FOUND;
    } catch (...) { return VGRE_GGUF_ERROR_PARSE_ERROR; }
}

int vgre_gguf_metadata_get_rope_freq_base(vgre_gguf_metadata* reader, float* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getRopeFreqBase();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_get_norm_eps(vgre_gguf_metadata* reader, float* out) {
    if (!validate_reader(reader) || !out) {
        return VGRE_GGUF_ERROR_INVALID_READER;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        auto result = impl->reader->getNormEps();
        if (result.has_value()) {
            *out = result.value();
            return VGRE_GGUF_SUCCESS;
        } else {
            update_cached_error(reader);
            return VGRE_GGUF_ERROR_NOT_FOUND;
        }
    } catch (...) {
        return VGRE_GGUF_ERROR_PARSE_ERROR;
    }
}

int vgre_gguf_metadata_has_attention_bias(vgre_gguf_metadata* reader) {
    if (!validate_reader(reader)) return VGRE_GGUF_ERROR_INVALID_READER;
    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        return impl->reader->hasAttentionBias() ? 1 : 0;
    } catch (...) { return VGRE_GGUF_ERROR_PARSE_ERROR; }
}

int vgre_gguf_metadata_has_output_weight(vgre_gguf_metadata* reader) {
    if (!validate_reader(reader)) return VGRE_GGUF_ERROR_INVALID_READER;
    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        return impl->reader->hasOutputWeight() ? 1 : 0;
    } catch (...) { return VGRE_GGUF_ERROR_PARSE_ERROR; }
}

const char* vgre_gguf_metadata_get_architecture(vgre_gguf_metadata* reader) {
    if (!validate_reader(reader)) return nullptr;
    auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
    return impl->architecture.empty() ? nullptr : impl->architecture.c_str();
}

const char* vgre_gguf_metadata_get_chat_template(vgre_gguf_metadata* reader) {
    if (!validate_reader(reader)) return nullptr;
    auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
    return impl->chat_template.empty() ? nullptr : impl->chat_template.c_str();
}

int vgre_gguf_metadata_is_valid(vgre_gguf_metadata* reader) {
    if (!reader) {
        return 0;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);
        return (impl->reader && impl->reader->isValid()) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

const char* vgre_gguf_metadata_get_last_error(vgre_gguf_metadata* reader) {
    if (!reader) {
        return nullptr;
    }

    try {
        auto impl = reinterpret_cast<vgre_gguf_metadata_impl*>(reader);

        // Ensure we have the latest error message cached
        update_cached_error(reader);

        // Return pointer to cached error string (owned by the reader)
        return impl->last_error.empty() ? nullptr : impl->last_error.c_str();
    } catch (...) {
        return nullptr;
    }
}

}  // extern "C"
