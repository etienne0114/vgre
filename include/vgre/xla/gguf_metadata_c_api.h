#ifndef VGRE_XLA_GGUF_METADATA_C_API_H
#define VGRE_XLA_GGUF_METADATA_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GGUF Metadata C API - Provides a stable C interface for Python bindings.
 *
 * This API wraps the C++ GGUFMetadataReader class to provide a stable ABI
 * for Python ctypes integration. It follows standard C conventions for
 * error handling and resource management.
 */

// Opaque handle to GGUFMetadataReader
typedef struct vgre_gguf_metadata vgre_gguf_metadata;

// API visibility macro
#ifndef VGRE_PUBLIC_API
#if defined(_WIN32) && defined(VGRE_SHARED_LIB)
    #ifdef VGRE_BUILDING_DLL
        #define VGRE_PUBLIC_API __declspec(dllexport)
    #else
        #define VGRE_PUBLIC_API __declspec(dllimport)
    #endif
#else
    #define VGRE_PUBLIC_API
#endif
#endif

// Return codes
#define VGRE_GGUF_SUCCESS 1
#define VGRE_GGUF_ERROR_NOT_FOUND 0
#define VGRE_GGUF_ERROR_INVALID_TYPE -1
#define VGRE_GGUF_ERROR_INVALID_READER -2
#define VGRE_GGUF_ERROR_FILE_ERROR -3
#define VGRE_GGUF_ERROR_PARSE_ERROR -4

/**
 * Open a GGUF file for metadata reading.
 *
 * @param path Path to the GGUF file (null-terminated string)
 * @return Handle to metadata reader, or NULL on error
 */
VGRE_PUBLIC_API vgre_gguf_metadata* vgre_gguf_metadata_open(const char* path);

/**
 * Free resources associated with a metadata reader.
 *
 * @param reader Handle to metadata reader (may be NULL)
 */
VGRE_PUBLIC_API void vgre_gguf_metadata_free(vgre_gguf_metadata* reader);

/**
 * Extract the number of transformer layers (blocks) from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_block_count(vgre_gguf_metadata* reader, int32_t* out);

/**
 * Extract the embedding/hidden dimension size from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_embedding_length(vgre_gguf_metadata* reader, int32_t* out);

/**
 * Extract the number of attention heads from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_head_count(vgre_gguf_metadata* reader, int32_t* out);

/**
 * Extract the number of key-value attention heads from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_head_count_kv(vgre_gguf_metadata* reader, int32_t* out);

/**
 * Extract the feed-forward network intermediate dimension from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_feed_forward_length(vgre_gguf_metadata* reader, int32_t* out);

VGRE_PUBLIC_API int vgre_gguf_metadata_get_vocabulary_size(vgre_gguf_metadata* reader, int32_t* out);

VGRE_PUBLIC_API int vgre_gguf_metadata_get_context_length(vgre_gguf_metadata* reader, int32_t* out);

/**
 * Extract the RoPE frequency base from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_rope_freq_base(vgre_gguf_metadata* reader, float* out);

/**
 * Extract the layer normalization epsilon from GGUF metadata.
 *
 * @param reader Handle to metadata reader
 * @param out Pointer to store the result (if successful)
 * @return VGRE_GGUF_SUCCESS if found, error code otherwise
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_get_norm_eps(vgre_gguf_metadata* reader, float* out);

VGRE_PUBLIC_API int vgre_gguf_metadata_has_attention_bias(vgre_gguf_metadata* reader);

VGRE_PUBLIC_API int vgre_gguf_metadata_has_output_weight(vgre_gguf_metadata* reader);

VGRE_PUBLIC_API const char* vgre_gguf_metadata_get_architecture(vgre_gguf_metadata* reader);

VGRE_PUBLIC_API const char* vgre_gguf_metadata_get_chat_template(vgre_gguf_metadata* reader);

/**
 * Check if the metadata reader handle is valid.
 *
 * @param reader Handle to metadata reader
 * @return 1 if valid, 0 if invalid or NULL
 */
VGRE_PUBLIC_API int vgre_gguf_metadata_is_valid(vgre_gguf_metadata* reader);

/**
 * Get the last error message from the metadata reader.
 *
 * @param reader Handle to metadata reader
 * @return Error message (null-terminated string), or NULL if no error
 *         The returned string is owned by the reader and should not be freed.
 */
VGRE_PUBLIC_API const char* vgre_gguf_metadata_get_last_error(vgre_gguf_metadata* reader);

#ifdef __cplusplus
}
#endif

#endif  // VGRE_XLA_GGUF_METADATA_C_API_H
