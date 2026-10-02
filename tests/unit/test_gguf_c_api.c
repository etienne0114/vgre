#include "vgre/xla/gguf_metadata_c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <process.h>
#define get_process_id _getpid
#else
#include <unistd.h>
#define get_process_id getpid
#endif

enum {
    GGUF_VERSION = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_STRING = 8
};

static int write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4];
    unsigned int i;
    for (i = 0; i < 4; ++i) bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int write_u64(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    unsigned int i;
    for (i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static int write_string(FILE *file, const char *value) {
    const size_t size = strlen(value);
    return write_u64(file, (uint64_t)size) && fwrite(value, 1, size, file) == size;
}

static int write_string_entry(FILE *file, const char *key, const char *value) {
    return write_string(file, key) && write_u32(file, GGUF_TYPE_STRING) &&
           write_string(file, value);
}

static int write_u32_entry(FILE *file, const char *key, uint32_t value) {
    return write_string(file, key) && write_u32(file, GGUF_TYPE_UINT32) &&
           write_u32(file, value);
}

static int write_f32_entry(FILE *file, const char *key, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return write_string(file, key) && write_u32(file, GGUF_TYPE_FLOAT32) &&
           write_u32(file, bits);
}

static int write_fixture(const char *path) {
    FILE *file = fopen(path, "wb");
    long position;
    int ok;
    if (!file) return 0;

    ok = fwrite("GGUF", 1, 4, file) == 4 &&
         write_u32(file, GGUF_VERSION) &&
         write_u64(file, 0) &&
         write_u64(file, 5) &&
         write_string_entry(file, "general.architecture", "llama") &&
         write_u32_entry(file, "llama.block_count", 3) &&
         write_u32_entry(file, "llama.embedding_length", 64) &&
         write_f32_entry(file, "llama.rope.freq_base", 500000.0f) &&
         write_string_entry(file, "tokenizer.chat_template", "{{ messages }}");

    position = ftell(file);
    if (position < 0) ok = 0;
    while (ok && position % 32 != 0) {
        if (fputc(0, file) == EOF) ok = 0;
        ++position;
    }
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int check(int condition, const char *description) {
    if (!condition) fprintf(stderr, "FAILED: %s\n", description);
    return condition;
}

int main(void) {
    char path[128];
    vgre_gguf_metadata *reader;
    int32_t integer_value = 0;
    float float_value = 0.0f;
    const char *text;
    int ok = 1;

    snprintf(path, sizeof(path), "vgre-gguf-c-api-%ld.gguf", (long)get_process_id());
    if (!write_fixture(path)) {
        fprintf(stderr, "FAILED: could not write typed GGUF fixture %s\n", path);
        remove(path);
        return 1;
    }

    reader = vgre_gguf_metadata_open(path);
    ok &= check(reader != NULL, "open a valid GGUF file from C");
    if (reader) {
        ok &= check(vgre_gguf_metadata_is_valid(reader) == 1,
                    "valid handle reports valid");
        ok &= check(vgre_gguf_metadata_get_block_count(reader, &integer_value) ==
                        VGRE_GGUF_SUCCESS && integer_value == 3,
                    "read a typed uint32 metadata value");
        ok &= check(vgre_gguf_metadata_get_embedding_length(reader, &integer_value) ==
                        VGRE_GGUF_SUCCESS && integer_value == 64,
                    "read another typed uint32 metadata value");
        ok &= check(vgre_gguf_metadata_get_rope_freq_base(reader, &float_value) ==
                        VGRE_GGUF_SUCCESS && float_value == 500000.0f,
                    "read a typed float32 metadata value");
        text = vgre_gguf_metadata_get_architecture(reader);
        ok &= check(text && strcmp(text, "llama") == 0, "read architecture string");
        text = vgre_gguf_metadata_get_chat_template(reader);
        ok &= check(text && strcmp(text, "{{ messages }}") == 0,
                    "read chat-template string");
        ok &= check(vgre_gguf_metadata_has_attention_bias(reader) == 0,
                    "report absent attention bias");

        integer_value = -1;
        ok &= check(vgre_gguf_metadata_get_head_count(reader, &integer_value) ==
                        VGRE_GGUF_ERROR_NOT_FOUND && integer_value == -1,
                    "report missing optional metadata without changing output");
        text = vgre_gguf_metadata_get_last_error(reader);
        ok &= check(text && strstr(text, "Metadata not found") != NULL,
                    "expose the missing-metadata error through the C API");
        vgre_gguf_metadata_free(reader);
    }

    reader = vgre_gguf_metadata_open("vgre-gguf-file-that-does-not-exist.gguf");
    ok &= check(reader == NULL, "reject a nonexistent GGUF path");
    vgre_gguf_metadata_free(reader);
    vgre_gguf_metadata_free(NULL);
    ok &= check(remove(path) == 0, "remove the temporary GGUF fixture");

    return ok ? 0 : 1;
}
