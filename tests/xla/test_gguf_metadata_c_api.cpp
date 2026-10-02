#include "vgre/xla/gguf_metadata_c_api.h"
#include "vgre/xla/model.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void writeU32(std::ostream& out, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.put(static_cast<char>((value >> (8 * i)) & 0xff));
}

void writeU64(std::ostream& out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.put(static_cast<char>((value >> (8 * i)) & 0xff));
}

void writeF32(std::ostream& out, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32(out, bits);
}

void writeString(std::ostream& out, const std::string& value) {
    writeU64(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void writeU32Entry(std::ostream& out, const std::string& key, uint32_t value) {
    writeString(out, key);
    writeU32(out, 4);
    writeU32(out, value);
}

void writeF32Entry(std::ostream& out, const std::string& key, float value) {
    writeString(out, key);
    writeU32(out, 6);
    writeF32(out, value);
}

void writeStringEntry(std::ostream& out, const std::string& key,
                      const std::string& value) {
    writeString(out, key);
    writeU32(out, 8);
    writeString(out, value);
}

std::filesystem::path makeFixture() {
    const auto path = std::filesystem::temp_directory_path() /
                      std::filesystem::u8path("vgre-typed-metadata-\xE6\xA8\xA1.gguf");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write("GGUF", 4);
    writeU32(out, 3);
    writeU64(out, 1);
    writeU64(out, 10);
    writeStringEntry(out, "general.architecture", "llama");
    writeU32Entry(out, "llama.block_count", 1);
    writeU32Entry(out, "llama.embedding_length", 64);
    writeU32Entry(out, "llama.attention.head_count", 1);
    writeU32Entry(out, "llama.attention.head_count_kv", 1);
    writeU32Entry(out, "llama.feed_forward_length", 128);
    writeU32Entry(out, "llama.context_length", 8192);
    writeF32Entry(out, "llama.rope.freq_base", 500000.0f);
    writeF32Entry(out, "llama.attention.layer_norm_rms_epsilon", 1e-6f);
    writeStringEntry(out, "tokenizer.chat_template", "<|im_start|>user");

    writeString(out, "token_embd.weight");
    writeU32(out, 2);
    writeU64(out, 64);
    writeU64(out, 256);
    writeU32(out, 0);
    writeU64(out, 0);

    const auto header_end = static_cast<uint64_t>(out.tellp());
    const uint64_t data_start = (header_end + 31) & ~uint64_t(31);
    while (static_cast<uint64_t>(out.tellp()) < data_start) out.put('\0');
    std::vector<char> tensor_data(256 * 64 * sizeof(float), 0);
    out.write(tensor_data.data(), static_cast<std::streamsize>(tensor_data.size()));
    return path;
}

bool check(bool condition, const char* description) {
    if (!condition) std::cerr << "FAILED: " << description << '\n';
    return condition;
}

}  // namespace

int main() {
    const auto path = makeFixture();
    const std::string utf8_path = path.u8string();
    bool ok = true;
    vgre_gguf_metadata* reader = vgre_gguf_metadata_open(utf8_path.c_str());
    ok &= check(reader != nullptr, "open typed GGUF metadata fixture");
    if (reader) {
        int32_t value = 0;
        ok &= check(vgre_gguf_metadata_get_block_count(reader, &value) == 1 && value == 1,
                    "read u32 block_count");
        ok &= check(vgre_gguf_metadata_get_embedding_length(reader, &value) == 1 && value == 64,
                    "read u32 embedding_length");
        ok &= check(vgre_gguf_metadata_get_head_count(reader, &value) == 1 && value == 1,
                    "read u32 head_count");
        ok &= check(vgre_gguf_metadata_get_head_count_kv(reader, &value) == 1 && value == 1,
                    "read u32 head_count_kv");
        ok &= check(vgre_gguf_metadata_get_feed_forward_length(reader, &value) == 1 && value == 128,
                    "read u32 feed_forward_length");
        ok &= check(vgre_gguf_metadata_get_vocabulary_size(reader, &value) == 1 && value == 256,
                    "read vocabulary size from token embedding tensor shape");
        ok &= check(vgre_gguf_metadata_get_context_length(reader, &value) == 1 && value == 8192,
                    "read u32 context_length");

        float float_value = 0.0f;
        ok &= check(vgre_gguf_metadata_get_rope_freq_base(reader, &float_value) == 1 &&
                        float_value == 500000.0f,
                    "read f32 RoPE base");
        ok &= check(vgre_gguf_metadata_get_norm_eps(reader, &float_value) == 1 &&
                        float_value == 1e-6f,
                    "read f32 normalization epsilon");
        const char* architecture = vgre_gguf_metadata_get_architecture(reader);
        ok &= check(architecture && std::string(architecture) == "llama",
                    "read architecture");
        const char* chat_template = vgre_gguf_metadata_get_chat_template(reader);
        ok &= check(chat_template && std::string(chat_template) == "<|im_start|>user",
                    "read chat template");
        ok &= check(vgre_gguf_metadata_has_attention_bias(reader) == 0,
                    "detect absent attention bias");
        ok &= check(vgre_gguf_metadata_has_output_weight(reader) == 0,
                    "detect tied output embedding");
        vgre_gguf_metadata_free(reader);
    }

    vgre::xla::model::Config config;
    config.vocab = 256;
    config.n_layer = 1;
    config.d_model = 64;
    config.n_head = 1;
    config.d_ff = 128;
    vgre::xla::model::GPT model(config, 1);
    model.set_rope_norm(500000.0f, 1e-6f);
    ok &= check(model.config().rope_base == 500000.0f && model.config().norm_eps == 1e-6f,
                "apply RoPE and normalization settings to C++ model config");

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return ok ? 0 : 1;
}
