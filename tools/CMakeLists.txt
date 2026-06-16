# Checkpoint tools — load real safetensors / GGUF model files through the VGRE
# loaders. Used to validate the loaders against the reference libraries and to
# run a real model forward pass on the engine (see tools/*.py).

# checkpoint_probe: dump per-tensor stats from a safetensors/GGUF file.
add_executable(checkpoint_probe checkpoint_probe.cpp)
target_link_libraries(checkpoint_probe PRIVATE vgre)

# gpt2_infer: real GPT-2 (124M) inference — weights via vgre::xla::SafeTensors,
# matmuls via vgre::xla::gemm_f32. Matches Hugging Face logits (tools/gpt2_reference.py).
add_executable(gpt2_infer gpt2_infer.cpp)
target_link_libraries(gpt2_infer PRIVATE vgre)

# Optional reproducible real-checkpoint validation (skipped unless the model
# files exist + VGRE_RUN_REAL_CHECKPOINT_TESTS=ON, so the default suite needs no
# multi-hundred-MB download). Reproduce locally:
#   python tools/make_probe_gguf.py  $ENV{VGRE_GGUF_PROBE}
#   ./checkpoint_probe gguf $ENV{VGRE_GGUF_PROBE}
#   python tools/validate_safetensors.py ./checkpoint_probe $ENV{VGRE_GPT2_SAFETENSORS}
if(VGRE_BUILD_TESTS AND DEFINED ENV{VGRE_RUN_REAL_CHECKPOINT_TESTS}
   AND EXISTS "$ENV{VGRE_GPT2_SAFETENSORS}")
    add_test(NAME RealGpt2Safetensors
        COMMAND ${CMAKE_COMMAND} -E env
            python3 ${CMAKE_CURRENT_SOURCE_DIR}/validate_safetensors.py
            $<TARGET_FILE:checkpoint_probe> "$ENV{VGRE_GPT2_SAFETENSORS}")
    set_tests_properties(RealGpt2Safetensors PROPERTIES TIMEOUT 300)
endif()
