# §  vgre_pqc — post-quantum crypto (FIPS 202 Keccak, FIPS 203 ML-KEM-768, hybrid)
add_library(vgre_pqc OBJECT
    keccak.cpp
    mlkem.cpp
    hybrid_kem.cpp
)

target_include_directories(vgre_pqc PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_pqc PROPERTIES POSITION_INDEPENDENT_CODE ON)

# hybrid_kem.cpp uses the X25519 primitives (secure_channel, OpenSSL-backed);
# VGRE_ENABLE_SSL must be set here so their declarations are visible. The
# libcrypto link is provided by the final shared targets.
if(_VGRE_HAS_SSL)
    target_compile_definitions(vgre_pqc PRIVATE VGRE_ENABLE_SSL)
    target_include_directories(vgre_pqc PRIVATE ${OPENSSL_INCLUDE_DIR})
endif()

if(TARGET Threads::Threads)
    target_link_libraries(vgre_pqc PUBLIC Threads::Threads)
endif()
