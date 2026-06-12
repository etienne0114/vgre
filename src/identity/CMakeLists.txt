# §  vgre_identity — OIDC/JWT verification + RBAC
add_library(vgre_identity OBJECT
    jwt_verifier.cpp
    rbac.cpp
)

target_include_directories(vgre_identity PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_identity PROPERTIES POSITION_INDEPENDENT_CODE ON)

# JWT signatures are verified through OpenSSL EVP.  VGRE_ENABLE_SSL + the OpenSSL
# include dir must be set on this OBJECT library (the #ifdef guards run here);
# the actual libcrypto link happens on the final shared targets via
# _vgre_configure_library().
if(_VGRE_HAS_SSL)
    target_compile_definitions(vgre_identity PRIVATE VGRE_ENABLE_SSL)
    target_include_directories(vgre_identity PRIVATE ${OPENSSL_INCLUDE_DIR})
endif()

if(TARGET Threads::Threads)
    target_link_libraries(vgre_identity PUBLIC Threads::Threads)
endif()
