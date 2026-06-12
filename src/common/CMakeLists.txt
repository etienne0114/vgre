# Common utilities CMakeLists.txt

add_library(vgre_common OBJECT
    input_validation.cpp
    siphash.cpp
    metrics_registry.cpp
    config_registry.cpp
    json.cpp
)

target_include_directories(vgre_common PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
