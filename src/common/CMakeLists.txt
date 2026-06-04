# Common utilities CMakeLists.txt

add_library(vgre_common OBJECT
    input_validation.cpp
    siphash.cpp
)

target_include_directories(vgre_common PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
