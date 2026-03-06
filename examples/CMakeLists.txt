# Examples CMakeLists.txt

add_executable(vector_addition vector_addition.cpp)
target_link_libraries(vector_addition PRIVATE vgre)

add_executable(matrix_multiply matrix_multiply.cpp)
target_link_libraries(matrix_multiply PRIVATE vgre)
