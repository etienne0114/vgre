#include "vgre/api/vgre_c_api.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

struct LargeData {
    float a;
    float b;
    int c;
    int d;
    float e; // Total 20 bytes
};

const char *STRUCT_KERNEL = R"(
struct LargeData {
    float a;
    float b;
    int c;
    int d;
    float e;
};

extern "C" __global__ void struct_kernel(LargeData data, float* out) {
    out[0] = data.a + data.b + (float)data.c + (float)data.d + data.e;
}
)";

int main() {
    vgre_init();
    LargeData data = {1.0f, 2.0f, 3, 4, 5.0f}; // Sum = 15.0f
    
    void *d_out;
    vgre_malloc(&d_out, sizeof(float));
    
    uint64_t kernel_id;
    int reg_res = vgre_register_kernel("struct_kernel", STRUCT_KERNEL, &kernel_id);
    if (reg_res != VGRE_SUCCESS) {
        std::cerr << "Failed to register kernel" << std::endl;
        return 1;
    }
    
    void *args[] = {&data, &d_out};
    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    
    int launch_res = vgre_launch_kernel(kernel_id, grid, block, args, 2, 0, 0);
    if (launch_res != VGRE_SUCCESS) {
        std::cerr << "Failed to launch kernel" << std::endl;
        return 1;
    }
    vgre_synchronize();
    
    float h_out = 0;
    vgre_memcpy(&h_out, d_out, sizeof(float), VGRE_MEMCPY_DEVICE_TO_HOST);
    
    std::cout << "Result: " << h_out << " (Expected: 15.0)" << std::endl;
    if (std::abs(h_out - 15.0f) > 1e-5f) {
        std::cerr << "Verification FAILED!" << std::endl;
        return 1;
    }
    
    std::cout << "Struct Argument Support verification SUCCESSFUL!" << std::endl;
    
    vgre_free(d_out);
    vgre_shutdown();
    return 0;
}
