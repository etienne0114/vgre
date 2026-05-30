// cuDNN Spatial Transformer Network tests: grid generator + sampler fwd/bwd.
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((float)(a)-(float)(b)) < (float)(eps))

// Minimal type declarations matching cudnn_internal.h / cudnn_spatial_tf.cpp
typedef int  cudnnStatus_t;
typedef void* cudnnHandle_t;
struct TensorDesc { int n,c,h,w; int dtype; int fmt; };
static constexpr cudnnStatus_t CUDNN_STATUS_SUCCESS = 0;

struct SpatialTfDesc;
typedef SpatialTfDesc* cudnnSpatialTransformerDescriptor_t;

extern "C" {
cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t*);
cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t);
cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t,
    int, int, int, const int[]);
cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t, cudnnSpatialTransformerDescriptor_t,
    const void*, void*);
cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t, cudnnSpatialTransformerDescriptor_t,
    const void*, void*, const void*, const void*, const void*, void*, void*);
cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t, cudnnSpatialTransformerDescriptor_t,
    const void*,              // alpha
    void*, const void*,       // xDesc, x
    const void*,              // beta
    void*, void*,             // dxDesc, dx
    const void*,              // alphaDgrid
    const void*,              // grid
    const void*, void*,       // betaDgrid, dgrid
    void*, const void*);      // dyDesc, dy
}

// ── 1. Identity transform: grid corners at ±1 ─────────────────────────────
int test_identity_grid() {
    cudnnSpatialTransformerDescriptor_t st = nullptr;
    cudnnCreateSpatialTransformerDescriptor(&st);
    int dims[4] = {1,1,2,2};
    cudnnSetSpatialTransformerNdDescriptor(st, 0, 0, 4, dims);

    float theta[6] = {1,0,0, 0,1,0};
    float grid[8]  = {};
    cudnnStatus_t r = cudnnSpatialTfGridGeneratorForward(nullptr, st, theta, grid);
    if (r != CUDNN_STATUS_SUCCESS) FAIL("grid generator failed");

    // (h=0,w=0): x_norm=-1,y_norm=-1 → gx=-1,gy=-1
    if (!NEAR(grid[0],-1,1e-5f)) FAIL("grid[0] gx");
    if (!NEAR(grid[1],-1,1e-5f)) FAIL("grid[1] gy");
    // (h=0,w=1): x_norm=+1,y_norm=-1 → gx=+1,gy=-1
    if (!NEAR(grid[2], 1,1e-5f)) FAIL("grid[2] gx");
    if (!NEAR(grid[3],-1,1e-5f)) FAIL("grid[3] gy");
    // (h=1,w=1): gx=+1,gy=+1
    if (!NEAR(grid[6], 1,1e-5f)) FAIL("grid[6] gx");
    if (!NEAR(grid[7], 1,1e-5f)) FAIL("grid[7] gy");

    cudnnDestroySpatialTransformerDescriptor(st);
    PASS("SpatialTf identity grid generator");
    return 0;
}

// ── 2. Sampler forward: identity transform on 3×3 ─────────────────────────
int test_sampler_forward_identity() {
    cudnnSpatialTransformerDescriptor_t st = nullptr;
    cudnnCreateSpatialTransformerDescriptor(&st);
    int dims[4] = {1,1,3,3};
    cudnnSetSpatialTransformerNdDescriptor(st, 0, 0, 4, dims);

    float x[9] = {1,2,3,4,5,6,7,8,9};
    float theta[6] = {1,0,0, 0,1,0};
    float grid[3*3*2] = {};
    cudnnSpatialTfGridGeneratorForward(nullptr, st, theta, grid);

    TensorDesc xDesc = {1,1,3,3,0,0};
    TensorDesc yDesc = {1,1,3,3,0,0};
    float y[9] = {};
    float alpha = 1.f, beta = 0.f;
    cudnnStatus_t r = cudnnSpatialTfSamplerForward(nullptr, st, &alpha,
        &xDesc, x, grid, &beta, &yDesc, y);
    if (r != CUDNN_STATUS_SUCCESS) FAIL("sampler forward failed");

    // Identity transform — output matches input (bilinear at exact pixel coords)
    for (int i = 0; i < 9; ++i)
        if (!NEAR(y[i], x[i], 0.1f)) FAIL("identity output mismatch at " + std::to_string(i));

    cudnnDestroySpatialTransformerDescriptor(st);
    PASS("SpatialTf sampler forward identity");
    return 0;
}

// ── 3. Backward: non-null dx gradients ────────────────────────────────────
int test_sampler_backward() {
    cudnnSpatialTransformerDescriptor_t st = nullptr;
    cudnnCreateSpatialTransformerDescriptor(&st);
    int dims[4] = {1,1,2,2};
    cudnnSetSpatialTransformerNdDescriptor(st, 0, 0, 4, dims);

    float x[4]    = {1,2,3,4};
    float dx[4]   = {};
    float dy[4]   = {1,1,1,1};
    float theta[6]= {1,0,0, 0,1,0};
    float grid[8] = {};
    float dgrid[8]= {};
    cudnnSpatialTfGridGeneratorForward(nullptr, st, theta, grid);

    TensorDesc xDesc  = {1,1,2,2,0,0};
    TensorDesc dxDesc = {1,1,2,2,0,0};
    TensorDesc dyDesc = {1,1,2,2,0,0};
    float alpha=1.f, beta=0.f, alphaDg=1.f, betaDg=0.f;
    cudnnStatus_t r = cudnnSpatialTfSamplerBackward(nullptr, st,
        &alpha, &xDesc, x,
        &beta,  &dxDesc, dx,
        &alphaDg, grid, &betaDg, dgrid,
        &dyDesc, dy);
    if (r != CUDNN_STATUS_SUCCESS) FAIL("sampler backward failed");

    // dx should have non-zero gradients
    float dxSum = 0.f;
    for (int i = 0; i < 4; ++i) dxSum += std::fabs(dx[i]);
    if (dxSum < 1e-6f) FAIL("backward dx is all zero");

    cudnnDestroySpatialTransformerDescriptor(st);
    PASS("SpatialTf sampler backward non-zero dx");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_identity_grid();
    fails += test_sampler_forward_identity();
    fails += test_sampler_backward();
    return fails;
}
