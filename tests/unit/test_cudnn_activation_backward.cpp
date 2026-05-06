#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

// cuDNN shim externs (minimal)
extern "C" {
  typedef int cudnnStatus_t;
  typedef void* cudnnActivationDescriptor_t;
  typedef void* cudnnTensorDescriptor_t;
  typedef void* cudnnHandle_t;

  cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* d);
  cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d, int mode, int /*nan*/, double coeff);
  cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d);

  cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
  cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d, int fmt, int dtype, int n,int c,int h,int w);
  cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);

  cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
      const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
      const void* beta,  cudnnTensorDescriptor_t yDesc, void* y);

  cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
      const void* alpha, cudnnTensorDescriptor_t yDesc, const void* y,
      cudnnTensorDescriptor_t dyDesc, const void* dy,
      cudnnTensorDescriptor_t xDesc, const void* x,
      const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx);
}

// cudnn activation enums (mirror shim)
static constexpr int CUDNN_ACTIVATION_SIGMOID = 0;
static constexpr int CUDNN_ACTIVATION_RELU = 1;
static constexpr int CUDNN_ACTIVATION_TANH = 2;
static constexpr int CUDNN_ACTIVATION_CLIPPED_RELU = 3;
static constexpr int CUDNN_ACTIVATION_ELU = 4;
static constexpr int CUDNN_ACTIVATION_IDENTITY = 5;
static constexpr int CUDNN_ACTIVATION_SWISH = 6;
static constexpr int CUDNN_ACTIVATION_GELU = 7;
static constexpr int CUDNN_ACTIVATION_SELU = 8;
static constexpr int CUDNN_ACTIVATION_MISH = 9;

// Run a numerical gradient check for a descriptor shape
static void run_activation_test_shape(int mode, const std::string &name,
                                     int N, int C, int H, int W,
                                     float eps, float abs_tol, float rel_tol,
                                     float value_min = -10.0f, float value_max = 10.0f) {
  int total = N * C * H * W;
  std::vector<float> x(total);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(value_min, value_max);
  for (int i = 0; i < total; ++i) x[i] = dist(rng);

  // descriptors
  cudnnActivationDescriptor_t act{};
  cudnnCreateActivationDescriptor(&act);
  cudnnSetActivationDescriptor(act, mode, 0, 1.0);

  cudnnTensorDescriptor_t xd{}, yd{};
  cudnnCreateTensorDescriptor(&xd);
  cudnnCreateTensorDescriptor(&yd);
  cudnnSetTensor4dDescriptor(xd, 0, 0, N, C, H, W);
  cudnnSetTensor4dDescriptor(yd, 0, 0, N, C, H, W);

  std::vector<float> y(total), dy(total, 1.0f), dx(total, 0.0f);
  float alpha = 1.0f, beta = 0.0f;

  // forward
  cudnnActivationForward(nullptr, act, &alpha, xd, x.data(), &beta, yd, y.data());

  // backward (correct argument order)
  cudnnActivationBackward(nullptr, act, &alpha, yd, y.data(), yd, dy.data(), xd, x.data(), &beta, xd, dx.data());

  // numerical derivative via central difference
  std::vector<float> dx_num(total);
  for (int i = 0; i < total; ++i) {
    std::vector<float> xp = x, xm = x;
    xp[i] += eps; xm[i] -= eps;
    std::vector<float> yp(total), ym(total);
    cudnnActivationForward(nullptr, act, &alpha, xd, xp.data(), &beta, yd, yp.data());
    cudnnActivationForward(nullptr, act, &alpha, xd, xm.data(), &beta, yd, ym.data());
    float deriv = (yp[i] - ym[i]) / (2.0f * eps);
    dx_num[i] = deriv * dy[i];
  }

  // compare
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  for (int i = 0; i < total; ++i) {
    float a = dx[i];
    float b = dx_num[i];
    float abs_err = std::fabs(a - b);
    float rel_err = abs_err / (std::fabs(b) + 1e-12f);
    max_abs_err = std::max(max_abs_err, abs_err);
    max_rel_err = std::max(max_rel_err, rel_err);
    // Accept if abs error small or relative error small.
    if (!(abs_err <= abs_tol || rel_err <= rel_tol)) {
      std::cerr << "Activation " << name << " gradient mismatch at idx=" << i
                << " x=" << x[i] << " dx=" << a << " dx_num=" << b
                << " abs_err=" << abs_err << " rel_err=" << rel_err << "\n";
      assert(false && "Activation backward numerical check failed");
    }
  }

  std::cout << "[PASS] " << name << " shape(" << N << "," << C << "," << H << "," << W << ")"
            << " max_abs_err=" << max_abs_err << " max_rel_err=" << max_rel_err << "\n";

  cudnnDestroyActivationDescriptor(act);
  cudnnDestroyTensorDescriptor(xd);
  cudnnDestroyTensorDescriptor(yd);
}

int main() {
  // Small vector baseline
  run_activation_test_shape(CUDNN_ACTIVATION_RELU, "ReLU", 20,1,1,1, 1e-4f, 5e-3f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_SIGMOID, "Sigmoid", 20,1,1,1, 1e-4f, 5e-3f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_TANH, "Tanh", 20,1,1,1, 1e-4f, 5e-3f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_CLIPPED_RELU, "ClippedReLU", 20,1,1,1, 1e-4f, 5e-3f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_ELU, "ELU", 20,1,1,1, 1e-4f, 1e-2f, 5e-2f, -10.f, 10.f);
  run_activation_test_shape(CUDNN_ACTIVATION_SWISH, "Swish", 20,1,1,1, 1e-4f, 1e-2f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_GELU, "GELU", 20,1,1,1, 1e-4f, 1e-2f, 5e-2f, -30.f, 30.f);
  run_activation_test_shape(CUDNN_ACTIVATION_SELU, "SELU", 20,1,1,1, 1e-4f, 1e-2f, 5e-2f, -30.f, 30.f);
  run_activation_test_shape(CUDNN_ACTIVATION_MISH, "Mish", 20,1,1,1, 1e-4f, 1e-2f, 5e-2f, -30.f, 30.f);

  // Larger vector (stress)
  run_activation_test_shape(CUDNN_ACTIVATION_GELU, "GELU-large", 200,1,1,1, 1e-4f, 2e-2f, 1e-1f, -30.f, 30.f);

  // Batched shapes
  run_activation_test_shape(CUDNN_ACTIVATION_SWISH, "Swish-batched", 4,3,5,5, 1e-4f, 2e-2f, 5e-2f);
  run_activation_test_shape(CUDNN_ACTIVATION_MISH, "Mish-batched", 2,4,8,8, 1e-4f, 2e-2f, 5e-2f, -30.f, 30.f);

  // Extreme values (stress numerical stability)
  run_activation_test_shape(CUDNN_ACTIVATION_GELU, "GELU-extreme", 50,1,1,1, 1e-3f, 5e-2f, 2e-1f, -100.f, 100.f);
  run_activation_test_shape(CUDNN_ACTIVATION_MISH, "Mish-extreme", 50,1,1,1, 1e-3f, 5e-2f, 2e-1f, -100.f, 100.f);

  std::cout << "All activation backward tests passed." << std::endl;
  return 0;
}
