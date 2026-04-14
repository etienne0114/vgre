/**
 * VGRE Unit Tests — Vector Engine
 */
#include "vgre/runtime/vector_engine.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre::runtime;

[[maybe_unused]] static constexpr float EPSILON = 1e-2f;

void test_capabilities() {
  VectorEngine engine;
  auto caps = engine.getCapabilities();
  (void)caps;
  std::cout << "[INFO] " << engine.getCapabilityString() << std::endl;
  std::cout << "[PASS] Capability detection" << std::endl;
}

void test_float_add() {
  VectorEngine engine;
  const int N = 1024;
  std::vector<float> a(N), b(N), c(N);

  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  engine.vectorAdd(a.data(), b.data(), c.data(), N);

  for (int i = 0; i < N; ++i) {
    assert(std::fabs(c[i] - (a[i] + b[i])) < EPSILON);
  }

  std::cout << "[PASS] Float vector add (N=" << N << ")" << std::endl;
}

void test_float_mul() {
  VectorEngine engine;
  const int N = 512;
  std::vector<float> a(N), b(N), c(N);

  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i) * 0.1f;
    b[i] = static_cast<float>(i) * 0.2f;
  }

  engine.vectorMul(a.data(), b.data(), c.data(), N);

  for (int i = 0; i < N; ++i) {
    assert(std::fabs(c[i] - (a[i] * b[i])) < EPSILON);
  }

  std::cout << "[PASS] Float vector mul" << std::endl;
}

void test_float_fma() {
  VectorEngine engine;
  const int N = 256;
  std::vector<float> a(N), b(N), c(N), out(N);

  for (int i = 0; i < N; ++i) {
    a[i] = 2.0f;
    b[i] = 3.0f;
    c[i] = 1.0f;
  }

  engine.vectorFMA(a.data(), b.data(), c.data(), out.data(), N);

  for (int i = 0; i < N; ++i) {
    // out = a*b + c = 2*3 + 1 = 7
    assert(std::fabs(out[i] - 7.0f) < EPSILON);
  }

  std::cout << "[PASS] Float FMA" << std::endl;
}

void test_float_dot() {
  VectorEngine engine;
  const int N = 128;
  std::vector<float> a(N, 1.0f), b(N, 2.0f);

  float dot = engine.vectorDot(a.data(), b.data(), N);
  assert(std::fabs(dot - 256.0f) < EPSILON);

  std::cout << "[PASS] Float dot product = " << dot << std::endl;
}

void test_float_sum() {
  VectorEngine engine;
  const int N = 100;
  std::vector<float> a(N, 1.0f);

  float sum = engine.vectorSum(a.data(), N);
  assert(std::fabs(sum - 100.0f) < EPSILON);

  std::cout << "[PASS] Float sum = " << sum << std::endl;
}

void test_double_add() {
  VectorEngine engine;
  const int N = 256;
  std::vector<double> a(N), b(N), c(N);

  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<double>(i);
    b[i] = static_cast<double>(i * 3);
  }

  engine.vectorAdd(a.data(), b.data(), c.data(), N);

  for (int i = 0; i < N; ++i) {
    assert(std::fabs(c[i] - (a[i] + b[i])) < 1e-10);
  }

  std::cout << "[PASS] Double vector add" << std::endl;
}

void test_fill_copy() {
  VectorEngine engine;
  const int N = 512;
  std::vector<float> a(N), b(N);

  engine.vectorFill(a.data(), 42.0f, N);
  for (int i = 0; i < N; ++i) {
    assert(a[i] == 42.0f);
  }

  engine.vectorCopy(a.data(), b.data(), N);
  for (int i = 0; i < N; ++i) {
    assert(b[i] == 42.0f);
  }

  std::cout << "[PASS] Fill and copy" << std::endl;
}

void test_vector_min() {
  VectorEngine engine;
  const int N = 256;
  std::vector<float> a(N), b(N), out(N);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(N - 1 - i);
  }
  engine.vectorMin(a.data(), b.data(), out.data(), N);
  for (int i = 0; i < N; ++i) {
    float expected = a[i] < b[i] ? a[i] : b[i];
    assert(std::fabs(out[i] - expected) < EPSILON);
  }
  std::cout << "[PASS] Float vectorMin" << std::endl;
}

void test_vector_max() {
  VectorEngine engine;
  const int N = 256;
  std::vector<float> a(N), b(N), out(N);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(N - 1 - i);
  }
  engine.vectorMax(a.data(), b.data(), out.data(), N);
  for (int i = 0; i < N; ++i) {
    float expected = a[i] > b[i] ? a[i] : b[i];
    assert(std::fabs(out[i] - expected) < EPSILON);
  }
  std::cout << "[PASS] Float vectorMax" << std::endl;
}

void test_vector_relu() {
  VectorEngine engine;
  const int N = 100;
  std::vector<float> a(N), out(N);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i - 50);  // -50..49
  }
  engine.vectorReLU(a.data(), out.data(), N);
  for (int i = 0; i < N; ++i) {
    float expected = a[i] > 0.0f ? a[i] : 0.0f;
    assert(std::fabs(out[i] - expected) < EPSILON);
  }
  std::cout << "[PASS] Float vectorReLU" << std::endl;
}

void test_vector_abs() {
  VectorEngine engine;
  const int N = 100;
  std::vector<float> a(N), out(N);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i - 50);  // -50..49
  }
  engine.vectorAbs(a.data(), out.data(), N);
  for (int i = 0; i < N; ++i) {
    assert(std::fabs(out[i] - std::fabs(a[i])) < EPSILON);
  }
  std::cout << "[PASS] Float vectorAbs" << std::endl;
}

void test_vector_clamp() {
  VectorEngine engine;
  const int N = 100;
  std::vector<float> a(N), out(N);
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);  // 0..99
  }
  engine.vectorClamp(a.data(), 10.0f, 80.0f, out.data(), N);
  for (int i = 0; i < N; ++i) {
    float expected = a[i] < 10.0f ? 10.0f : (a[i] > 80.0f ? 80.0f : a[i]);
    assert(std::fabs(out[i] - expected) < EPSILON);
  }
  std::cout << "[PASS] Float vectorClamp" << std::endl;
}

int main() {
  std::cout << "=== VGRE Vector Engine Unit Tests ===" << std::endl;

  test_capabilities();
  test_float_add();
  test_float_mul();
  test_float_fma();
  test_float_dot();
  test_float_sum();
  test_double_add();
  test_fill_copy();
  test_vector_min();
  test_vector_max();
  test_vector_relu();
  test_vector_abs();
  test_vector_clamp();

  std::cout << "\nAll vector engine tests passed!" << std::endl;
  return 0;
}
