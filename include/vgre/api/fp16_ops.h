#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace vgre {

// FP16 (half-precision) type definition
struct fp16_t {
  uint16_t data;

  fp16_t() : data(0) {}
  fp16_t(float f) { from_float(f); }
  fp16_t(double d) { from_float(static_cast<float>(d)); }

  operator float() const { return to_float(); }
  
  float to_float() const {
    uint32_t bits = data;
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;

    if (exponent == 0) {
      if (mantissa == 0) {
        return sign ? -0.0f : 0.0f;
      }
      // Denormalized number
      return static_cast<float>(sign ? -1 : 1) * 
             (1.0f / 16384.0f) * 
             (mantissa / 1024.0f);
    } else if (exponent == 0x1F) {
      if (mantissa == 0) {
        return sign ? -INFINITY : INFINITY;
      }
      return NAN;
    }

    // Normalized number
    uint32_t fp32_exp = exponent + 112; // Bias adjustment
    uint32_t fp32_mantissa = mantissa << 13;
    uint32_t result = (sign << 31) | (fp32_exp << 23) | fp32_mantissa;
    
    float res;
    memcpy(&res, &result, sizeof(float));
    return res;
  }

  void from_float(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 31) & 1;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent == 0xFF) {
      // Infinity or NaN
      data = (sign << 15) | 0x7C00 | (mantissa ? 0x3FF : 0);
    } else if (exponent == 0) {
      // Zero or denormalized
      data = sign << 15;
    } else if (exponent < 113) {
      // Rounds to zero
      data = sign << 15;
    } else if (exponent > 142) {
      // Overflow to infinity
      data = (sign << 15) | 0x7C00;
    } else {
      // Normalized
      uint32_t new_exp = exponent - 112;
      uint32_t new_mantissa = (mantissa + 0x1000) >> 13;
      if (new_mantissa & 0x400) {
        new_exp++;
        new_mantissa &= 0x3FF;
      }
      data = (sign << 15) | (new_exp << 10) | (new_mantissa & 0x3FF);
    }
  }
};

// Dynamic loss scaling for mixed-precision training
class DynamicLossScaler {
 public:
  DynamicLossScaler(float init_scale = 65536.0f,
                   uint32_t scale_factor = 2,
                   uint32_t scale_window = 2000,
                   float min_loss_scale = 1.0f)
      : scale_(init_scale),
        scale_factor_(scale_factor),
        scale_window_(scale_window),
        min_loss_scale_(min_loss_scale),
        overflow_count_(0),
        iteration_(0) {}

  float get_scale() const { return scale_; }

  void scale_loss(float* loss, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      loss[i] *= scale_;
    }
  }

  void unscale_gradients(float* grads, uint32_t count) {
    float inv_scale = 1.0f / scale_;
    for (uint32_t i = 0; i < count; ++i) {
      grads[i] *= inv_scale;
    }
  }

  void update(bool overflow) {
    ++iteration_;
    if (overflow) {
      overflow_count_++;
      // Scale down on overflow
      scale_ = std::max(scale_ / scale_factor_, min_loss_scale_);
      overflow_count_ = 0; // Reset counter
    } else if (iteration_ % scale_window_ == 0) {
      // Scale up if no overflow in window
      scale_ *= scale_factor_;
      overflow_count_ = 0;
    }
  }

  uint32_t get_iteration() const { return iteration_; }
  uint32_t get_overflow_count() const { return overflow_count_; }

 private:
  float scale_;
  uint32_t scale_factor_;
  uint32_t scale_window_;
  float min_loss_scale_;
  uint32_t overflow_count_;
  uint32_t iteration_;
};

// FP16 conversion utilities (SIMD-accelerated if available)
class FP16Converter {
 public:
  // Convert batch of float32 to float16 (AVX2-optimized)
  static void float32_to_float16(const float* src, fp16_t* dst, size_t count) {
#ifdef __AVX2__
    size_t simd_count = count & ~7; // Process 8 elements per iteration
    for (size_t i = 0; i < simd_count; i += 8) {
      __m256 v = _mm256_loadu_ps(&src[i]);
      // Simple conversion: truncate exponent and mantissa
      for (int j = 0; j < 8; ++j) {
        dst[i + j] = fp16_t(src[i + j]);
      }
    }
    // Handle remainder
    for (size_t i = simd_count; i < count; ++i) {
      dst[i] = fp16_t(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
      dst[i] = fp16_t(src[i]);
    }
#endif
  }

  // Convert batch of float16 to float32
  static void float16_to_float32(const fp16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }

  // Check for NaN or Inf in FP16 batch
  static bool has_overflow(const fp16_t* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      uint16_t bits = data[i].data;
      uint32_t exponent = (bits >> 10) & 0x1F;
      if (exponent == 0x1F) { // Inf or NaN
        return true;
      }
    }
    return false;
  }
};

} // namespace vgre
