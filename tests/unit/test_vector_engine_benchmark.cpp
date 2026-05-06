#include "vgre/runtime/vector_engine.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  bool csv = (argc > 1 && std::string(argv[1]) == "--csv");
  constexpr size_t kN = 4 * 1024 * 1024;
  constexpr int kIters = 20;

  std::vector<float> a(kN, 1.25f);
  std::vector<float> b(kN, 0.75f);
  auto &ve = vgre::runtime::VectorEngine::instance();

  volatile float sink = 0.0f;
  auto t0 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    sink += ve.vectorDot(a.data(), b.data(), kN);
  }
  auto t1 = Clock::now();
  (void)sink;

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double gflops = (2.0 * static_cast<double>(kN) * static_cast<double>(kIters)) /
                  ((sec > 0.0 ? sec : 1e-9) * 1e9);

  if (csv) {
    std::cout << "metric,value\n";
    std::cout << "n," << kN << "\n";
    std::cout << "iters," << kIters << "\n";
    std::cout << "achieved_gflops," << gflops << "\n";
    return 0;
  }

  std::cout << "[BENCH] VectorDot\n";
  std::cout << "  n=" << kN << " iters=" << kIters << "\n";
  std::cout << "  achieved_gflops=" << gflops << "\n";
  return 0;
}
