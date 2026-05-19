/**
 * VGRE Integration Test — CUDA Driver API parity (minimal)
 */
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
int cuInit(unsigned int flags);
int cuDeviceGetCount(int *count);
int cuDeviceGet(int *device, int ordinal);
int cuDeviceGetName(char *name, int len, int dev);
int cuDeviceTotalMem(size_t *bytes, int dev);
int cuDeviceGetAttribute(int *pi, int attrib, int dev);
int cuCtxCreate(void **pctx, unsigned int flags, int dev);
int cuCtxDestroy(void *ctx);
int cuCtxSynchronize(void);
int cuMemAlloc(void **dptr, size_t bytesize);
int cuMemFree(void *dptr);
int cuMemcpyHtoD(void *dst, const void *src, size_t bytes);
int cuMemcpyDtoH(void *dst, void *src, size_t bytes);
int cuMemcpyDtoD(void *dst, void *src, size_t bytes);
int cuStreamCreate(unsigned long long *stream, unsigned int flags);
int cuStreamSynchronize(unsigned long long stream);
int cuStreamDestroy(unsigned long long stream);
int cuEventCreate(void **evt, unsigned int flags);
int cuEventRecord(void *evt, unsigned long long stream);
int cuEventSynchronize(void *evt);
int cuEventElapsedTime(float *ms, void *start, void *end);
int cuEventDestroy(void *evt);
int cuModuleLoad(void **module, const char *fname);
int cuModuleGetFunction(int *func, void *module, const char *name);
int cuModuleUnload(void *module);
int cuLaunchKernel(int func,
                   unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                   unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                   unsigned int shared, unsigned long long stream,
                   void **params, void **extra);
int cuModuleLoadData(void **module, const void *image);
int cuModuleLoadDataEx(void **module, const void *image,
                       unsigned int numOptions, int *options, void **optionValues);
}

int main() {
  std::cout << "\n--- Test: CUDA Driver API ---\n";

  int r = cuInit(0);
  assert(r == 0);

  int count = 0;
  r = cuDeviceGetCount(&count);
  assert(r == 0);
  assert(count >= 1);

  int dev = -1;
  r = cuDeviceGet(&dev, 0);
  assert(r == 0);

  char name[256] = {0};
  r = cuDeviceGetName(name, sizeof(name), dev);
  assert(r == 0);
  assert(strlen(name) > 0);

  size_t total = 0;
  r = cuDeviceTotalMem(&total, dev);
  assert(r == 0);
  assert(total > 0);

  int attr = 0;
  r = cuDeviceGetAttribute(&attr, 1, dev);
  assert(r == 0);
  assert(attr > 0);

  void *ctx = nullptr;
  r = cuCtxCreate(&ctx, 0, dev);
  assert(r == 0);

  void *dptr = nullptr;
  r = cuMemAlloc(&dptr, 1024);
  assert(r == 0);

  std::vector<unsigned char> h(1024, 7);
  r = cuMemcpyHtoD(dptr, h.data(), h.size());
  assert(r == 0);

  std::vector<unsigned char> h2(1024, 0);
  r = cuMemcpyDtoH(h2.data(), dptr, h2.size());
  assert(r == 0);
  assert(h2[0] == 7);

  void *dptr2 = nullptr;
  r = cuMemAlloc(&dptr2, 1024);
  assert(r == 0);
  r = cuMemcpyDtoD(dptr2, dptr, 1024);
  assert(r == 0);

  unsigned long long stream = 0;
  r = cuStreamCreate(&stream, 0);
  assert(r == 0);
  r = cuStreamSynchronize(stream);
  assert(r == 0);
  r = cuStreamDestroy(stream);
  assert(r == 0);

  void *evt = nullptr;
  r = cuEventCreate(&evt, 0);
  assert(r == 0);
  r = cuEventRecord(evt, 0);
  assert(r == 0);
  r = cuEventSynchronize(evt);
  assert(r == 0);
  float ms = 0.0f;
  r = cuEventElapsedTime(&ms, evt, evt);
  assert(r == 0);
  r = cuEventDestroy(evt);
  assert(r == 0);

  r = cuMemFree(dptr);
  assert(r == 0);
  r = cuMemFree(dptr2);
  assert(r == 0);

  r = cuCtxSynchronize();
  assert(r == 0);
  r = cuCtxDestroy(ctx);
  assert(r == 0);

  // Module load from IR file is supported; data load is not.
  void *module = nullptr;
  r = cuModuleLoad(&module, "nonexistent.ll");
  assert(r != 0);

  const char *ptx = R"(\n  .version 7.0\n  .target sm_52\n  .address_size 64\n  .visible .entry dummy() {\n    ret;\n  }\n)";
  void *mod2 = nullptr;
  r = cuModuleLoadData(&mod2, ptx);
  assert(r == 0);
  void *mod3 = nullptr;
  r = cuModuleLoadDataEx(&mod3, ptx, 0, nullptr, nullptr);
  assert(r == 0);

  std::cout << "[PASS] CUDA Driver API parity\n";
  return 0;
}
