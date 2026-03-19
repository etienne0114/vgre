/**
 * VGRE Integration Test — cubin/ELF loading
 *
 * Verifies that the ELFReader correctly parses a mock cubin image,
 * extracting PTX source and global symbol metadata.
 */
#include "vgre/api/cuda_interceptor.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

// We need to access the internal C-API symbols exported by libvgre_cudart
extern "C" {
  void *vgre_register_module_data(const void *data, size_t size);
  bool vgre_unregister_module_data(void *handle);
  const char *vgre_get_module_source(void *handle);
  void *vgre_lookup_symbol(void *handle, const char *name, size_t *size);
}

// Minimal ELF structures for mocking
struct Elf64_Header {
  uint8_t  magic[4];
  uint8_t  cls, data, ver, abi, abiver;
  uint8_t  pad[7];
  uint16_t type, machine;
  uint32_t version;
  uint64_t entry, phoff, shoff;
  uint32_t flags;
  uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};

struct Elf64_Shdr {
  uint32_t name, type;
  uint64_t flags, addr, offset, size;
  uint32_t link, info;
  uint64_t addralign, entsize;
};

struct Elf64_Sym {
  uint32_t name;
  uint8_t  info, other;
  uint16_t shndx;
  uint64_t value, size;
};

void test_cubin_loading() {
  std::cout << "\n--- Test: Mock Cubin Loading ---\n";

  // Construct a minimal ELF buffer
  std::vector<uint8_t> buffer(4096, 0);
  
  Elf64_Header* h = reinterpret_cast<Elf64_Header*>(buffer.data());
  h->magic[0] = 0x7f; h->magic[1] = 'E'; h->magic[2] = 'L'; h->magic[3] = 'F';
  h->shoff = sizeof(Elf64_Header);
  h->shentsize = sizeof(Elf64_Shdr);
  h->shnum = 5; // NULL, .shstrtab, .nv_ptx, .symtab, .strtab
  h->shstrndx = 1;

  Elf64_Shdr* sh = reinterpret_cast<Elf64_Shdr*>(buffer.data() + h->shoff);
  
  // Section 1: .shstrtab
  size_t shstr_off = 1024;
  const char* shstr_names = "\0.shstrtab\0.nv_ptx\0.symtab\0.strtab\0";
  std::memcpy(buffer.data() + shstr_off, shstr_names, 40);
  sh[1].name = 1; sh[1].type = 3; sh[1].offset = shstr_off; sh[1].size = 40;

  // Section 2: .nv_ptx
  size_t ptx_off = 2048;
  const char* ptx_src = ".version 7.0\n.target sm_75\n.global .b32 my_global[10];\n";
  std::memcpy(buffer.data() + ptx_off, ptx_src, std::strlen(ptx_src) + 1);
  sh[2].name = 11; sh[2].type = 1; sh[2].offset = ptx_off; sh[2].size = std::strlen(ptx_src) + 1;

  // Section 3: .symtab
  size_t sym_off = 3072;
  Elf64_Sym* syms = reinterpret_cast<Elf64_Sym*>(buffer.data() + sym_off);
  syms[1].name = 1; syms[1].info = (1 << 4) | 1; // GLOBAL, OBJECT
  syms[1].size = 40; // 10 * b32
  sh[3].name = 19; sh[3].type = 2; sh[3].offset = sym_off; sh[3].size = 2 * sizeof(Elf64_Sym); sh[3].entsize = sizeof(Elf64_Sym); sh[3].link = 4;

  // Section 4: .strtab
  size_t str_off = 3500;
  const char* sym_names = "\0my_global\0";
  std::memcpy(buffer.data() + str_off, sym_names, 12);
  sh[4].name = 27; sh[4].type = 3; sh[4].offset = str_off; sh[4].size = 12;

  // Register the module
  void* handle = vgre_register_module_data(buffer.data(), buffer.size());
  assert(handle != nullptr);

  // Verify PTX extraction
  const char* extracted = vgre_get_module_source(handle);
  assert(extracted != nullptr);
  assert(std::strstr(extracted, ".version 7.0") != nullptr);
  std::cout << "  ✓ PTX extraction verified\n";

  // Verify Global Symbol extraction
  size_t symSize = 0;
  void* symPtr = vgre_lookup_symbol(handle, "my_global", &symSize);
  assert(symPtr != nullptr);
  assert(symSize == 40);
  std::cout << "  ✓ Global symbol 'my_global' (40 bytes) verified\n";

  vgre_unregister_module_data(handle);
  std::cout << "[PASS] Cubin loading and symbol resolution\n";
}

int main() {
  test_cubin_loading();
  return 0;
}
