#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

namespace vgre {
namespace common {

class ELFReader {
public:
  struct Header {
    uint8_t  magic[4]; // 0x7f, 'E', 'L', 'F'
    uint8_t  cls;
    uint8_t  data_encoding;
    uint8_t  version;
    uint8_t  osabi;
    uint8_t  abiversion;
    uint8_t  pad[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint64_t entry;
    uint64_t phoff;
    uint64_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
  };

  struct SectionHeader {
    uint32_t name;
    uint32_t type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t addralign;
    uint64_t entsize;
  };

  struct Symbol {
    uint32_t name;
    uint8_t  info;
    uint8_t  other;
    uint16_t shndx;
    uint64_t value;
    uint64_t size;
  };

  ELFReader(const void* data, size_t size) : data_(reinterpret_cast<const uint8_t*>(data)), size_(size) {}

  bool isValid() const {
    if (size_ < sizeof(Header)) return false;
    const Header* h = reinterpret_cast<const Header*>(data_);
    return h->magic[0] == 0x7f && h->magic[1] == 'E' && h->magic[2] == 'L' && h->magic[3] == 'F';
  }

  size_t getTotalSize() const {
    if (!isValid()) return 0;
    const Header* h = reinterpret_cast<const Header*>(data_);
    
    // The total size of an ELF image is determined by the end of the last section
    // or the end of the section header table, whichever is further.
    size_t maxSize = h->shoff + (h->shnum * h->shentsize);
    
    // Also check section boundaries to be absolutely sure
    for (int i = 0; i < h->shnum; ++i) {
      if (h->shoff + (i + 1) * h->shentsize > size_) break;
      const SectionHeader* sh = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + i * h->shentsize);
      if (sh->offset + sh->size > maxSize) {
        maxSize = sh->offset + sh->size;
      }
    }
    return maxSize;
  }

  const char* getSectionData(const char* targetName, size_t& outSize) {
    if (!isValid()) return nullptr;
    const Header* h = reinterpret_cast<const Header*>(data_);
    if (h->shoff == 0 || h->shnum == 0 || h->shstrndx >= h->shnum) return nullptr;

    const SectionHeader* shstr = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + h->shstrndx * h->shentsize);
    if (shstr->offset + shstr->size > size_) return nullptr;
    const char* strtab = reinterpret_cast<const char*>(data_ + shstr->offset);

    for (int i = 0; i < h->shnum; ++i) {
      const SectionHeader* sh = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + i * h->shentsize);
      const char* name = strtab + sh->name;
      if (std::strcmp(name, targetName) == 0) {
        outSize = sh->size;
        return reinterpret_cast<const char*>(data_ + sh->offset);
      }
    }
    return nullptr;
  }

  struct SymbolInfo {
    std::string name;
    uint64_t value;
    uint64_t size;
  };

  std::vector<SymbolInfo> getGlobalSymbols() {
    std::vector<SymbolInfo> results;
    if (!isValid()) return results;
    const Header* h = reinterpret_cast<const Header*>(data_);
    
    const SectionHeader* symtab_sh = nullptr;
    const SectionHeader* strtab_sh = nullptr;

    const SectionHeader* shstr = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + h->shstrndx * h->shentsize);
    const char* shstrtab = reinterpret_cast<const char*>(data_ + shstr->offset);

    for (int i = 0; i < h->shnum; ++i) {
      const SectionHeader* sh = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + i * h->shentsize);
      const char* name = shstrtab + sh->name;
      if (sh->type == 2) { // SHT_SYMTAB
        symtab_sh = sh;
      } else if (std::strcmp(name, ".strtab") == 0) {
        strtab_sh = sh;
      }
    }

    if (symtab_sh && strtab_sh) {
      const Symbol* syms = reinterpret_cast<const Symbol*>(data_ + symtab_sh->offset);
      const char* strings = reinterpret_cast<const char*>(data_ + strtab_sh->offset);
      size_t numSyms = symtab_sh->size / sizeof(Symbol);

      for (size_t i = 0; i < numSyms; ++i) {
        const Symbol& s = syms[i];
        uint8_t bind = s.info >> 4;
        uint8_t type = s.info & 0xf;
        // STB_GLOBAL and (STT_OBJECT or STT_FUNC)
        if (bind == 1 && (type == 1 || type == 2) && s.name != 0) {
          results.push_back({strings + s.name, s.value, s.size});
        }
      }
    }
    return results;
  }

private:
  const uint8_t* data_;
  size_t size_;
};

} // namespace common
} // namespace vgre
