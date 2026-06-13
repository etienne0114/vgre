// VGRE XLA backend — HLO module (de)serialization (binary blob).
#ifndef VGRE_XLA_HLO_SERIALIZE_H
#define VGRE_XLA_HLO_SERIALIZE_H

#include "vgre/xla/hlo.h"

#include <string>

namespace vgre {
namespace xla {

// Serialize a module to a self-describing binary blob (the "compiled program").
std::string serialize(const HloModule& m);
// Rebuild a module from a blob. Returns false on malformed input.
bool deserialize(const std::string& blob, HloModule& out);

} // namespace xla
} // namespace vgre

#endif // VGRE_XLA_HLO_SERIALIZE_H
