#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/graph_optimizer.h"
#include <algorithm>
#include <cstring>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace vgre {
namespace core {


// ── Graph Serialization ────────────────────────────────────────────────────
// Produces a minimal JSON that captures the DAG topology (types, names,
// grid/block dims, dependency edges). Process-specific pointers and live
// memory are not serialized — they are re-registered after deserialization.

static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

vgre::VGREResult GraphManager::serializeGraph(GraphId id, std::string &outJson) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end()) {
        return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    const auto &g = *it->second;

    std::ostringstream j;
    j << "{\"id\":" << g.id
      << ",\"nextNodeId\":" << g.nextNodeId
      << ",\"nodes\":[";

    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
        const auto &n = g.nodes[ni];
        if (ni > 0) j << ",";
        const char *typeStr = (n.type == GraphNodeType::KERNEL)     ? "KERNEL"
                             : (n.type == GraphNodeType::MEMCPY)     ? "MEMCPY"
                                                                      : "CONDITIONAL";
        j << "{\"nodeId\":" << n.nodeId
          << ",\"type\":\"" << typeStr << "\""
          << ",\"kernelId\":" << n.kernelId
          << ",\"kernelName\":\"" << jsonEscape(n.kernelName) << "\""
          << ",\"gridDim\":[" << n.gridDim.x << "," << n.gridDim.y << "," << n.gridDim.z << "]"
          << ",\"blockDim\":[" << n.blockDim.x << "," << n.blockDim.y << "," << n.blockDim.z << "]"
          << ",\"memcpyCount\":" << n.count
          << ",\"memcpyKind\":" << n.kind
          << ",\"condType\":" << static_cast<int>(n.condType)
          << ",\"bodyGraphId\":" << n.bodyGraphId
          << ",\"maxIterations\":" << n.maxIterations
          << ",\"deps\":[";
        for (size_t di = 0; di < n.deps.size(); ++di) {
            if (di > 0) j << ",";
            j << n.deps[di];
        }
        j << "]}";
    }
    j << "]}";

    outJson = j.str();
    VGRE_LOG_INFO("GraphManager", "Serialized graph " + std::to_string(id) +
                  " (" + std::to_string(g.nodes.size()) + " nodes, " +
                  std::to_string(outJson.size()) + " bytes)");
    return vgre::VGREResult::SUCCESS;
}

// ── Graph Deserialization ──────────────────────────────────────────────────
// Minimal JSON parser — no external dependency required.

namespace {

// Skip whitespace and return a reference to the next non-space char.
static size_t skipWS(const std::string &s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
    return pos;
}

// Expect a literal character; advance past it or return false.
static bool expectChar(const std::string &s, size_t &pos, char c) {
    pos = skipWS(s, pos);
    if (pos >= s.size() || s[pos] != c) return false;
    ++pos;
    return true;
}

// Parse a quoted JSON string; pos must point at the opening '"'.
static bool parseString(const std::string &s, size_t &pos, std::string &out) {
    pos = skipWS(s, pos);
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
            switch (s[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        ++pos;
    }
    if (pos >= s.size()) return false;
    ++pos; // consume closing '"'
    return true;
}

// Parse an integer (uint64_t); pos points at the first digit or '-'.
static bool parseInt(const std::string &s, size_t &pos, uint64_t &out) {
    pos = skipWS(s, pos);
    if (pos >= s.size()) return false;
    size_t start = pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos == start) return false;
    out = std::stoull(s.substr(start, pos - start));
    return true;
}

} // namespace

vgre::VGREResult GraphManager::deserializeGraph(const std::string &json, GraphId &outId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t p = 0;
    if (!expectChar(json, p, '{')) return vgre::VGREResult::ERR_INVALID_VALUE;

    auto newGraph = std::make_shared<Graph>();
    newGraph->id = nextGraphId_++;

    // Parse top-level fields until we hit the nodes array.
    while (p < json.size() && json[p] != '}') {
        std::string key;
        if (!parseString(json, p, key)) return vgre::VGREResult::ERR_INVALID_VALUE;
        if (!expectChar(json, p, ':'))  return vgre::VGREResult::ERR_INVALID_VALUE;

        if (key == "nextNodeId") {
            uint64_t v = 0;
            if (!parseInt(json, p, v)) return vgre::VGREResult::ERR_INVALID_VALUE;
            newGraph->nextNodeId = v;
        } else if (key == "id") {
            uint64_t v = 0; parseInt(json, p, v); // original id — ignored, we assign a new one
        } else if (key == "nodes") {
            // Parse array of node objects
            if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
            p = skipWS(json, p);
            while (p < json.size() && json[p] != ']') {
                if (!expectChar(json, p, '{')) return vgre::VGREResult::ERR_INVALID_VALUE;

                GraphNode node;
                while (p < json.size() && json[p] != '}') {
                    std::string nk;
                    if (!parseString(json, p, nk)) return vgre::VGREResult::ERR_INVALID_VALUE;
                    if (!expectChar(json, p, ':'))  return vgre::VGREResult::ERR_INVALID_VALUE;

                    if (nk == "nodeId") {
                        uint64_t v = 0; parseInt(json, p, v); node.nodeId = v;
                    } else if (nk == "type") {
                        std::string tv; parseString(json, p, tv);
                        if (tv == "KERNEL") node.type = GraphNodeType::KERNEL;
                        else if (tv == "CONDITIONAL") node.type = GraphNodeType::CONDITIONAL;
                        else node.type = GraphNodeType::MEMCPY;
                    } else if (nk == "kernelId") {
                        uint64_t v = 0; parseInt(json, p, v); node.kernelId = static_cast<KernelId>(v);
                    } else if (nk == "kernelName") {
                        parseString(json, p, node.kernelName);
                    } else if (nk == "gridDim" || nk == "blockDim") {
                        // parse [x,y,z]
                        if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
                        uint64_t x = 0, y = 0, z = 0;
                        parseInt(json, p, x); expectChar(json, p, ',');
                        parseInt(json, p, y); expectChar(json, p, ',');
                        parseInt(json, p, z); expectChar(json, p, ']');
                        if (nk == "gridDim")  node.gridDim  = {static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
                        else                   node.blockDim = {static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
                    } else if (nk == "memcpyCount") {
                        uint64_t v = 0; parseInt(json, p, v); node.count = static_cast<size_t>(v);
                    } else if (nk == "memcpyKind") {
                        uint64_t v = 0; parseInt(json, p, v); node.kind = static_cast<int>(v);
                    } else if (nk == "condType") {
                        uint64_t v = 0; parseInt(json, p, v);
                        if (v == 1) node.condType = GraphCondType::WHILE;
                        else if (v == 2) node.condType = GraphCondType::SWITCH;
                        else node.condType = GraphCondType::IF;
                    } else if (nk == "bodyGraphId") {
                        uint64_t v = 0; parseInt(json, p, v); node.bodyGraphId = static_cast<GraphId>(v);
                    } else if (nk == "maxIterations") {
                        uint64_t v = 0; parseInt(json, p, v); node.maxIterations = static_cast<unsigned int>(v);
                    } else if (nk == "deps") {
                        if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
                        p = skipWS(json, p);
                        while (p < json.size() && json[p] != ']') {
                            uint64_t dep = 0;
                            if (!parseInt(json, p, dep)) return vgre::VGREResult::ERR_INVALID_VALUE;
                            node.deps.push_back(dep);
                            p = skipWS(json, p);
                            if (p < json.size() && json[p] == ',') ++p;
                        }
                        expectChar(json, p, ']');
                    } else {
                        // Unknown field — skip a simple value (string or number)
                        p = skipWS(json, p);
                        if (p < json.size() && json[p] == '"') {
                            std::string tmp; parseString(json, p, tmp);
                        } else {
                            while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
                        }
                    }

                    p = skipWS(json, p);
                    if (p < json.size() && json[p] == ',') ++p;
                    p = skipWS(json, p);
                }
                if (!expectChar(json, p, '}')) return vgre::VGREResult::ERR_INVALID_VALUE;
                newGraph->nodes.push_back(std::move(node));
                p = skipWS(json, p);
                if (p < json.size() && json[p] == ',') ++p;
                p = skipWS(json, p);
            }
            expectChar(json, p, ']');
        } else {
            // Unknown top-level key — skip value
            p = skipWS(json, p);
            if (p < json.size() && json[p] == '"') {
                std::string tmp; parseString(json, p, tmp);
            } else {
                while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
            }
        }

        p = skipWS(json, p);
        if (p < json.size() && json[p] == ',') ++p;
        p = skipWS(json, p);
    }

    graphs_[newGraph->id] = newGraph;
    outId = newGraph->id;

    VGRE_LOG_INFO("GraphManager", "Deserialized graph -> id=" + std::to_string(outId) +
                  " (" + std::to_string(newGraph->nodes.size()) + " nodes)");
    return vgre::VGREResult::SUCCESS;
}


} // namespace core
} // namespace vgre
