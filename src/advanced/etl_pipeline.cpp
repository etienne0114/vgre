// VGRE ETL Pipeline (impl).  See etl_pipeline.h.

#include "vgre/data/etl_pipeline.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace vgre {
namespace data {

namespace {
bool isInt(const std::string& s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i) if (!std::isdigit((unsigned char)s[i])) return false;
    return true;
}
bool isDouble(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
} // namespace

bool validateRecord(const Record& r, const Schema& s, std::string& reason) {
    for (const auto& f : s.fields) {
        auto it = r.find(f.name);
        bool present = it != r.end() && !it->second.empty();
        if (!present) {
            if (f.required) { reason = "missing required field '" + f.name + "'"; return false; }
            continue;
        }
        if (f.type == SchemaField::INT && !isInt(it->second)) {
            reason = "field '" + f.name + "' is not an integer: '" + it->second + "'";
            return false;
        }
        if (f.type == SchemaField::DOUBLE && !isDouble(it->second)) {
            reason = "field '" + f.name + "' is not a number: '" + it->second + "'";
            return false;
        }
    }
    return true;
}

ETLPipeline& ETLPipeline::setSchema(const Schema& s) { schema_ = s; hasSchema_ = true; return *this; }
ETLPipeline& ETLPipeline::addTransform(const std::string& name, TransformFn fn) {
    transforms_.emplace_back(name, std::move(fn));
    return *this;
}

QualityReport ETLPipeline::run(const std::vector<Record>& input, std::vector<Record>& output) {
    lineage_.clear();
    output.clear();
    QualityReport rep;
    rep.total = static_cast<int>(input.size());
    lineage_.push_back("extract: " + std::to_string(rep.total) + " records");

    std::vector<Record> stage;
    for (const auto& r : input) {
        if (hasSchema_) {
            std::string reason;
            if (!validateRecord(r, schema_, reason)) {
                ++rep.rejected;
                rep.rejectReasons.push_back(reason);
                continue;
            }
        }
        ++rep.passed;
        stage.push_back(r);
    }
    lineage_.push_back("validate: " + std::to_string(rep.passed) + " passed, " +
                       std::to_string(rep.rejected) + " rejected");

    for (const auto& t : transforms_) {
        std::vector<Record> next;
        int dropped = 0;
        for (auto& r : stage) {
            Record rec = r;
            if (t.second(rec)) next.push_back(std::move(rec));
            else ++dropped;
        }
        rep.filtered += dropped;
        lineage_.push_back("transform '" + t.first + "': " + std::to_string(stage.size()) +
                           " in, " + std::to_string(next.size()) + " out");
        stage = std::move(next);
    }

    output = std::move(stage);
    rep.output = static_cast<int>(output.size());
    lineage_.push_back("load: " + std::to_string(rep.output) + " records");
    return rep;
}

std::vector<Record> ETLPipeline::extractCSV(const std::string& path) {
    std::vector<Record> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    std::vector<std::string> header;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cells.push_back(trim(cell));
        if (first) { header = cells; first = false; continue; }
        if (cells.empty()) continue;
        Record r;
        for (size_t i = 0; i < header.size() && i < cells.size(); ++i) r[header[i]] = cells[i];
        out.push_back(std::move(r));
    }
    return out;
}

bool ETLPipeline::loadCSV(const std::string& path, const std::vector<Record>& records,
                          const std::vector<std::string>& columns) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    for (size_t i = 0; i < columns.size(); ++i) f << (i ? "," : "") << columns[i];
    f << "\n";
    for (const auto& r : records) {
        for (size_t i = 0; i < columns.size(); ++i) {
            auto it = r.find(columns[i]);
            f << (i ? "," : "") << (it != r.end() ? it->second : "");
        }
        f << "\n";
    }
    return static_cast<bool>(f);
}

} // namespace data
} // namespace vgre
