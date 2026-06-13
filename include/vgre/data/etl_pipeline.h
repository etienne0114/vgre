// VGRE ETL Pipeline — docs/missingFeatures.md §9.2.
// A self-contained extract→validate→transform→load engine with schema-based data
// quality validation and a lineage trace. The in-tree connector is the local
// filesystem (CSV); S3 / Azure Data Lake / GCS connectors implement the same
// extract/load interface but require live cloud endpoints.
#ifndef VGRE_DATA_ETL_PIPELINE_H
#define VGRE_DATA_ETL_PIPELINE_H

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace vgre {
namespace data {

using Record = std::map<std::string, std::string>; // one row: column → value

// Mutates a record in place; returns false to DROP the record (filter).
using TransformFn = std::function<bool(Record&)>;

struct SchemaField {
    enum Type { STRING, INT, DOUBLE } type = STRING;
    std::string name;
    bool        required = false;
};
struct Schema {
    std::vector<SchemaField> fields;
};

struct QualityReport {
    int total = 0;
    int passed = 0;      // passed validation
    int rejected = 0;    // failed validation (quarantined)
    int filtered = 0;    // dropped by a transform
    int output = 0;      // surviving records written
    std::vector<std::string> rejectReasons;
};

// Validate one record against a schema. Returns true if valid; else fills reason.
bool validateRecord(const Record& r, const Schema& s, std::string& reason);

class ETLPipeline {
public:
    ETLPipeline& setSchema(const Schema& s);
    ETLPipeline& addTransform(const std::string& name, TransformFn fn);

    // Extract→validate→transform; surviving records go to `output`. The quality
    // report + lineage trace describe what happened at each stage.
    QualityReport run(const std::vector<Record>& input, std::vector<Record>& output);

    const std::vector<std::string>& lineage() const { return lineage_; }

    // Local filesystem connector (CSV; simple comma-split, first line = header).
    static std::vector<Record> extractCSV(const std::string& path);
    static bool loadCSV(const std::string& path, const std::vector<Record>& records,
                        const std::vector<std::string>& columns);

private:
    Schema schema_;
    bool   hasSchema_ = false;
    std::vector<std::pair<std::string, TransformFn>> transforms_;
    std::vector<std::string> lineage_;
};

} // namespace data
} // namespace vgre

#endif // VGRE_DATA_ETL_PIPELINE_H
