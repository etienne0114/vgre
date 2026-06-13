// ETL pipeline — extract/validate/transform/load + data quality + lineage.

#include "vgre/data/etl_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <process.h>
#define VGRE_GETPID _getpid
#else
#include <unistd.h>
#define VGRE_GETPID getpid
#endif

using namespace vgre::data;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== ETL pipeline (extract/validate/transform/load) ===\n");

    std::string base = "/tmp/vgre_etl_" + std::to_string(VGRE_GETPID());
    std::string inPath = base + "_in.csv", outPath = base + "_out.csv";

    // ── CSV connector round-trip ─────────────────────────────────────────
    {
        FILE* f = std::fopen(inPath.c_str(), "w");
        std::fputs("name,age,score\n"
                   "alice,30,9.5\n"
                   "bob,17,8.0\n"          // under-age (filtered by transform)
                   "carol,notanumber,7\n"  // bad age (rejected by validation)
                   "dave,40,6.5\n", f);
        std::fclose(f);
        auto recs = ETLPipeline::extractCSV(inPath);
        check("extractCSV reads 4 data rows", recs.size() == 4);
        check("extractCSV maps columns", recs[0]["name"] == "alice" && recs[0]["age"] == "30");
    }

    // ── pipeline: schema validation + filter + map ───────────────────────
    {
        Schema schema;
        schema.fields = {
            {SchemaField::STRING, "name", true},
            {SchemaField::INT, "age", true},
            {SchemaField::DOUBLE, "score", true},
        };

        ETLPipeline p;
        p.setSchema(schema);
        // filter: drop under-18
        p.addTransform("adults_only", [](Record& r) { return std::atoi(r["age"].c_str()) >= 18; });
        // map: uppercase the name's first letter into an UPPER column
        p.addTransform("upper_name", [](Record& r) {
            std::string n = r["name"];
            std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c){ return (char)std::toupper(c); });
            r["NAME"] = n;
            return true;
        });

        auto input = ETLPipeline::extractCSV(inPath);
        std::vector<Record> output;
        QualityReport q = p.run(input, output);

        check("validation rejected the bad-age row", q.rejected == 1 && q.passed == 3);
        check("transform filtered the under-age row", q.filtered == 1);
        check("output has 2 surviving records", q.output == 2 && output.size() == 2);
        check("map transform applied (uppercased name)",
              output[0]["NAME"] == "ALICE" || output[1]["NAME"] == "ALICE");
        check("reject reason captured", !q.rejectReasons.empty() &&
              q.rejectReasons[0].find("age") != std::string::npos);

        // ── lineage trace ────────────────────────────────────────────────
        const auto& lin = p.lineage();
        check("lineage records every stage",
              lin.size() == 5 &&                    // extract, validate, 2 transforms, load
              lin.front().find("extract") != std::string::npos &&
              lin.back().find("load") != std::string::npos);

        // ── load to CSV ──────────────────────────────────────────────────
        check("loadCSV writes output", ETLPipeline::loadCSV(outPath, output, {"name", "age", "NAME"}));
        auto reloaded = ETLPipeline::extractCSV(outPath);
        check("loaded CSV round-trips", reloaded.size() == 2 && reloaded[0].count("NAME") == 1);
    }

    std::remove(inPath.c_str());
    std::remove(outPath.c_str());
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
