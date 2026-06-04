// Tests for QUEUE-49: cudaEventCreateWithFlags + cudaEventElapsedTime.
#include "vgre/core/event.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/error_codes.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <thread>
#include <chrono>
using namespace vgre;
using namespace vgre::core;

static void init_engine() {
    RuntimeEngine::instance().initialize();
}

static void test_default_event() {
    Event e;
    auto r = e.record(0);
    assert(r == VGREResult::SUCCESS);
    assert(e.synchronize() == VGREResult::SUCCESS);
    printf("[PASS] QUEUE-49 default event record/sync\n");
}

static void test_disable_timing() {
    Event e(kEventDisableTiming);
    assert(e.record(0) == VGREResult::SUCCESS);
    assert(e.synchronize() == VGREResult::SUCCESS);
    assert(e.isQueryReady());
    printf("[PASS] QUEUE-49 kEventDisableTiming\n");
}

static void test_blocking_sync() {
    Event e(kEventBlockingSync);
    assert(e.record(0) == VGREResult::SUCCESS);
    assert(e.synchronize() == VGREResult::SUCCESS);
    printf("[PASS] QUEUE-49 kEventBlockingSync\n");
}

static void test_elapsed_time() {
    Event start, end;
    start.record(0); start.synchronize();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    end.record(0); end.synchronize();
    float ms = 0.f;
    assert(end.elapsedTime(start, ms) == VGREResult::SUCCESS);
    assert(ms >= 0.f);
    printf("[PASS] QUEUE-49 elapsedTime = %.3f ms\n", ms);
}

int main() {
    init_engine();
    test_default_event();
    test_disable_timing();
    test_blocking_sync();
    test_elapsed_time();
    printf("All QUEUE-49 event flag tests passed.\n");
    return 0;
}
