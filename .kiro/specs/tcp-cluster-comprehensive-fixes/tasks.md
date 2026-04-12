# Implementation Tasks: TCP Cluster Comprehensive Fixes

## Overview

This task list implements all fixes for tcp_cluster.cpp following the bugfix requirements-first workflow. The implementation is organized into 7 phases covering missing implementations, code duplication elimination, stub replacement, business logic fixes, modular architecture, testability improvements, and documentation.

## Phase 1: Implement Missing Methods (Priority: CRITICAL)

- [x] 1. Write bug condition exploration test for missing implementations
  - **Property 1: Bug Condition** - Missing Method Linker Errors
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface linker errors that demonstrate missing implementations
  - **Scoped PBT Approach**: Test concrete cases - reportComputeFromWorker() and allReduce() calls
  - Create test file `tests/advanced/test_tcp_cluster_missing_methods.cpp`
  - Test that calling reportComputeFromWorker(1.5, 4, 12345) links successfully
  - Test that calling allReduce(ptr, 100, FLOAT32) links successfully
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with linker errors (this is correct - it proves the bug exists)
  - Document linker error messages to understand root cause
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2_

- [x] 2. Write preservation property tests for existing TCP cluster functionality (BEFORE implementing fix)
  - **Property 2: Preservation** - Existing Cluster Operations
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for existing methods (launchRemoteKernel, barrier, etc.)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Test that master initialization binds to port successfully
  - Test that worker connection to master succeeds
  - Test that kernel registration broadcasts to all workers
  - Test that remote kernel launch sends correct packets
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_


- [x] 3. Implement missing methods

  - [x] 3.1 Implement reportComputeFromWorker()
    - Add method implementation in `src/advanced/tcp_cluster.cpp`
    - Create CreditReportPacket with compute_seconds, cpu_cores, kernel_id, timestamp
    - Send packet using send_packet() with PacketType::CREDIT_REPORT
    - Add validation: return early if !enabled_ or is_master_
    - Add debug logging for compute metrics
    - _Bug_Condition: isMissingImplementation("reportComputeFromWorker") returns true_
    - _Expected_Behavior: Method links successfully and reports metrics to master_
    - _Preservation: Existing packet sending and telemetry aggregation unchanged_
    - _Requirements: 2.1_

  - [x] 3.2 Implement allReduce() - worker path
    - Add worker-side implementation in `src/advanced/tcp_cluster.cpp`
    - Create CollectiveOpPacket with op_type=0, datatype, count, sequence
    - Send CollectiveOpPacket followed by RAW_DATA packet with local values
    - Wait on reduction_cv_ with 30-second timeout
    - Copy result from active_reduction_buffer_ back to ptr
    - Return SUCCESS or ERR_TIMEOUT
    - _Bug_Condition: isMissingImplementation("allReduce") returns true_
    - _Expected_Behavior: Worker sends data to master and receives reduced result_
    - _Preservation: Existing collective operation synchronization unchanged_
    - _Requirements: 2.2_

  - [x] 3.3 Implement allReduce() - master path
    - Add master-side implementation in `src/advanced/tcp_cluster.cpp`
    - Initialize reduction state: is_reducing_=true, reduction_count_=0
    - Copy master's local data to active_reduction_buffer_
    - Wait for all workers to send their data (wait on reduction_cv_)
    - Broadcast final result to all workers via RAW_DATA packet
    - Copy result back to master's ptr
    - Reset is_reducing_=false and notify all waiters
    - _Bug_Condition: isMissingImplementation("allReduce") returns true_
    - _Expected_Behavior: Master collects data from all workers, reduces, and broadcasts result_
    - _Preservation: Existing broadcast and synchronization mechanisms unchanged_
    - _Requirements: 2.2_

  - [x] 3.4 Add COLLECTIVE_OP packet handler
    - Add handler in processClientStagingBuffer() for PacketType::COLLECTIVE_OP
    - Parse CollectiveOpPacket from payload
    - Set flag to expect RAW_DATA next
    - Add handler for RAW_DATA when is_reducing_ is true
    - Call sum_reduce_simd() based on datatype (FLOAT32, FLOAT64, INT32, INT64)
    - Increment reduction_count_ and notify reduction_cv_
    - _Bug_Condition: COLLECTIVE_OP packets are not handled_
    - _Expected_Behavior: Master processes reduction data from workers_
    - _Preservation: Existing packet processing loop unchanged_
    - _Requirements: 2.2_

  - [x] 3.5 Add getTypeSizeFromDatatype() helper
    - Add private method to TCPClusterManager
    - Return 4 for INT32, UINT32, FLOAT32
    - Return 8 for INT64, UINT64, FLOAT64, default
    - Use in allReduce() for buffer sizing
    - _Bug_Condition: Type size calculation is duplicated_
    - _Expected_Behavior: Centralized type size lookup_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.2_

  - [x] 3.6 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Missing Methods Implemented
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify reportComputeFromWorker() links and executes
    - Verify allReduce() links and executes
    - _Requirements: 2.1, 2.2_

  - [x] 3.7 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing Cluster Operations
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all existing cluster operations still work
    - Confirm protocol compatibility maintained
    - Confirm performance characteristics unchanged

- [x] 4. Checkpoint - Phase 1 Complete
  - Ensure all Phase 1 tests pass
  - Verify no linker errors for reportComputeFromWorker() and allReduce()
  - Verify existing functionality preserved
  - Ask user if questions arise


## Phase 2: Eliminate Code Duplication (Priority: HIGH)

- [x] 5. Write bug condition exploration test for code duplication
  - **Property 1: Bug Condition** - Duplicated Code Patterns
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Detect duplicated packet construction, delta-sync, and argument serialization logic
  - **Scoped PBT Approach**: Test that unified functions exist and are used consistently
  - Create test file `tests/advanced/test_tcp_cluster_duplication.cpp`
  - Test that constructPacket() method exists and is called by send_packet() and send_packet_direct()
  - Test that syncPointerToWorker() method exists and handles delta-sync logic
  - Test that streamArgumentsToWorker() method exists and handles all argument types
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (methods don't exist - confirms duplication bug)
  - Document duplicated code locations
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 6. Write preservation property tests for packet sending and memory sync (BEFORE implementing fix)
  - **Property 2: Preservation** - Packet and Memory Operations
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for send_packet(), send_packet_direct(), memory sync
  - Write property-based tests capturing observed behavior patterns
  - Test that packets are constructed with correct VSBP headers
  - Test that priority queuing works correctly (HIGH before LOW)
  - Test that delta-sync detects dirty ranges correctly
  - Test that full-sync transfers complete data
  - Test that SHM fallback to TCP works
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.8, 3.9, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16_

- [x] 7. Extract common packet construction logic

  - [x] 7.1 Add constructPacket() private method
    - Add method to TCPClusterManager in `src/advanced/tcp_cluster.cpp`
    - Accept PacketType, payload pointer, payload length
    - Create VSBPHeader with VSBP_MAGIC, VSBP_VERSION, type, sequence, payloadSize
    - Use atomic sequence counter for packet ordering
    - Copy payload after header
    - Return std::vector<uint8_t> with complete packet
    - _Bug_Condition: isCodeDuplication(packet_construction) returns true_
    - _Expected_Behavior: Single unified packet construction function_
    - _Preservation: Packet format and structure unchanged_
    - _Requirements: 2.3_

  - [x] 7.2 Refactor send_packet() to use constructPacket()
    - Replace inline packet construction with call to constructPacket()
    - Keep validation logic (validatePacketSize)
    - Keep priority determination logic
    - Keep TSS2 enqueue logic
    - Remove duplicated header construction code
    - _Bug_Condition: send_packet() duplicates packet construction_
    - _Expected_Behavior: send_packet() delegates to constructPacket()_
    - _Preservation: Packet sending behavior unchanged_
    - _Requirements: 2.3_

  - [x] 7.3 Refactor send_packet_direct() to use constructPacket()
    - Replace inline packet construction with call to constructPacket()
    - Keep direct send logic (bypass TSS2 queue)
    - Keep encryption logic (SecureChannel)
    - Remove duplicated header construction code
    - _Bug_Condition: send_packet_direct() duplicates packet construction_
    - _Expected_Behavior: send_packet_direct() delegates to constructPacket()_
    - _Preservation: Direct packet sending behavior unchanged_
    - _Requirements: 2.3_


  - [x] 7.4 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Unified Packet Construction
    - **IMPORTANT**: Re-run the SAME test from task 5 - do NOT write a new test
    - Run bug condition exploration test from step 5
    - **EXPECTED OUTCOME**: Test PASSES (confirms constructPacket() exists and is used)
    - _Requirements: 2.3_

  - [x] 7.5 Verify preservation tests still pass
    - **Property 2: Preservation** - Packet Operations
    - **IMPORTANT**: Re-run the SAME tests from task 6 - do NOT write new tests
    - Run preservation property tests from step 6
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)

- [x] 8. Extract delta-sync logic

  - [x] 8.1 Add syncPointerToWorker() private method
    - Add method to TCPClusterManager
    - Accept void* ptr, uint64_t handle, shared_ptr<ClientConnection>
    - Get allocation size from MemoryManager
    - Check if pointer was previously synced (in synced_handles set)
    - If synced, get dirty ranges from MemoryManager
    - If dirty ranges exist, call sendDeltaSync()
    - Otherwise, call sendFullSync()
    - Clear dirty pages after successful sync
    - Return VGREResult
    - _Bug_Condition: isCodeDuplication(delta_sync_logic) returns true_
    - _Expected_Behavior: Single unified delta-sync decision function_
    - _Preservation: Memory synchronization behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.2 Add sendDeltaSync() private method
    - Add method to TCPClusterManager
    - Accept ptr, handle, dirty ranges, client connection
    - Branch on client->is_local: call sendDeltaSyncSHM() or sendDeltaSyncTCP()
    - Return VGREResult
    - _Bug_Condition: Delta-sync logic is duplicated_
    - _Expected_Behavior: Unified delta-sync dispatcher_
    - _Preservation: Delta-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.3 Add sendDeltaSyncSHM() private method
    - Add method to TCPClusterManager
    - Calculate total size of dirty ranges
    - Check SHM space availability
    - Update client->shm_offset
    - Send DataShmDirtyPacket with offset and range count
    - For each range: send DirtyRangePacket and copy data to SHM
    - Return VGREResult
    - _Bug_Condition: SHM delta-sync is duplicated_
    - _Expected_Behavior: Unified SHM delta-sync implementation_
    - _Preservation: SHM delta-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.4 Add sendDeltaSyncTCP() private method
    - Add method to TCPClusterManager
    - Send DataHeaderDirtyPacket with range count
    - For each range: send DirtyRangePacket and DATA_BODY with range data
    - Return VGREResult
    - _Bug_Condition: TCP delta-sync is duplicated_
    - _Expected_Behavior: Unified TCP delta-sync implementation_
    - _Preservation: TCP delta-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.5 Add sendFullSync() private method
    - Add method to TCPClusterManager
    - Branch on client->is_local: call sendFullSyncSHM() or sendFullSyncTCP()
    - Return VGREResult
    - _Bug_Condition: Full-sync logic is duplicated_
    - _Expected_Behavior: Unified full-sync dispatcher_
    - _Preservation: Full-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.6 Add sendFullSyncSHM() private method
    - Add method to TCPClusterManager
    - Check SHM space availability
    - Update client->shm_offset
    - Copy data to SHM at offset
    - Send DataShmPacket with offset and size
    - Return VGREResult
    - _Bug_Condition: SHM full-sync is duplicated_
    - _Expected_Behavior: Unified SHM full-sync implementation_
    - _Preservation: SHM full-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.7 Add sendFullSyncTCP() private method
    - Add method to TCPClusterManager
    - Send DataHeaderPacket with size
    - Send DATA_BODY with complete data
    - Return VGREResult
    - _Bug_Condition: TCP full-sync is duplicated_
    - _Expected_Behavior: Unified TCP full-sync implementation_
    - _Preservation: TCP full-sync behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.8 Refactor launchRemoteKernel() to use syncPointerToWorker()
    - Replace inline delta-sync logic with call to syncPointerToWorker()
    - Remove duplicated dirty range detection
    - Remove duplicated SHM/TCP branching
    - Keep kernel launch logic unchanged
    - _Bug_Condition: launchRemoteKernel() duplicates delta-sync_
    - _Expected_Behavior: launchRemoteKernel() delegates to syncPointerToWorker()_
    - _Preservation: Remote kernel launch behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.9 Refactor launchPartitionedKernel() to use syncPointerToWorker()
    - Replace inline delta-sync logic with call to syncPointerToWorker()
    - Remove duplicated dirty range detection
    - Remove duplicated SHM/TCP branching
    - Keep partitioned dispatch logic unchanged
    - _Bug_Condition: launchPartitionedKernel() duplicates delta-sync_
    - _Expected_Behavior: launchPartitionedKernel() delegates to syncPointerToWorker()_
    - _Preservation: Partitioned kernel launch behavior unchanged_
    - _Requirements: 2.4_

  - [x] 8.10 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Unified Delta-Sync
    - **IMPORTANT**: Re-run the SAME test from task 5 - do NOT write a new test
    - Run bug condition exploration test from step 5
    - **EXPECTED OUTCOME**: Test PASSES (confirms syncPointerToWorker() exists and is used)
    - _Requirements: 2.4_

  - [x] 8.11 Verify preservation tests still pass
    - **Property 2: Preservation** - Memory Sync Operations
    - **IMPORTANT**: Re-run the SAME tests from task 6 - do NOT write new tests
    - Run preservation property tests from step 6
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)


- [x] 9. Extract argument serialization logic

  - [x] 9.1 Add streamArgumentsToWorker() private method
    - Add method to TCPClusterManager
    - Accept void** args, int num_args, uint64_t kernel_id, client connection
    - Get argument types from RuntimeEngine
    - For each argument: dispatch to sendStructArg(), sendPointerArg(), or sendScalarArg()
    - Return VGREResult on first error or SUCCESS
    - _Bug_Condition: isCodeDuplication(argument_serialization) returns true_
    - _Expected_Behavior: Single unified argument streaming function_
    - _Preservation: Argument serialization behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.2 Add sendStructArg() private method
    - Add method to TCPClusterManager
    - Get struct size from kernel IR
    - Send StructDataPacket with arg_index and size
    - Send DATA_BODY with struct contents
    - Return VGREResult
    - _Bug_Condition: Struct argument sending is duplicated_
    - _Expected_Behavior: Unified struct argument sender_
    - _Preservation: Struct argument behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.3 Add sendPointerArg() private method
    - Add method to TCPClusterManager
    - Dereference pointer to get actual address
    - Call syncPointerToWorker() to sync memory
    - Send ArgScalarPacket with ARG_POINTER type and handle
    - Return VGREResult
    - _Bug_Condition: Pointer argument sending is duplicated_
    - _Expected_Behavior: Unified pointer argument sender_
    - _Preservation: Pointer argument behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.4 Add sendScalarArg() private method
    - Add method to TCPClusterManager
    - Determine argument size based on type (4 or 8 bytes)
    - Copy value to ArgScalarPacket
    - Send ArgScalarPacket with ARG_SCALAR type
    - Return VGREResult
    - _Bug_Condition: Scalar argument sending is duplicated_
    - _Expected_Behavior: Unified scalar argument sender_
    - _Preservation: Scalar argument behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.5 Refactor launchRemoteKernel() to use streamArgumentsToWorker()
    - Replace inline argument loop with call to streamArgumentsToWorker()
    - Remove duplicated type dispatch logic
    - Keep kernel launch command sending unchanged
    - _Bug_Condition: launchRemoteKernel() duplicates argument serialization_
    - _Expected_Behavior: launchRemoteKernel() delegates to streamArgumentsToWorker()_
    - _Preservation: Remote kernel launch behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.6 Refactor launchPartitionedKernel() to use streamArgumentsToWorker()
    - Replace inline argument loop with call to streamArgumentsToWorker()
    - Remove duplicated type dispatch logic
    - Keep partitioned dispatch logic unchanged
    - _Bug_Condition: launchPartitionedKernel() duplicates argument serialization_
    - _Expected_Behavior: launchPartitionedKernel() delegates to streamArgumentsToWorker()_
    - _Preservation: Partitioned kernel launch behavior unchanged_
    - _Requirements: 2.5_

  - [x] 9.7 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Unified Argument Serialization
    - **IMPORTANT**: Re-run the SAME test from task 5 - do NOT write a new test
    - Run bug condition exploration test from step 5
    - **EXPECTED OUTCOME**: Test PASSES (confirms streamArgumentsToWorker() exists and is used)
    - _Requirements: 2.5_

  - [x] 9.8 Verify preservation tests still pass
    - **Property 2: Preservation** - Argument Operations
    - **IMPORTANT**: Re-run the SAME tests from task 6 - do NOT write new tests
    - Run preservation property tests from step 6
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)

- [x] 10. Replace goto-based SHM fallback with RAII

  - [x] 10.1 Add ShmFallbackGuard RAII class
    - Add class definition in tcp_cluster.cpp anonymous namespace
    - Constructor: save current shm_offset, attempt SHM allocation
    - Destructor: rollback shm_offset if !succeeded_
    - Methods: succeeded(), offset()
    - _Bug_Condition: isCodeDuplication(shm_fallback_goto) returns true_
    - _Expected_Behavior: RAII-based SHM fallback management_
    - _Preservation: SHM fallback behavior unchanged_
    - _Requirements: 2.6_

  - [x] 10.2 Replace goto patterns with ShmFallbackGuard
    - Find all instances of SHM fallback goto patterns
    - Replace with ShmFallbackGuard usage
    - Remove goto labels and manual rollback code
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: goto-based error handling is duplicated_
    - _Expected_Behavior: RAII ensures cleanup on all paths_
    - _Preservation: SHM fallback behavior unchanged_
    - _Requirements: 2.6_

  - [x] 10.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - RAII SHM Fallback
    - **IMPORTANT**: Re-run the SAME test from task 5 - do NOT write a new test
    - Run bug condition exploration test from step 5
    - **EXPECTED OUTCOME**: Test PASSES (confirms RAII pattern used)
    - _Requirements: 2.6_

  - [x] 10.4 Verify preservation tests still pass
    - **Property 2: Preservation** - SHM Fallback Operations
    - **IMPORTANT**: Re-run the SAME tests from task 6 - do NOT write new tests
    - Run preservation property tests from step 6
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)

- [x] 11. Checkpoint - Phase 2 Complete
  - Ensure all Phase 2 tests pass
  - Verify code duplication reduced by 60%
  - Verify all refactored code uses unified functions
  - Verify no regressions in packet sending, memory sync, or argument serialization
  - Ask user if questions arise


## Phase 3: Replace Stubs with Production Code (Priority: HIGH)

- [x] 12. Write bug condition exploration test for stub implementations
  - **Property 1: Bug Condition** - Stub/Mock Implementations
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Detect stub implementations that lack production-ready optimizations
  - **Scoped PBT Approach**: Test that sum_reduce uses SIMD and security handshake enforces policy
  - Create test file `tests/advanced/test_tcp_cluster_stubs.cpp`
  - Test that sum_reduce() uses SIMD instructions (check for AVX2/SSE2 code paths)
  - Test that security handshake failure rejects connection (no plaintext fallback)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (stubs lack optimizations - confirms bug)
  - Document stub limitations
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 3.1, 3.2_

- [x] 13. Write preservation property tests for reduction and security (BEFORE implementing fix)
  - **Property 2: Preservation** - Reduction and Security Operations
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for sum_reduce() and security handshake
  - Write property-based tests capturing observed behavior patterns
  - Test that sum_reduce() produces correct results (even if slow)
  - Test that security handshake succeeds when credentials match
  - Test that security handshake times out appropriately
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.6, 3.19, 3.20_

- [x] 14. Implement SIMD-optimized sum_reduce

  - [x] 14.1 Add SIMD intrinsics headers
    - Add conditional includes for immintrin.h (AVX2) and emmintrin.h (SSE2)
    - Add compile-time detection macros
    - _Bug_Condition: isStubImplementation(sum_reduce) returns true_
    - _Expected_Behavior: SIMD headers available for optimization_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.7_

  - [x] 14.2 Implement AVX2 path for float
    - Add AVX2 code path using _mm256_loadu_ps, _mm256_add_ps, _mm256_storeu_ps
    - Process 8 floats per iteration
    - Use if constexpr for type dispatch
    - _Bug_Condition: sum_reduce uses simple loop_
    - _Expected_Behavior: AVX2 vectorized addition for float arrays_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.3 Implement AVX2 path for double
    - Add AVX2 code path using _mm256_loadu_pd, _mm256_add_pd, _mm256_storeu_pd
    - Process 4 doubles per iteration
    - Use if constexpr for type dispatch
    - _Bug_Condition: sum_reduce uses simple loop_
    - _Expected_Behavior: AVX2 vectorized addition for double arrays_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.4 Implement AVX2 path for int32_t
    - Add AVX2 code path using _mm256_loadu_si256, _mm256_add_epi32, _mm256_storeu_si256
    - Process 8 int32_t per iteration
    - Use if constexpr for type dispatch
    - _Bug_Condition: sum_reduce uses simple loop_
    - _Expected_Behavior: AVX2 vectorized addition for int32_t arrays_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.5 Implement AVX2 path for int64_t
    - Add AVX2 code path using _mm256_loadu_si256, _mm256_add_epi64, _mm256_storeu_si256
    - Process 4 int64_t per iteration
    - Use if constexpr for type dispatch
    - _Bug_Condition: sum_reduce uses simple loop_
    - _Expected_Behavior: AVX2 vectorized addition for int64_t arrays_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.6 Implement SSE2 fallback paths
    - Add SSE2 code paths for float, double, int32_t, int64_t
    - Use _mm_loadu_ps, _mm_add_ps, _mm_storeu_ps for float (4 per iteration)
    - Use _mm_loadu_pd, _mm_add_pd, _mm_storeu_pd for double (2 per iteration)
    - Use _mm_loadu_si128, _mm_add_epi32, _mm_storeu_si128 for int32_t (4 per iteration)
    - Use _mm_loadu_si128, _mm_add_epi64, _mm_storeu_si128 for int64_t (2 per iteration)
    - _Bug_Condition: No SSE2 fallback for non-AVX2 systems_
    - _Expected_Behavior: SSE2 vectorized addition when AVX2 unavailable_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.7 Keep scalar fallback for remaining elements
    - Add scalar loop for elements not covered by SIMD
    - Handle tail elements (count % vector_width)
    - _Bug_Condition: Tail elements not handled_
    - _Expected_Behavior: Complete reduction including tail_
    - _Preservation: Reduction correctness unchanged_
    - _Requirements: 2.7_

  - [x] 14.8 Add performance benchmark
    - Create benchmark comparing old vs new sum_reduce
    - Measure throughput (GB/s) for different array sizes
    - Verify 4-8x speedup with SIMD
    - _Bug_Condition: No performance validation_
    - _Expected_Behavior: Measurable SIMD performance improvement_
    - _Preservation: Correctness maintained_
    - _Requirements: 2.7_

  - [x] 14.9 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - SIMD-Optimized Reduction
    - **IMPORTANT**: Re-run the SAME test from task 12 - do NOT write a new test
    - Run bug condition exploration test from step 12
    - **EXPECTED OUTCOME**: Test PASSES (confirms SIMD code paths exist)
    - _Requirements: 2.7_

  - [x] 14.10 Verify preservation tests still pass
    - **Property 2: Preservation** - Reduction Correctness
    - **IMPORTANT**: Re-run the SAME tests from task 13 - do NOT write new tests
    - Run preservation property tests from step 13
    - **EXPECTED OUTCOME**: Tests PASS (confirms correctness preserved)


- [x] 15. Enforce security policy (no plaintext fallback)

  - [x] 15.1 Update performSecureHandshake() to enforce policy
    - Check if security_enabled_ is true
    - If enabled and auth_token_str_ is empty, return ERR_AUTH_FAILED
    - Perform handshake with timeout
    - If handshake fails, log error and return ERR_AUTH_FAILED
    - Remove any plaintext fallback code
    - Close connection on handshake failure
    - _Bug_Condition: isStubImplementation(security_handshake) returns true_
    - _Expected_Behavior: Security policy enforced, no silent fallback_
    - _Preservation: Successful handshake behavior unchanged_
    - _Requirements: 2.8_

  - [x] 15.2 Add explicit logging for security enforcement
    - Log "Security enabled but VGRE_TCP_AUTH_TOKEN not set" on missing token
    - Log "Security handshake failed - closing connection (no plaintext fallback)" on failure
    - Log "Security handshake completed with [ip]" on success
    - _Bug_Condition: Silent fallback without user notification_
    - _Expected_Behavior: Clear logging of security decisions_
    - _Preservation: Logging behavior enhanced_
    - _Requirements: 2.8, 2.24_

  - [x] 15.3 Update packet type validation in handshake
    - Check that response packet type is SECURE_HANDSHAKE_ACK
    - If not, log error with actual packet type received
    - Return ERR_AUTH_FAILED without fallback
    - _Bug_Condition: Unexpected packet types accepted_
    - _Expected_Behavior: Strict packet type validation_
    - _Preservation: Valid handshake unchanged_
    - _Requirements: 2.8_

  - [x] 15.4 Add security policy test
    - Create test that enables security without setting auth token
    - Verify ERR_AUTH_FAILED returned
    - Create test that simulates handshake failure
    - Verify connection closed without plaintext fallback
    - _Bug_Condition: Security policy not enforced_
    - _Expected_Behavior: Policy violations rejected_
    - _Preservation: Valid security flows unchanged_
    - _Requirements: 2.8, 2.24_

  - [x] 15.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Security Policy Enforced
    - **IMPORTANT**: Re-run the SAME test from task 12 - do NOT write a new test
    - Run bug condition exploration test from step 12
    - **EXPECTED OUTCOME**: Test PASSES (confirms policy enforcement)
    - _Requirements: 2.8_

  - [x] 15.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Security Operations
    - **IMPORTANT**: Re-run the SAME tests from task 13 - do NOT write new tests
    - Run preservation property tests from step 13
    - **EXPECTED OUTCOME**: Tests PASS (confirms valid handshakes work)

- [x] 16. Checkpoint - Phase 3 Complete
  - Ensure all Phase 3 tests pass
  - Verify SIMD optimization provides 4-8x speedup
  - Verify security policy enforced without silent fallback
  - Verify reduction correctness and security handshake success preserved
  - Ask user if questions arise


## Phase 4: Fix Business Logic Issues (Priority: CRITICAL)

- [x] 17. Write bug condition exploration test for business logic issues
  - **Property 1: Bug Condition** - Poor Business Logic
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Detect magic numbers, race conditions, resource leaks, and inconsistent error handling
  - **Scoped PBT Approach**: Test for named constants, RAII wrappers, atomic operations, consistent return types
  - Create test file `tests/advanced/test_tcp_cluster_business_logic.cpp`
  - Test that timeout constants are named (not magic numbers)
  - Test that duplicate connection check is atomic (no TOCTOU)
  - Test that sockets are wrapped in RAII (no leaks)
  - Test that error-prone operations return VGREResult consistently
  - Test that I/O uses blocking with timeout (not busy-wait)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (business logic issues exist - confirms bug)
  - Document all business logic issues found
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 4.1-4.16_

- [x] 18. Write preservation property tests for core functionality (BEFORE implementing fix)
  - **Property 2: Preservation** - Core Cluster Functionality
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for connection management, packet I/O, error handling
  - Write property-based tests capturing observed behavior patterns
  - Test that connections are established correctly
  - Test that packets are sent and received correctly
  - Test that errors are handled (even if inconsistently)
  - Test that timeouts work (even if hardcoded)
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1-3.25_

- [x] 19. Add named constants

  - [x] 19.1 Create Constants namespace
    - Add namespace Constants inside anonymous namespace in tcp_cluster.cpp
    - Group constants by category: Timeouts, Flow Control, Memory, Backoff, Discovery, Retry
    - _Bug_Condition: hasPoorBusinessLogic(magic_numbers) returns true_
    - _Expected_Behavior: All constants named and documented_
    - _Preservation: No functional changes_
    - _Requirements: 2.9_

  - [x] 19.2 Add timeout constants
    - SEND_TIMEOUT_SECONDS = 5
    - HANDSHAKE_TIMEOUT_SECONDS = 5
    - PEEK_TIMEOUT_MS = 200
    - POLL_TIMEOUT_MS = 50
    - RECV_TIMEOUT_MS = 1000
    - BACKOFF_AFTER_DISCONNECT_SECONDS = 8
    - HANDSHAKE_STUCK_TIMEOUT_MS = 15000
    - _Bug_Condition: Timeout values are magic numbers_
    - _Expected_Behavior: Named timeout constants_
    - _Preservation: Timeout values unchanged_
    - _Requirements: 2.9, 2.13_

  - [x] 19.3 Add flow control constants
    - MAX_IN_FLIGHT_KERNELS = 16
    - Add comment explaining tuning guidance
    - _Bug_Condition: Flow control limit is magic number_
    - _Expected_Behavior: Named flow control constant_
    - _Preservation: Flow control behavior unchanged_
    - _Requirements: 2.11_

  - [x] 19.4 Add memory constants
    - SHM_RESULT_OFFSET_BASE = 128 * 1024 * 1024 (128MB)
    - DEFAULT_SHM_SIZE = 256 * 1024 * 1024 (256MB)
    - Add comments explaining offset rationale
    - _Bug_Condition: Memory offsets are magic numbers_
    - _Expected_Behavior: Named memory constants with explanations_
    - _Preservation: Memory layout unchanged_
    - _Requirements: 2.12_

  - [x] 19.5 Add backoff constants
    - INITIAL_BACKOFF_SECONDS = 5
    - MAX_BACKOFF_SECONDS = 300
    - BACKOFF_MULTIPLIER = 4
    - PROACTIVE_BACKOFF_SECONDS = 4
    - _Bug_Condition: Backoff values are magic numbers_
    - _Expected_Behavior: Named backoff constants_
    - _Preservation: Backoff behavior unchanged_
    - _Requirements: 2.14_

  - [x] 19.6 Add discovery constants
    - UDP_ANNOUNCE_INTERVAL_SECONDS = 2
    - UDP_DISCOVERY_PORT = 7778
    - UDP_WORKER_PORT = 7779
    - _Bug_Condition: Discovery values are magic numbers_
    - _Expected_Behavior: Named discovery constants_
    - _Preservation: Discovery behavior unchanged_
    - _Requirements: 2.9_

  - [x] 19.7 Add retry constants
    - MAX_DELTA_SYNC_RETRIES = 3
    - INITIAL_RETRY_BACKOFF_MS = 100
    - MAX_RETRY_BACKOFF_MS = 5000
    - _Bug_Condition: Retry values are magic numbers_
    - _Expected_Behavior: Named retry constants_
    - _Preservation: Retry behavior unchanged_
    - _Requirements: 2.22_

  - [x] 19.8 Add PacketPriority enum
    - Create enum class PacketPriority : uint32_t { HIGH = 0, LOW = 1 }
    - Replace magic numbers 0/1 with enum values
    - _Bug_Condition: Priority values are magic numbers_
    - _Expected_Behavior: Type-safe priority enum_
    - _Preservation: Priority values unchanged_
    - _Requirements: 2.10_

  - [x] 19.9 Replace all magic numbers with named constants
    - Search for all numeric literals in tcp_cluster.cpp
    - Replace with appropriate named constants
    - Verify no hardcoded values remain
    - _Bug_Condition: Magic numbers throughout codebase_
    - _Expected_Behavior: All values named and documented_
    - _Preservation: Functional behavior unchanged_
    - _Requirements: 2.9, 2.10, 2.11, 2.12, 2.13_


- [x] 20. Fix TOCTOU race in duplicate connection detection

  - [x] 20.1 Add addClientIfNotDuplicate() method
    - Add private method to TCPClusterManager
    - Accept ip address, socket fd, sockaddr_in address
    - Perform atomic check-and-insert under clients_mutex_
    - Check for duplicate: same IP and (active or is_authenticating)
    - If duplicate found: close new socket, log warning, return false
    - If no duplicate: create ClientConnection, add to clients_, return true
    - _Bug_Condition: hasPoorBusinessLogic(toctou_race) returns true_
    - _Expected_Behavior: Atomic duplicate check prevents race_
    - _Preservation: Duplicate detection behavior unchanged_
    - _Requirements: 2.16_

  - [x] 20.2 Refactor serverLoop() to use addClientIfNotDuplicate()
    - Replace separate check and insert with single call
    - Remove manual duplicate detection loop
    - Remove manual ClientConnection creation
    - Keep accept() and address parsing logic
    - _Bug_Condition: serverLoop() has TOCTOU race_
    - _Expected_Behavior: serverLoop() uses atomic check-and-insert_
    - _Preservation: Server accept behavior unchanged_
    - _Requirements: 2.16_

  - [x] 20.3 Add test for concurrent duplicate connections
    - Create test that simulates two threads accepting from same IP
    - Verify only one connection is added
    - Verify second connection is closed
    - _Bug_Condition: Race condition allows duplicates_
    - _Expected_Behavior: Atomic operation prevents duplicates_
    - _Preservation: Single connection per IP maintained_
    - _Requirements: 2.16_

- [x] 21. Add RAII wrappers for resource management

  - [x] 21.1 Add SocketGuard RAII class
    - Add class definition in tcp_cluster.cpp or utility header
    - Constructor: accept socket fd (default VGRE_INVALID_SOCKET)
    - Destructor: close socket if valid
    - Delete copy constructor and assignment
    - Implement move constructor and assignment
    - Methods: get(), release()
    - _Bug_Condition: hasPoorBusinessLogic(socket_leaks) returns true_
    - _Expected_Behavior: RAII ensures socket cleanup_
    - _Preservation: Socket lifecycle unchanged_
    - _Requirements: 2.17_

  - [x] 21.2 Use SocketGuard in udpAnnouncerLoop()
    - Wrap UDP socket creation with SocketGuard
    - Remove manual close() calls
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: UDP socket leaks on error paths_
    - _Expected_Behavior: SocketGuard prevents leaks_
    - _Preservation: UDP announcer behavior unchanged_
    - _Requirements: 2.17_

  - [x] 21.3 Use SocketGuard in udpMasterDiscoveryLoop()
    - Wrap UDP socket creation with SocketGuard
    - Remove manual close() calls
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: UDP socket leaks on error paths_
    - _Expected_Behavior: SocketGuard prevents leaks_
    - _Preservation: UDP discovery behavior unchanged_
    - _Requirements: 2.17_

  - [x] 21.4 Use SocketGuard in udpDiscoveryLoop()
    - Wrap UDP socket creation with SocketGuard
    - Remove manual close() calls
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: UDP socket leaks on error paths_
    - _Expected_Behavior: SocketGuard prevents leaks_
    - _Preservation: UDP discovery behavior unchanged_
    - _Requirements: 2.17_

  - [x] 21.5 Use SocketGuard in udpWorkerAnnouncerLoop()
    - Wrap UDP socket creation with SocketGuard
    - Remove manual close() calls
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: UDP socket leaks on error paths_
    - _Expected_Behavior: SocketGuard prevents leaks_
    - _Preservation: UDP worker announcer behavior unchanged_
    - _Requirements: 2.17_

  - [x] 21.6 Add JoinableThread RAII class
    - Add class definition in tcp_cluster.cpp or utility header
    - Constructor: accept callable and arguments
    - Destructor: join if joinable
    - Delete copy constructor and assignment
    - Implement move constructor and assignment
    - Methods: joinable(), join()
    - _Bug_Condition: hasPoorBusinessLogic(thread_leaks) returns true_
    - _Expected_Behavior: RAII ensures thread cleanup_
    - _Preservation: Thread lifecycle unchanged_
    - _Requirements: 2.18_

  - [x] 21.7 Replace std::thread with JoinableThread
    - Replace all std::thread members with JoinableThread
    - Remove manual join() calls in shutdown()
    - Verify automatic join on destruction
    - _Bug_Condition: Threads leak when shutdown called before join_
    - _Expected_Behavior: JoinableThread prevents leaks_
    - _Preservation: Thread behavior unchanged_
    - _Requirements: 2.18_

  - [x] 21.8 Add ShmSegment RAII class
    - Add class definition in tcp_cluster.cpp or utility header
    - Constructor: default
    - Destructor: close ShmManager if valid
    - Delete copy constructor and assignment
    - Implement move constructor and assignment
    - Methods: open(), getBasePtr(), getSize()
    - _Bug_Condition: hasPoorBusinessLogic(shm_leaks) returns true_
    - _Expected_Behavior: RAII ensures SHM cleanup_
    - _Preservation: SHM lifecycle unchanged_
    - _Requirements: 2.19_

  - [x] 21.9 Use ShmSegment for client SHM management
    - Replace raw ShmManager pointers with ShmSegment
    - Remove manual close() calls
    - Verify automatic cleanup on all exit paths
    - _Bug_Condition: SHM leaks on abnormal termination_
    - _Expected_Behavior: ShmSegment prevents leaks_
    - _Preservation: SHM behavior unchanged_
    - _Requirements: 2.19_


- [x] 22. Standardize error handling to VGREResult

  - [x] 22.1 Convert send_packet() to return VGREResult
    - Change return type from bool to VGREResult
    - Return ERR_INVALID_VALUE for invalid socket or payload size
    - Return ERR_IO for send failures
    - Return SUCCESS for successful send
    - _Bug_Condition: hasPoorBusinessLogic(inconsistent_errors) returns true_
    - _Expected_Behavior: Consistent VGREResult return type_
    - _Preservation: Error detection unchanged_
    - _Requirements: 2.15_

  - [x] 22.2 Convert send_packet_direct() to return VGREResult
    - Change return type from bool to VGREResult
    - Return ERR_INVALID_VALUE for invalid socket
    - Return ERR_IO for send failures
    - Return SUCCESS for successful send
    - _Bug_Condition: send_packet_direct() returns bool_
    - _Expected_Behavior: Consistent VGREResult return type_
    - _Preservation: Error detection unchanged_
    - _Requirements: 2.15_

  - [x] 22.3 Update all send_packet() call sites
    - Replace bool checks with VGREResult checks
    - Change `if (!send_packet(...))` to `if (send_packet(...) != VGREResult::SUCCESS)`
    - Propagate VGREResult instead of converting to bool
    - _Bug_Condition: Call sites expect bool_
    - _Expected_Behavior: Call sites use VGREResult_
    - _Preservation: Error handling behavior unchanged_
    - _Requirements: 2.15_

  - [x] 22.4 Update all send_packet_direct() call sites
    - Replace bool checks with VGREResult checks
    - Change `if (!send_packet_direct(...))` to `if (send_packet_direct(...) != VGREResult::SUCCESS)`
    - Propagate VGREResult instead of converting to bool
    - _Bug_Condition: Call sites expect bool_
    - _Expected_Behavior: Call sites use VGREResult_
    - _Preservation: Error handling behavior unchanged_
    - _Requirements: 2.15_

  - [x] 22.5 Update broadcastPacket() to return VGREResult
    - Change return type from bool to VGREResult
    - Return ERR_NOT_SUPPORTED if not master
    - Return last error encountered during broadcast
    - Update call sites if needed
    - _Bug_Condition: broadcastPacket() returns bool_
    - _Expected_Behavior: Consistent VGREResult return type_
    - _Preservation: Broadcast behavior unchanged_
    - _Requirements: 2.15_
    - Update call sites to check VGREResult
    - _Bug_Condition: Mixed return types for error-prone operations_
    - _Expected_Behavior: Consistent VGREResult usage_
    - _Preservation: Error detection unchanged_
    - _Requirements: 2.15_

- [x] 23. Replace busy-wait with blocking I/O

  - [x] 23.1 Add waitForData() helper method
    - Add private method to TCPClusterManager
    - Accept socket fd and timeout_ms
    - Use vgre_poll() to wait for POLLIN event
    - Return SUCCESS if data available, ERR_TIMEOUT if timeout, ERR_IO if error
    - _Bug_Condition: hasPoorBusinessLogic(busy_wait) returns true_
    - _Expected_Behavior: Blocking I/O with timeout_
    - _Preservation: Data availability detection unchanged_
    - _Requirements: 2.21_

  - [x] 23.2 Replace busy-wait in performSecureHandshake()
    - Replace sleep loop with waitForData() + recv()
    - Calculate remaining timeout for each poll
    - Remove manual sleep calls
    - _Bug_Condition: Handshake busy-waits consuming CPU_
    - _Expected_Behavior: Handshake blocks efficiently_
    - _Preservation: Handshake behavior unchanged_
    - _Requirements: 2.21_

  - [x] 23.3 Replace busy-wait in packet receive loops
    - Find all recv() loops with sleep
    - Replace with waitForData() + recv()
    - Remove manual sleep calls
    - _Bug_Condition: Packet receive busy-waits consuming CPU_
    - _Expected_Behavior: Packet receive blocks efficiently_
    - _Preservation: Packet receive behavior unchanged_
    - _Requirements: 2.21_

  - [x] 23.4 Add CPU usage test
    - Create test that measures CPU usage during idle wait
    - Verify CPU usage < 5% when waiting for data
    - Compare old busy-wait vs new blocking I/O
    - _Bug_Condition: High CPU usage during idle_
    - _Expected_Behavior: Low CPU usage with blocking I/O_
    - _Preservation: Functionality unchanged_
    - _Requirements: 2.21_

- [x] 24. Add exponential backoff for delta-sync retry

  - [x] 24.1 Add ExponentialBackoff class
    - Add class definition in tcp_cluster.cpp or utility header
    - Constructor: accept initial_ms, max_ms, multiplier (default 2.0)
    - Method next(): return current delay, update for next iteration
    - Method reset(): reset to initial delay
    - _Bug_Condition: hasPoorBusinessLogic(no_backoff) returns true_
    - _Expected_Behavior: Exponential backoff strategy_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.14, 2.22_

  - [x] 24.2 Add sendDeltaSyncWithRetry() method
    - Add private method to TCPClusterManager
    - Accept ptr, handle, dirty ranges, client connection
    - Create ExponentialBackoff with INITIAL_RETRY_BACKOFF_MS and MAX_RETRY_BACKOFF_MS
    - Loop up to MAX_DELTA_SYNC_RETRIES attempts
    - Call sendDeltaSync(), return SUCCESS if successful
    - If ERR_IO, return immediately (connection lost)
    - For transient failures, sleep with backoff and retry
    - After all retries exhausted, fall back to sendFullSync()
    - _Bug_Condition: hasPoorBusinessLogic(no_retry) returns true_
    - _Expected_Behavior: Retry with exponential backoff before full-sync fallback_
    - _Preservation: Delta-sync behavior enhanced_
    - _Requirements: 2.22_

  - [x] 24.3 Update syncPointerToWorker() to use sendDeltaSyncWithRetry()
    - Replace direct sendDeltaSync() call with sendDeltaSyncWithRetry()
    - Remove manual fallback to sendFullSync()
    - _Bug_Condition: No retry before fallback_
    - _Expected_Behavior: Automatic retry with backoff_
    - _Preservation: Sync behavior enhanced_
    - _Requirements: 2.22_


- [x] 25. Early authentication validation

  - [x] 25.1 Update handleRemoteCommand() to validate auth first
    - Move auth token check to top of function (before any processing)
    - Check auth_token_ == 0 || pkt.auth_token != auth_token_
    - If invalid: log error, clear pending_args_, return immediately
    - Add packet structure validation after auth check
    - _Bug_Condition: hasPoorBusinessLogic(late_auth_check) returns true_
    - _Expected_Behavior: Auth validated before processing_
    - _Preservation: Valid commands processed unchanged_
    - _Requirements: 2.23_

  - [x] 25.2 Update handlePartitionDispatch() to validate auth first
    - Move auth token check to top of function (before any processing)
    - Check auth_token_ == 0 || pkt.auth_token != auth_token_
    - If invalid: log error, clear pending_args_, return immediately
    - Add packet structure validation after auth check
    - _Bug_Condition: Auth checked after processing_
    - _Expected_Behavior: Auth validated before processing_
    - _Preservation: Valid dispatches processed unchanged_
    - _Requirements: 2.23_

  - [x] 25.3 Add security test for early validation
    - Create test that sends command with invalid auth token
    - Verify command is rejected before processing
    - Verify pending_args_ is cleared
    - Verify no kernel execution occurs
    - _Bug_Condition: Invalid auth processed before rejection_
    - _Expected_Behavior: Invalid auth rejected immediately_
    - _Preservation: Valid auth processed unchanged_
    - _Requirements: 2.23_

- [x] 26. Preserve buffer contents for diagnostics

  - [x] 26.1 Update protocol violation handling
    - Before clearing rx_buffer on protocol error, log diagnostic info
    - Log first 64 bytes as hex dump
    - Log magic, version, buffer size
    - Log client IP address
    - Then clear buffer
    - _Bug_Condition: hasPoorBusinessLogic(lost_diagnostics) returns true_
    - _Expected_Behavior: Diagnostic info preserved before cleanup_
    - _Preservation: Error handling behavior enhanced_
    - _Requirements: 2.20_

  - [x] 26.2 Add diagnostic logging helper
    - Add private method hexDump() to format bytes as hex string
    - Accept buffer and max bytes to dump
    - Return formatted string
    - Use in protocol violation logging
    - _Bug_Condition: No diagnostic formatting_
    - _Expected_Behavior: Formatted hex dumps for debugging_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.20_

- [x] 27. Verify bug condition exploration test now passes
  - **Property 1: Expected Behavior** - Proper Business Logic
  - **IMPORTANT**: Re-run the SAME test from task 17 - do NOT write a new test
  - Run bug condition exploration test from step 17
  - **EXPECTED OUTCOME**: Test PASSES (confirms all business logic fixes applied)
  - Verify named constants used throughout
  - Verify TOCTOU race fixed
  - Verify RAII wrappers prevent leaks
  - Verify consistent VGREResult usage
  - Verify blocking I/O instead of busy-wait
  - _Requirements: 2.9-2.24_

- [x] 28. Verify preservation tests still pass
  - **Property 2: Preservation** - Core Functionality
  - **IMPORTANT**: Re-run the SAME tests from task 18 - do NOT write new tests
  - Run preservation property tests from step 18
  - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
  - Verify connections still work
  - Verify packets still sent/received correctly
  - Verify errors still detected
  - Verify timeouts still work

- [x] 29. Checkpoint - Phase 4 Complete
  - Ensure all Phase 4 tests pass
  - Verify all magic numbers replaced with named constants
  - Verify TOCTOU race fixed with atomic check-and-insert
  - Verify RAII wrappers prevent all resource leaks
  - Verify consistent VGREResult error handling
  - Verify blocking I/O replaces busy-wait
  - Verify exponential backoff for retries
  - Verify early auth validation
  - Verify diagnostic info preserved
  - Verify no regressions in core functionality
  - Ask user if questions arise


## Phase 5: Refactor into Modular Architecture (Priority: MEDIUM)

- [~] 30. Write bug condition exploration test for monolithic architecture
  - **Property 1: Bug Condition** - Monolithic File Structure
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Detect monolithic 3,267-line file that should be modular
  - **Scoped PBT Approach**: Test that separate module files exist with focused responsibilities
  - Create test file `tests/advanced/test_tcp_cluster_architecture.cpp`
  - Test that ConnectionManager class exists in separate file
  - Test that PacketHandler class exists in separate file
  - Test that SecurityManager class exists in separate file
  - Test that DispatchManager class exists in separate file
  - Test that MemorySyncManager class exists in separate file
  - Test that CollectiveOpsManager class exists in separate file
  - Test that each module file is < 500 lines
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (modules don't exist - confirms monolithic bug)
  - Document current file size and lack of modularity
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 5.1_

- [~] 31. Write preservation property tests for all cluster operations (BEFORE implementing fix)
  - **Property 2: Preservation** - Complete Cluster Functionality
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for all cluster operations
  - Write comprehensive property-based tests capturing all observed behaviors
  - Test master initialization, worker connection, kernel dispatch, memory sync, collective ops
  - Test security handshake, UDP discovery, packet I/O, error handling
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1-3.25_

- [~] 32. Create module directory structure

  - [ ] 32.1 Create source directory structure
    - Create `src/advanced/tcp_cluster/` directory
    - Create placeholder files for each module
    - Update CMakeLists.txt to include new directory
    - _Bug_Condition: Single monolithic file_
    - _Expected_Behavior: Modular directory structure_
    - _Preservation: Build system updated_
    - _Requirements: 2.25_

  - [ ] 32.2 Create header directory structure
    - Create `include/vgre/advanced/tcp_cluster/` directory
    - Create `include/vgre/advanced/tcp_cluster/internal/` directory
    - Create placeholder headers for each module
    - _Bug_Condition: Single monolithic header_
    - _Expected_Behavior: Modular header structure_
    - _Preservation: Include paths updated_
    - _Requirements: 2.25_

- [~] 33. Extract ConnectionManager module

  - [ ] 33.1 Create ConnectionManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/connection_manager.h`
    - Define ConnectionManager class with TCPClusterManager* parent pointer
    - Declare methods: acceptConnection, connectToMaster, closeConnection
    - Declare methods: getClient, getActiveClients, addClientIfNotDuplicate, purgeDeadClients
    - Declare private members: clients_ vector, clients_mutex_
    - _Bug_Condition: Connection management embedded in monolithic class_
    - _Expected_Behavior: Separate ConnectionManager class_
    - _Preservation: Connection management interface unchanged_
    - _Requirements: 2.25_

  - [ ] 33.2 Implement ConnectionManager methods
    - Create `src/advanced/tcp_cluster/connection_manager.cpp`
    - Move acceptConnection logic from serverLoop()
    - Move connectToMaster logic from initialize()
    - Move closeConnection logic from shutdown()
    - Implement getClient, getActiveClients, addClientIfNotDuplicate, purgeDeadClients
    - Target < 400 lines
    - _Bug_Condition: Connection logic scattered in monolithic file_
    - _Expected_Behavior: Focused ConnectionManager implementation_
    - _Preservation: Connection behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 33.3 Update TCPClusterManager to use ConnectionManager
    - Add connection_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate connection operations to connection_manager_
    - Remove old connection management code
    - _Bug_Condition: TCPClusterManager handles connections directly_
    - _Expected_Behavior: TCPClusterManager delegates to ConnectionManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 33.4 Write unit tests for ConnectionManager
    - Create `tests/advanced/tcp_cluster/test_connection_manager.cpp`
    - Test addClientIfNotDuplicate with no duplicate
    - Test addClientIfNotDuplicate with duplicate (rejected)
    - Test purgeDeadClients removes inactive
    - Test getActiveClients returns only active
    - _Bug_Condition: No isolated connection manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Connection behavior verified_
    - _Requirements: 2.25_


- [-] 34. Extract PacketHandler module

  - [x] 34.1 Create PacketHandler class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/packet_handler.h`
    - Define PacketHandler class
    - Declare methods: constructPacket, sendPacket, sendPacketDirect, recvPacket, parseVSBPHeader
    - Declare private members: sequence_counter_
    - _Bug_Condition: Packet handling embedded in monolithic class_
    - _Expected_Behavior: Separate PacketHandler class_
    - _Preservation: Packet handling interface unchanged_
    - _Requirements: 2.25_

  - [ ] 34.2 Implement PacketHandler methods
    - Create `src/advanced/tcp_cluster/packet_handler.cpp`
    - Move constructPacket implementation
    - Move send_packet implementation (rename to sendPacket)
    - Move send_packet_direct implementation (rename to sendPacketDirect)
    - Move packet receive logic
    - Move VSBP header parsing
    - Target < 300 lines
    - _Bug_Condition: Packet logic scattered in monolithic file_
    - _Expected_Behavior: Focused PacketHandler implementation_
    - _Preservation: Packet behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 34.3 Update TCPClusterManager to use PacketHandler
    - Add packet_handler_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate packet operations to packet_handler_
    - Remove old packet handling code
    - _Bug_Condition: TCPClusterManager handles packets directly_
    - _Expected_Behavior: TCPClusterManager delegates to PacketHandler_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 34.4 Write unit tests for PacketHandler
    - Create `tests/advanced/tcp_cluster/test_packet_handler.cpp`
    - Test constructPacket creates valid VSBP header
    - Test sendPacket with socket error returns ERR_IO
    - Test sendPacket with success returns SUCCESS
    - Test parseVSBPHeader validates magic and version
    - _Bug_Condition: No isolated packet handler tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Packet behavior verified_
    - _Requirements: 2.25_

- [~] 35. Extract SecurityManager module

  - [ ] 35.1 Create SecurityManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/security_manager.h`
    - Define SecurityManager class with TCPClusterManager* parent pointer
    - Declare methods: enableSecurity, isSecurityEnabled, getSecurityInfo
    - Declare methods: performServerHandshake, performClientHandshake, rotateSessionKey
    - Declare private members: security_enabled_, auth_token_str_, auth_token_
    - _Bug_Condition: Security logic embedded in monolithic class_
    - _Expected_Behavior: Separate SecurityManager class_
    - _Preservation: Security interface unchanged_
    - _Requirements: 2.25_

  - [ ] 35.2 Implement SecurityManager methods
    - Create `src/advanced/tcp_cluster/security_manager.cpp`
    - Move performSecureHandshake implementation
    - Move security configuration logic
    - Move key rotation logic
    - Target < 300 lines
    - _Bug_Condition: Security logic scattered in monolithic file_
    - _Expected_Behavior: Focused SecurityManager implementation_
    - _Preservation: Security behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 35.3 Update TCPClusterManager to use SecurityManager
    - Add security_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate security operations to security_manager_
    - Remove old security code
    - _Bug_Condition: TCPClusterManager handles security directly_
    - _Expected_Behavior: TCPClusterManager delegates to SecurityManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 35.4 Write unit tests for SecurityManager
    - Create `tests/advanced/tcp_cluster/test_security_manager.cpp`
    - Test handshake success with valid credentials
    - Test handshake failure with invalid credentials
    - Test plaintext fallback prevention
    - Test key derivation
    - _Bug_Condition: No isolated security manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Security behavior verified_
    - _Requirements: 2.25_

- [~] 36. Extract DiscoveryManager module

  - [ ] 36.1 Create DiscoveryManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/discovery_manager.h`
    - Define DiscoveryManager class with TCPClusterManager* parent pointer
    - Declare methods: startMasterAnnouncer, startWorkerDiscovery, startProactiveConnections, stopAll
    - Declare methods: addProactiveAddress, removeProactiveAddress
    - Declare private members: announcer_thread_, discovery_thread_, proactive_thread_, stop_flag_
    - _Bug_Condition: Discovery logic embedded in monolithic class_
    - _Expected_Behavior: Separate DiscoveryManager class_
    - _Preservation: Discovery interface unchanged_
    - _Requirements: 2.25_

  - [ ] 36.2 Implement DiscoveryManager methods
    - Create `src/advanced/tcp_cluster/discovery_manager.cpp`
    - Move udpAnnouncerLoop implementation
    - Move udpMasterDiscoveryLoop implementation
    - Move udpDiscoveryLoop implementation
    - Move udpWorkerAnnouncerLoop implementation
    - Move proactiveConnectionLoop implementation
    - Target < 300 lines
    - _Bug_Condition: Discovery logic scattered in monolithic file_
    - _Expected_Behavior: Focused DiscoveryManager implementation_
    - _Preservation: Discovery behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 36.3 Update TCPClusterManager to use DiscoveryManager
    - Add discovery_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate discovery operations to discovery_manager_
    - Remove old discovery code
    - _Bug_Condition: TCPClusterManager handles discovery directly_
    - _Expected_Behavior: TCPClusterManager delegates to DiscoveryManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 36.4 Write unit tests for DiscoveryManager
    - Create `tests/advanced/tcp_cluster/test_discovery_manager.cpp`
    - Test UDP announcer sends broadcasts
    - Test UDP discovery receives announcements
    - Test proactive connection attempts
    - Test stop functionality
    - _Bug_Condition: No isolated discovery manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Discovery behavior verified_
    - _Requirements: 2.25_


- [~] 37. Extract DispatchManager module

  - [ ] 37.1 Create DispatchManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/dispatch_manager.h`
    - Define DispatchManager class with TCPClusterManager* parent pointer
    - Declare methods: launchRemoteKernel, launchPartitionedKernel, collectPartitionResults
    - Declare methods: broadcastKernelRegistration, waitForRemoteResult
    - Declare private members: remote_results_, results_mutex_, results_cv_
    - Declare private members: partition_results_, partition_mutex_, partition_cv_
    - _Bug_Condition: Dispatch logic embedded in monolithic class_
    - _Expected_Behavior: Separate DispatchManager class_
    - _Preservation: Dispatch interface unchanged_
    - _Requirements: 2.25_

  - [ ] 37.2 Implement DispatchManager methods
    - Create `src/advanced/tcp_cluster/dispatch_manager.cpp`
    - Move launchRemoteKernel implementation
    - Move launchPartitionedKernel implementation
    - Move collectPartitionResults implementation
    - Move broadcastKernelRegistration implementation
    - Move waitForRemoteResult implementation
    - Move handleRemoteCommand implementation
    - Move handlePartitionDispatch implementation
    - Target < 400 lines
    - _Bug_Condition: Dispatch logic scattered in monolithic file_
    - _Expected_Behavior: Focused DispatchManager implementation_
    - _Preservation: Dispatch behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 37.3 Update TCPClusterManager to use DispatchManager
    - Add dispatch_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate dispatch operations to dispatch_manager_
    - Remove old dispatch code
    - _Bug_Condition: TCPClusterManager handles dispatch directly_
    - _Expected_Behavior: TCPClusterManager delegates to DispatchManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 37.4 Write unit tests for DispatchManager
    - Create `tests/advanced/tcp_cluster/test_dispatch_manager.cpp`
    - Test launchRemoteKernel sends correct packets
    - Test launchPartitionedKernel distributes work
    - Test collectPartitionResults waits for all partitions
    - Test waitForRemoteResult with timeout
    - _Bug_Condition: No isolated dispatch manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Dispatch behavior verified_
    - _Requirements: 2.25_

- [~] 38. Extract MemorySyncManager module

  - [ ] 38.1 Create MemorySyncManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/memory_sync_manager.h`
    - Define MemorySyncManager class with TCPClusterManager* parent pointer
    - Declare methods: streamArgumentsToWorker, syncPointerToWorker, syncPointerFromWorker
    - Declare methods: initializeShmForClient
    - Declare private methods: sendDeltaSync, sendFullSync, sendScalarArg, sendPointerArg, sendStructArg
    - _Bug_Condition: Memory sync logic embedded in monolithic class_
    - _Expected_Behavior: Separate MemorySyncManager class_
    - _Preservation: Memory sync interface unchanged_
    - _Requirements: 2.25_

  - [ ] 38.2 Implement MemorySyncManager methods
    - Create `src/advanced/tcp_cluster/memory_sync_manager.cpp`
    - Move streamArgumentsToWorker implementation
    - Move syncPointerToWorker implementation
    - Move syncPointerFromWorker implementation
    - Move sendDeltaSync, sendDeltaSyncSHM, sendDeltaSyncTCP implementations
    - Move sendFullSync, sendFullSyncSHM, sendFullSyncTCP implementations
    - Move sendScalarArg, sendPointerArg, sendStructArg implementations
    - Target < 400 lines
    - _Bug_Condition: Memory sync logic scattered in monolithic file_
    - _Expected_Behavior: Focused MemorySyncManager implementation_
    - _Preservation: Memory sync behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 38.3 Update TCPClusterManager to use MemorySyncManager
    - Add memory_sync_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate memory sync operations to memory_sync_manager_
    - Remove old memory sync code
    - _Bug_Condition: TCPClusterManager handles memory sync directly_
    - _Expected_Behavior: TCPClusterManager delegates to MemorySyncManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 38.4 Write unit tests for MemorySyncManager
    - Create `tests/advanced/tcp_cluster/test_memory_sync_manager.cpp`
    - Test delta-sync vs full-sync decision
    - Test SHM vs TCP path selection
    - Test retry logic with exponential backoff
    - Test argument serialization for all types
    - _Bug_Condition: No isolated memory sync manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Memory sync behavior verified_
    - _Requirements: 2.25_

- [~] 39. Extract CollectiveOpsManager module

  - [ ] 39.1 Create CollectiveOpsManager class definition
    - Create `include/vgre/advanced/tcp_cluster/internal/collective_ops_manager.h`
    - Define CollectiveOpsManager class with TCPClusterManager* parent pointer
    - Declare methods: allReduce, barrier, sumReduce
    - Declare private methods: masterAllReduce, workerAllReduce
    - Declare private members: reduction_mutex_, reduction_cv_, reduction_count_
    - Declare private members: active_reduction_buffer_, is_reducing_, reduction_datatype_, reduction_element_count_
    - _Bug_Condition: Collective ops logic embedded in monolithic class_
    - _Expected_Behavior: Separate CollectiveOpsManager class_
    - _Preservation: Collective ops interface unchanged_
    - _Requirements: 2.25_

  - [ ] 39.2 Implement CollectiveOpsManager methods
    - Create `src/advanced/tcp_cluster/collective_ops_manager.cpp`
    - Move allReduce implementation (master and worker paths)
    - Move barrier implementation
    - Move sum_reduce_simd implementation
    - Target < 300 lines
    - _Bug_Condition: Collective ops logic scattered in monolithic file_
    - _Expected_Behavior: Focused CollectiveOpsManager implementation_
    - _Preservation: Collective ops behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 39.3 Update TCPClusterManager to use CollectiveOpsManager
    - Add collective_ops_manager_ member to TCPClusterManager
    - Initialize in constructor
    - Delegate collective operations to collective_ops_manager_
    - Remove old collective ops code
    - _Bug_Condition: TCPClusterManager handles collective ops directly_
    - _Expected_Behavior: TCPClusterManager delegates to CollectiveOpsManager_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 39.4 Write unit tests for CollectiveOpsManager
    - Create `tests/advanced/tcp_cluster/test_collective_ops_manager.cpp`
    - Test allReduce with different datatypes
    - Test SIMD reduction correctness
    - Test timeout handling
    - Test master/worker coordination
    - _Bug_Condition: No isolated collective ops manager tests_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: Collective ops behavior verified_
    - _Requirements: 2.25_


- [~] 40. Update TCPClusterManager to be thin coordinator

  - [ ] 40.1 Refactor TCPClusterManager class
    - Keep only public API methods
    - Add module member variables (connection_manager_, packet_handler_, etc.)
    - Initialize all modules in constructor
    - Delegate all operations to appropriate modules
    - Remove all implementation code (moved to modules)
    - Target < 500 lines for tcp_cluster_manager.cpp
    - _Bug_Condition: TCPClusterManager is monolithic_
    - _Expected_Behavior: TCPClusterManager is thin coordinator_
    - _Preservation: Public API unchanged_
    - _Requirements: 2.25_

  - [ ] 40.2 Update public API methods to delegate
    - launchRemoteKernel() delegates to dispatch_manager_
    - allReduce() delegates to collective_ops_manager_
    - barrier() delegates to collective_ops_manager_
    - getConnectedNodes() delegates to connection_manager_
    - enableSecurity() delegates to security_manager_
    - All other methods delegate appropriately
    - _Bug_Condition: Public methods contain implementation_
    - _Expected_Behavior: Public methods delegate to modules_
    - _Preservation: Public API behavior unchanged_
    - _Requirements: 2.25_

  - [ ] 40.3 Remove old monolithic tcp_cluster.cpp
    - Verify all code moved to modules
    - Delete old src/advanced/tcp_cluster.cpp
    - Update CMakeLists.txt to remove old file
    - _Bug_Condition: Old monolithic file still exists_
    - _Expected_Behavior: Only modular files remain_
    - _Preservation: Functionality preserved in modules_
    - _Requirements: 2.25_

- [~] 41. Verify bug condition exploration test now passes
  - **Property 1: Expected Behavior** - Modular Architecture
  - **IMPORTANT**: Re-run the SAME test from task 30 - do NOT write a new test
  - Run bug condition exploration test from step 30
  - **EXPECTED OUTCOME**: Test PASSES (confirms modular architecture exists)
  - Verify ConnectionManager exists in separate file < 500 lines
  - Verify PacketHandler exists in separate file < 500 lines
  - Verify SecurityManager exists in separate file < 500 lines
  - Verify DiscoveryManager exists in separate file < 500 lines
  - Verify DispatchManager exists in separate file < 500 lines
  - Verify MemorySyncManager exists in separate file < 500 lines
  - Verify CollectiveOpsManager exists in separate file < 500 lines
  - Verify TCPClusterManager is thin coordinator < 500 lines
  - _Requirements: 2.25_

- [~] 42. Verify preservation tests still pass
  - **Property 2: Preservation** - Complete Cluster Functionality
  - **IMPORTANT**: Re-run the SAME tests from task 31 - do NOT write new tests
  - Run preservation property tests from step 31
  - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
  - Verify all cluster operations still work after refactoring
  - Verify master initialization, worker connection, kernel dispatch unchanged
  - Verify memory sync, collective ops, security, discovery unchanged

- [~] 43. Checkpoint - Phase 5 Complete
  - Ensure all Phase 5 tests pass
  - Verify monolithic file split into 8 focused modules
  - Verify each module < 500 lines
  - Verify TCPClusterManager is thin coordinator
  - Verify all unit tests pass for each module
  - Verify no regressions in cluster functionality
  - Ask user if questions arise


## Phase 6: Make Code Testable (Priority: MEDIUM)

- [~] 44. Write bug condition exploration test for testability issues
  - **Property 1: Bug Condition** - Untestable Design
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Detect singleton pattern, lack of interfaces, and global state preventing testing
  - **Scoped PBT Approach**: Test that interface abstractions exist and dependency injection is supported
  - Create test file `tests/advanced/test_tcp_cluster_testability.cpp`
  - Test that ISocketFactory interface exists
  - Test that IMemoryManager interface exists
  - Test that ISecureChannelFactory interface exists
  - Test that TCPClusterManager constructor accepts injected dependencies
  - Test that mock implementations can be created
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (interfaces don't exist - confirms untestable bug)
  - Document testability issues
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 6.1, 6.2, 6.3_

- [~] 45. Write preservation property tests for cluster operations (BEFORE implementing fix)
  - **Property 2: Preservation** - Cluster Operations with Real Dependencies
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code with real dependencies
  - Write property-based tests capturing observed behavior patterns
  - Test all cluster operations work with real sockets, memory manager, security
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1-3.25_

- [~] 46. Define interface abstractions

  - [ ] 46.1 Create interfaces.h header
    - Create `include/vgre/advanced/tcp_cluster/internal/interfaces.h`
    - Add header guards and namespace declarations
    - _Bug_Condition: No interface abstractions_
    - _Expected_Behavior: Interfaces header exists_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.30, 2.31_

  - [ ] 46.2 Define ISocketFactory interface
    - Add pure virtual interface for socket operations
    - Methods: createSocket, bind, listen, accept, connect, send, recv, poll, close
    - All methods return appropriate types (vgre_socket_t, int, etc.)
    - _Bug_Condition: Socket operations not abstracted_
    - _Expected_Behavior: ISocketFactory interface for testing_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.31_

  - [ ] 46.3 Define RealSocketFactory implementation
    - Implement ISocketFactory using real system calls
    - createSocket calls socket()
    - bind calls ::bind()
    - listen calls ::listen()
    - accept calls ::accept()
    - connect calls ::connect()
    - send calls ::send()
    - recv calls ::recv()
    - poll calls vgre_poll()
    - close calls vgre_close_socket()
    - _Bug_Condition: No real socket factory_
    - _Expected_Behavior: RealSocketFactory for production use_
    - _Preservation: Socket behavior unchanged_
    - _Requirements: 2.31_

  - [ ] 46.4 Define IMemoryManager interface
    - Add pure virtual interface for memory operations
    - Methods: getAllocationSize, getPointer, isValidHandle, allocateManagedAt
    - Methods: getDirtyPages, clearDirtyPages
    - _Bug_Condition: Memory operations not abstracted_
    - _Expected_Behavior: IMemoryManager interface for testing_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.31_

  - [ ] 46.5 Define RealMemoryManager implementation
    - Implement IMemoryManager delegating to RuntimeEngine
    - getAllocationSize delegates to RuntimeEngine::getMemoryManager()
    - getPointer delegates to RuntimeEngine::getMemoryManager()
    - All methods delegate appropriately
    - _Bug_Condition: No real memory manager wrapper_
    - _Expected_Behavior: RealMemoryManager for production use_
    - _Preservation: Memory behavior unchanged_
    - _Requirements: 2.31_

  - [ ] 46.6 Define ISecureChannelFactory interface
    - Add pure virtual interface for secure channel creation
    - Method: create() returns unique_ptr<SecureChannel>
    - _Bug_Condition: Secure channel creation not abstracted_
    - _Expected_Behavior: ISecureChannelFactory interface for testing_
    - _Preservation: No impact on existing code_
    - _Requirements: 2.31_

  - [ ] 46.7 Define RealSecureChannelFactory implementation
    - Implement ISecureChannelFactory
    - create() returns std::make_unique<SecureChannel>()
    - _Bug_Condition: No real secure channel factory_
    - _Expected_Behavior: RealSecureChannelFactory for production use_
    - _Preservation: Security behavior unchanged_
    - _Requirements: 2.31_


- [~] 47. Add dependency injection to TCPClusterManager

  - [ ] 47.1 Add constructor with dependency injection
    - Add constructor accepting unique_ptr<ISocketFactory>, unique_ptr<IMemoryManager>, unique_ptr<ISecureChannelFactory>
    - Store as private members: socket_factory_, memory_manager_, security_factory_
    - Pass to modules that need them
    - _Bug_Condition: TCPClusterManager uses singleton pattern_
    - _Expected_Behavior: Constructor accepts injected dependencies_
    - _Preservation: Existing singleton still available_
    - _Requirements: 2.30_

  - [ ] 47.2 Update singleton instance() to use real implementations
    - Keep static instance() method for production use
    - Create with RealSocketFactory, RealMemoryManager, RealSecureChannelFactory
    - _Bug_Condition: Singleton prevents testing_
    - _Expected_Behavior: Singleton uses real implementations, tests use injected mocks_
    - _Preservation: Production code unchanged_
    - _Requirements: 2.30_

  - [ ] 47.3 Update modules to use injected dependencies
    - Pass socket_factory_ to ConnectionManager, PacketHandler, DiscoveryManager
    - Pass memory_manager_ to MemorySyncManager
    - Pass security_factory_ to SecurityManager
    - Remove direct calls to system functions
    - Use factory methods instead
    - _Bug_Condition: Modules use global functions_
    - _Expected_Behavior: Modules use injected dependencies_
    - _Preservation: Module behavior unchanged_
    - _Requirements: 2.30, 2.31_

  - [ ] 47.4 Eliminate global state
    - Move all static variables to instance members
    - Remove static instance() from modules (if any)
    - Ensure all state is instance-specific
    - _Bug_Condition: Global state prevents parallel testing_
    - _Expected_Behavior: All state is instance-specific_
    - _Preservation: Single-instance behavior unchanged_
    - _Requirements: 2.32_

- [~] 48. Create mock implementations

  - [ ] 48.1 Create mocks.h header
    - Create `tests/advanced/tcp_cluster/mocks.h`
    - Add header guards and namespace declarations
    - Include gmock/gmock.h
    - _Bug_Condition: No mock implementations_
    - _Expected_Behavior: Mocks header exists_
    - _Preservation: No impact on production code_
    - _Requirements: 2.31_

  - [ ] 48.2 Define MockSocketFactory
    - Implement ISocketFactory using Google Mock
    - MOCK_METHOD for all interface methods
    - Allow test to configure behavior
    - _Bug_Condition: Cannot mock socket operations_
    - _Expected_Behavior: MockSocketFactory for unit tests_
    - _Preservation: No impact on production code_
    - _Requirements: 2.31_

  - [ ] 48.3 Define MockMemoryManager
    - Implement IMemoryManager using Google Mock
    - MOCK_METHOD for all interface methods
    - Allow test to configure behavior
    - _Bug_Condition: Cannot mock memory operations_
    - _Expected_Behavior: MockMemoryManager for unit tests_
    - _Preservation: No impact on production code_
    - _Requirements: 2.31_

  - [ ] 48.4 Define MockSecureChannelFactory
    - Implement ISecureChannelFactory using Google Mock
    - MOCK_METHOD for create()
    - Allow test to return mock SecureChannel
    - _Bug_Condition: Cannot mock secure channel creation_
    - _Expected_Behavior: MockSecureChannelFactory for unit tests_
    - _Preservation: No impact on production code_
    - _Requirements: 2.31_

  - [ ] 48.5 Define FakeSocketFactory for integration tests
    - Implement ISocketFactory with in-memory socket simulation
    - Track connected sockets, send/recv buffers
    - Provide test helpers: injectRecvData(), getSentData()
    - _Bug_Condition: Integration tests need real sockets_
    - _Expected_Behavior: FakeSocketFactory for integration tests_
    - _Preservation: No impact on production code_
    - _Requirements: 2.31_

- [~] 49. Write comprehensive unit tests

  - [ ] 49.1 Write PacketHandler unit tests with mocks
    - Test constructPacket with valid/invalid inputs
    - Test sendPacket with socket errors (mock returns -1)
    - Test sendPacket with success (mock returns bytes sent)
    - Test parseVSBPHeader with valid/invalid headers
    - Use MockSocketFactory to control behavior
    - _Bug_Condition: PacketHandler not unit tested_
    - _Expected_Behavior: Comprehensive unit tests with mocks_
    - _Preservation: PacketHandler behavior verified_
    - _Requirements: 2.30, 2.31_

  - [ ] 49.2 Write ConnectionManager unit tests with fakes
    - Test addClientIfNotDuplicate with no duplicate
    - Test addClientIfNotDuplicate with duplicate (rejected)
    - Test purgeDeadClients removes inactive
    - Test getActiveClients returns only active
    - Use FakeSocketFactory for integration-style tests
    - _Bug_Condition: ConnectionManager not unit tested_
    - _Expected_Behavior: Comprehensive unit tests with fakes_
    - _Preservation: ConnectionManager behavior verified_
    - _Requirements: 2.30, 2.31_

  - [ ] 49.3 Write SecurityManager unit tests with mocks
    - Test handshake success with valid credentials
    - Test handshake failure with invalid credentials
    - Test plaintext fallback prevention
    - Test key derivation
    - Use MockSocketFactory to simulate handshake
    - _Bug_Condition: SecurityManager not unit tested_
    - _Expected_Behavior: Comprehensive unit tests with mocks_
    - _Preservation: SecurityManager behavior verified_
    - _Requirements: 2.30, 2.31_

  - [ ] 49.4 Write MemorySyncManager unit tests with mocks
    - Test delta-sync vs full-sync decision
    - Test SHM vs TCP path selection
    - Test retry logic with exponential backoff
    - Test argument serialization for all types
    - Use MockMemoryManager to control dirty pages
    - _Bug_Condition: MemorySyncManager not unit tested_
    - _Expected_Behavior: Comprehensive unit tests with mocks_
    - _Preservation: MemorySyncManager behavior verified_
    - _Requirements: 2.30, 2.31_

  - [ ] 49.5 Write CollectiveOpsManager unit tests
    - Test allReduce with different datatypes
    - Test SIMD reduction correctness
    - Test timeout handling
    - Test master/worker coordination
    - _Bug_Condition: CollectiveOpsManager not unit tested_
    - _Expected_Behavior: Comprehensive unit tests_
    - _Preservation: CollectiveOpsManager behavior verified_
    - _Requirements: 2.30_


- [~] 50. Write integration tests

  - [ ] 50.1 Create test_integration.cpp
    - Create `tests/advanced/tcp_cluster/test_integration.cpp`
    - Use FakeSocketFactory for in-memory communication
    - Test master-worker connection
    - Test kernel dispatch end-to-end
    - Test collective operations
    - _Bug_Condition: No integration tests_
    - _Expected_Behavior: Integration tests verify module interactions_
    - _Preservation: Cluster behavior verified_
    - _Requirements: 2.30_

  - [ ] 50.2 Test parallel instances without interference
    - Create two separate TCPClusterManager instances
    - Initialize both in parallel threads
    - Verify no interference between instances
    - Verify no shared global state
    - _Bug_Condition: Global state causes interference_
    - _Expected_Behavior: Parallel instances work independently_
    - _Preservation: Single-instance behavior unchanged_
    - _Requirements: 2.32_

  - [ ] 50.3 Test master-worker connection with fakes
    - Create master instance with FakeSocketFactory
    - Create worker instance with FakeSocketFactory
    - Simulate connection establishment
    - Verify connection successful
    - Verify nodes list updated
    - _Bug_Condition: Connection not integration tested_
    - _Expected_Behavior: Connection works with fake sockets_
    - _Preservation: Connection behavior verified_
    - _Requirements: 2.30, 2.31_

  - [ ] 50.4 Test kernel dispatch with fakes
    - Create master and worker with FakeSocketFactory
    - Register kernel on both
    - Launch kernel from master to worker
    - Verify packets sent correctly
    - Verify result returned
    - _Bug_Condition: Dispatch not integration tested_
    - _Expected_Behavior: Dispatch works with fake sockets_
    - _Preservation: Dispatch behavior verified_
    - _Requirements: 2.30, 2.31_

- [~] 51. Write system tests with real sockets

  - [ ] 51.1 Create test_system.cpp
    - Create `tests/advanced/tcp_cluster/test_system.cpp`
    - Use real sockets (localhost)
    - Test full cluster lifecycle
    - Test security end-to-end
    - Test performance characteristics
    - _Bug_Condition: No system tests_
    - _Expected_Behavior: System tests verify real-world behavior_
    - _Preservation: Cluster behavior verified_
    - _Requirements: 2.30_

  - [ ] 51.2 Test full cluster lifecycle
    - Initialize master on localhost:7777
    - Initialize worker connecting to localhost:7777
    - Wait for connection
    - Register and execute kernel
    - Shutdown gracefully
    - _Bug_Condition: Lifecycle not system tested_
    - _Expected_Behavior: Full lifecycle works with real sockets_
    - _Preservation: Lifecycle behavior verified_
    - _Requirements: 2.30_

  - [ ] 51.3 Test security end-to-end
    - Enable security with auth token
    - Perform handshake
    - Verify encrypted communication
    - Test key rotation
    - Test auth token validation
    - _Bug_Condition: Security not system tested_
    - _Expected_Behavior: Security works end-to-end_
    - _Preservation: Security behavior verified_
    - _Requirements: 2.30_

  - [ ] 51.4 Test performance characteristics
    - Measure throughput (packets/sec, MB/sec)
    - Measure latency (kernel dispatch round-trip)
    - Measure memory overhead
    - Measure CPU utilization
    - Verify performance meets requirements
    - _Bug_Condition: Performance not measured_
    - _Expected_Behavior: Performance characteristics documented_
    - _Preservation: Performance maintained_
    - _Requirements: 2.30_

- [~] 52. Verify bug condition exploration test now passes
  - **Property 1: Expected Behavior** - Testable Design
  - **IMPORTANT**: Re-run the SAME test from task 44 - do NOT write a new test
  - Run bug condition exploration test from step 44
  - **EXPECTED OUTCOME**: Test PASSES (confirms testable design exists)
  - Verify ISocketFactory interface exists
  - Verify IMemoryManager interface exists
  - Verify ISecureChannelFactory interface exists
  - Verify TCPClusterManager accepts injected dependencies
  - Verify mock implementations can be created
  - _Requirements: 2.30, 2.31, 2.32_

- [~] 53. Verify preservation tests still pass
  - **Property 2: Preservation** - Cluster Operations
  - **IMPORTANT**: Re-run the SAME tests from task 45 - do NOT write new tests
  - Run preservation property tests from step 45
  - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
  - Verify all cluster operations work with real dependencies
  - Verify production behavior unchanged

- [~] 54. Checkpoint - Phase 6 Complete
  - Ensure all Phase 6 tests pass
  - Verify interface abstractions defined
  - Verify dependency injection implemented
  - Verify mock implementations created
  - Verify comprehensive unit tests written (80%+ coverage)
  - Verify integration tests written
  - Verify system tests written
  - Verify parallel test execution works
  - Verify no regressions in cluster functionality
  - Ask user if questions arise


## Phase 7: Documentation and Cleanup (Priority: LOW)

- [~] 55. Update API documentation

  - [ ] 55.1 Document TCPClusterManager public API
    - Add Doxygen comments to all public methods
    - Document parameters, return values, exceptions
    - Add usage examples
    - Document thread safety
    - _Bug_Condition: API not documented_
    - _Expected_Behavior: Complete API documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 55.2 Document module interfaces
    - Add Doxygen comments to all module classes
    - Document responsibilities and interactions
    - Add usage examples
    - Document thread safety
    - _Bug_Condition: Modules not documented_
    - _Expected_Behavior: Complete module documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 55.3 Document interface abstractions
    - Add Doxygen comments to ISocketFactory, IMemoryManager, ISecureChannelFactory
    - Document contract and expectations
    - Add implementation examples
    - _Bug_Condition: Interfaces not documented_
    - _Expected_Behavior: Complete interface documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

- [~] 56. Add inline comments

  - [ ] 56.1 Add comments to complex algorithms
    - Document SIMD reduction logic
    - Document delta-sync decision logic
    - Document exponential backoff strategy
    - Document TSS2 priority queuing
    - _Bug_Condition: Complex code not commented_
    - _Expected_Behavior: Clear inline comments_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 56.2 Add comments to critical sections
    - Document locking strategy
    - Document TOCTOU fix
    - Document RAII resource management
    - Document security policy enforcement
    - _Bug_Condition: Critical sections not commented_
    - _Expected_Behavior: Clear inline comments_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 56.3 Add comments to packet handling
    - Document VSBP protocol format
    - Document packet types and payloads
    - Document encryption flow
    - Document priority assignment
    - _Bug_Condition: Packet handling not commented_
    - _Expected_Behavior: Clear inline comments_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

- [~] 57. Create architecture documentation

  - [ ] 57.1 Create architecture diagram
    - Create diagram showing module relationships
    - Show data flow between modules
    - Show dependency injection points
    - Show interface abstractions
    - Save as `docs/tcp_cluster_architecture.md`
    - _Bug_Condition: No architecture diagram_
    - _Expected_Behavior: Visual architecture documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 57.2 Document design decisions
    - Document why modular architecture chosen
    - Document why SIMD optimization used
    - Document why exponential backoff chosen
    - Document why RAII pattern used
    - Save as `docs/tcp_cluster_design_decisions.md`
    - _Bug_Condition: Design decisions not documented_
    - _Expected_Behavior: Design rationale documented_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 57.3 Document testing strategy
    - Document unit test approach
    - Document integration test approach
    - Document system test approach
    - Document mock/fake usage
    - Save as `docs/tcp_cluster_testing.md`
    - _Bug_Condition: Testing strategy not documented_
    - _Expected_Behavior: Testing approach documented_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

- [~] 58. Write migration guide

  - [ ] 58.1 Document breaking changes
    - List any API changes (if any)
    - Document migration steps
    - Provide code examples
    - _Bug_Condition: No migration guide_
    - _Expected_Behavior: Clear migration documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 58.2 Document new features
    - Document allReduce() usage
    - Document reportComputeFromWorker() usage
    - Document improved error handling
    - Document testability improvements
    - _Bug_Condition: New features not documented_
    - _Expected_Behavior: Feature documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

  - [ ] 58.3 Document performance improvements
    - Document SIMD speedup (4-8x)
    - Document blocking I/O CPU savings
    - Document delta-sync retry benefits
    - Provide benchmarks
    - _Bug_Condition: Performance improvements not documented_
    - _Expected_Behavior: Performance documentation_
    - _Preservation: No code changes_
    - _Requirements: Documentation_

- [~] 59. Remove deprecated code

  - [ ] 59.1 Search for TODO/FIXME comments
    - Find all TODO comments
    - Resolve or remove each one
    - Find all FIXME comments
    - Resolve or remove each one
    - _Bug_Condition: Unresolved TODOs_
    - _Expected_Behavior: All TODOs resolved_
    - _Preservation: Code cleanup_
    - _Requirements: Cleanup_

  - [ ] 59.2 Remove commented-out code
    - Find all commented-out code blocks
    - Remove if no longer needed
    - Document if kept for reference
    - _Bug_Condition: Dead code in comments_
    - _Expected_Behavior: Clean codebase_
    - _Preservation: Code cleanup_
    - _Requirements: Cleanup_

  - [ ] 59.3 Remove unused variables and functions
    - Run static analysis to find unused code
    - Remove unused private methods
    - Remove unused variables
    - _Bug_Condition: Unused code_
    - _Expected_Behavior: Clean codebase_
    - _Preservation: Code cleanup_
    - _Requirements: Cleanup_

- [~] 60. Final code review

  - [ ] 60.1 Review all module implementations
    - Check for code quality
    - Check for consistency
    - Check for best practices
    - Check for security issues
    - _Bug_Condition: Code not reviewed_
    - _Expected_Behavior: High-quality code_
    - _Preservation: Code quality_
    - _Requirements: Quality_

  - [ ] 60.2 Review all tests
    - Check for test coverage
    - Check for test quality
    - Check for edge cases
    - Check for flaky tests
    - _Bug_Condition: Tests not reviewed_
    - _Expected_Behavior: High-quality tests_
    - _Preservation: Test quality_
    - _Requirements: Quality_

  - [ ] 60.3 Review all documentation
    - Check for completeness
    - Check for accuracy
    - Check for clarity
    - Check for examples
    - _Bug_Condition: Documentation not reviewed_
    - _Expected_Behavior: High-quality documentation_
    - _Preservation: Documentation quality_
    - _Requirements: Quality_

- [~] 61. Checkpoint - Phase 7 Complete
  - Ensure all documentation complete
  - Verify API documentation comprehensive
  - Verify inline comments clear
  - Verify architecture documented
  - Verify migration guide complete
  - Verify deprecated code removed
  - Verify final code review complete
  - Ask user if questions arise

## Final Checkpoint - All Phases Complete

- [~] 62. Verify all requirements satisfied
  - Review bugfix.md requirements 2.1-2.32
  - Verify each requirement has corresponding implementation
  - Verify each requirement has corresponding tests
  - Verify preservation requirements 3.1-3.25 maintained

- [~] 63. Verify all tests pass
  - Run all unit tests
  - Run all integration tests
  - Run all system tests
  - Verify 80%+ code coverage
  - Verify no flaky tests

- [~] 64. Verify performance requirements met
  - Run performance benchmarks
  - Verify SIMD provides 4-8x speedup
  - Verify blocking I/O reduces CPU usage
  - Verify no performance regressions

- [~] 65. Verify production readiness
  - Verify no linker errors
  - Verify no memory leaks (valgrind)
  - Verify no race conditions (thread sanitizer)
  - Verify no undefined behavior (UBSan)
  - Verify security policy enforced

- [~] 66. Final sign-off
  - All 7 phases complete
  - All tests passing
  - All documentation complete
  - Production ready
  - Request user approval for deployment

