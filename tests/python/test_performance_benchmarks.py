#!/usr/bin/env python3
"""
Performance Benchmarks for GGUF Auto-Configuration

**Feature: gguf-auto-config, Task 7.3**
**Validates: Requirements 10.4, 10.5**

This benchmark suite validates specific performance requirements
for the GGUF auto-configuration system.

Key benchmarks:
1. Metadata extraction completes under 1 second for typical files
2. Cache effectiveness for repeated access
3. Minimal memory usage for metadata-only operations
4. Scalability with metadata complexity
"""

import os
import sys
import tempfile
import struct
import time
import gc
import statistics
from typing import List, Dict, Any, Tuple

# Set up paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bindings', 'python'))

# Set library path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
native_name = {'darwin': 'libvgre.dylib', 'win32': 'vgre.dll'}.get(sys.platform, 'libvgre.so')
lib_path = os.path.join(build_dir, native_name)
os.environ['LD_LIBRARY_PATH'] = build_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['VGRE_LIB_PATH'] = lib_path

import pytest

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import vgre
    from vgre import GGUFMetadataReader
    from vgre.auto_config import ModelAutoConfigurator
    VGRE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGRE not available: {e}")
    VGRE_AVAILABLE = False


# GGUF file format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8


def create_benchmark_gguf_file(metadata: Dict[str, Any]) -> bytes:
    """Create a valid metadata-only GGUF file for parser benchmarks."""
    metadata = {
        'llama.block_count': 1,
        'llama.embedding_length': 64,
        'llama.attention.head_count': 1,
        'llama.feed_forward_length': 128,
        'llama.vocab_size': 256,
        **metadata,
    }
    data = bytearray(struct.pack('<4sIQQ', GGUF_MAGIC, GGUF_VERSION, 0, len(metadata)))
    for key, value in metadata.items():
        key_bytes = key.encode('utf-8')
        data.extend(struct.pack('<Q', len(key_bytes)))
        data.extend(key_bytes)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            data.extend(struct.pack('<II', GGUF_TYPE_UINT32, value))
        elif isinstance(value, float):
            data.extend(struct.pack('<If', GGUF_TYPE_FLOAT32, value))
        else:
            value_bytes = str(value).encode('utf-8')
            data.extend(struct.pack('<I', GGUF_TYPE_STRING))
            data.extend(struct.pack('<Q', len(value_bytes)))
            data.extend(value_bytes)
    data.extend(b'\0' * ((32 - len(data) % 32) % 32))
    return bytes(data)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.memory_usage: List[int] = []
        self.success_count = 0
        self.failure_count = 0

    def add_timing(self, elapsed_time: float, memory_bytes: int = 0, success: bool = True):
        """Add a timing measurement."""
        self.times.append(elapsed_time)
        self.memory_usage.append(memory_bytes)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of benchmark results."""
        if not self.times:
            return {'count': 0}

        return {
            'count': len(self.times),
            'min_time': min(self.times),
            'max_time': max(self.times),
            'mean_time': statistics.mean(self.times),
            'median_time': statistics.median(self.times),
            'stddev_time': statistics.stdev(self.times) if len(self.times) > 1 else 0,
            'p95_time': statistics.quantiles(self.times, n=20)[18] if len(self.times) >= 20 else max(self.times),
            'success_rate': self.success_count / (self.success_count + self.failure_count),
            'avg_memory_mb': statistics.mean(self.memory_usage) / (1024 * 1024) if self.memory_usage else 0,
        }


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestPerformanceBenchmarks:
    """Performance benchmarks for GGUF auto-configuration."""

    def test_metadata_extraction_speed_requirement(self):
        """
        **Validates: Requirement 10.4**

        Benchmark: Metadata extraction MUST complete in under 1 second
        for typical GGUF files.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Metadata Extraction Speed")
        print("="*60)

        # Test increasing metadata complexity; these fixtures contain no weights.
        test_configs = [
            {'name': 'small_metadata', 'metadata_count': 10},
            {'name': 'medium_metadata', 'metadata_count': 20},
            {'name': 'large_metadata', 'metadata_count': 50},
            {'name': 'xlarge_metadata', 'metadata_count': 100},
        ]

        results = {}

        for config in test_configs:
            print(f"\nTesting {config['name']}:")
            print(f"  Metadata entries: {config['metadata_count']}")

            # Generate metadata
            metadata = {}
            for i in range(config['metadata_count']):
                if i < 10:
                    # Standard parameters
                    metadata.update({
                        'llama.block_count': 32,
                        'llama.embedding_length': 4096,
                        'llama.attention.head_count': 32,
                        'llama.attention.head_count_kv': 8,
                        'llama.feed_forward_length': 16384,
                        'llama.rope.freq_base': 10000.0,
                        'llama.attention.layer_norm_epsilon': 1e-5,
                    })
                else:
                    # Additional metadata
                    metadata[f'custom.param_{i}'] = f'value_{i}'

            gguf_data = create_benchmark_gguf_file(metadata)

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                benchmark = BenchmarkResult(config['name'])

                # Run multiple iterations
                for iteration in range(10):
                    gc.collect()  # Clean up before measurement

                    if PSUTIL_AVAILABLE:
                        process = psutil.Process()
                        initial_memory = process.memory_info().rss

                    start_time = time.perf_counter()
                    reader = None

                    try:
                        reader = GGUFMetadataReader(temp_path)

                        # Extract key parameters
                        block_count = reader.get_block_count()
                        embedding_length = reader.get_embedding_length()
                        head_count = reader.get_head_count()
                        rope_freq_base = reader.get_rope_freq_base()

                        success = True
                    except Exception as e:
                        print(f"    Iteration {iteration} failed: {e}")
                        success = False
                    finally:
                        if reader is not None:
                            reader.close()

                    end_time = time.perf_counter()
                    elapsed = end_time - start_time

                    if PSUTIL_AVAILABLE:
                        final_memory = process.memory_info().rss
                        memory_used = final_memory - initial_memory
                    else:
                        memory_used = 0

                    benchmark.add_timing(elapsed, memory_used, success)

                results[config['name']] = benchmark
                stats = benchmark.get_stats()

                print(f"  Results (10 iterations):")
                print(f"    Mean time: {stats['mean_time']:.4f}s")
                print(f"    Median time: {stats['median_time']:.4f}s")
                print(f"    P95 time: {stats['p95_time']:.4f}s")
                print(f"    Max time: {stats['max_time']:.4f}s")
                print(f"    Success rate: {stats['success_rate']:.1%}")
                if PSUTIL_AVAILABLE:
                    print(f"    Avg memory: {stats['avg_memory_mb']:.2f} MB")

                # Validate requirement: < 1 second
                assert stats['p95_time'] < 1.0, \
                    f"{config['name']}: P95 time {stats['p95_time']:.4f}s exceeds 1.0s requirement"

                print(f"    ✅ Meets 1.0s requirement")

            finally:
                os.unlink(temp_path)

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY: Metadata Extraction Speed")
        print(f"{'='*60}")
        for name, benchmark in results.items():
            stats = benchmark.get_stats()
            status = "✅ PASS" if stats['p95_time'] < 1.0 else "❌ FAIL"
            print(f"{name:15} P95: {stats['p95_time']:.4f}s {status}")

    def test_cache_performance_benchmark(self):
        """
        **Validates: Requirements 10.1, 10.2, 10.3**

        Benchmark: Cache effectiveness for repeated access patterns.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Cache Performance")
        print("="*60)

        # Create test file
        metadata = {
            'llama.block_count': 32,
            'llama.embedding_length': 4096,
            'llama.attention.head_count': 32,
            'llama.rope.freq_base': 10000.0,
        }

        gguf_data = create_benchmark_gguf_file(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            # Benchmark cold access (no cache)
            print("\nCold access (first time):")
            cold_times = []
            for i in range(5):
                # The cache is class-wide, so a new instance alone is not cold.
                ModelAutoConfigurator._metadata_cache.pop(temp_path, None)
                fresh_configurator = ModelAutoConfigurator()

                start_time = time.perf_counter()
                config = fresh_configurator.auto_configure(temp_path)
                end_time = time.perf_counter()

                cold_time = end_time - start_time
                cold_times.append(cold_time)
                print(f"  Run {i+1}: {cold_time:.4f}s")

            avg_cold_time = statistics.mean(cold_times)
            print(f"  Average cold time: {avg_cold_time:.4f}s")

            # Benchmark warm access (with cache)
            print("\nWarm access (cached):")
            warm_times = []
            for i in range(20):
                start_time = time.perf_counter()
                config = configurator.auto_configure(temp_path)
                end_time = time.perf_counter()

                warm_time = end_time - start_time
                warm_times.append(warm_time)
                if i < 5:  # Show first 5
                    print(f"  Run {i+1}: {warm_time:.4f}s")

            avg_warm_time = statistics.mean(warm_times)
            print(f"  Average warm time (20 runs): {avg_warm_time:.4f}s")

            # Calculate cache effectiveness
            cache_speedup = (avg_cold_time - avg_warm_time) / avg_cold_time
            print(f"\nCache Analysis:")
            print(f"  Cache speedup: {cache_speedup:.1%}")
            print(f"  Speedup factor: {avg_cold_time / avg_warm_time:.1f}x")

            # Validate cache provides benefit (at least not slower)
            assert avg_warm_time <= avg_cold_time * 1.1, \
                f"Cached access should not be significantly slower: cold={avg_cold_time:.4f}s, warm={avg_warm_time:.4f}s"

            print(f"  ✅ Cache does not degrade performance")

            # Both should still meet performance requirements
            assert avg_cold_time < 2.0, f"Cold access should be < 2.0s: {avg_cold_time:.4f}s"
            assert avg_warm_time < 1.5, f"Warm access should be < 1.5s: {avg_warm_time:.4f}s"

            print(f"  ✅ Both cold and warm access meet requirements")

        finally:
            ModelAutoConfigurator._metadata_cache.pop(temp_path, None)
            os.unlink(temp_path)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil required for memory benchmarks")
    def test_memory_usage_benchmark(self):
        """
        **Validates: Requirement 10.5**

        Benchmark: Memory usage for metadata-only operations should be minimal.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Memory Usage")
        print("="*60)

        process = psutil.Process()
        baseline_memory = process.memory_info().rss

        print(f"Baseline memory: {baseline_memory / 1024 / 1024:.2f} MB")

        # Compare parser allocations for valid metadata-only files of increasing size.
        metadata_counts = [10, 100, 500, 1000]

        for metadata_count in metadata_counts:
            print(f"\n{metadata_count} metadata entries:")

            metadata = {
                'llama.block_count': 32,
                'llama.embedding_length': 4096,
                'llama.attention.head_count': 32,
            }

            for index in range(len(metadata), metadata_count):
                metadata[f'custom.param_{index}'] = f'value_{index}'
            gguf_data = create_benchmark_gguf_file(metadata)

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                # Measure memory usage during metadata extraction
                memory_samples = []

                for iteration in range(5):
                    gc.collect()
                    pre_memory = process.memory_info().rss

                    # Perform metadata extraction
                    configurator = ModelAutoConfigurator()
                    config = configurator.auto_configure(temp_path)

                    post_memory = process.memory_info().rss
                    memory_increase = post_memory - pre_memory
                    memory_samples.append(memory_increase)

                avg_memory_increase = statistics.mean(memory_samples)
                max_memory_increase = max(memory_samples)

                print(f"  Average memory increase: {avg_memory_increase / 1024 / 1024:.2f} MB")
                print(f"  Peak memory increase: {max_memory_increase / 1024 / 1024:.2f} MB")

                # Metadata-only operations should use less than 100 MB.
                max_allowed_mb = 100

                assert max_memory_increase < max_allowed_mb * 1024 * 1024, \
                    f"{metadata_count} entries: Memory usage {max_memory_increase / 1024 / 1024:.2f}MB exceeds {max_allowed_mb}MB limit"

                print(f"  ✅ Memory usage within {max_allowed_mb}MB limit")

            finally:
                os.unlink(temp_path)

    def test_scalability_benchmark(self):
        """
        **Validates: Requirements 10.4, 10.5**

        Benchmark: Performance scalability with increasing complexity.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Scalability")
        print("="*60)

        # Test increasing metadata complexity in valid metadata-only files.
        metadata_counts = [10, 50, 100, 200, 500]
        scalability_results = []

        for metadata_count in metadata_counts:
            print(f"\nMetadata entries: {metadata_count}")

            # Generate metadata
            metadata = {
                'llama.block_count': 32,
                'llama.embedding_length': 4096,
                'llama.attention.head_count': 32,
            }

            # Add extra metadata entries
            for i in range(3, metadata_count):
                metadata[f'custom.param_{i}'] = f'value_{i}'

            gguf_data = create_benchmark_gguf_file(metadata)

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                times = []

                for iteration in range(10):
                    configurator = ModelAutoConfigurator()

                    start_time = time.perf_counter()
                    config = configurator.auto_configure(temp_path)
                    end_time = time.perf_counter()

                    times.append(end_time - start_time)

                avg_time = statistics.mean(times)
                scalability_results.append((metadata_count, avg_time))

                print(f"  Average time: {avg_time:.4f}s")

                # Should still meet performance requirement
                assert avg_time < 2.0, \
                    f"Performance degraded with {metadata_count} metadata entries: {avg_time:.4f}s"

                print(f"  ✅ Performance acceptable")

            finally:
                os.unlink(temp_path)

        # Analyze scalability
        print(f"\nScalability Analysis:")
        print(f"{'Metadata Count':<15} {'Time (s)':<10} {'Time/Entry (ms)':<15}")
        print(f"{'-'*40}")

        for metadata_count, avg_time in scalability_results:
            time_per_entry = (avg_time * 1000) / metadata_count
            print(f"{metadata_count:<15} {avg_time:<10.4f} {time_per_entry:<15.4f}")

        # Check that performance doesn't degrade drastically
        first_time = scalability_results[0][1]
        last_time = scalability_results[-1][1]
        scalability_ratio = last_time / first_time

        print(f"\nScalability ratio (last/first): {scalability_ratio:.2f}x")

        # Performance should scale reasonably (not more than 10x degradation)
        assert scalability_ratio < 10.0, \
            f"Performance degraded too much with scale: {scalability_ratio:.2f}x"

        print(f"✅ Scalability within acceptable bounds")

    def test_reader_and_auto_configuration_performance_benchmark(self):
        """
        **Validates: Requirements 10.4, 10.5**

        Benchmark: Metadata reading and model auto-configuration performance.
        """
        print("\n" + "="*60)
        print("BENCHMARK: Metadata and Auto-Configuration Performance")
        print("="*60)

        # Benchmark realistic dimensions without claiming the metadata fixtures contain weights.
        model_configs = [
            {
                'name': 'small_llama',
                'metadata': {
                    'llama.block_count': 12,
                    'llama.embedding_length': 768,
                    'llama.attention.head_count': 12,
                    'llama.attention.head_count_kv': 12,
                    'llama.feed_forward_length': 3072,
                    'llama.rope.freq_base': 10000.0,
                    'llama.attention.layer_norm_epsilon': 1e-5,
                }
            },
            {
                'name': 'medium_llama',
                'metadata': {
                    'llama.block_count': 24,
                    'llama.embedding_length': 2048,
                    'llama.attention.head_count': 16,
                    'llama.attention.head_count_kv': 8,
                    'llama.feed_forward_length': 8192,
                    'llama.rope.freq_base': 10000.0,
                    'llama.attention.layer_norm_epsilon': 1e-6,
                }
            },
            {
                'name': 'large_llama',
                'metadata': {
                    'llama.block_count': 48,
                    'llama.embedding_length': 4096,
                    'llama.attention.head_count': 32,
                    'llama.attention.head_count_kv': 8,
                    'llama.feed_forward_length': 16384,
                    'llama.rope.freq_base': 10000.0,
                    'llama.attention.layer_norm_epsilon': 1e-6,
                }
            }
        ]

        for config in model_configs:
            print(f"\n{config['name']}:")

            gguf_data = create_benchmark_gguf_file(config['metadata'])

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                # Benchmark complete workflow
                times = {
                    'metadata_reader': [],
                    'auto_configurator': [],
                }

                for iteration in range(5):
                    # 1. Direct metadata reader
                    start_time = time.perf_counter()
                    with GGUFMetadataReader(temp_path) as reader:
                        reader.get_block_count()
                    times['metadata_reader'].append(time.perf_counter() - start_time)

                    # 2. Auto-configurator
                    start_time = time.perf_counter()
                    configurator = ModelAutoConfigurator()
                    config = configurator.auto_configure(temp_path)
                    times['auto_configurator'].append(time.perf_counter() - start_time)

                # Report results
                for component, time_list in times.items():
                    avg_time = statistics.mean(time_list)
                    print(f"  {component:<20}: {avg_time:.4f}s (avg of 5 runs)")

                    # All components should meet reasonable performance requirements
                    if component == 'metadata_reader':
                        assert avg_time < 0.5, f"Metadata reader too slow: {avg_time:.4f}s"
                    elif component == 'auto_configurator':
                        assert avg_time < 1.0, f"Auto-configurator too slow: {avg_time:.4f}s"

                print(f"  ✅ All components meet performance requirements")

            finally:
                os.unlink(temp_path)

        print(f"\n{'='*60}")
        print("✅ ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output
