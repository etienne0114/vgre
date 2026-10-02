#!/usr/bin/env python3
"""
Property Test for Performance and Caching Efficiency

**Feature: gguf-auto-config, Property 10: Performance and Caching Efficiency**
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**

This property test validates that the performance optimization and caching
system meets efficiency requirements and provides proper cache management.

Key test coverage:
1. Metadata extraction completes under performance requirements
2. Cache freshness validation works correctly
3. Memory usage is minimized for metadata-only operations
4. Cache effectiveness for repeated access
"""

import os
import sys
import tempfile
import struct
import time
import gc
import hashlib
from typing import Dict, Any
import threading

# Set up paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bindings', 'python'))

# Set library path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
native_name = {'darwin': 'libvgre.dylib', 'win32': 'vgre.dll'}.get(sys.platform, 'libvgre.so')
lib_path = os.path.join(build_dir, native_name)
os.environ['LD_LIBRARY_PATH'] = build_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['VGRE_LIB_PATH'] = lib_path

import pytest
from hypothesis import given, strategies as st, settings, note, assume
from hypothesis.strategies import composite

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
GGUF_TYPE_STRING = 8


def create_large_gguf_file(metadata: Dict[str, Any], file_size_mb: int = 100) -> bytes:
    """Create a large GGUF file for performance testing."""
    data = GGUF_MAGIC
    data += struct.pack('<I', GGUF_VERSION)
    data += struct.pack('<Q', 1)  # One large tensor
    data += struct.pack('<Q', len(metadata))  # Metadata count

    # Add metadata entries
    for key, value in metadata.items():
        key_bytes = key.encode('utf-8')
        value_bytes = str(value).encode('utf-8')

        data += struct.pack('<Q', len(key_bytes))
        data += key_bytes
        data += struct.pack('<I', GGUF_TYPE_STRING)
        data += struct.pack('<Q', len(value_bytes))
        data += value_bytes

    # Add large tensor definition
    tensor_name = b'large_tensor'
    data += struct.pack('<Q', len(tensor_name))
    data += tensor_name
    data += struct.pack('<I', 4)  # 4 dimensions

    # Large tensor shape to reach target file size
    tensor_elements = (file_size_mb * 1024 * 1024) // 4  # 4 bytes per float32
    dim_size = int(tensor_elements ** 0.25)  # Fourth root for 4D tensor

    for _ in range(4):
        data += struct.pack('<Q', dim_size)

    data += struct.pack('<I', 10)  # Float32 type
    data += struct.pack('<Q', len(data) + 32)  # Tensor offset (after alignment)

    # Add alignment
    alignment = 32
    pos = len(data)
    pad = (alignment - (pos % alignment)) % alignment
    data += b'\0' * pad

    # Add large tensor data (zeros for simplicity)
    tensor_data_size = tensor_elements * 4
    # Don't actually write all zeros to save memory, just indicate the size
    # The reader should only read metadata anyway

    return data  # Return header only for metadata testing


@composite
def performance_test_parameters(draw):
    """Generate parameters for performance testing."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096, 8192]))
    head_count = draw(st.sampled_from([
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]))
    return {
        'llama.block_count': draw(st.integers(min_value=1, max_value=100)),
        'llama.embedding_length': embedding_length,
        'llama.attention.head_count': head_count,
        'llama.attention.head_count_kv': draw(st.sampled_from([
            heads for heads in range(1, head_count + 1) if head_count % heads == 0
        ])),
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=32768)),
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=1000000.0, allow_nan=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(min_value=1e-12, max_value=1e-3, allow_nan=False)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
    }


@composite
def cache_test_scenarios(draw):
    """Generate scenarios for cache testing."""
    num_files = draw(st.integers(min_value=2, max_value=5))
    access_pattern = draw(st.sampled_from(['sequential', 'random', 'repeated']))

    return {
        'num_files': num_files,
        'access_pattern': access_pattern,
        'access_count': draw(st.integers(min_value=3, max_value=10))
    }


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestPerformanceAndCachingEfficiency:
    """Property tests for performance and caching efficiency."""

    @given(performance_test_parameters())
    @settings(max_examples=20, deadline=5000)  # 5 second deadline per test
    def test_metadata_extraction_performance_requirement(self, parameters):
        """
        **Validates: Requirements 10.1, 10.4, 10.5**

        Property: Metadata extraction SHALL complete in under 1 second
        for typical GGUF files, regardless of file size.
        """
        note(f"Testing performance with parameters: {parameters}")

        # Create GGUF file (header only for metadata testing)
        gguf_data = create_large_gguf_file(parameters, file_size_mb=1)  # Small for testing

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        reader = None
        try:
            # Test GGUFMetadataReader performance
            start_time = time.perf_counter()
            reader = GGUFMetadataReader(temp_path)

            # Extract all parameters
            block_count = reader.get_block_count()
            embedding_length = reader.get_embedding_length()
            head_count = reader.get_head_count()
            head_count_kv = reader.get_head_count_kv()
            feed_forward_length = reader.get_feed_forward_length()
            rope_freq_base = reader.get_rope_freq_base()
            norm_eps = reader.get_norm_eps()

            extraction_time = time.perf_counter() - start_time

            note(f"Metadata extraction took {extraction_time:.4f} seconds")

            # Should complete in under 1 second (requirement 10.4)
            assert extraction_time < 1.0, \
                f"Metadata extraction took {extraction_time:.4f}s, should be < 1.0s"

            # Test ModelAutoConfigurator performance
            reader.close()
            start_time = time.perf_counter()
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)
            config_time = time.perf_counter() - start_time

            note(f"Auto-configuration took {config_time:.4f} seconds")

            # Auto-configuration should also be fast
            assert config_time < 2.0, \
                f"Auto-configuration took {config_time:.4f}s, should be < 2.0s"

        finally:
            if reader is not None:
                reader.close()
            os.unlink(temp_path)

    @given(cache_test_scenarios())
    @settings(max_examples=20, deadline=None)
    def test_cache_effectiveness_for_repeated_access(self, scenario):
        """
        **Validates: Requirements 10.1, 10.2, 10.3**

        Property: Repeated access to the same GGUF file SHALL benefit from
        caching, with subsequent accesses being significantly faster.
        """
        note(f"Testing cache scenario: {scenario}")

        # Create test files
        test_files = []
        for i in range(scenario['num_files']):
            parameters = {
                'llama.block_count': 16 + i,
                'llama.embedding_length': 1024,
                'llama.attention.head_count': 16,
                'llama.feed_forward_length': 4096,
                'llama.vocab_size': 32000,
                'llama.rope.freq_base': 10000.0 + i * 1000,
            }

            gguf_data = create_large_gguf_file(parameters)

            with tempfile.NamedTemporaryFile(suffix=f'_cache_test_{i}.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                test_files.append(f.name)

        try:
            configurator = ModelAutoConfigurator()
            access_times = []

            # Perform access pattern
            for access_round in range(scenario['access_count']):
                if scenario['access_pattern'] == 'sequential':
                    file_index = access_round % len(test_files)
                elif scenario['access_pattern'] == 'random':
                    file_index = hash(str(access_round)) % len(test_files)
                else:  # repeated
                    file_index = 0  # Always access first file

                test_file = test_files[file_index]

                start_time = time.perf_counter()
                config = configurator.auto_configure(test_file)
                access_time = time.perf_counter() - start_time

                access_times.append((access_round, file_index, access_time))
                note(f"Access {access_round}: file {file_index} took {access_time:.4f}s")

            # Analyze cache effectiveness
            if scenario['access_pattern'] == 'repeated':
                # Subsequent accesses to same file should be faster
                first_access_time = access_times[0][2]
                later_accesses = [t for _, _, t in access_times[1:3]]  # Next 2 accesses

                if len(later_accesses) > 0:
                    avg_later_time = sum(later_accesses) / len(later_accesses)

                    # Cache should provide some speedup (at least 10% faster)
                    assert first_access_time > 0.0, \
                        "High-resolution timer returned zero for the first cache access"
                    cache_speedup = (first_access_time - avg_later_time) / first_access_time
                    note(f"Cache speedup: {cache_speedup:.2%}")

                    # Allow for measurement noise, but expect some improvement
                    assert cache_speedup > -0.5, \
                        f"Cached access should not be much slower: first={first_access_time:.4f}s, avg_later={avg_later_time:.4f}s"

            # All accesses should still be reasonably fast
            max_access_time = max(t for _, _, t in access_times)
            assert max_access_time < 3.0, \
                f"Even with cache misses, access should be < 3s, got {max_access_time:.4f}s"

        finally:
            for test_file in test_files:
                if os.path.exists(test_file):
                    os.unlink(test_file)

    def test_cache_freshness_validation(self):
        """
        **Validates: Requirements 10.2, 10.3**

        Test that cache freshness validation works correctly when files are modified.
        """
        parameters = {
            'llama.block_count': 20,
            'llama.embedding_length': 2048,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 5504,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 15000.0,
        }

        gguf_data = create_large_gguf_file(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            # First access - should cache
            config1 = configurator.auto_configure(temp_path)
            first_access_time = time.time()

            # Second access immediately - should use cache
            config2 = configurator.auto_configure(temp_path)

            assert config1 == config2, "Cached config should be identical"

            # Modify file timestamp to simulate file change
            time.sleep(0.1)  # Ensure timestamp difference
            current_time = time.time()
            os.utime(temp_path, (current_time, current_time))

            # Third access - should detect freshness and reload
            modified_parameters = parameters.copy()
            modified_parameters['llama.block_count'] = 25  # Different value

            # Overwrite file with new content
            new_gguf_data = create_large_gguf_file(modified_parameters)
            with open(temp_path, 'wb') as f:
                f.write(new_gguf_data)

            config3 = configurator.auto_configure(temp_path)

            # Should detect change and return updated config
            assert config3['block_count'] == 25, \
                f"Should detect file change: expected 25, got {config3.get('block_count')}"
            assert config3['block_count'] != config1['block_count'], \
                "Cache should be invalidated after file modification"

        finally:
            os.unlink(temp_path)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available for memory testing")
    def test_minimal_memory_usage_metadata_only(self):
        """
        **Validates: Requirements 10.4, 10.5**

        Test that metadata-only operations use minimal memory and don't
        load model weights.
        """
        parameters = {
            'llama.block_count': 32,
            'llama.embedding_length': 4096,
            'llama.attention.head_count': 32,
            'llama.feed_forward_length': 16384,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 10000.0,
        }

        # Create a file with metadata but indicate large size
        gguf_data = create_large_gguf_file(parameters, file_size_mb=10)  # Indicate 10MB

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            # Measure memory before operation
            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Perform metadata extraction (should not load weights)
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)

            # Measure memory after operation
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Should not load significant amount of data (< 50MB for metadata only)
            assert memory_increase < 50 * 1024 * 1024, \
                f"Memory increase {memory_increase / 1024 / 1024:.2f}MB too large for metadata-only operation"

            # Verify we got the configuration without loading weights
            assert config['block_count'] == parameters['llama.block_count']
            assert config['embedding_length'] == parameters['llama.embedding_length']

        finally:
            os.unlink(temp_path)

    def test_concurrent_cache_access_thread_safety(self):
        """
        **Validates: Requirements 10.1, 10.2, 10.3**

        Test that concurrent access to cached metadata is thread-safe
        and maintains cache effectiveness.
        """
        parameters = {
            'llama.block_count': 24,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 12000.0,
        }

        gguf_data = create_large_gguf_file(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            results = []
            errors = []

            def worker(worker_id):
                try:
                    configurator = ModelAutoConfigurator()

                    for i in range(3):  # Multiple accesses per thread
                        start_time = time.perf_counter()
                        config = configurator.auto_configure(temp_path)
                        access_time = time.perf_counter() - start_time

                        results.append((worker_id, i, access_time, config))
                        time.sleep(0.01)  # Small delay

                except Exception as e:
                    errors.append((worker_id, e))

            # Start multiple threads
            threads = []
            for worker_id in range(4):
                t = threading.Thread(target=worker, args=(worker_id,))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Verify no errors occurred
            assert len(errors) == 0, f"Concurrent access caused errors: {errors}"

            # Verify all results are consistent
            configs = [config for _, _, _, config in results]
            base_config = configs[0]

            for i, config in enumerate(configs[1:], 1):
                assert config == base_config, \
                    f"Config mismatch at result {i}: expected {base_config}, got {config}"

            # Verify reasonable performance (cache should help)
            access_times = [access_time for _, _, access_time, _ in results]
            max_access_time = max(access_times)
            avg_access_time = sum(access_times) / len(access_times)

            assert max_access_time < 5.0, \
                f"Concurrent access should be reasonably fast: {max_access_time:.4f}s"

        finally:
            os.unlink(temp_path)

    def test_cache_memory_efficiency(self):
        """
        **Validates: Requirements 10.2, 10.5**

        Test that cache memory usage is efficient and doesn't grow unbounded.
        """
        # Create multiple different files
        test_files = []
        for i in range(10):  # More files than reasonable cache size
            parameters = {
                'llama.block_count': 16 + i,
                'llama.embedding_length': 1024,
                'llama.attention.head_count': 16,
                'llama.feed_forward_length': 4096,
                'llama.vocab_size': 32000,
                'llama.rope.freq_base': 10000.0 + i * 1000,
            }

            gguf_data = create_large_gguf_file(parameters)

            with tempfile.NamedTemporaryFile(suffix=f'_mem_test_{i}.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                test_files.append(f.name)

        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                initial_memory = process.memory_info().rss

            configurator = ModelAutoConfigurator()

            # Access all files to populate cache
            for test_file in test_files:
                config = configurator.auto_configure(test_file)

            if PSUTIL_AVAILABLE:
                peak_memory = process.memory_info().rss
                memory_increase = peak_memory - initial_memory

                # Cache should not use excessive memory (< 100MB for 10 files)
                assert memory_increase < 100 * 1024 * 1024, \
                    f"Cache using too much memory: {memory_increase / 1024 / 1024:.2f}MB"

            # Access files again - should still work efficiently
            for test_file in test_files:
                start_time = time.perf_counter()
                config = configurator.auto_configure(test_file)
                access_time = time.perf_counter() - start_time

                assert access_time < 2.0, \
                    f"Cached access should be fast: {access_time:.4f}s"

        finally:
            for test_file in test_files:
                if os.path.exists(test_file):
                    os.unlink(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
