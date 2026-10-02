#!/usr/bin/env python3
"""
Property Test for Auto-Configuration Determinism

**Feature: gguf-auto-config, Property 5: Auto-Configuration Determinism**
**Validates: Requirements 4.1, 4.2, 4.4, 4.5**

This property test validates that the auto-configuration system produces
deterministic and consistent results for the same input files and parameters.

Key test coverage:
1. Same GGUF file produces identical configuration across multiple runs
2. Parameter extraction is deterministic regardless of system state
3. Default parameter application is consistent
4. Configuration merging (GGUF + defaults) is deterministic
"""

import os
import sys
import tempfile
import struct
import hashlib
from typing import Dict, Any

# Set up paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bindings', 'python'))

# Set library path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
native_name = {'darwin': 'libvgre.dylib', 'win32': 'vgre.dll'}.get(sys.platform, 'libvgre.so')
lib_path = os.path.join(build_dir, native_name)
os.environ['LD_LIBRARY_PATH'] = build_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['VGRE_LIB_PATH'] = lib_path

import pytest
from hypothesis import given, strategies as st, settings, note
from hypothesis.strategies import composite

try:
    import vgre
    from vgre.auto_config import ModelAutoConfigurator
    from vgre.lm import LanguageModel
    VGRE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGRE not available: {e}")
    VGRE_AVAILABLE = False


# GGUF file format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8
FLOAT32_EPS_MIN = struct.unpack('<f', struct.pack('<f', 1e-12))[0]
FLOAT32_EPS_MAX = struct.unpack('<f', struct.pack('<f', 1e-3))[0]


def create_gguf_metadata_entry(key: str, value) -> bytes:
    """Create a single GGUF metadata entry."""
    key_bytes = key.encode('utf-8')
    key_len = len(key_bytes)

    entry = struct.pack('<Q', key_len)  # Key length
    entry += key_bytes  # Key string
    entry += struct.pack('<I', GGUF_TYPE_STRING)  # Type (string)

    value_bytes = str(value).encode('utf-8')
    entry += struct.pack('<Q', len(value_bytes))
    entry += value_bytes

    return entry


def create_gguf_file_with_metadata(metadata: Dict[str, Any]) -> bytes:
    """Create a complete GGUF file with specified metadata."""
    data = GGUF_MAGIC
    data += struct.pack('<I', GGUF_VERSION)
    data += struct.pack('<Q', 0)  # Tensor count
    data += struct.pack('<Q', len(metadata))  # Metadata count

    for key, value in metadata.items():
        data += create_gguf_metadata_entry(key, value)

    # Add alignment
    alignment = 32
    pos = len(data)
    pad = (alignment - (pos % alignment)) % alignment
    data += b'\0' * pad

    return data


@composite
def valid_model_parameters(draw):
    """Generate valid model parameters for testing determinism."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096]))
    valid_heads = [
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]
    head_count = draw(st.sampled_from(valid_heads))
    kv_heads = draw(st.sampled_from([heads for heads in range(1, head_count + 1)
                                     if head_count % heads == 0]))
    return {
        'llama.block_count': draw(st.integers(min_value=1, max_value=100)),
        'llama.embedding_length': embedding_length,
        'llama.attention.head_count': head_count,
        'llama.attention.head_count_kv': kv_heads,
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=16384)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=100000.0,
                                               width=32, allow_nan=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(min_value=FLOAT32_EPS_MIN, max_value=FLOAT32_EPS_MAX,
                                                             width=32, allow_nan=False)),
    }


@composite
def partial_model_parameters(draw):
    """Generate partial model parameters (some missing) for default testing."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096]))
    valid_heads = [
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]
    head_count = draw(st.sampled_from(valid_heads))
    kv_heads = draw(st.sampled_from([heads for heads in range(1, head_count + 1)
                                     if head_count % heads == 0]))
    all_params = {
        'llama.block_count': draw(st.integers(min_value=1, max_value=100)),
        'llama.embedding_length': embedding_length,
        'llama.attention.head_count': head_count,
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=16384)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
    }

    # Randomly omit some optional parameters
    optional_params = {
        'llama.attention.head_count_kv': kv_heads,
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=16384)),
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=100000.0,
                                               width=32, allow_nan=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(min_value=FLOAT32_EPS_MIN, max_value=FLOAT32_EPS_MAX,
                                                             width=32, allow_nan=False)),
    }

    # Include some but not all optional parameters
    include_keys = draw(st.lists(st.sampled_from(list(optional_params.keys())),
                                min_size=0, max_size=len(optional_params), unique=True))

    for key in include_keys:
        all_params[key] = optional_params[key]

    return all_params


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestAutoConfigurationDeterminism:
    """Property tests for auto-configuration determinism."""

    @given(valid_model_parameters())
    @settings(max_examples=100, deadline=None)
    def test_identical_file_produces_identical_config(self, parameters):
        """
        **Validates: Requirements 4.1, 4.2, 4.4, 4.5**

        Property: For any valid GGUF file, the auto-configuration system SHALL
        produce identical configuration parameters across multiple independent runs.
        """
        note(f"Testing with parameters: {parameters}")

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configs = []

            # Run auto-configuration multiple times
            for run in range(5):
                configurator = ModelAutoConfigurator()
                config = configurator.auto_configure(temp_path)
                configs.append(config)
                note(f"Run {run}: {config}")

            # All configurations should be identical
            base_config = configs[0]
            for i, config in enumerate(configs[1:], 1):
                assert config == base_config, \
                    f"Configuration mismatch at run {i}: expected {base_config}, got {config}"

            # Verify that the configuration contains expected values
            for key, expected_value in parameters.items():
                if 'block_count' in key:
                    assert config['block_count'] == expected_value
                elif 'embedding_length' in key:
                    assert config['embedding_length'] == expected_value
                elif 'head_count_kv' in key:
                    assert config['head_count_kv'] == expected_value
                elif 'head_count' in key and 'head_count_kv' not in key:
                    assert config['head_count'] == expected_value
                elif 'feed_forward_length' in key:
                    assert config['feed_forward_length'] == expected_value
                elif 'rope.freq_base' in key:
                    assert config['rope_theta'] == expected_value
                elif 'layer_norm_epsilon' in key:
                    assert config['norm_eps'] == expected_value

        finally:
            os.unlink(temp_path)

    @given(partial_model_parameters())
    @settings(max_examples=50, deadline=None)
    def test_deterministic_default_application(self, partial_parameters):
        """
        **Validates: Requirements 4.1, 4.2, 4.4, 4.5**

        Property: For any GGUF file with missing parameters, the auto-configuration
        system SHALL apply defaults deterministically and consistently.
        """
        note(f"Testing with partial parameters: {partial_parameters}")

        gguf_data = create_gguf_file_with_metadata(partial_parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configs = []

            # Run auto-configuration multiple times
            for run in range(3):
                configurator = ModelAutoConfigurator()
                config = configurator.auto_configure(temp_path)
                configs.append(config)
                note(f"Run {run}: {config}")

            # All configurations should be identical
            base_config = configs[0]
            for i, config in enumerate(configs[1:], 1):
                assert config == base_config, \
                    f"Default application not deterministic at run {i}: expected {base_config}, got {config}"

            # Verify defaults were applied consistently
            expected_defaults = {
                'rope_theta': 10000.0,
                'norm_eps': 1e-5,
            }

            for param, default_value in expected_defaults.items():
                corresponding_gguf_key = None
                if param == 'rope_theta':
                    corresponding_gguf_key = 'llama.rope.freq_base'
                elif param == 'norm_eps':
                    corresponding_gguf_key = 'llama.attention.layer_norm_epsilon'

                if corresponding_gguf_key not in partial_parameters:
                    # Should have default value
                    assert config.get(param) == default_value, \
                        f"Default {param} should be {default_value}, got {config.get(param)}"

        finally:
            os.unlink(temp_path)

    @given(valid_model_parameters())
    @settings(max_examples=50, deadline=None)
    def test_configuration_hash_consistency(self, parameters):
        """
        **Validates: Requirements 4.1, 4.2**

        Property: For any given GGUF file, the configuration hash SHALL be
        consistent across multiple runs, indicating deterministic processing.
        """
        note(f"Testing hash consistency with: {parameters}")

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            hashes = []

            # Generate configuration hashes multiple times
            for run in range(3):
                configurator = ModelAutoConfigurator()
                config = configurator.auto_configure(temp_path)

                # Create deterministic hash of configuration
                config_str = str(sorted(config.items()))
                config_hash = hashlib.sha256(config_str.encode()).hexdigest()
                hashes.append(config_hash)
                note(f"Run {run} hash: {config_hash}")

            # All hashes should be identical
            base_hash = hashes[0]
            for i, hash_value in enumerate(hashes[1:], 1):
                assert hash_value == base_hash, \
                    f"Configuration hash not consistent at run {i}: expected {base_hash}, got {hash_value}"

        finally:
            os.unlink(temp_path)

    def test_order_independence(self):
        """
        **Validates: Requirements 4.1, 4.2**

        Property: The order of metadata entries in the GGUF file SHALL NOT
        affect the final configuration result.
        """
        base_metadata = {
            'llama.block_count': 32,
            'llama.embedding_length': 4096,
            'llama.attention.head_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 10000.0,
        }

        # Create the same metadata in different orders
        metadata_orders = [
            dict(list(base_metadata.items())),  # Original order
            dict(list(reversed(base_metadata.items()))),  # Reversed
            dict(sorted(base_metadata.items())),  # Alphabetically sorted
        ]

        configs = []
        temp_files = []

        try:
            for i, metadata in enumerate(metadata_orders):
                gguf_data = create_gguf_file_with_metadata(metadata)

                with tempfile.NamedTemporaryFile(suffix=f'_order_{i}.gguf', delete=False) as f:
                    f.write(gguf_data)
                    f.flush()
                    temp_files.append(f.name)

                configurator = ModelAutoConfigurator()
                config = configurator.auto_configure(f.name)
                configs.append(config)

            # All configurations should be identical regardless of metadata order
            base_config = configs[0]
            for i, config in enumerate(configs[1:], 1):
                assert config == base_config, \
                    f"Order-dependent configuration at order {i}: expected {base_config}, got {config}"

        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def test_concurrent_determinism(self):
        """
        **Validates: Requirements 4.1, 4.2**

        Property: Concurrent auto-configuration operations on the same file
        SHALL produce identical results (thread safety of determinism).
        """
        metadata = {
            'llama.block_count': 24,
            'llama.embedding_length': 2048,
            'llama.attention.head_count': 16,
            'llama.attention.head_count_kv': 8,
            'llama.feed_forward_length': 5504,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 10000.0,
            'llama.attention.layer_norm_epsilon': 1e-6,
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            import threading
            import queue

            results = queue.Queue()

            def worker():
                try:
                    configurator = ModelAutoConfigurator()
                    results.put(configurator.auto_configure(temp_path))
                except Exception as error:
                    results.put(error)

            # Start multiple threads
            threads = []
            for i in range(3):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Collect results
            configs = []
            while not results.empty():
                configs.append(results.get())

            errors = [result for result in configs if isinstance(result, Exception)]
            assert not errors, f"Concurrent configuration raised: {errors}"

            # All results should be identical
            assert len(configs) == 3, f"Expected 3 results, got {len(configs)}"

            base_config = configs[0]
            for i, config in enumerate(configs[1:], 1):
                assert config == base_config, \
                    f"Concurrent determinism failure at thread {i}: expected {base_config}, got {config}"

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
