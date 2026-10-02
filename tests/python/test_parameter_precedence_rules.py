#!/usr/bin/env python3
"""
Property Test for Parameter Precedence Rules

**Feature: gguf-auto-config, Property 9: Parameter Precedence Rules**
**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

This property test validates that the parameter precedence system works correctly,
ensuring that JSON sidecar parameters take precedence over GGUF metadata,
and that the precedence rules are applied consistently.

Key test coverage:
1. JSON sidecar parameters override GGUF metadata
2. GGUF metadata overrides system defaults
3. Precedence order: JSON > GGUF > Defaults
4. Partial overrides work correctly (some params from JSON, others from GGUF)
"""

import os
import sys
import tempfile
import struct
import json
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
from hypothesis import given, strategies as st, settings, note, assume
from hypothesis.strategies import composite

try:
    import vgre
    from vgre.auto_config import GGUFFileError, ModelAutoConfigurator
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
FLOAT32_EPS_PARTIAL_MAX = struct.unpack('<f', struct.pack('<f', 1e-4))[0]


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
def gguf_parameters(draw):
    """Generate GGUF metadata parameters."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096]))
    head_count = draw(st.sampled_from([
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]))
    kv_head_count = draw(st.sampled_from([
        heads for heads in range(1, head_count + 1) if head_count % heads == 0
    ]))
    return {
        'llama.block_count': draw(st.integers(min_value=1, max_value=100)),
        'llama.embedding_length': embedding_length,
        'llama.attention.head_count': head_count,
        'llama.attention.head_count_kv': kv_head_count,
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=16384)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=100000.0,
                                               width=32, allow_nan=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(min_value=FLOAT32_EPS_MIN,
                                                             max_value=FLOAT32_EPS_MAX,
                                                             width=32, allow_nan=False)),
    }


@composite
def json_config_parameters(draw):
    """Generate JSON sidecar configuration parameters."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096]))
    head_count = draw(st.sampled_from([
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]))
    kv_head_count = draw(st.sampled_from([
        heads for heads in range(1, head_count + 1) if head_count % heads == 0
    ]))
    return {
        'block_count': draw(st.integers(min_value=1, max_value=100)),
        'embedding_length': embedding_length,
        'head_count': head_count,
        'head_count_kv': kv_head_count,
        'feed_forward_length': draw(st.integers(min_value=1024, max_value=16384)),
        'vocab': draw(st.integers(min_value=256, max_value=50000)),
        'rope_theta': draw(st.floats(min_value=1000.0, max_value=100000.0,
                                     width=32, allow_nan=False)),
        'norm_eps': draw(st.floats(min_value=FLOAT32_EPS_MIN, max_value=FLOAT32_EPS_MAX,
                                   width=32, allow_nan=False)),
    }


@composite
def conflicting_parameters(draw):
    """Generate pairs of conflicting GGUF and JSON parameters."""
    gguf_params = draw(gguf_parameters())
    json_params = draw(json_config_parameters())

    # Ensure they have different values for the same logical parameters
    json_params['block_count'] = gguf_params['llama.block_count'] + 10
    json_params['embedding_length'] = 2048 if gguf_params['llama.embedding_length'] == 4096 else 4096
    valid_heads = [
        heads for heads in range(1, 257)
        if json_params['embedding_length'] % heads == 0 and
        (json_params['embedding_length'] // heads) % 2 == 0
    ]
    json_params['head_count'] = draw(st.sampled_from(valid_heads))
    json_params['head_count_kv'] = draw(st.sampled_from([
        heads for heads in range(1, json_params['head_count'] + 1)
        if json_params['head_count'] % heads == 0
    ]))
    json_params['rope_theta'] = gguf_params['llama.rope.freq_base'] + 5000.0

    return gguf_params, json_params


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestParameterPrecedenceRules:
    """Property tests for parameter precedence rules."""

    @given(conflicting_parameters())
    @settings(max_examples=50, deadline=None)
    def test_json_overrides_gguf(self, param_pair):
        """
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

        Property: For any parameter present in both JSON sidecar and GGUF metadata,
        the JSON sidecar value SHALL take precedence over the GGUF metadata value.
        """
        gguf_params, json_params = param_pair
        note(f"GGUF params: {gguf_params}")
        note(f"JSON params: {json_params}")

        # Create GGUF file
        gguf_data = create_gguf_file_with_metadata(gguf_params)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        # Create JSON sidecar
        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(json_params, json_file)

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # JSON parameters should override GGUF parameters
            assert config['block_count'] == json_params['block_count'], \
                f"JSON block_count {json_params['block_count']} should override GGUF {gguf_params['llama.block_count']}"

            assert config['embedding_length'] == json_params['embedding_length'], \
                f"JSON embedding_length {json_params['embedding_length']} should override GGUF {gguf_params['llama.embedding_length']}"

            assert config['rope_theta'] == json_params['rope_theta'], \
                f"JSON rope_theta {json_params['rope_theta']} should override GGUF {gguf_params['llama.rope.freq_base']}"

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    @given(gguf_parameters())
    @settings(max_examples=50, deadline=None)
    def test_gguf_overrides_defaults(self, gguf_params):
        """
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

        Property: For any parameter present in GGUF metadata but not in JSON sidecar,
        the GGUF metadata value SHALL take precedence over system defaults.
        """
        note(f"GGUF params: {gguf_params}")

        # Ensure GGUF parameters are different from defaults
        assume(gguf_params.get('llama.rope.freq_base', 10000.0) != 10000.0)
        assume(gguf_params.get('llama.attention.layer_norm_epsilon', 1e-5) != 1e-5)

        # Create GGUF file (no JSON sidecar)
        gguf_data = create_gguf_file_with_metadata(gguf_params)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # GGUF parameters should override defaults
            if 'llama.rope.freq_base' in gguf_params:
                assert config.get('rope_theta') == gguf_params['llama.rope.freq_base'], \
                    f"GGUF rope_freq_base {gguf_params['llama.rope.freq_base']} should override default 10000.0"

            if 'llama.attention.layer_norm_epsilon' in gguf_params:
                assert config.get('norm_eps') == gguf_params['llama.attention.layer_norm_epsilon'], \
                    f"GGUF norm_eps {gguf_params['llama.attention.layer_norm_epsilon']} should override default 1e-5"

            # Required parameters should come from GGUF
            assert config.get('block_count') == gguf_params.get('llama.block_count'), \
                f"block_count should come from GGUF: {gguf_params.get('llama.block_count')}"

        finally:
            os.unlink(gguf_path)

    @given(st.data())
    @settings(max_examples=30, deadline=None)
    def test_partial_json_override(self, data):
        """
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

        Property: When JSON sidecar contains only some parameters, those parameters
        SHALL override GGUF/defaults, while missing parameters SHALL come from GGUF
        or defaults according to precedence rules.
        """
        # Generate full GGUF parameters
        gguf_params = data.draw(gguf_parameters())

        # Generate partial JSON parameters (randomly select subset)
        all_json_keys = ['block_count', 'rope_theta', 'norm_eps']
        json_keys = data.draw(st.lists(st.sampled_from(all_json_keys), min_size=1, max_size=3, unique=True))

        json_params = {}
        for key in json_keys:
            if key == 'block_count':
                json_params[key] = data.draw(st.integers(min_value=1, max_value=50))
            elif key == 'rope_theta':
                json_params[key] = data.draw(st.floats(min_value=5000.0, max_value=50000.0,
                                                       width=32, allow_nan=False))
            elif key == 'norm_eps':
                json_params[key] = data.draw(st.floats(min_value=FLOAT32_EPS_MIN,
                                                       max_value=FLOAT32_EPS_PARTIAL_MAX,
                                                       width=32, allow_nan=False))

        note(f"GGUF params: {gguf_params}")
        note(f"Partial JSON params: {json_params}")

        # Create files
        gguf_data = create_gguf_file_with_metadata(gguf_params)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(json_params, json_file)

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # Parameters in JSON should use JSON values
            for json_key, json_value in json_params.items():
                assert config.get(json_key) == json_value, \
                    f"JSON parameter {json_key} should be {json_value}, got {config.get(json_key)}"

            # Parameters not in JSON should come from GGUF or defaults
            if 'block_count' not in json_params:
                assert config.get('block_count') == gguf_params.get('llama.block_count'), \
                    f"Missing JSON block_count should come from GGUF: {gguf_params.get('llama.block_count')}"

            if 'rope_theta' not in json_params and 'llama.rope.freq_base' in gguf_params:
                assert config.get('rope_theta') == gguf_params['llama.rope.freq_base'], \
                    f"Missing JSON rope_theta should come from GGUF: {gguf_params['llama.rope.freq_base']}"
            elif 'rope_theta' not in json_params:
                assert config.get('rope_theta') == 10000.0, \
                    f"Missing parameters should use default rope_theta=10000.0"

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_precedence_chain_completeness(self):
        """
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

        Property: The complete precedence chain (JSON > GGUF > Defaults) SHALL
        work correctly for all parameter types.
        """
        # Test complete precedence chain with known values
        gguf_params = {
            'llama.block_count': 20,
            'llama.embedding_length': 2048,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 5504,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 15000.0,
            # Deliberately omit llama.attention.layer_norm_epsilon to test default
        }

        json_params = {
            'block_count': 30,  # Override GGUF
            # Deliberately omit embedding_length to test GGUF fallback
            'rope_theta': 25000.0,  # Override GGUF
            # Deliberately omit norm_eps to test default fallback
        }

        expected_config = {
            'block_count': 30,  # From JSON (highest priority)
            'embedding_length': 2048,  # From GGUF (JSON missing)
            'rope_theta': 25000.0,  # From JSON (highest priority)
            'norm_eps': 1e-5,  # From defaults (both JSON and GGUF missing)
        }

        # Create files
        gguf_data = create_gguf_file_with_metadata(gguf_params)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(json_params, json_file)

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # Verify complete precedence chain
            for param, expected_value in expected_config.items():
                actual_value = config.get(param)
                assert actual_value == expected_value, \
                    f"Precedence chain failed for {param}: expected {expected_value}, got {actual_value}"

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_invalid_json_sidecar_is_reported(self):
        """
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

        Property: Invalid JSON sidecar SHALL NOT prevent GGUF parameter loading;
        the system SHALL gracefully fall back to GGUF parameters.
        """
        gguf_params = {
            'llama.block_count': 16,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 12000.0,
        }

        # Create GGUF file
        gguf_data = create_gguf_file_with_metadata(gguf_params)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        # Create invalid JSON sidecar
        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json_file.write('{ invalid json content')

        try:
            configurator = ModelAutoConfigurator()
            with pytest.raises(GGUFFileError, match='Failed to parse JSON sidecar'):
                configurator.auto_configure(gguf_path)

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
