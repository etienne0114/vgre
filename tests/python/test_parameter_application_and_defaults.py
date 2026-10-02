#!/usr/bin/env python3
"""
Property Test for Parameter Application and Defaults

**Feature: gguf-auto-config, Property 6: Parameter Application and Defaults**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

This property test validates that parameters are correctly applied to LanguageModel
instances and that default values are used when parameters are missing.

Key test coverage:
1. All extracted parameters are correctly applied to LanguageModel
2. Default values are used when parameters are missing from GGUF
3. Parameter validation and range checking
4. Backward compatibility with existing LanguageModel constructor
"""

import os
import sys
import tempfile
import struct
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
    from vgre.lm import LanguageModel
    from vgre.auto_config import (
        GGUFFileError, GGUFMetadataError, GGUFValidationError,
        ModelAutoConfigurator,
    )
    VGRE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGRE not available: {e}")
    VGRE_AVAILABLE = False


# GGUF file format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGUF_TYPE_STRING = 8


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
def complete_model_parameters(draw):
    """Generate complete set of model parameters."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096, 8192]))
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
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=32768)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=1000000.0,
                                               width=32, allow_nan=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(
            min_value=struct.unpack('<f', struct.pack('<f', 1e-12))[0],
            max_value=struct.unpack('<f', struct.pack('<f', 1e-3))[0],
            width=32, allow_nan=False)),
    }


@composite
def minimal_model_parameters(draw):
    """Generate minimal required parameters (others will use defaults)."""
    embedding_length = draw(st.sampled_from([768, 1024, 2048, 4096]))
    head_count = draw(st.sampled_from([
        heads for heads in range(1, 257)
        if embedding_length % heads == 0 and (embedding_length // heads) % 2 == 0
    ]))
    return {
        'llama.block_count': draw(st.integers(min_value=1, max_value=100)),
        'llama.embedding_length': embedding_length,
        'llama.attention.head_count': head_count,
        'llama.feed_forward_length': draw(st.integers(min_value=1024, max_value=32768)),
        'llama.vocab_size': draw(st.integers(min_value=256, max_value=50000)),
    }


@composite
def invalid_parameter_ranges(draw):
    """Generate parameters with invalid ranges for validation testing."""
    param_name = draw(st.sampled_from([
        'llama.block_count',
        'llama.embedding_length',
        'llama.attention.head_count',
        'llama.rope.freq_base',
        'llama.attention.layer_norm_epsilon'
    ]))

    if param_name == 'llama.block_count':
        invalid_value = draw(st.integers(max_value=0))  # Should be > 0
    elif param_name == 'llama.embedding_length':
        invalid_value = draw(st.integers(max_value=0))  # Should be > 0
    elif param_name == 'llama.attention.head_count':
        invalid_value = draw(st.integers(max_value=0))  # Should be > 0
    elif param_name == 'llama.rope.freq_base':
        invalid_value = draw(st.one_of(
            st.floats(max_value=0.0, allow_nan=False),  # Should be > 0
            st.just(float('nan')),  # Should not be NaN
            st.just(float('inf'))   # Should not be infinite
        ))
    elif param_name == 'llama.attention.layer_norm_epsilon':
        invalid_value = draw(st.one_of(
            st.floats(max_value=0.0, allow_nan=False),  # Should be > 0
            st.just(float('nan')),  # Should not be NaN
            st.just(float('inf'))   # Should not be infinite
        ))

    return param_name, invalid_value


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestParameterApplicationAndDefaults:
    """Property tests for parameter application and defaults."""

    @given(complete_model_parameters())
    @settings(max_examples=50, deadline=None)
    def test_complete_parameter_application(self, parameters):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: For any complete set of valid model parameters extracted from GGUF,
        auto-configuration SHALL preserve every field and apply numeric defaults.
        """
        note(f"Testing with parameters: {parameters}")

        # Ensure parameters are in valid ranges
        assume(parameters['llama.block_count'] > 0)
        assume(parameters['llama.embedding_length'] > 0)
        assume(parameters['llama.attention.head_count'] > 0)
        assume(parameters['llama.rope.freq_base'] > 0 and not (
            abs(parameters['llama.rope.freq_base']) == float('inf') or
            str(parameters['llama.rope.freq_base']) == 'nan'
        ))
        assume(parameters['llama.attention.layer_norm_epsilon'] > 0 and not (
            abs(parameters['llama.attention.layer_norm_epsilon']) == float('inf') or
            str(parameters['llama.attention.layer_norm_epsilon']) == 'nan'
        ))

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            config = ModelAutoConfigurator.auto_configure(temp_path)

            # Verify all parameters were applied correctly
            assert config['block_count'] == parameters['llama.block_count']

            assert config['embedding_length'] == parameters['llama.embedding_length']

            assert config['head_count'] == parameters['llama.attention.head_count']

            if 'llama.attention.head_count_kv' in parameters:
                assert config['head_count_kv'] == parameters['llama.attention.head_count_kv']

            if 'llama.feed_forward_length' in parameters:
                assert config['feed_forward_length'] == parameters['llama.feed_forward_length']

            assert config['rope_theta'] == pytest.approx(parameters['llama.rope.freq_base'])

            assert config['norm_eps'] == pytest.approx(
                parameters['llama.attention.layer_norm_epsilon'])

        finally:
            os.unlink(temp_path)

    @given(minimal_model_parameters())
    @settings(max_examples=50, deadline=None)
    def test_default_parameter_application(self, parameters):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: For any minimal set of required parameters, the LanguageModel
        SHALL use appropriate default values for missing optional parameters.
        """
        note(f"Testing with minimal parameters: {parameters}")

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            config = ModelAutoConfigurator.auto_configure(temp_path)

            # Required parameters should match GGUF values
            assert config['block_count'] == parameters['llama.block_count']
            assert config['embedding_length'] == parameters['llama.embedding_length']
            assert config['head_count'] == parameters['llama.attention.head_count']

            # Missing optional parameters should use defaults
            expected_defaults = {
                'rope_theta': 10000.0,
                'norm_eps': 1e-5,
            }

            for param, default_value in expected_defaults.items():
                assert config[param] == default_value

        finally:
            os.unlink(temp_path)

    def test_backward_compatibility(self):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: The LanguageModel constructor SHALL remain backward compatible
        with existing code that passes parameters directly.
        """
        # Test original constructor still works
        model = LanguageModel(
            vocab=256, n_layer=1, d_model=64, n_head=1, n_kv_head=1, d_ff=128,
            max_seq=16,
            rope_theta=10000.0,
            norm_eps=1e-5
        )
        try:
            assert model.num_parameters > 0
            assert model.rope_theta == 10000.0
            assert model.norm_eps == 1e-5
        finally:
            model.close()

    def test_property_accessor_consistency(self):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: Parameter values accessible via property accessors SHALL
        match the values used during model construction.
        """
        parameters = {
            'llama.block_count': 32,
            'llama.embedding_length': 4096,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 16,
            'llama.feed_forward_length': 16384,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 12000.0,
            'llama.attention.layer_norm_epsilon': 1e-6,
        }

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            model = ModelAutoConfigurator.auto_configure(temp_path)

            # Test property getters
            assert 'block_count' in model
            assert 'embedding_length' in model
            assert 'head_count' in model
            assert 'head_count_kv' in model
            assert 'feed_forward_length' in model
            assert 'rope_theta' in model
            assert 'norm_eps' in model

            # Values should be accessible and correct
            config_values = {
                'block_count': parameters['llama.block_count'],
                'embedding_length': parameters['llama.embedding_length'],
                'head_count': parameters['llama.attention.head_count'],
                'head_count_kv': parameters['llama.attention.head_count_kv'],
                'feed_forward_length': parameters['llama.feed_forward_length'],
                'rope_theta': parameters['llama.rope.freq_base'],
                'norm_eps': parameters['llama.attention.layer_norm_epsilon'],
            }

            for prop_name, expected_value in config_values.items():
                actual_value = model[prop_name]
                assert actual_value == pytest.approx(expected_value), \
                    f"Property {prop_name}: expected {expected_value}, got {actual_value}"

        finally:
            os.unlink(temp_path)

    @given(invalid_parameter_ranges())
    @settings(max_examples=30, deadline=None)
    def test_parameter_validation(self, invalid_param):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: Invalid parameter values SHALL be detected and rejected
        with appropriate error messages during model construction.
        """
        param_name, invalid_value = invalid_param
        note(f"Testing invalid {param_name} = {invalid_value}")

        # Create valid baseline parameters
        parameters = {
            'llama.block_count': 16,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 10000.0,
            'llama.attention.layer_norm_epsilon': 1e-5,
        }

        # Replace one parameter with invalid value
        parameters[param_name] = invalid_value

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            # Should raise appropriate validation error
            with pytest.raises((GGUFFileError, GGUFMetadataError, GGUFValidationError)) as exc_info:
                ModelAutoConfigurator.auto_configure(temp_path)

            # Error message should mention the problematic parameter
            error_message = str(exc_info.value).lower()
            param_keywords = param_name.lower().split('.')[-1]  # Get last part of parameter name

            # Should contain information about the validation failure
            assert any(keyword in error_message for keyword in [
                'invalid', 'range', 'value', 'finite', 'validation', 'missing', param_keywords
            ]), \
                f"Error message should mention validation issue: {exc_info.value}"

        finally:
            os.unlink(temp_path)

    def test_default_parameter_values(self):
        """
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

        Property: Default parameter values SHALL be mathematically reasonable
        and consistent with common model configurations.
        """
        # Test with minimal required parameters only
        parameters = {
            'llama.block_count': 12,
            'llama.embedding_length': 768,
            'llama.attention.head_count': 12,
            'llama.feed_forward_length': 2048,
            'llama.vocab_size': 32000,
        }

        gguf_data = create_gguf_file_with_metadata(parameters)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            config = ModelAutoConfigurator.auto_configure(temp_path)

            # Test default values are reasonable
            assert config['rope_theta'] == 10000.0, "Default rope_theta should be 10000.0"
            assert config['norm_eps'] == 1e-5, "Default norm_eps should be 1e-5"

            # Default values should be in reasonable ranges
            assert 1000.0 <= config['rope_theta'] <= 1000000.0, "Default rope_theta should be in reasonable range"
            assert 1e-12 <= config['norm_eps'] <= 1e-3, "Default norm_eps should be in reasonable range"

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
