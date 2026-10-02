#!/usr/bin/env python3
"""
Unit Tests for Auto-Configuration Edge Cases

**Feature: gguf-auto-config, Task 4.6**
**Validates: Requirements 4.4, 4.5, 8.2, 8.3, 9.5**

This test suite validates edge cases in the auto-configuration system,
including missing metadata scenarios, validation errors, and JSON sidecar behavior.

Key test coverage:
1. Missing metadata scenarios with parameter inference
2. Validation error handling for out-of-range parameters
3. JSON sidecar loading and precedence behavior
4. Corrupted or invalid file handling
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

try:
    import vgre
    from vgre.auto_config import (
        GGUFFileError, GGUFMetadataError, GGUFValidationError,
        ModelAutoConfigurator,
    )
    from vgre.lm import LanguageModel
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

    entry = struct.pack('<Q', key_len)
    entry += key_bytes
    entry += struct.pack('<I', GGUF_TYPE_STRING)

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


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestAutoConfigurationEdgeCases:
    """Unit tests for auto-configuration edge cases."""

    def test_missing_required_metadata(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test handling when required metadata is completely missing from GGUF file.
        """
        # Create GGUF with no useful metadata
        metadata = {
            'general.name': 'test_model',
            'general.description': 'A test model'
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            # Should raise error for missing required parameters
            with pytest.raises(GGUFMetadataError) as exc_info:
                config = configurator.auto_configure(temp_path)

            error_message = str(exc_info.value).lower()
            assert any(keyword in error_message for keyword in ['missing', 'required', 'parameter']), \
                f"Error should indicate missing required parameters: {exc_info.value}"

        finally:
            os.unlink(temp_path)

    def test_partially_missing_metadata(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test handling when some required metadata is missing but others are present.
        """
        # Missing embedding_length and head_count
        metadata = {
            'llama.block_count': 16,
            'llama.rope.freq_base': 15000.0,
            'llama.attention.layer_norm_epsilon': 1e-6,
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            # Should fail due to missing critical parameters
            with pytest.raises(GGUFMetadataError) as exc_info:
                config = configurator.auto_configure(temp_path)

            error_message = str(exc_info.value).lower()
            assert any(keyword in error_message for keyword in ['missing', 'required']), \
                f"Error should indicate missing parameters: {exc_info.value}"

        finally:
            os.unlink(temp_path)

    def test_out_of_range_parameters(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test validation error handling for parameters with out-of-range values.
        """
        test_cases = [
            # (parameter_name, invalid_value, description)
            ('llama.block_count', -5, 'negative block count'),
            ('llama.embedding_length', 0, 'zero embedding length'),
            ('llama.attention.head_count', -1, 'negative head count'),
            ('llama.rope.freq_base', -1000.0, 'negative rope frequency'),
            ('llama.attention.layer_norm_epsilon', 0.0, 'zero norm epsilon'),
        ]

        for param_name, invalid_value, description in test_cases:
            metadata = {
                'llama.block_count': 16,
                'llama.embedding_length': 1024,
                'llama.attention.head_count': 16,
                'llama.feed_forward_length': 4096,
                'llama.vocab_size': 32000,
                param_name: invalid_value  # Override with invalid value
            }

            gguf_data = create_gguf_file_with_metadata(metadata)

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                configurator = ModelAutoConfigurator()

                with pytest.raises(GGUFValidationError) as exc_info:
                    config = configurator.auto_configure(temp_path)

                error_message = str(exc_info.value).lower()
                assert any(keyword in error_message for keyword in ['invalid', 'range', 'value']), \
                    f"Error for {description} should indicate validation failure: {exc_info.value}"

            finally:
                os.unlink(temp_path)

    def test_json_sidecar_precedence(self):
        """
        **Validates: Requirements 9.5**

        Test that JSON sidecar parameters correctly override GGUF metadata.
        """
        gguf_metadata = {
            'llama.block_count': 20,
            'llama.embedding_length': 2048,
            'llama.attention.head_count': 16,
            'llama.rope.freq_base': 12000.0,
            'llama.feed_forward_length': 5504,
            'llama.vocab_size': 32000,
        }

        json_config = {
            'block_count': 30,  # Override GGUF
            'rope_theta': 18000.0,  # Override GGUF
            # embedding_length and head_count should come from GGUF
        }

        gguf_data = create_gguf_file_with_metadata(gguf_metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(json_config, json_file)

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # JSON values should override GGUF
            assert config['block_count'] == json_config['block_count'], \
                f"JSON block_count should override GGUF: expected {json_config['block_count']}, got {config['block_count']}"

            assert config['rope_theta'] == json_config['rope_theta'], \
                f"JSON rope_theta should override GGUF: expected {json_config['rope_theta']}, got {config['rope_theta']}"

            # GGUF values should be used for non-JSON parameters
            assert config['embedding_length'] == gguf_metadata['llama.embedding_length'], \
                f"GGUF embedding_length should be used: expected {gguf_metadata['llama.embedding_length']}, got {config['embedding_length']}"

            assert config['head_count'] == gguf_metadata['llama.attention.head_count'], \
                f"GGUF head_count should be used: expected {gguf_metadata['llama.attention.head_count']}, got {config['head_count']}"

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_invalid_json_sidecar_is_reported(self):
        """
        **Validates: Requirements 9.5**

        Test that invalid JSON sidecar gracefully falls back to GGUF metadata.
        """
        metadata = {
            'llama.block_count': 24,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.rope.freq_base': 11000.0,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        # Create invalid JSON sidecar
        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json_file.write('{ invalid json content }'[:-1])  # Truncated JSON

        try:
            configurator = ModelAutoConfigurator()
            with pytest.raises(GGUFFileError, match='Failed to parse JSON sidecar'):
                configurator.auto_configure(gguf_path)

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_json_with_invalid_parameter_values(self):
        """
        **Validates: Requirements 8.2, 8.3, 9.5**

        Test validation when JSON sidecar contains invalid parameter values.
        """
        metadata = {
            'llama.block_count': 16,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
        }

        invalid_json_configs = [
            {'block_count': -5},  # Negative value
            {'embedding_length': 0},  # Zero value
            {'head_count': 'invalid'},  # String instead of number
            {'rope_theta': float('inf')},  # Infinite value
            {'norm_eps': float('nan')},  # NaN value
        ]

        for invalid_config in invalid_json_configs:
            gguf_data = create_gguf_file_with_metadata(metadata)

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
                gguf_file.write(gguf_data)
                gguf_file.flush()
                gguf_path = gguf_file.name

            json_path = gguf_path.replace('.gguf', '.json')
            with open(json_path, 'w') as json_file:
                json.dump(invalid_config, json_file)

            try:
                configurator = ModelAutoConfigurator()

                with pytest.raises(GGUFValidationError) as exc_info:
                    config = configurator.auto_configure(gguf_path)

                error_message = str(exc_info.value).lower()
                assert any(keyword in error_message for keyword in ['invalid', 'value', 'type', 'range', 'finite']), \
                    f"Should reject invalid JSON value {invalid_config}: {exc_info.value}"

            finally:
                os.unlink(gguf_path)
                if os.path.exists(json_path):
                    os.unlink(json_path)

    def test_empty_gguf_file(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test handling of empty or minimal GGUF files.
        """
        # Create minimal GGUF file with no metadata
        metadata = {}  # Empty metadata

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            with pytest.raises(GGUFMetadataError) as exc_info:
                config = configurator.auto_configure(temp_path)

            error_message = str(exc_info.value).lower()
            assert any(keyword in error_message for keyword in ['missing', 'empty', 'metadata']), \
                f"Should handle empty GGUF file: {exc_info.value}"

        finally:
            os.unlink(temp_path)

    def test_malformed_gguf_metadata(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test handling of malformed GGUF metadata entries.
        """
        # A malformed numeric field is rejected rather than guessed.
        metadata = {
            'llama.block_count': 'not_a_number',
            'llama.embedding_length': '1024',  # String instead of int
            'llama.attention.head_count': 16,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()

            with pytest.raises(GGUFFileError, match='Invalid numeric value') as exc_info:
                configurator.auto_configure(temp_path)

        finally:
            os.unlink(temp_path)

    def test_large_parameter_values(self):
        """
        **Validates: Requirements 4.4, 4.5, 8.2, 8.3**

        Test accepting values at the configured supported limits.
        """
        metadata = {
            'llama.block_count': 128,
            'llama.embedding_length': 16384,
            'llama.attention.head_count': 256,  # Very large
            'llama.rope.freq_base': 1000000.0,  # Very large
            'llama.feed_forward_length': 65536,
            'llama.vocab_size': 100000,
        }

        gguf_data = create_gguf_file_with_metadata(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)

            # Should accept large but valid values
            assert config['block_count'] == metadata['llama.block_count']
            assert config['embedding_length'] == metadata['llama.embedding_length']
            assert config['head_count'] == metadata['llama.attention.head_count']
            assert config['rope_theta'] == metadata['llama.rope.freq_base']

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
