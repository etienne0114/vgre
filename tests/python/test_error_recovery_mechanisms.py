#!/usr/bin/env python3
"""
Unit Tests for Error Recovery Mechanisms

**Feature: gguf-auto-config, Task 5.3**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

This test suite validates error recovery mechanisms in the GGUF auto-configuration
system, including parameter inference, graceful degradation, and user guidance.

Key test coverage:
1. Parameter inference from tensor shapes when metadata is missing
2. Error message formatting and user guidance
3. Graceful degradation scenarios
4. Recovery from partial corruption
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
    from vgre import GGUFMetadataReader
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
GGUF_TYPE_UINT32 = 6
GGUF_TYPE_FLOAT32 = 10


def create_gguf_with_tensor_shapes(metadata: Dict[str, Any], tensor_shapes: Dict[str, tuple]) -> bytes:
    """Create a valid float32-tensor GGUF fixture for metadata inference tests."""
    header = bytearray(GGUF_MAGIC)
    header += struct.pack('<I', GGUF_VERSION)
    header += struct.pack('<Q', len(tensor_shapes))
    header += struct.pack('<Q', len(metadata))

    for key, value in metadata.items():
        key_bytes = key.encode('utf-8')
        value_bytes = str(value).encode('utf-8')
        header += struct.pack('<Q', len(key_bytes)) + key_bytes
        header += struct.pack('<I', GGUF_TYPE_STRING)
        header += struct.pack('<Q', len(value_bytes)) + value_bytes

    offsets = {}
    payload = bytearray()
    for tensor_name, shape in tensor_shapes.items():
        padding = (-len(payload)) % 32
        payload.extend(b'\0' * padding)
        offsets[tensor_name] = len(payload)
        elements = 1
        for dim in shape:
            elements *= dim
        payload.extend(b'\0' * (elements * 4))

    for tensor_name, shape in tensor_shapes.items():
        name_bytes = tensor_name.encode('utf-8')
        header += struct.pack('<Q', len(name_bytes)) + name_bytes
        header += struct.pack('<I', len(shape))
        header += struct.pack(f'<{len(shape)}Q', *shape)
        header += struct.pack('<IQ', 0, offsets[tensor_name])

    header.extend(b'\0' * ((-len(header)) % 32))
    return bytes(header + payload)


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestErrorRecoveryMechanisms:
    """Unit tests for error recovery mechanisms."""

    def test_parameter_inference_from_tensor_shapes(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

        Test that missing parameters can be inferred from tensor shapes
        when metadata is incomplete.
        """
        # Metadata with some missing parameters
        metadata = {
            'llama.block_count': 1,
            'llama.attention.head_count': 16,
        }

        # Tensor shapes that allow parameter inference
        tensor_shapes = {
            'token_embd.weight': (1024, 256),
            'blk.0.ffn_down.weight': (128, 1024),
        }

        gguf_data = create_gguf_with_tensor_shapes(metadata, tensor_shapes)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)

            # The native reader derives only dimensions represented unambiguously.
            assert config.get('embedding_length') == 1024, \
                f"Should infer embedding_length=1024 from tensor shapes, got {config.get('embedding_length')}"
            assert config.get('feed_forward_length') == 128
            assert config.get('vocab') == 256
            assert config.get('head_count_kv') == 16

        finally:
            os.unlink(temp_path)

    def test_graceful_degradation_partial_metadata(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        Test graceful degradation when only partial metadata is available.
        """
        # Minimal metadata that should allow basic operation
        metadata = {
            'llama.block_count': 12,
            'llama.embedding_length': 768,
            'llama.attention.head_count': 12,
            'llama.feed_forward_length': 2048,
            'llama.vocab_size': 32000,
            # Missing rope_freq_base, layer_norm_epsilon, etc.
        }

        gguf_data = create_gguf_with_tensor_shapes(metadata, {})

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)

            # Should succeed with defaults for missing parameters
            assert config['block_count'] == 12
            assert config['embedding_length'] == 768
            assert config['head_count'] == 12

            # Should have applied reasonable defaults
            assert config.get('rope_theta') == 10000.0, "Should apply default rope_theta"
            assert config.get('norm_eps') == 1e-5, "Should apply default norm_eps"

            assert config['feed_forward_length'] == 2048
            assert config['vocab'] == 32000

        finally:
            os.unlink(temp_path)

    def test_error_message_formatting_and_guidance(self):
        """
        **Validates: Requirements 8.4, 8.5**

        Test that error messages are well-formatted and provide actionable
        user guidance for common error scenarios.
        """
        error_scenarios = [
            {
                'name': 'missing_critical_metadata',
                'metadata': {'general.name': 'test'},  # No model parameters
                'expected_guidance': ['missing', 'required', 'parameter']
            },
            {
                'name': 'invalid_parameter_values',
                'metadata': {
                    'llama.block_count': -5,  # Invalid negative value
                    'llama.embedding_length': 0,  # Invalid zero value
                    'llama.attention.head_count': 16,
                    'llama.feed_forward_length': 4096,
                    'llama.vocab_size': 32000,
                },
                'expected_guidance': ['invalid', 'value', 'range']
            },
            {
                'name': 'inconsistent_parameters',
                'metadata': {
                    'llama.block_count': 16,
                    'llama.embedding_length': 1024,
                    'llama.attention.head_count': 16,
                    'llama.attention.head_count_kv': 100,  # Much larger than head_count
                    'llama.feed_forward_length': 4096,
                    'llama.vocab_size': 32000,
                },
                'expected_guidance': ['divide', 'divisible', 'n_kv_head']
            }
        ]

        for scenario in error_scenarios:
            gguf_data = create_gguf_with_tensor_shapes(scenario['metadata'], {})

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(gguf_data)
                f.flush()
                temp_path = f.name

            try:
                configurator = ModelAutoConfigurator()

                with pytest.raises((GGUFFileError, GGUFMetadataError, GGUFValidationError)) as exc_info:
                    config = configurator.auto_configure(temp_path)

                error_message = str(exc_info.value).lower()

                # Should contain helpful guidance keywords
                guidance_found = any(
                    keyword in error_message
                    for keyword in scenario['expected_guidance']
                )

                assert guidance_found, \
                    f"Error message for {scenario['name']} should contain guidance keywords {scenario['expected_guidance']}: {exc_info.value}"

                # Should be reasonably informative (not just a generic error)
                assert len(error_message) > 10, \
                    f"Error message should be informative for {scenario['name']}: {exc_info.value}"

            finally:
                os.unlink(temp_path)

    def test_recovery_from_corrupted_json_sidecar(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

        Test that corrupted JSON is reported instead of silently ignored.
        """
        # Valid GGUF metadata
        metadata = {
            'llama.block_count': 20,
            'llama.embedding_length': 2048,
            'llama.attention.head_count': 16,
            'llama.rope.freq_base': 12000.0,
            'llama.attention.layer_norm_epsilon': 1e-6,
            'llama.attention.head_count_kv': 8,
            'llama.feed_forward_length': 5504,
            'llama.vocab_size': 32000,
        }

        gguf_data = create_gguf_with_tensor_shapes(metadata, {})

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        # Create corrupted JSON sidecar
        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json_file.write('{ "block_count": 30, "invalid_json":')  # Truncated

        try:
            configurator = ModelAutoConfigurator()
            with pytest.raises(GGUFFileError, match='Failed to parse JSON sidecar'):
                configurator.auto_configure(gguf_path)

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_recovery_from_partial_tensor_corruption(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

        Test recovery when some tensor information is corrupted but metadata is valid.
        """
        # Complete metadata (should not need tensor inference)
        metadata = {
            'llama.block_count': 16,
            'llama.embedding_length': 1024,
            'llama.attention.head_count': 16,
            'llama.attention.head_count_kv': 8,
            'llama.feed_forward_length': 4096,
            'llama.vocab_size': 32000,
            'llama.rope.freq_base': 10000.0,
            'llama.attention.layer_norm_epsilon': 1e-5,
        }

        # Some valid tensors and some with invalid shapes
        tensor_shapes = {
            'token_embd.weight': (1024, 256),  # Valid
            'blk.0.attn_q.weight': (1024, 0),    # Invalid dimension
            'corrupted_tensor': (0, 0, 0),       # Invalid shape
        }

        gguf_data = create_gguf_with_tensor_shapes(metadata, tensor_shapes)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(temp_path)

            # Should succeed using metadata instead of corrupted tensor shapes
            assert config['block_count'] == metadata['llama.block_count']
            assert config['embedding_length'] == metadata['llama.embedding_length']
            assert config['head_count'] == metadata['llama.attention.head_count']
            assert config['head_count_kv'] == metadata['llama.attention.head_count_kv']


        finally:
            os.unlink(temp_path)

    def test_user_guidance_for_common_issues(self):
        """
        **Validates: Requirements 8.4, 8.5**

        Test that common user issues receive helpful guidance messages.
        """
        common_issues = [
            {
                'issue': 'file_not_found',
                'test_path': '/nonexistent/model.gguf',
                'expected_guidance': ['file', 'open', 'path']
            },
            {
                'issue': 'invalid_format',
                'setup': lambda: self._create_invalid_format_file(),
                'expected_guidance': ['gguf', 'open']
            }
        ]

        for issue_info in common_issues:
            issue = issue_info['issue']

            if issue == 'file_not_found':
                test_path = issue_info['test_path']
                cleanup_path = None
            else:
                test_path = issue_info['setup']()
                cleanup_path = test_path

            try:
                configurator = ModelAutoConfigurator()

                with pytest.raises((ValueError, RuntimeError, FileNotFoundError, OSError)) as exc_info:
                    config = configurator.auto_configure(test_path)

                error_message = str(exc_info.value).lower()

                # Should contain helpful guidance
                guidance_found = any(
                    keyword in error_message
                    for keyword in issue_info['expected_guidance']
                )

                assert guidance_found, \
                    f"Error for {issue} should contain guidance: {exc_info.value}"

                # Should suggest actionable steps
                assert len(error_message) > 15, \
                    f"Error message should be detailed for {issue}: {exc_info.value}"

            finally:
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.chmod(cleanup_path, 0o644)  # Restore permissions
                        os.unlink(cleanup_path)
                    except:
                        pass

    def test_progressive_fallback_strategy(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

        Test that the system uses a progressive fallback strategy:
        JSON sidecar → GGUF metadata → tensor inference → defaults
        """
        # GGUF with minimal metadata
        gguf_metadata = {
            'llama.block_count': 16,
            # The JSON sidecar supplies the required attention head count.
        }

        # Tensor shapes that could provide some inference
        tensor_shapes = {
            'token_embd.weight': (1024, 256),
            'blk.0.ffn_down.weight': (128, 1024),
        }

        gguf_data = create_gguf_with_tensor_shapes(gguf_metadata, tensor_shapes)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as gguf_file:
            gguf_file.write(gguf_data)
            gguf_file.flush()
            gguf_path = gguf_file.name

        # Create partial JSON sidecar
        json_config = {
            'head_count': 16,  # Override via JSON
            # Other parameters will fall back through the chain
        }

        json_path = gguf_path.replace('.gguf', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(json_config, json_file)

        try:
            configurator = ModelAutoConfigurator()
            config = configurator.auto_configure(gguf_path)

            # Should use JSON value (highest priority)
            assert config['head_count'] == json_config['head_count'], \
                "Should use JSON head_count"

            # Should use GGUF metadata and unambiguous tensor dimensions.
            assert config['block_count'] == gguf_metadata['llama.block_count'], \
                "Should use GGUF block_count"

            # Should infer from tensors (third priority)
            assert config.get('embedding_length') == 1024, \
                "Should infer embedding_length from tensors"
            assert config.get('feed_forward_length') == 128
            assert config.get('vocab') == 256

            # Should use defaults (lowest priority)
            assert config.get('rope_theta') == 10000.0, \
                "Should use default rope_theta"
            assert config.get('norm_eps') == 1e-5, \
                "Should use default norm_eps"

        finally:
            os.unlink(gguf_path)
            if os.path.exists(json_path):
                os.unlink(json_path)

    def _create_permission_denied_file(self):
        """Helper to create a file with denied permissions."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(GGUF_MAGIC + b'\x00' * 100)
            f.flush()
            temp_path = f.name

        # Remove read permissions
        os.chmod(temp_path, 0o000)
        return temp_path

    def _create_invalid_format_file(self):
        """Helper to create a file with invalid format."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'NOT_GGUF_FORMAT' + b'\x00' * 100)
            f.flush()
            temp_path = f.name

        return temp_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
