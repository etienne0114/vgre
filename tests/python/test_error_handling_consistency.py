#!/usr/bin/env python3
"""
Property Test for Error Handling Consistency

**Feature: gguf-auto-config, Property 2: Error Handling Consistency**
**Validates: Requirements 1.7, 1.8, 2.3, 2.5, 2.6, 8.1, 8.2, 8.3, 8.4**

This property test validates that error handling is consistent across all
components of the GGUF auto-configuration system, including C++ metadata
extraction, C API, Python bindings, and auto-configuration logic.

Key test coverage:
1. Consistent error types and messages across language boundaries
2. Proper error propagation from C++ to C API to Python
3. Structured error information preservation
4. Graceful degradation and recovery mechanisms
"""

import os
import sys
import tempfile
import struct
import gc
import threading
import time
from typing import List, Dict, Any, Optional

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
    from vgre import GGUFMetadataReader
    from vgre.auto_config import ModelAutoConfigurator
    from vgre.lm import LanguageModel
    VGRE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGRE not available: {e}")
    VGRE_AVAILABLE = False


# GGUF file format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3


def create_corrupted_gguf_file(corruption_type: str) -> bytes:
    """Create various types of corrupted GGUF files for testing."""
    if corruption_type == 'invalid_magic':
        return b'INVALID_MAGIC_NUMBER' + b'\x00' * 100
    elif corruption_type == 'truncated_header':
        return GGUF_MAGIC + b'\x03\x00\x00'  # Incomplete version
    elif corruption_type == 'invalid_version':
        return GGUF_MAGIC + b'\xFF\xFF\xFF\xFF' + b'\x00' * 100
    elif corruption_type == 'corrupted_metadata_count':
        data = GGUF_MAGIC + b'\x03\x00\x00\x00'  # Valid version
        data += b'\x00\x00\x00\x00\x00\x00\x00\x00'  # Tensor count (0)
        data += b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'  # Invalid metadata count
        return data
    elif corruption_type == 'truncated_metadata':
        data = GGUF_MAGIC + b'\x03\x00\x00\x00'  # Valid version
        data += b'\x00\x00\x00\x00\x00\x00\x00\x00'  # Tensor count (0)
        data += b'\x01\x00\x00\x00\x00\x00\x00\x00'  # Metadata count (1)
        data += b'\x05\x00\x00\x00\x00\x00\x00\x00'  # Key length (5)
        data += b'test'  # Truncated key (should be 5 bytes)
        return data
    else:
        return b''  # Empty file


@composite
def error_inducing_files(draw):
    """Generate various error-inducing file scenarios."""
    scenario = draw(st.sampled_from([
        'nonexistent_file',
        'invalid_magic',
        'truncated_header',
        'invalid_version',
        'corrupted_metadata_count',
        'truncated_metadata',
        'empty_file'
    ]))

    if scenario == 'nonexistent_file':
        return scenario, f'/nonexistent/path/{draw(st.text(min_size=1, max_size=20))}.gguf'
    else:
        return scenario, None  # Will create corrupted file content


@composite
def error_contexts(draw):
    """Generate different contexts where errors might occur."""
    return draw(st.sampled_from([
        'metadata_reader_init',
        'parameter_extraction',
        'auto_configuration',
        'model_construction'
    ]))


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestErrorHandlingConsistency:
    """Property tests for error handling consistency."""

    @given(error_inducing_files())
    @settings(max_examples=50, deadline=None)
    def test_consistent_error_types_across_components(self, error_scenario):
        """
        **Validates: Requirements 1.7, 1.8, 2.3, 2.5, 2.6, 8.1, 8.2, 8.3, 8.4**

        Property: For any error condition, all components (C++, C API, Python bindings)
        SHALL raise consistent error types that properly represent the error category.
        """
        scenario_type, file_path = error_scenario
        note(f"Testing error scenario: {scenario_type}")

        if scenario_type == 'nonexistent_file':
            test_path = file_path
        else:
            # Create corrupted file
            corrupted_data = create_corrupted_gguf_file(scenario_type)
            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(corrupted_data)
                f.flush()
                test_path = f.name

        errors_caught = []
        reader = None

        try:
            # Test GGUFMetadataReader (direct C API access)
            try:
                reader = GGUFMetadataReader(test_path)
                reader.get_block_count()  # Try to extract parameter
            except Exception as e:
                errors_caught.append(('GGUFMetadataReader', type(e), str(e)))

            # Test ModelAutoConfigurator (higher-level interface)
            try:
                configurator = ModelAutoConfigurator()
                configurator.auto_configure(test_path)
            except Exception as e:
                errors_caught.append(('ModelAutoConfigurator', type(e), str(e)))

            # Test LanguageModel.from_gguf (end-user interface)
            try:
                model = LanguageModel.from_gguf(test_path)
            except Exception as e:
                errors_caught.append(('LanguageModel.from_gguf', type(e), str(e)))

            # Should have caught errors from all components
            assert len(errors_caught) >= 2, f"Should catch errors from multiple components, got: {errors_caught}"

            # All error types should be in the same category
            error_types = [error_type for _, error_type, _ in errors_caught]

            # Should all be appropriate exception types (RuntimeError, ValueError, FileNotFoundError, etc.)
            for component, error_type, message in errors_caught:
                assert issubclass(error_type, (RuntimeError, ValueError, FileNotFoundError, OSError)), \
                    f"Component {component} should raise standard exception type, got {error_type}: {message}"

            # Error messages should be meaningful and contain relevant information
            for component, error_type, message in errors_caught:
                message_lower = message.lower()
                if scenario_type == 'nonexistent_file':
                    assert any(keyword in message_lower for keyword in ['file', 'not found', 'open']), \
                        f"Error message for {component} should indicate file issue: {message}"
                elif 'invalid' in scenario_type or 'corrupted' in scenario_type:
                    assert any(keyword in message_lower for keyword in [
                        'invalid', 'corrupt', 'format', 'parse', 'gguf', 'open'
                    ]), \
                        f"Error message for {component} should indicate format issue: {message}"

        finally:
            if reader is not None:
                reader.close()
            if scenario_type != 'nonexistent_file' and os.path.exists(test_path):
                os.unlink(test_path)

    @given(error_contexts())
    @settings(max_examples=30, deadline=None)
    def test_error_information_preservation(self, context):
        """
        **Validates: Requirements 1.7, 1.8, 2.3, 2.5, 2.6, 8.1, 8.2, 8.3, 8.4**

        Property: Error information SHALL be preserved as it propagates up
        through the component stack (C++ → C API → Python bindings).
        """
        note(f"Testing error preservation in context: {context}")

        # Create a file with a specific error condition
        if context == 'metadata_reader_init':
            # File with invalid magic number
            test_data = b'BADMAGIC' + b'\x00' * 100
        elif context == 'parameter_extraction':
            # Valid GGUF header but corrupted metadata
            test_data = GGUF_MAGIC + b'\x03\x00\x00\x00'  # Valid header
            test_data += b'\x00\x00\x00\x00\x00\x00\x00\x00'  # Tensor count (0)
            test_data += b'\x01\x00\x00\x00\x00\x00\x00\x00'  # Metadata count (1)
            test_data += b'\xFF' * 50  # Corrupted metadata
        else:
            # Empty file
            test_data = b''

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(test_data)
            f.flush()
            test_path = f.name

        try:
            # Test at different levels to ensure error information is preserved
            low_level_error = None
            high_level_error = None
            reader = None

            # Low-level access (GGUFMetadataReader)
            try:
                reader = GGUFMetadataReader(test_path)
            except Exception as e:
                low_level_error = e

            # High-level access (ModelAutoConfigurator)
            try:
                configurator = ModelAutoConfigurator()
                config = configurator.auto_configure(test_path)
            except Exception as e:
                high_level_error = e

            # Both should have failed with related errors
            assert low_level_error is not None, "Low-level component should have failed"
            assert high_level_error is not None, "High-level component should have failed"

            # Error messages should contain overlapping information
            low_msg = str(low_level_error).lower()
            high_msg = str(high_level_error).lower()

            # At least one significant keyword should be preserved across levels
            common_keywords = []
            for word in low_msg.split():
                if len(word) > 3 and word in high_msg:  # Significant words
                    common_keywords.append(word)

            # Should have some preserved error context
            assert len(common_keywords) > 0 or any(
                keyword in low_msg and keyword in high_msg
                for keyword in ['invalid', 'corrupt', 'file', 'format', 'parse', 'error']
            ), f"Error information should be preserved: low='{low_level_error}', high='{high_level_error}'"

        finally:
            if reader is not None:
                reader.close()
            os.unlink(test_path)

    def test_concurrent_error_handling_consistency(self):
        """
        **Validates: Requirements 1.7, 1.8, 2.3, 2.5, 2.6, 8.1, 8.2, 8.3, 8.4**

        Property: Error handling SHALL be consistent when multiple threads
        encounter errors simultaneously (thread safety of error handling).
        """
        # Create an invalid file that will cause consistent errors
        test_data = b'INVALID' + b'\x00' * 50

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(test_data)
            f.flush()
            test_path = f.name

        errors = []

        def worker():
            reader = None
            try:
                reader = GGUFMetadataReader(test_path)
            except Exception as e:
                errors.append((type(e), str(e)))
            finally:
                if reader is not None:
                    reader.close()

        try:
            # Start multiple threads that will all fail
            threads = []
            for i in range(5):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            # Wait for all to complete
            for t in threads:
                t.join()

            # All should have failed with consistent error types and messages
            assert len(errors) == 5, f"Expected 5 errors, got {len(errors)}"

            base_error_type, base_message = errors[0]
            for i, (error_type, message) in enumerate(errors[1:], 1):
                assert error_type == base_error_type, \
                    f"Error type inconsistent at thread {i}: expected {base_error_type}, got {error_type}"

                # Messages should be similar (same error condition)
                # Allow for minor variations but core content should match
                assert len(message) > 0, f"Error message should not be empty at thread {i}"

        finally:
            os.unlink(test_path)

    def test_error_recovery_mechanisms(self):
        """
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

        Property: Error recovery mechanisms SHALL provide consistent
        graceful degradation across all component levels.
        """
        # Test various recovery scenarios
        recovery_scenarios = [
            ('missing_optional_params', {
                'llama.block_count': 16,
                'llama.embedding_length': 1024,
                'llama.attention.head_count': 16,
                'llama.feed_forward_length': 4096,
                'llama.vocab_size': 32000,
                # Missing rope_freq_base and layer_norm_epsilon (should use defaults)
            }),
            ('partial_corruption', {
                'llama.block_count': 'invalid_string',  # Should be recoverable
                'llama.embedding_length': 1024,
                'llama.attention.head_count': 16,
                'llama.feed_forward_length': 4096,
                'llama.vocab_size': 32000,
                'llama.rope.freq_base': 10000.0,
            }),
        ]

        for scenario_name, metadata in recovery_scenarios:
            # Create GGUF file with the specified metadata
            data = GGUF_MAGIC + b'\x03\x00\x00\x00'  # Valid header
            data += b'\x00\x00\x00\x00\x00\x00\x00\x00'  # Tensor count (0)
            data += struct.pack('<Q', len(metadata))  # Metadata count

            for key, value in metadata.items():
                key_bytes = key.encode('utf-8')
                value_bytes = str(value).encode('utf-8')

                data += struct.pack('<Q', len(key_bytes))
                data += key_bytes
                data += struct.pack('<I', 8)  # String type
                data += struct.pack('<Q', len(value_bytes))
                data += value_bytes

            # Add alignment
            pos = len(data)
            pad = (32 - (pos % 32)) % 32
            data += b'\0' * pad

            with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                f.write(data)
                f.flush()
                test_path = f.name

            try:
                if scenario_name == 'missing_optional_params':
                    # Should succeed with defaults applied
                    configurator = ModelAutoConfigurator()
                    config = configurator.auto_configure(test_path)

                    # Should have applied defaults for missing parameters
                    assert config.get('rope_theta') == 10000.0, "Should apply default rope_theta"
                    assert config.get('norm_eps') == 1e-5, "Should apply default norm_eps"

                elif scenario_name == 'partial_corruption':
                    # Should either recover gracefully or fail consistently
                    configurator = ModelAutoConfigurator()

                    try:
                        config = configurator.auto_configure(test_path)
                        # If it succeeds, should have valid recovered values
                        assert config.get('embedding_length') == 1024, "Should recover valid parameters"
                        assert config.get('head_count') == 16, "Should recover valid parameters"
                    except (ValueError, RuntimeError, OSError) as e:
                        # If it fails, should give meaningful error about the corruption
                        error_msg = str(e).lower()
                        assert any(keyword in error_msg for keyword in ['invalid', 'parse', 'corrupt']), \
                            f"Should indicate parsing issue: {e}"

            finally:
                os.unlink(test_path)

    def test_structured_error_information(self):
        """
        **Validates: Requirements 1.7, 1.8, 2.5, 2.6, 8.1, 8.4**

        Property: Structured error information SHALL be consistently
        available across all error scenarios and component levels.
        """
        error_categories = [
            ('file_access', '/nonexistent/test.gguf'),
            ('format_validation', b'INVALID_FORMAT'),
            ('parameter_validation', 'valid_gguf_invalid_params'),
        ]

        for category, test_input in error_categories:
            if category == 'file_access':
                test_path = test_input
            elif category == 'format_validation':
                with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                    f.write(test_input)
                    f.flush()
                    test_path = f.name
            else:  # parameter_validation
                # Create valid GGUF with invalid parameter values
                metadata = {
                    'llama.block_count': -5,  # Invalid
                    'llama.embedding_length': 0,  # Invalid
                    'llama.attention.head_count': 16,
                    'llama.feed_forward_length': 4096,
                    'llama.vocab_size': 32000,
                }

                data = GGUF_MAGIC + b'\x03\x00\x00\x00'
                data += b'\x00\x00\x00\x00\x00\x00\x00\x00'
                data += struct.pack('<Q', len(metadata))

                for key, value in metadata.items():
                    key_bytes = key.encode('utf-8')
                    value_bytes = str(value).encode('utf-8')

                    data += struct.pack('<Q', len(key_bytes))
                    data += key_bytes
                    data += struct.pack('<I', 8)
                    data += struct.pack('<Q', len(value_bytes))
                    data += value_bytes

                pos = len(data)
                pad = (32 - (pos % 32)) % 32
                data += b'\0' * pad

                with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
                    f.write(data)
                    f.flush()
                    test_path = f.name

            try:
                # Test that errors contain structured information
                components_tested = []
                reader = None

                try:
                    reader = GGUFMetadataReader(test_path)
                    if category == 'parameter_validation':
                        reader.get_block_count()  # Should fail on validation
                    components_tested.append('reader_success')
                except Exception as e:
                    components_tested.append(('GGUFMetadataReader', type(e).__name__, str(e)))

                try:
                    configurator = ModelAutoConfigurator()
                    config = configurator.auto_configure(test_path)
                    components_tested.append('configurator_success')
                except Exception as e:
                    components_tested.append(('ModelAutoConfigurator', type(e).__name__, str(e)))

                # Should have caught structured errors
                error_components = [c for c in components_tested if isinstance(c, tuple)]
                assert len(error_components) > 0, f"Should have caught errors for {category}"

                # Each error should have meaningful structured information
                for component, error_type, error_message in error_components:
                    assert len(error_message) > 0, f"Error message should not be empty for {component}"
                    assert error_type in [
                        'RuntimeError', 'ValueError', 'FileNotFoundError', 'OSError',
                        'GGUFFileError', 'GGUFMetadataError', 'GGUFValidationError',
                    ], \
                        f"Error type should be appropriate for {component}: {error_type}"

                    # Error message should contain category-appropriate information
                    msg_lower = error_message.lower()
                    if category == 'file_access':
                        assert any(kw in msg_lower for kw in ['file', 'not found', 'open']), \
                            f"File access error should mention file issue: {error_message}"
                    elif category == 'format_validation':
                        assert any(kw in msg_lower for kw in ['format', 'invalid', 'magic', 'gguf']), \
                            f"Format error should mention format issue: {error_message}"
                    elif category == 'parameter_validation':
                        assert any(kw in msg_lower for kw in ['parameter', 'invalid', 'value', 'range']), \
                            f"Parameter error should mention parameter issue: {error_message}"

            finally:
                if reader is not None:
                    reader.close()
                if category != 'file_access' and os.path.exists(test_path):
                    os.unlink(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
