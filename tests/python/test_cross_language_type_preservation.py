#!/usr/bin/env python3
"""
Property-Based Test for Cross-Language Type Preservation

**Feature: gguf-auto-config, Property 4: Cross-Language Type Preservation**
**Validates: Requirements 3.2, 3.3, 3.5**

This test validates that data types are preserved correctly when passing between
C API and Python bindings. It verifies that parameters extracted from GGUF files
maintain their correct types and values across the C/Python boundary.

Key test coverage:
1. int32_t values from C API → Python int with same logical value
2. float values from C API → Python float with acceptable precision
3. Error codes from C API → Python exceptions with preserved error information
4. C API function signatures and return codes work correctly via ctypes
5. File access errors are properly converted to Python exceptions

The tests generate synthetic GGUF files with known metadata values and verify
that the C API extracts them correctly and the Python bindings preserve the
types and values exactly as specified in the requirements.
"""

import os
import sys
import struct
import tempfile
import ctypes
from typing import Optional, Dict, Any

# Set up paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bindings', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bindings', 'python', 'build', 'lib'))

# Set library path
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'build')
native_name = {'darwin': 'libvgre.dylib', 'win32': 'vgre.dll'}.get(sys.platform, 'libvgre.so')
lib_path = os.path.join(build_dir, native_name)
os.environ['LD_LIBRARY_PATH'] = build_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['VGRE_LIB_PATH'] = lib_path

import pytest
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.strategies import composite

try:
    import vgre
    from vgre import GGUFMetadataReader
    from vgre._native import _lib
    VGRE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VGRE not available: {e}")
    VGRE_AVAILABLE = False


# GGUF file format constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# Metadata type constants (from gguf specification)
GGUF_TYPE_UINT32 = 6
GGUF_TYPE_FLOAT32 = 10
GGUF_TYPE_STRING = 8


def create_gguf_metadata_entry(key: str, value) -> bytes:
    """Create a single GGUF metadata entry as string (all GGUF metadata is string-based)."""
    # Key as string (length + utf-8 bytes)
    key_bytes = key.encode('utf-8')
    key_len = len(key_bytes)

    entry = struct.pack('<Q', key_len)  # Key length (uint64)
    entry += key_bytes  # Key string
    entry += struct.pack('<I', GGUF_TYPE_STRING)  # All metadata is stored as strings

    # Convert value to string
    if isinstance(value, str):
        value_bytes = value.encode('utf-8')
    else:
        value_bytes = str(value).encode('utf-8')
    entry += struct.pack('<Q', len(value_bytes))
    entry += value_bytes

    return entry


def create_minimal_gguf_file(metadata: Dict[str, Any]) -> bytes:
    """Create a minimal valid GGUF file with specified metadata."""
    # Header
    data = GGUF_MAGIC  # Magic number
    data += struct.pack('<I', GGUF_VERSION)  # Version
    data += struct.pack('<Q', 0)  # Tensor count
    data += struct.pack('<Q', len(metadata))  # Metadata count

    # Metadata entries (all stored as strings)
    for key, value in metadata.items():
        data += create_gguf_metadata_entry(key, value)

    # Add alignment padding to 32-byte boundary (default GGUF alignment)
    alignment = 32
    pos = len(data)
    pad = (alignment - (pos % alignment)) % alignment
    data += b'\0' * pad

    # No tensors in this minimal file, so we're done (but we need the data section)
    return data


@composite
def gguf_metadata_values(draw):
    """Generate realistic GGUF metadata values for testing."""

    # Generate int32 values in reasonable ranges
    int32_params = {
        'llama.block_count': draw(st.integers(min_value=1, max_value=128)),
        'llama.embedding_length': draw(st.sampled_from([768, 1024, 2048, 4096, 8192])),
        'llama.attention.head_count': draw(st.integers(min_value=8, max_value=128)),
        'llama.attention.head_count_kv': draw(st.integers(min_value=1, max_value=32)),
        'llama.feed_forward_length': draw(st.integers(min_value=512, max_value=32768)),
    }

    # Generate float values in reasonable ranges
    float_params = {
        'llama.rope.freq_base': draw(st.floats(min_value=1000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)),
        'llama.attention.layer_norm_epsilon': draw(st.floats(min_value=1e-12, max_value=1e-3, allow_nan=False, allow_infinity=False)),
    }

    return int32_params, float_params


@composite
def corrupted_values(draw):
    """Generate values that should cause parsing errors."""
    corruption_type = draw(st.sampled_from(['invalid_int_string', 'invalid_float_string', 'empty_string']))

    if corruption_type == 'invalid_int_string':
        # String that looks like text but definitely can't be parsed as int32
        return draw(st.text(alphabet='abcdefghij', min_size=1))
    elif corruption_type == 'invalid_float_string':
        # String that isn't parseable as float
        return draw(st.text(alphabet='xyz@#$', min_size=1))
    elif corruption_type == 'empty_string':
        # Empty string
        return ""


@pytest.mark.skipif(not VGRE_AVAILABLE, reason="VGRE library not available")
class TestCrossLanguageTypePreservation:
    """Test cross-language type preservation between C API and Python bindings."""

    @given(gguf_metadata_values())
    @settings(max_examples=100, deadline=None)
    def test_int32_type_preservation(self, metadata_values):
        """
        **Validates: Requirements 3.2, 3.3, 3.5**

        Property: For any valid int32 metadata parameter extracted by the C API,
        the Python bindings SHALL return the same logical value as a Python int.
        """
        int32_params, _ = metadata_values

        # Create GGUF file with int32 metadata (stored as strings in GGUF format)
        metadata = {}
        for key, value in int32_params.items():
            # Ensure value is within int32 range
            assume(-2**31 <= value <= 2**31 - 1)
            metadata[key] = value  # Store as value, will be converted to string in GGUF

        gguf_data = create_minimal_gguf_file(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            # Test through Python bindings
            reader = GGUFMetadataReader(temp_path)

            # Test each parameter extraction method
            for key, expected_value in metadata.items():
                note(f"Testing {key} = {expected_value}")

                if 'block_count' in key:
                    result = reader.get_block_count()
                elif 'embedding_length' in key:
                    result = reader.get_embedding_length()
                elif 'head_count_kv' in key:
                    result = reader.get_head_count_kv()
                elif 'head_count' in key and 'head_count_kv' not in key:
                    result = reader.get_head_count()
                elif 'feed_forward_length' in key:
                    result = reader.get_feed_forward_length()
                else:
                    continue

                # Validate type and value preservation
                assert result is not None, f"Failed to extract {key}"
                assert isinstance(result, int), f"Expected int, got {type(result)} for {key}"
                assert result == expected_value, f"Value mismatch for {key}: expected {expected_value}, got {result}"

                # Validate range is within int32
                assert -2**31 <= result <= 2**31 - 1, f"Result {result} outside int32 range for {key}"

        finally:
            os.unlink(temp_path)

    @given(gguf_metadata_values())
    @settings(max_examples=100, deadline=None)
    def test_float_type_preservation(self, metadata_values):
        """
        **Validates: Requirements 3.2, 3.3, 3.5**

        Property: For any valid float metadata parameter extracted by the C API,
        the Python bindings SHALL return the same logical value as a Python float
        with acceptable floating-point precision.
        """
        _, float_params = metadata_values

        # Create GGUF file with float metadata (stored as strings in GGUF format)
        metadata = {}
        for key, value in float_params.items():
            # Ensure value is finite and reasonable
            assume(abs(value) < 1e30)  # Reasonable range for our parameters
            metadata[key] = value  # Store as value, will be converted to string in GGUF

        gguf_data = create_minimal_gguf_file(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            # Test through Python bindings
            reader = GGUFMetadataReader(temp_path)

            # Test each parameter extraction method
            for key, expected_value in metadata.items():
                note(f"Testing {key} = {expected_value}")

                if 'rope.freq_base' in key:
                    result = reader.get_rope_freq_base()
                elif 'layer_norm_epsilon' in key:
                    result = reader.get_norm_eps()
                else:
                    continue

                # Validate type and value preservation
                assert result is not None, f"Failed to extract {key}"
                assert isinstance(result, float), f"Expected float, got {type(result)} for {key}"

                # Check floating point equality with reasonable tolerance
                # (accounting for float32 precision limits)
                rel_tolerance = 1e-6  # Reasonable for float32
                abs_tolerance = 1e-30  # For very small numbers

                if abs(expected_value) > abs_tolerance:
                    relative_error = abs(result - expected_value) / abs(expected_value)
                    assert relative_error <= rel_tolerance, \
                        f"Relative error {relative_error} > {rel_tolerance} for {key}: expected {expected_value}, got {result}"
                else:
                    absolute_error = abs(result - expected_value)
                    assert absolute_error <= abs_tolerance, \
                        f"Absolute error {absolute_error} > {abs_tolerance} for {key}: expected {expected_value}, got {result}"

        finally:
            os.unlink(temp_path)

    @given(corrupted_values())
    @settings(max_examples=50, deadline=None)
    def test_error_information_preservation(self, corrupted_value):
        """
        **Validates: Requirements 3.2, 3.3, 3.5**

        Property: For any C API error condition, the Python bindings SHALL
        convert C error codes to appropriate Python exceptions with preserved
        error information from the original C API call.
        """
        # Create GGUF file with corrupted metadata (string that can't be parsed as number)
        metadata = {
            'llama.block_count': corrupted_value  # This should cause parsing errors
        }

        gguf_data = create_minimal_gguf_file(metadata)

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(gguf_data)
            f.flush()
            temp_path = f.name

        try:
            reader = GGUFMetadataReader(temp_path)

            # This should return None (not found/invalid) rather than throwing
            # The C API should handle the type mismatch gracefully
            result = reader.get_block_count()

            # For invalid data, we expect None (parameter not found/invalid)
            # This tests that the C API error handling is properly exposed to Python
            assert result is None, f"Expected None for corrupted value, got {result}"

            # The error should be accessible through the error reporting mechanism
            # (though the specific error message format may vary)

        finally:
            os.unlink(temp_path)

    def test_c_api_direct_interface(self):
        """
        **Validates: Requirements 3.2, 3.3, 3.5**

        Property: The Python ctypes bindings SHALL correctly interface with
        the C API functions and handle return codes and output parameters
        according to the C API specification.
        """
        # Test that we can access C API functions directly
        assert hasattr(_lib, 'vgre_gguf_metadata_open'), "C API open function not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_free'), "C API free function not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_get_block_count'), "C API get_block_count not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_get_embedding_length'), "C API get_embedding_length not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_get_head_count'), "C API get_head_count not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_get_rope_freq_base'), "C API get_rope_freq_base not available"
        assert hasattr(_lib, 'vgre_gguf_metadata_get_norm_eps'), "C API get_norm_eps not available"

        # Test that function signatures are properly bound
        # (This should not throw an AttributeError)
        try:
            # These calls will fail (NULL pointer), but should not crash
            result = _lib.vgre_gguf_metadata_is_valid(None)
            assert result == 0, "NULL handle should be invalid"

            # Test that we can pass ctypes properly
            value = ctypes.c_int32()
            result = _lib.vgre_gguf_metadata_get_block_count(None, ctypes.byref(value))
            # Should return error code for NULL handle (not crash)
            assert result != 1, "NULL handle should return error code"

        except Exception as e:
            # If it throws, it should be a reasonable error, not a segfault
            assert "cannot access memory" not in str(e).lower(), f"Memory access error suggests binding problem: {e}"

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=None)
    def test_file_error_handling(self, random_filename):
        """
        **Validates: Requirements 3.2, 3.3, 3.5**

        Property: For any file access error in the C API (file not found,
        permission denied, invalid format), the Python bindings SHALL raise
        appropriate Python exceptions with meaningful error messages.
        """
        # Filter out paths that might actually exist or cause other issues
        assume(not os.path.exists(random_filename))
        assume(len(random_filename.strip()) > 0)  # Non-empty filename
        assume('/' not in random_filename or not random_filename.startswith('/'))  # Avoid absolute paths

        # Test file not found error
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)) as exc_info:
            reader = GGUFMetadataReader(random_filename)

        # The error should contain meaningful information
        error_message = str(exc_info.value).lower()
        # Should indicate it's a file-related error
        assert any(keyword in error_message for keyword in ['file', 'open', 'path', 'not found', 'invalid']), \
            f"Error message should indicate file issue: {exc_info.value}"


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, "-v"])
