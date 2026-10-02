"""
VGRE Model Auto-Configuration - Automatically configure LanguageModel from GGUF metadata.

This module implements the ModelAutoConfigurator class that creates LanguageModel
instances automatically from GGUF files without requiring manual JSON sidecars.
It extracts all necessary parameters from GGUF metadata and applies sensible defaults
for missing values.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union

from .lm import LanguageModel, GGUFMetadataReader


@dataclass
class GGUFMetadataCache:
    """Cached GGUF metadata with freshness validation."""

    path: str
    file_mtime: float  # File modification time
    file_size: int     # File size for additional validation
    parameters: Dict[str, Any]  # Extracted parameters
    created_at: float  # Cache creation timestamp

    def is_fresh(self) -> bool:
        """Check if cache is still valid against file system."""
        try:
            stat = os.stat(self.path)
            return (stat.st_mtime == self.file_mtime and
                    stat.st_size == self.file_size)
        except OSError:
            return False  # File no longer exists

    @classmethod
    def create(cls, path: str, parameters: Dict[str, Any]) -> 'GGUFMetadataCache':
        """Create cache entry for the given file."""
        stat = os.stat(path)
        return cls(
            path=path,
            file_mtime=stat.st_mtime,
            file_size=stat.st_size,
            parameters=parameters,
            created_at=time.time()
        )


class GGUFAutoConfigError(Exception):
    """Base exception for GGUF auto-configuration errors."""
    pass


class GGUFFileError(GGUFAutoConfigError, OSError):
    """GGUF file access or format errors."""
    pass


class GGUFMetadataError(GGUFAutoConfigError):
    """Missing or invalid metadata errors."""

    def __init__(self, message: str, missing_params: List[str] = None):
        super().__init__(message)
        self.missing_params = missing_params or []


class GGUFValidationError(GGUFAutoConfigError):
    """Parameter validation errors."""

    def __init__(self, message: str, invalid_params: Dict[str, str] = None):
        super().__init__(message)
        self.invalid_params = invalid_params or {}


class ModelAutoConfigurator:
    """Automatically configures LanguageModel from GGUF metadata."""

    PARAMETER_ALIASES = {
        'n_layer': ('block_count',),
        'd_model': ('embedding_length',),
        'n_head': ('head_count',),
        'n_kv_head': ('head_count_kv',),
        'd_ff': ('feed_forward_length',),
        'vocab': ('vocab_size',),
        'rope_theta': ('rope_freq_base',),
        'norm_eps': ('layer_norm_epsilon',),
        'max_seq': ('context_length',),
    }

    # Default values for missing parameters
    DEFAULT_VALUES = {
        'rope_theta': 10000.0,  # Traditional RoPE frequency
        'norm_eps': 1e-5,       # Standard layer norm epsilon
        'max_seq': 2048,        # Conservative initial KV-cache allocation
        'dropout': 0.0,         # No dropout for inference
        'tie_embeddings': False, # Most models don't tie embeddings
        'attn_bias': False,     # Most models don't use attention bias
        'seed': 1234,           # Default seed
    }

    PARAMETER_VALIDATION = {
        'vocab': (1, 1000000),           # Reasonable vocab size range
        'n_layer': (1, 128),             # Reasonable layer count
        'd_model': (64, 16384),          # Reasonable embedding dimensions
        'n_head': (1, 256),              # Reasonable head count
        'n_kv_head': (1, 256),           # Reasonable KV head count
        'rope_theta': (1.0, 1000000.0),  # Reasonable frequency base
        'norm_eps': (1e-15, 1e-1),      # Positive values representable by the C float API
        'd_ff': (64, 65536),             # Reasonable FF dimension
        'max_seq': (1, 131072),          # Runtime sequence limit
        'context_length': (1, 131072),
    }

    # Cache for metadata to avoid repeated parsing
    _metadata_cache: Dict[str, GGUFMetadataCache] = {}

    @classmethod
    def from_gguf(cls, gguf_path: str,
                  json_sidecar_path: Optional[str] = None) -> LanguageModel:
        """Create LanguageModel from GGUF file with optional JSON override.

        Args:
            gguf_path: Path to the GGUF file
            json_sidecar_path: Optional path to JSON sidecar (overrides GGUF metadata)

        Returns:
            Configured LanguageModel instance with weights loaded

        Raises:
            GGUFFileError: If GGUF file cannot be read
            GGUFMetadataError: If required metadata is missing
            GGUFValidationError: If parameters are invalid
        """

        params = cls.auto_configure(gguf_path, json_sidecar_path)

        # Step 5: Create LanguageModel with enhanced parameters
        model = cls._create_language_model(params)
        model._gguf_chat_template = params.get('chat_template')

        # Step 6: Load weights from GGUF
        try:
            model.load_gguf(gguf_path)
        except Exception as e:
            model.close()
            raise GGUFFileError(f"Failed to load GGUF weights from {gguf_path}: {e}")

        return model

    @classmethod
    def auto_configure(cls, gguf_path: str,
                       json_sidecar_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract, merge, and validate model parameters without loading weights.

        A neighboring ``<model>.json`` or ``<model>.gguf.json`` sidecar overrides
        GGUF metadata. An explicit sidecar path takes precedence over both.
        """
        try:
            gguf_params = cls._extract_gguf_metadata_cached(gguf_path)
        except OSError:
            raise
        except Exception as e:
            raise GGUFFileError(f"Failed to read GGUF metadata from {gguf_path}: {e}")

        sidecar_path = json_sidecar_path
        if sidecar_path is None:
            candidates = [gguf_path + ".json", os.path.splitext(gguf_path)[0] + ".json"]
            sidecar_path = next((path for path in candidates if os.path.isfile(path)), None)

        json_params = {}
        if sidecar_path is not None and os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path, encoding="utf-8") as sidecar:
                    json_params = json.load(sidecar)
            except Exception as e:
                raise GGUFFileError(f"Failed to parse JSON sidecar {sidecar_path}: {e}")

        params = cls._merge_parameters(
            cls._canonicalize_parameters(gguf_params),
            cls._canonicalize_parameters(json_params),
        )
        architecture = params.get('architecture')
        if architecture and architecture not in ('llama', 'qwen2'):
            raise GGUFMetadataError(
                f"GGUF architecture '{architecture}' is not supported by the current tensor loader. "
                "Supported architectures are llama and qwen2."
            )
        params = cls._validate_and_default(params)
        return cls._add_parameter_aliases(params)

    @classmethod
    def _extract_gguf_metadata_cached(cls, gguf_path: str) -> Dict[str, Any]:
        """Extract GGUF metadata with caching for performance."""

        # Check cache first
        if gguf_path in cls._metadata_cache:
            cached = cls._metadata_cache[gguf_path]
            if cached.is_fresh():
                return cached.parameters.copy()
            else:
                # Remove stale cache entry
                del cls._metadata_cache[gguf_path]

        # Extract metadata
        try:
            with GGUFMetadataReader(gguf_path) as reader:
                parameters = reader.get_model_parameters()
        except OSError:
            raise
        except Exception as e:
            raise GGUFFileError(f"Failed to read GGUF metadata: {e}")

        # Cache the result
        try:
            cls._metadata_cache[gguf_path] = GGUFMetadataCache.create(gguf_path, parameters)
        except OSError:
            # If we can't stat the file for caching, just proceed without caching
            pass

        return parameters

    @classmethod
    def _merge_parameters(cls, gguf_params: Dict, json_params: Dict) -> Dict:
        """Merge GGUF and JSON parameters, with JSON taking precedence."""
        merged = gguf_params.copy()
        merged.update(json_params)
        return merged

    @classmethod
    def _canonicalize_parameters(cls, params: Dict) -> Dict:
        """Map GGUF and historical JSON names to LanguageModel field names."""
        normalized = params.copy()
        for canonical, aliases in cls.PARAMETER_ALIASES.items():
            if canonical not in params or params[canonical] is None:
                for alias in aliases:
                    if alias in params and params[alias] is not None:
                        normalized[canonical] = params[alias]
                        break
        return normalized

    @classmethod
    def _add_parameter_aliases(cls, params: Dict) -> Dict:
        """Expose familiar GGUF names alongside canonical model fields."""
        result = params.copy()
        for canonical, aliases in cls.PARAMETER_ALIASES.items():
            if canonical in params:
                for alias in aliases:
                    result[alias] = params[canonical]
        return result

    @classmethod
    def _validate_and_default(cls, params: Dict) -> Dict:
        """Validate parameters and apply sensible defaults."""
        # Check required parameters
        required = ['n_layer', 'd_model', 'n_head', 'd_ff', 'vocab']
        missing = [p for p in required if p not in params or params[p] is None]
        if missing:
            raise GGUFMetadataError(
                f"Missing required parameters: {missing}. "
                f"Available parameters: {list(params.keys())}",
                missing_params=missing
            )

        # Validate ranges
        invalid_params = {}
        for param, value in params.items():
            if param in cls.PARAMETER_VALIDATION and value is not None:
                min_val, max_val = cls.PARAMETER_VALIDATION[param]
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value)):
                    invalid_params[param] = f"{value!r} is not a finite number"
                elif not (min_val <= value <= max_val):
                    invalid_params[param] = f"{value} outside range [{min_val}, {max_val}]"

        if invalid_params:
            raise GGUFValidationError(
                f"Parameter validation failed: {invalid_params}",
                invalid_params=invalid_params
            )

        if 'context_length' in params and ('max_seq' not in params or params['max_seq'] is None):
            params['max_seq'] = min(params['context_length'], cls.DEFAULT_VALUES['max_seq'])

        # Apply defaults
        for param, default in cls.DEFAULT_VALUES.items():
            if param not in params or params[param] is None:
                params[param] = default

        # Ensure n_kv_head is valid
        if 'n_kv_head' not in params or params['n_kv_head'] is None:
            # Default to n_head (standard multi-head attention)
            params['n_kv_head'] = params['n_head']
        elif (params['n_kv_head'] > params['n_head'] or
              params['n_head'] % params['n_kv_head'] != 0):
            raise GGUFValidationError(
                f"n_kv_head ({params['n_kv_head']}) must divide n_head ({params['n_head']})"
            )

        if params['d_model'] % params['n_head'] != 0:
            raise GGUFValidationError(
                f"d_model ({params['d_model']}) must be divisible by n_head ({params['n_head']})"
            )

        # Ensure head dimensions are valid
        head_dim = params['d_model'] // params['n_head']
        if head_dim <= 0 or head_dim % 2 != 0:
            raise GGUFValidationError(
                f"Invalid head dimension {head_dim} (d_model={params['d_model']}, n_head={params['n_head']}). "
                "RoPE requires a positive even head dimension"
            )

        return params

    @classmethod
    def _create_language_model(cls, params: Dict[str, Any]) -> LanguageModel:
        """Create LanguageModel from validated parameters."""

        # Create model with all parameters
        model = LanguageModel(
            vocab=params['vocab'],
            n_layer=params['n_layer'],
            d_model=params['d_model'],
            n_head=params['n_head'],
            n_kv_head=params['n_kv_head'],
            d_ff=params['d_ff'],
            max_seq=params['max_seq'],
            dropout=params['dropout'],
            tie_embeddings=params['tie_embeddings'],
            seed=params['seed'],
            attn_bias=params['attn_bias'],
            rope_theta=params['rope_theta'],
            norm_eps=params['norm_eps']
        )

        return model


class ErrorRecovery:
    """Handles common error scenarios with automatic recovery."""

    @staticmethod
    def handle_missing_metadata(gguf_path: str, missing_params: List[str]) -> None:
        """Reject incomplete configs instead of guessing model dimensions."""
        raise GGUFMetadataError(
            f"Cannot infer exact model dimensions for {gguf_path}; missing {missing_params}. "
            "Use a GGUF with complete metadata or provide an explicit JSON sidecar.",
            missing_params=missing_params,
        )

    @staticmethod
    def suggest_fixes(error: GGUFAutoConfigError) -> List[str]:
        """Provide actionable suggestions for common errors."""

        suggestions = []

        if isinstance(error, GGUFFileError):
            suggestions.extend([
                "Check that the file path is correct",
                "Verify the file is a valid GGUF format",
                "Ensure you have read permissions for the file",
            ])

        elif isinstance(error, GGUFMetadataError):
            if error.missing_params:
                suggestions.append(f"Consider creating a JSON sidecar with missing parameters: {error.missing_params}")
                suggestions.append("Check if this is a non-standard GGUF variant")

        elif isinstance(error, GGUFValidationError):
            suggestions.extend([
                "Verify the model dimensions match expected ranges",
                "Check if parameters need adjustment for your use case",
            ])

        return suggestions


def format_user_error(error: Exception, gguf_path: str) -> str:
    """Format technical errors into user-friendly messages."""

    if isinstance(error, GGUFFileError):
        return f"""
Failed to open GGUF file: {gguf_path}

{error}

Possible solutions:
• Check that the file exists and is readable
• Verify this is a valid GGUF file (not corrupted)
• Try re-downloading the model file
        """.strip()

    elif isinstance(error, GGUFMetadataError):
        missing = ", ".join(error.missing_params) if error.missing_params else "unknown parameters"
        return f"""
Model auto-configuration failed - missing required metadata: {missing}

This GGUF file doesn't contain all the information needed for automatic setup.
You can create a JSON sidecar file to provide the missing information:

    echo '{{"n_layer": 32, "d_model": 4096, "n_head": 32}}' > {gguf_path}.json
    # Then use: vgre.ModelAutoConfigurator.from_gguf("{gguf_path}")

Common parameter names: n_layer, d_model, n_head, n_kv_head, d_ff, vocab
        """.strip()

    elif isinstance(error, GGUFValidationError):
        return f"""
Model configuration validation failed:

{error}

The extracted parameters don't look reasonable for a language model.
Check if this is the correct model file or if it needs different parameters.
        """.strip()

    else:
        return f"Unexpected error: {error}"
