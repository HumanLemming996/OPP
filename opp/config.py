"""
OPP Configuration — Pydantic Settings for all tunable parameters.

All values have sensible defaults. Override via environment variables
prefixed with OPP_, e.g. OPP_QDRANT_URL=http://myhost:6333
"""

from pydantic_settings import BaseSettings


class OPPConfig(BaseSettings):
    """Central configuration for OPP server and tools."""

    model_config = {"env_prefix": "OPP_"}

    # --- CLIP Model ---
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "opp_signatures"
    qdrant_vector_size: int = 768  # ViT-L/14 output dim

    # --- SQLite ---
    sqlite_path: str = "opp_metadata.db"

    # --- Similarity Thresholds ---
    threshold_exact: float = 0.97
    threshold_likely: float = 0.90
    threshold_possible: float = 0.80

    # PDQ hamming distance threshold (out of 256 bits)
    # <= 31 bits different = likely same image
    pdq_hamming_threshold: int = 31

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Signature ---
    # Generate multi-resolution signatures (full + quadrants)
    enable_multi_resolution: bool = True


def get_config() -> OPPConfig:
    """Get the global OPP configuration."""
    return OPPConfig()
