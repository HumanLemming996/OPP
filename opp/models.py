"""
OPP Data Models — Pydantic v2 models for the protocol.

These define the wire format for mint/verify operations and the
internal signature data structures.
"""

from __future__ import annotations

import datetime
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MatchLevel(str, Enum):
    """Classification of how closely two signatures match."""
    EXACT = "exact"        # >0.97 cosine similarity
    LIKELY = "likely"      # >0.90
    POSSIBLE = "possible"  # >0.80
    NONE = "none"          # <=0.80


# ---------------------------------------------------------------------------
# Core Signature Models
# ---------------------------------------------------------------------------

class SignatureLayer(BaseModel):
    """A single layer of an OPP signature."""
    algorithm: str = Field(..., description="Algorithm identifier, e.g. 'sha256', 'pdq', 'clip-vit-l-14'")
    version: str = Field(default="1.0", description="Algorithm version for forward compat")
    value: str = Field(..., description="Hex-encoded hash or base64-encoded embedding vector")


class OPPSignature(BaseModel):
    """
    Complete OPP signature for an image.

    Contains three layers:
    - L1 (exact): SHA-256 hash for exact dedup
    - L2 (perceptual): PDQ 256-bit hash for near-duplicate filtering
    - L3 (semantic): CLIP ViT-L/14 768-dim embedding for robust matching

    Multi-resolution variants (quadrants) are stored separately.
    """
    protocol_version: str = Field(default="OPP/1.0", description="Protocol version")
    l1_exact: SignatureLayer = Field(..., description="SHA-256 exact hash")
    l2_perceptual: SignatureLayer = Field(..., description="PDQ perceptual hash")
    l3_semantic: SignatureLayer = Field(..., description="CLIP embedding vector")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")


class QuadrantSignatures(BaseModel):
    """Multi-resolution signatures for partial image matching."""
    full: OPPSignature
    top_left: Optional[OPPSignature] = None
    top_right: Optional[OPPSignature] = None
    bottom_left: Optional[OPPSignature] = None
    bottom_right: Optional[OPPSignature] = None


# ---------------------------------------------------------------------------
# Provenance Metadata
# ---------------------------------------------------------------------------

class ProvenanceMetadata(BaseModel):
    """Metadata about the AI generation of an image."""
    generator: str = Field(..., description="Generator name, e.g. 'openai-dall-e-3', 'midjourney-v6'")
    model_version: Optional[str] = Field(default=None, description="Specific model version")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="When the image was generated (UTC)"
    )
    tags: list[str] = Field(default_factory=list, description="Optional tags")
    extra: dict = Field(default_factory=dict, description="Additional metadata (flexible)")


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------

class MintRequest(BaseModel):
    """Request to mint a new provenance signature.

    Primary mode: generator submits a pre-computed signature.
    Generators compute the 3-layer signature on their own infrastructure
    using the open-source OPP library, then submit only the signature
    and metadata. No raw image data is transmitted.
    """
    signature: OPPSignature = Field(..., description="Pre-computed 3-layer OPP signature")
    metadata: ProvenanceMetadata


class MintResponse(BaseModel):
    """Response after successfully minting a signature."""
    signature_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "minted"
    protocol_version: str = "OPP/1.0"
    l1_hash: str = Field(..., description="SHA-256 hash for reference")
    l2_hash: str = Field(..., description="PDQ hash for reference")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class MatchResult(BaseModel):
    """A single match result from verification."""
    signature_id: str
    match_level: MatchLevel
    cosine_similarity: float = Field(..., ge=0.0, le=1.0)
    pdq_hamming_distance: Optional[int] = Field(default=None, ge=0, le=256)
    metadata: ProvenanceMetadata


class VerifyRequest(BaseModel):
    """Request to verify an image against the registry."""
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image data")
    image_url: Optional[str] = Field(default=None, description="URL to fetch the image from")
    max_results: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.80, ge=0.0, le=1.0)


class VerifyResponse(BaseModel):
    """Response with matching results."""
    query_l1_hash: str
    query_l2_hash: str
    matches: list[MatchResult]
    total_signatures_searched: int
    protocol_version: str = "OPP/1.0"


# ---------------------------------------------------------------------------
# Storage Record
# ---------------------------------------------------------------------------

class SignatureRecord(BaseModel):
    """Complete record stored in the database."""
    signature_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signature: OPPSignature
    quadrants: Optional[QuadrantSignatures] = None
    metadata: ProvenanceMetadata
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
