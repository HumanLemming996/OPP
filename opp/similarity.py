"""
OPP Similarity Engine — Tiered matching pipeline.

Match flow (fast → accurate):
1. SHA-256 exact match           → O(1), instant
2. PDQ hamming distance filter   → narrows billions to thousands
3. CLIP cosine similarity rank   → accurate final ordering

This module provides the scoring and classification logic.
The actual database queries happen in storage.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from opp.config import OPPConfig, get_config
from opp.models import MatchLevel, OPPSignature
from opp.signature import clip_embedding_to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDQ Hamming Distance
# ---------------------------------------------------------------------------

def pdq_hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert PDQ hex string to a 256-element boolean array."""
    raw_bytes = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))


def pdq_hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute hamming distance between two PDQ hashes.

    Returns the number of differing bits (0-256).
    Lower = more similar.
    """
    bits_a = pdq_hex_to_bits(hash_a)
    bits_b = pdq_hex_to_bits(hash_b)
    return int(np.sum(bits_a != bits_b))


# ---------------------------------------------------------------------------
# CLIP Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Both vectors should already be L2-normalized (which our signature
    generator does), so this simplifies to a dot product.
    """
    # Ensure normalized
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb_a / norm_a, emb_b / norm_b))


def cosine_similarity_from_base64(emb_a_b64: str, emb_b_b64: str) -> float:
    """Compute cosine similarity between two base64-encoded CLIP embeddings."""
    vec_a = clip_embedding_to_numpy(emb_a_b64)
    vec_b = clip_embedding_to_numpy(emb_b_b64)
    return cosine_similarity(vec_a, vec_b)


# ---------------------------------------------------------------------------
# Match Classification
# ---------------------------------------------------------------------------

def classify_match(
    cosine_sim: float,
    config: Optional[OPPConfig] = None,
) -> MatchLevel:
    """Classify a cosine similarity score into a match level."""
    config = config or get_config()

    if cosine_sim >= config.threshold_exact:
        return MatchLevel.EXACT
    elif cosine_sim >= config.threshold_likely:
        return MatchLevel.LIKELY
    elif cosine_sim >= config.threshold_possible:
        return MatchLevel.POSSIBLE
    else:
        return MatchLevel.NONE


# ---------------------------------------------------------------------------
# Full Matching Pipeline
# ---------------------------------------------------------------------------

def compute_match_score(
    query_sig: OPPSignature,
    candidate_sig: OPPSignature,
    config: Optional[OPPConfig] = None,
) -> dict:
    """Compute full match score between query and candidate signatures.

    Returns a dict with:
        - exact_match: bool (SHA-256)
        - pdq_hamming_distance: int (0-256)
        - cosine_similarity: float (0-1)
        - match_level: MatchLevel
    """
    config = config or get_config()

    # L1: Exact match check
    exact = query_sig.l1_exact.value == candidate_sig.l1_exact.value

    # L2: PDQ hamming distance
    pdq_dist = pdq_hamming_distance(
        query_sig.l2_perceptual.value,
        candidate_sig.l2_perceptual.value,
    )

    # L3: CLIP cosine similarity
    cos_sim = cosine_similarity_from_base64(
        query_sig.l3_semantic.value,
        candidate_sig.l3_semantic.value,
    )

    level = classify_match(cos_sim, config)

    return {
        "exact_match": exact,
        "pdq_hamming_distance": pdq_dist,
        "cosine_similarity": cos_sim,
        "match_level": level,
    }
