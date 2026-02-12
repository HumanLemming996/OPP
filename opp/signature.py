"""
OPP Signature Generator — 3-layer image signature production.

Layer 1 (Exact):     SHA-256 cryptographic hash
Layer 2 (Perceptual): PDQ 256-bit perceptual hash (Meta)
Layer 3 (Semantic):   CLIP ViT-L/14 768-dim embedding (OpenAI)

Multi-resolution: generates signatures for full image + 4 quadrants.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pdqhash
import torch
from PIL import Image

from opp.config import OPPConfig, get_config
from opp.models import OPPSignature, QuadrantSignatures, SignatureLayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIP Model Management (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip_model(config: Optional[OPPConfig] = None):
    """Lazy-load the CLIP model. Cached for reuse."""
    global _clip_model, _clip_preprocess, _clip_device

    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device

    import open_clip

    config = config or get_config()
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Loading CLIP model {config.clip_model_name} "
        f"(pretrained={config.clip_pretrained}) on {_clip_device}"
    )

    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        config.clip_model_name,
        pretrained=config.clip_pretrained,
        device=_clip_device,
    )
    _clip_model.eval()

    logger.info("CLIP model loaded successfully")
    return _clip_model, _clip_preprocess, _clip_device


# ---------------------------------------------------------------------------
# Layer 1: SHA-256 Exact Hash
# ---------------------------------------------------------------------------

def _compute_sha256(image: Image.Image) -> str:
    """Compute SHA-256 hash of the raw image pixels (canonical form).

    We normalize to PNG bytes at original resolution to ensure
    deterministic hashing regardless of input format.
    """
    buf = io.BytesIO()
    # Convert to RGB to strip alpha and normalize
    image_rgb = image.convert("RGB")
    image_rgb.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# Layer 2: PDQ 256-bit Perceptual Hash
# ---------------------------------------------------------------------------

def _compute_pdq(image: Image.Image) -> str:
    """Compute PDQ perceptual hash (256-bit) using Meta's algorithm.

    Returns the hash as a 64-character hex string.
    """
    # pdqhash expects a numpy array
    img_array = np.array(image.convert("RGB"))
    hash_vector, quality = pdqhash.compute(img_array)
    # hash_vector is a numpy array of 256 bools, pack to hex
    # Convert bool array to bytes then to hex
    hash_bits = np.packbits(hash_vector)
    return hash_bits.tobytes().hex()


# ---------------------------------------------------------------------------
# Layer 3: CLIP ViT-L/14 Embedding
# ---------------------------------------------------------------------------

def _compute_clip_embedding(image: Image.Image, config: Optional[OPPConfig] = None) -> str:
    """Compute CLIP ViT-L/14 768-dim embedding.

    Returns the embedding as a base64-encoded float32 numpy array.
    """
    model, preprocess, device = _get_clip_model(config)

    image_rgb = image.convert("RGB")
    image_tensor = preprocess(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device if device != "cpu" else "cpu"):
        embedding = model.encode_image(image_tensor)

    # Normalize to unit vector (important for cosine similarity)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    # Convert to numpy, then base64
    emb_np = embedding.cpu().float().numpy().flatten()
    emb_bytes = emb_np.tobytes()
    return base64.b64encode(emb_bytes).decode("ascii")


def clip_embedding_to_numpy(base64_str: str) -> np.ndarray:
    """Decode a base64-encoded CLIP embedding back to numpy array."""
    raw = base64.b64decode(base64_str)
    return np.frombuffer(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined Signature Generation
# ---------------------------------------------------------------------------

def generate_signature(
    image: Image.Image,
    config: Optional[OPPConfig] = None,
) -> OPPSignature:
    """Generate a complete 3-layer OPP signature for an image.

    Args:
        image: PIL Image to generate signature for.
        config: Optional OPP configuration.

    Returns:
        OPPSignature with all three layers populated.
    """
    config = config or get_config()

    logger.debug("Generating L1 (SHA-256)...")
    sha256_hash = _compute_sha256(image)

    logger.debug("Generating L2 (PDQ)...")
    pdq_hash = _compute_pdq(image)

    logger.debug("Generating L3 (CLIP ViT-L/14)...")
    clip_emb = _compute_clip_embedding(image, config)

    return OPPSignature(
        l1_exact=SignatureLayer(algorithm="sha256", version="1.0", value=sha256_hash),
        l2_perceptual=SignatureLayer(algorithm="pdq", version="1.0", value=pdq_hash),
        l3_semantic=SignatureLayer(
            algorithm=f"clip-{config.clip_model_name.lower()}",
            version="1.0",
            value=clip_emb,
        ),
        image_width=image.width,
        image_height=image.height,
    )


def generate_multi_resolution_signature(
    image: Image.Image,
    config: Optional[OPPConfig] = None,
) -> QuadrantSignatures:
    """Generate signatures for full image + 4 quadrants.

    This enables partial image matching — if someone crops and uses
    just a portion of an AI-generated image, we can still match it.

    Args:
        image: PIL Image to generate signatures for.
        config: Optional OPP configuration.

    Returns:
        QuadrantSignatures with full + 4 quadrant signatures.
    """
    config = config or get_config()

    logger.info("Generating full-image signature...")
    full_sig = generate_signature(image, config)

    if not config.enable_multi_resolution:
        return QuadrantSignatures(full=full_sig)

    w, h = image.size
    mid_w, mid_h = w // 2, h // 2

    quadrant_boxes = {
        "top_left": (0, 0, mid_w, mid_h),
        "top_right": (mid_w, 0, w, mid_h),
        "bottom_left": (0, mid_h, mid_w, h),
        "bottom_right": (mid_w, mid_h, w, h),
    }

    quadrant_sigs = {}
    for name, box in quadrant_boxes.items():
        logger.info(f"Generating {name} quadrant signature...")
        crop = image.crop(box)
        quadrant_sigs[name] = generate_signature(crop, config)

    return QuadrantSignatures(full=full_sig, **quadrant_sigs)


# ---------------------------------------------------------------------------
# Utility: Load image from various sources
# ---------------------------------------------------------------------------

def load_image_from_base64(b64_string: str) -> Image.Image:
    """Load a PIL Image from a base64-encoded string."""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def load_image_from_path(path: str) -> Image.Image:
    """Load a PIL Image from a file path."""
    return Image.open(path)


async def load_image_from_url(url: str) -> Image.Image:
    """Load a PIL Image from a URL (async)."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content))
