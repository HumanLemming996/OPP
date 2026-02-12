"""Tests for OPP signature generation."""

import numpy as np
from PIL import Image, ImageDraw

import pytest


def _make_test_image(width=256, height=256, color=(100, 150, 200)):
    """Create a simple test image with some features."""
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, width - 50, height - 50], fill=(200, 100, 50))
    draw.rectangle([80, 80, width - 80, height - 80], outline=(255, 255, 255))
    return img


class TestSignatureGeneration:
    """Test the 3-layer signature generation."""

    def test_generates_all_layers(self):
        """Signature should have L1, L2, and L3 layers."""
        from opp.signature import generate_signature

        img = _make_test_image()
        sig = generate_signature(img)

        assert sig.l1_exact.algorithm == "sha256"
        assert sig.l2_perceptual.algorithm == "pdq"
        assert "clip" in sig.l3_semantic.algorithm

    def test_sha256_deterministic(self):
        """Same image should produce same SHA-256 hash."""
        from opp.signature import generate_signature

        img = _make_test_image()
        sig1 = generate_signature(img)
        sig2 = generate_signature(img)

        assert sig1.l1_exact.value == sig2.l1_exact.value

    def test_pdq_hash_length(self):
        """PDQ hash should be 64 hex chars (256 bits)."""
        from opp.signature import generate_signature

        img = _make_test_image()
        sig = generate_signature(img)

        assert len(sig.l2_perceptual.value) == 64

    def test_clip_embedding_decodable(self):
        """CLIP embedding should decode to 768-dim float32 vector."""
        from opp.signature import generate_signature, clip_embedding_to_numpy

        img = _make_test_image()
        sig = generate_signature(img)

        emb = clip_embedding_to_numpy(sig.l3_semantic.value)
        assert emb.shape == (768,)
        assert emb.dtype == np.float32

    def test_clip_embedding_normalized(self):
        """CLIP embedding should be L2-normalized."""
        from opp.signature import generate_signature, clip_embedding_to_numpy

        img = _make_test_image()
        sig = generate_signature(img)

        emb = clip_embedding_to_numpy(sig.l3_semantic.value)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01

    def test_image_dimensions_stored(self):
        """Signature should record original image dimensions."""
        from opp.signature import generate_signature

        img = _make_test_image(300, 200)
        sig = generate_signature(img)

        assert sig.image_width == 300
        assert sig.image_height == 200

    def test_protocol_version(self):
        """Signature should include protocol version."""
        from opp.signature import generate_signature

        img = _make_test_image()
        sig = generate_signature(img)

        assert sig.protocol_version == "OPP/1.0"
