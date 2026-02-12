"""Tests for OPP similarity engine."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import pytest


def _make_test_image(width=256, height=256, seed=42):
    """Create a test image with deterministic patterns."""
    img = Image.new("RGB", (width, height), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, width - 50, height - 50], fill=(200, 100, 50))
    draw.rectangle([80, 80, width - 80, height - 80], outline=(255, 255, 255))
    return img


def _make_different_image(width=256, height=256):
    """Create a visually different test image."""
    img = Image.new("RGB", (width, height), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, width - 20, height - 20], fill=(200, 50, 50))
    return img


class TestPDQHamming:
    """Test PDQ hamming distance computation."""

    def test_identical_hashes(self):
        """Identical hashes should have distance 0."""
        from opp.similarity import pdq_hamming_distance

        hash_val = "a" * 64  # 256-bit hex string
        assert pdq_hamming_distance(hash_val, hash_val) == 0

    def test_completely_different(self):
        """Opposite hashes should have distance 256."""
        from opp.similarity import pdq_hamming_distance

        hash_a = "f" * 64
        hash_b = "0" * 64
        assert pdq_hamming_distance(hash_a, hash_b) == 256


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        from opp.similarity import cosine_similarity

        vec = np.random.randn(768).astype(np.float32)
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        from opp.similarity import cosine_similarity

        vec_a = np.zeros(768, dtype=np.float32)
        vec_b = np.zeros(768, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b[1] = 1.0
        assert abs(cosine_similarity(vec_a, vec_b)) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        from opp.similarity import cosine_similarity

        vec = np.random.randn(768).astype(np.float32)
        assert abs(cosine_similarity(vec, -vec) - (-1.0)) < 0.001


class TestMatchClassification:
    """Test match level classification."""

    def test_exact_match(self):
        """Score >= 0.97 should be EXACT."""
        from opp.similarity import classify_match
        from opp.models import MatchLevel

        assert classify_match(0.99) == MatchLevel.EXACT
        assert classify_match(0.97) == MatchLevel.EXACT

    def test_likely_match(self):
        """Score >= 0.90 and < 0.97 should be LIKELY."""
        from opp.similarity import classify_match
        from opp.models import MatchLevel

        assert classify_match(0.95) == MatchLevel.LIKELY
        assert classify_match(0.90) == MatchLevel.LIKELY

    def test_possible_match(self):
        """Score >= 0.80 and < 0.90 should be POSSIBLE."""
        from opp.similarity import classify_match
        from opp.models import MatchLevel

        assert classify_match(0.85) == MatchLevel.POSSIBLE
        assert classify_match(0.80) == MatchLevel.POSSIBLE

    def test_no_match(self):
        """Score < 0.80 should be NONE."""
        from opp.similarity import classify_match
        from opp.models import MatchLevel

        assert classify_match(0.79) == MatchLevel.NONE
        assert classify_match(0.5) == MatchLevel.NONE


class TestFullMatchPipeline:
    """Test the complete match scoring pipeline."""

    def test_identical_images_high_similarity(self):
        """Same image should produce very high similarity."""
        from opp.signature import generate_signature
        from opp.similarity import compute_match_score

        img = _make_test_image()
        sig = generate_signature(img)
        result = compute_match_score(sig, sig)

        assert result["exact_match"] is True
        assert result["cosine_similarity"] > 0.99
        assert result["pdq_hamming_distance"] == 0

    def test_resized_image_high_similarity(self):
        """Resized image should still match strongly."""
        from opp.signature import generate_signature
        from opp.similarity import compute_match_score

        img = _make_test_image(512, 512)
        resized = img.resize((256, 256))

        sig_orig = generate_signature(img)
        sig_resized = generate_signature(resized)
        result = compute_match_score(sig_orig, sig_resized)

        assert result["cosine_similarity"] > 0.85

    def test_compressed_image_high_similarity(self):
        """JPEG-compressed image should still match."""
        import io
        from opp.signature import generate_signature
        from opp.similarity import compute_match_score

        img = _make_test_image(512, 512)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        compressed = Image.open(buf)

        sig_orig = generate_signature(img)
        sig_compressed = generate_signature(compressed)
        result = compute_match_score(sig_orig, sig_compressed)

        assert result["cosine_similarity"] > 0.80

    def test_different_images_low_similarity(self):
        """Completely different images should have low similarity."""
        from opp.signature import generate_signature
        from opp.similarity import compute_match_score

        img1 = _make_test_image()
        img2 = _make_different_image()

        sig1 = generate_signature(img1)
        sig2 = generate_signature(img2)
        result = compute_match_score(sig1, sig2)

        assert result["exact_match"] is False
        assert result["cosine_similarity"] < 0.90
