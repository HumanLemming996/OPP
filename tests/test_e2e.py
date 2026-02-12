"""End-to-end tests for OPP mint → verify flow.

These tests require Qdrant to be running. Skip if not available.
Run with: pytest tests/test_e2e.py -v
"""

import base64
import io

import pytest
from PIL import Image, ImageDraw, ImageFilter

from opp.config import OPPConfig


def _make_test_image(width=256, height=256):
    """Create a test image."""
    img = Image.new("RGB", (width, height), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, width - 50, height - 50], fill=(200, 100, 50))
    draw.rectangle([80, 80, width - 80, height - 80], outline=(255, 255, 255))
    return img


def _image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _qdrant_available():
    """Check if Qdrant is available."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


skip_no_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant not available at localhost:6333"
)


@skip_no_qdrant
class TestEndToEnd:
    """End-to-end mint → verify tests."""

    def test_mint_then_verify_exact(self):
        """Mint an image, then verify the same image returns exact match."""
        from fastapi.testclient import TestClient
        from opp.server import app

        client = TestClient(app)

        img = _make_test_image()
        img_b64 = _image_to_base64(img)

        # Mint
        mint_response = client.post("/v1/mint", json={
            "image_base64": img_b64,
            "metadata": {
                "generator": "test-generator",
                "model_version": "1.0",
            }
        })
        assert mint_response.status_code == 200
        mint_data = mint_response.json()
        assert mint_data["status"] == "minted"
        sig_id = mint_data["signature_id"]

        # Verify same image
        verify_response = client.post("/v1/verify", json={
            "image_base64": img_b64,
            "min_similarity": 0.80,
        })
        assert verify_response.status_code == 200
        verify_data = verify_response.json()

        # Should find the minted signature
        assert len(verify_data["matches"]) > 0
        match_ids = [m["signature_id"] for m in verify_data["matches"]]
        assert sig_id in match_ids

        # First match should be exact
        first_match = verify_data["matches"][0]
        assert first_match["cosine_similarity"] > 0.95

    def test_mint_then_verify_modified(self):
        """Mint an image, then verify a modified version still matches."""
        from fastapi.testclient import TestClient
        from opp.server import app

        client = TestClient(app)

        img = _make_test_image(512, 512)
        img_b64 = _image_to_base64(img)

        # Mint original
        mint_response = client.post("/v1/mint", json={
            "image_base64": img_b64,
            "metadata": {
                "generator": "test-generator",
            }
        })
        assert mint_response.status_code == 200

        # Modify: resize + blur
        modified = img.resize((256, 256)).filter(ImageFilter.GaussianBlur(2))
        mod_b64 = _image_to_base64(modified)

        # Verify modified version
        verify_response = client.post("/v1/verify", json={
            "image_base64": mod_b64,
            "min_similarity": 0.70,
        })
        assert verify_response.status_code == 200
        verify_data = verify_response.json()

        # Should still find a match
        assert len(verify_data["matches"]) > 0
        assert verify_data["matches"][0]["cosine_similarity"] > 0.75
