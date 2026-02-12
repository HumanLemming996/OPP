"""Tests for OPP server API endpoints."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock


def _make_test_image_base64():
    """Create a test image and return as base64."""
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 200, 200], fill=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Note: Full server tests require Qdrant running.
# These tests validate the API structure and error handling.

class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_version(self):
        """Health endpoint should return version info even if storage fails."""
        from opp.server import app
        client = TestClient(app)

        # This may fail if Qdrant isn't running, but should not crash
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "status" in data


class TestMintValidation:
    """Test mint endpoint input validation."""

    def test_mint_requires_image(self):
        """Mint should reject requests without image data."""
        from opp.server import app
        client = TestClient(app)

        response = client.post("/v1/mint", json={
            "metadata": {
                "generator": "test",
            }
        })
        # Should get 400 or 422 for missing image
        assert response.status_code in (400, 422)


class TestVerifyValidation:
    """Test verify endpoint input validation."""

    def test_verify_requires_image(self):
        """Verify should reject requests without image data."""
        from opp.server import app
        client = TestClient(app)

        response = client.post("/v1/verify", json={
            "max_results": 10,
        })
        # Should get 400 or 422 for missing image
        assert response.status_code in (400, 422)
