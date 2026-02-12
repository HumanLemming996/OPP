"""
OPP Reference Server — FastAPI application.

Endpoints:
    POST /v1/mint     - Mint a new provenance signature
    POST /v1/verify   - Verify an image against the registry
    GET  /v1/signatures/{id} - Lookup a specific signature
    GET  /health      - Health check + stats
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from opp import __version__, __protocol_version__
from opp.config import OPPConfig, get_config
from opp.models import (
    MintRequest,
    MintResponse,
    MatchResult,
    SignatureRecord,
    VerifyRequest,
    VerifyResponse,
)
from opp.signature import (
    generate_signature,
    load_image_from_base64,
    load_image_from_url,
)
from opp.storage import OPPStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global storage instance
# ---------------------------------------------------------------------------
_storage: Optional[OPPStorage] = None


def get_storage() -> OPPStorage:
    """Get the global storage instance."""
    if _storage is None:
        raise RuntimeError("Storage not initialized. Server not started properly.")
    return _storage


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage on startup, close on shutdown."""
    global _storage
    config = get_config()
    _storage = OPPStorage(config)
    _storage.initialize()
    logger.info(f"OPP Server v{__version__} started ({__protocol_version__})")
    yield
    _storage.close()
    logger.info("OPP Server shut down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OPP — Open Provenance Protocol",
    description=(
        "Shazam for AI-generated images. An open protocol for minting and "
        "verifying provenance signatures of AI-generated content."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/mint", response_model=MintResponse)
async def mint_signature(request: MintRequest):
    """Mint a new provenance signature for an AI-generated image.

    Generators compute the 3-layer signature on their own infrastructure
    using the open-source OPP library, then submit the pre-computed
    signature and metadata. No raw image data is transmitted.

    The server validates signature format but does NOT recompute it.
    """
    storage = get_storage()
    start_time = time.time()

    signature = request.signature

    # Validate signature format
    if len(signature.l1_exact.value) != 64:
        raise HTTPException(
            status_code=400,
            detail="L1 SHA-256 hash must be 64 hex characters",
        )
    if len(signature.l2_perceptual.value) != 64:
        raise HTTPException(
            status_code=400,
            detail="L2 PDQ hash must be 64 hex characters",
        )

    # Validate CLIP embedding is decodable and correct dimension
    try:
        import base64
        import numpy as np

        raw = base64.b64decode(signature.l3_semantic.value)
        vec = np.frombuffer(raw, dtype=np.float32)
        if vec.shape[0] != 768:
            raise HTTPException(
                status_code=400,
                detail=f"L3 CLIP embedding must be 768-dim, got {vec.shape[0]}",
            )
        norm = float(np.linalg.norm(vec))
        if abs(norm - 1.0) > 0.05:
            raise HTTPException(
                status_code=400,
                detail=f"L3 CLIP embedding must be L2-normalized (norm={norm:.3f})",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"L3 CLIP embedding is not valid base64-encoded float32: {e}",
        )

    # Create record and store
    record = SignatureRecord(
        signature=signature,
        metadata=request.metadata,
    )
    sig_id = storage.store_signature(record)

    elapsed = time.time() - start_time
    logger.info(f"Minted {sig_id} in {elapsed:.2f}s")

    return MintResponse(
        signature_id=sig_id,
        l1_hash=signature.l1_exact.value,
        l2_hash=signature.l2_perceptual.value,
    )


@app.post("/v1/verify", response_model=VerifyResponse)
async def verify_image(request: VerifyRequest):
    """Verify an image against the provenance registry.

    Runs the tiered matching pipeline:
    1. SHA-256 exact match (instant)
    2. CLIP cosine similarity via Qdrant HNSW search
    3. PDQ hamming distance enrichment
    """
    storage = get_storage()
    start_time = time.time()

    # Load image
    if request.image_base64:
        image = load_image_from_base64(request.image_base64)
    elif request.image_url:
        image = await load_image_from_url(request.image_url)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either image_base64 or image_url must be provided",
        )

    # Generate query signature
    query_sig = generate_signature(image)

    matches: list[MatchResult] = []

    # L1: Check exact SHA-256 match first
    exact = storage.find_exact_match(query_sig.l1_exact.value)
    if exact:
        matches.append(exact)

    # L2+L3: Qdrant vector search + PDQ enrichment
    similar = storage.search_similar(
        query_sig,
        max_results=request.max_results,
        min_similarity=request.min_similarity,
    )

    # Merge results (avoid duplicates from exact match)
    exact_ids = {m.signature_id for m in matches}
    for m in similar:
        if m.signature_id not in exact_ids:
            matches.append(m)

    stats = storage.get_stats()
    elapsed = time.time() - start_time
    logger.info(f"Verified in {elapsed:.2f}s, {len(matches)} matches found")

    return VerifyResponse(
        query_l1_hash=query_sig.l1_exact.value,
        query_l2_hash=query_sig.l2_perceptual.value,
        matches=matches[:request.max_results],
        total_signatures_searched=stats["total_signatures"],
    )


@app.get("/v1/signatures/{signature_id}")
async def get_signature(signature_id: str):
    """Lookup a specific signature by ID."""
    storage = get_storage()
    record = storage.get_signature(signature_id)
    if not record:
        raise HTTPException(status_code=404, detail="Signature not found")
    return record


@app.get("/health")
async def health_check():
    """Health check with storage stats."""
    storage = get_storage()
    try:
        stats = storage.get_stats()
        return {
            "status": "healthy",
            "version": __version__,
            "protocol": __protocol_version__,
            **stats,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "version": __version__,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    """Run the OPP server."""
    import uvicorn
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "opp.server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
