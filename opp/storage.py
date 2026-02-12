"""
OPP Storage Backend — Qdrant (vectors) + SQLite (metadata).

Qdrant handles CLIP vector storage + HNSW-indexed cosine similarity search.
SQLite handles metadata, PDQ hash lookups, and exact SHA-256 lookups.

Designed with a clean interface so horizontal scaling (sharding, replication)
can be added without changing the API.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from opp.config import OPPConfig, get_config
from opp.models import (
    MatchLevel,
    MatchResult,
    OPPSignature,
    ProvenanceMetadata,
    SignatureRecord,
)
from opp.signature import clip_embedding_to_numpy
from opp.similarity import classify_match, pdq_hamming_distance

logger = logging.getLogger(__name__)


class OPPStorage:
    """Combined Qdrant + SQLite storage backend for OPP signatures."""

    def __init__(self, config: Optional[OPPConfig] = None):
        self.config = config or get_config()
        self._qdrant: Optional[QdrantClient] = None
        self._sqlite: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize both storage backends."""
        self._init_qdrant()
        self._init_sqlite()
        logger.info("Storage backends initialized")

    def _init_qdrant(self) -> None:
        """Connect to Qdrant and ensure collection exists."""
        self._qdrant = QdrantClient(url=self.config.qdrant_url)

        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self.config.qdrant_collection not in collections:
            self._qdrant.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config=VectorParams(
                    size=self.config.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.config.qdrant_collection}")
        else:
            logger.info(f"Qdrant collection exists: {self.config.qdrant_collection}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database with schema."""
        self._sqlite = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA synchronous=NORMAL")

        self._sqlite.executescript("""
            CREATE TABLE IF NOT EXISTS signatures (
                signature_id TEXT PRIMARY KEY,
                sha256_hash TEXT NOT NULL,
                pdq_hash TEXT NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                generator TEXT NOT NULL,
                model_version TEXT,
                metadata_json TEXT NOT NULL,
                protocol_version TEXT NOT NULL DEFAULT 'OPP/1.0',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_sha256 ON signatures(sha256_hash);
            CREATE INDEX IF NOT EXISTS idx_pdq ON signatures(pdq_hash);
            CREATE INDEX IF NOT EXISTS idx_generator ON signatures(generator);
            CREATE INDEX IF NOT EXISTS idx_created_at ON signatures(created_at);
        """)
        self._sqlite.commit()
        logger.info(f"SQLite initialized at {self.config.sqlite_path}")

    # ------------------------------------------------------------------
    # Mint (Store)
    # ------------------------------------------------------------------

    def store_signature(self, record: SignatureRecord) -> str:
        """Store a complete signature record.

        Writes the CLIP vector to Qdrant and metadata to SQLite.
        Returns the signature_id.
        """
        sig = record.signature

        # --- Qdrant: store CLIP embedding ---
        clip_vector = clip_embedding_to_numpy(sig.l3_semantic.value).tolist()

        self._qdrant.upsert(
            collection_name=self.config.qdrant_collection,
            points=[
                PointStruct(
                    id=record.signature_id,
                    vector=clip_vector,
                    payload={
                        "signature_id": record.signature_id,
                        "sha256_hash": sig.l1_exact.value,
                        "pdq_hash": sig.l2_perceptual.value,
                        "generator": record.metadata.generator,
                    },
                )
            ],
        )

        # --- SQLite: store metadata ---
        self._sqlite.execute(
            """
            INSERT OR REPLACE INTO signatures
            (signature_id, sha256_hash, pdq_hash, image_width, image_height,
             generator, model_version, metadata_json, protocol_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.signature_id,
                sig.l1_exact.value,
                sig.l2_perceptual.value,
                sig.image_width,
                sig.image_height,
                record.metadata.generator,
                record.metadata.model_version,
                record.metadata.model_dump_json(),
                sig.protocol_version,
                record.created_at.isoformat(),
            ),
        )
        self._sqlite.commit()

        logger.info(f"Stored signature {record.signature_id}")
        return record.signature_id

    # ------------------------------------------------------------------
    # Verify (Search)
    # ------------------------------------------------------------------

    def find_exact_match(self, sha256_hash: str) -> Optional[MatchResult]:
        """L1: Check for exact SHA-256 match. O(1)."""
        row = self._sqlite.execute(
            "SELECT signature_id, metadata_json FROM signatures WHERE sha256_hash = ? LIMIT 1",
            (sha256_hash,),
        ).fetchone()

        if row:
            sig_id, meta_json = row
            metadata = ProvenanceMetadata.model_validate_json(meta_json)
            return MatchResult(
                signature_id=sig_id,
                match_level=MatchLevel.EXACT,
                cosine_similarity=1.0,
                pdq_hamming_distance=0,
                metadata=metadata,
            )
        return None

    def search_similar(
        self,
        query_signature: OPPSignature,
        max_results: int = 10,
        min_similarity: float = 0.80,
    ) -> list[MatchResult]:
        """L2+L3: Search for similar signatures using the tiered pipeline.

        1. Qdrant vector search (CLIP cosine similarity) with HNSW
        2. Re-rank and filter results
        3. Enrich with metadata from SQLite
        """
        # Extract CLIP vector for Qdrant search
        clip_vector = clip_embedding_to_numpy(query_signature.l3_semantic.value).tolist()

        # Search Qdrant — this uses HNSW index, sub-10ms even at billions
        qdrant_results = self._qdrant.search(
            collection_name=self.config.qdrant_collection,
            query_vector=clip_vector,
            limit=max_results * 2,  # oversample for post-filtering
            score_threshold=min_similarity,
        )

        if not qdrant_results:
            return []

        # Enrich with metadata and compute PDQ distances
        matches = []
        for result in qdrant_results:
            sig_id = result.payload.get("signature_id", str(result.id))
            cos_sim = float(result.score)

            # Compute PDQ hamming distance for additional context
            candidate_pdq = result.payload.get("pdq_hash", "")
            pdq_dist = None
            if candidate_pdq:
                pdq_dist = pdq_hamming_distance(
                    query_signature.l2_perceptual.value,
                    candidate_pdq,
                )

            # Get full metadata from SQLite
            row = self._sqlite.execute(
                "SELECT metadata_json FROM signatures WHERE signature_id = ?",
                (sig_id,),
            ).fetchone()

            if row:
                metadata = ProvenanceMetadata.model_validate_json(row[0])
            else:
                # Fallback if SQLite doesn't have it (shouldn't happen)
                metadata = ProvenanceMetadata(
                    generator=result.payload.get("generator", "unknown")
                )

            match_level = classify_match(cos_sim, self.config)

            matches.append(
                MatchResult(
                    signature_id=sig_id,
                    match_level=match_level,
                    cosine_similarity=round(cos_sim, 4),
                    pdq_hamming_distance=pdq_dist,
                    metadata=metadata,
                )
            )

        # Sort by similarity descending, take top N
        matches.sort(key=lambda m: m.cosine_similarity, reverse=True)
        return matches[:max_results]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_signature(self, signature_id: str) -> Optional[dict]:
        """Get a specific signature record by ID."""
        row = self._sqlite.execute(
            """
            SELECT signature_id, sha256_hash, pdq_hash, image_width, image_height,
                   generator, model_version, metadata_json, protocol_version, created_at
            FROM signatures WHERE signature_id = ?
            """,
            (signature_id,),
        ).fetchone()

        if row:
            return {
                "signature_id": row[0],
                "sha256_hash": row[1],
                "pdq_hash": row[2],
                "image_width": row[3],
                "image_height": row[4],
                "generator": row[5],
                "model_version": row[6],
                "metadata": json.loads(row[7]),
                "protocol_version": row[8],
                "created_at": row[9],
            }
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get storage statistics."""
        sqlite_count = self._sqlite.execute(
            "SELECT COUNT(*) FROM signatures"
        ).fetchone()[0]

        qdrant_info = self._qdrant.get_collection(self.config.qdrant_collection)

        return {
            "total_signatures": sqlite_count,
            "qdrant_points": qdrant_info.points_count,
            "qdrant_status": qdrant_info.status.value,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close storage connections."""
        if self._sqlite:
            self._sqlite.close()
        if self._qdrant:
            self._qdrant.close()
