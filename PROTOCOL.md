# OPP/1.0 — Open Provenance Protocol Specification

**Version:** 1.0  
**Status:** Draft  
**Date:** 2025-02-12  
**Authors:** OPP Contributors  
**License:** Apache 2.0  

---

## 1. Abstract

The Open Provenance Protocol (OPP) is an open standard for registering and
verifying the provenance of AI-generated images. It provides a mechanism for
AI image generators to **mint** cryptographic and perceptual signatures of
their outputs to a distributed registry, and for anyone to **verify** whether
an image has a registered provenance record.

OPP uses a 3-layer signature system combining exact, perceptual, and semantic
matching - enabling verification even after an image has been screenshotted,
cropped, compressed, filtered, or partially reused.

---

## 2. Motivation

Existing approaches to AI content provenance have significant limitations:

- **C2PA/Content Credentials**: Embeds metadata inside image files. Stripped
  on screenshot, re-upload, or social media compression. Requires generator
  cooperation to embed.
- **AI Detectors**: Probabilistic binary classifiers ("is this AI?"). Unreliable,
  adversarial arms race, no provenance data.
- **Blockchain provenance**: Over-engineered. Consensus is unnecessary for a
  registry. Gas fees make billion-scale minting impractical.
- **Proprietary watermarks** (SynthID, etc.): No interoperability between
  generators. Can be attacked/removed.

OPP addresses these by providing an **external**, **federated**, **open**
registry that works regardless of image format, metadata stripping, or
generator cooperation.

---

## 3. Signature Format

### 3.1 Three-Layer Architecture

Every OPP signature consists of three layers:

| Layer | Name | Algorithm | Output | Purpose |
|-------|------|-----------|--------|---------|
| L1 | Exact | SHA-256 | 256-bit hash | Exact duplicate detection, O(1) lookup |
| L2 | Perceptual | PDQ (Meta) | 256-bit hash | Near-duplicate matching at billion scale |
| L3 | Semantic | CLIP ViT-L/14 | 768-dim float32 | Robust matching across transformations |

### 3.2 Signature Object

```json
{
  "protocol_version": "OPP/1.0",
  "l1_exact": {
    "algorithm": "sha256",
    "version": "1.0",
    "value": "a1b2c3d4..."
  },
  "l2_perceptual": {
    "algorithm": "pdq",
    "version": "1.0",
    "value": "f0e1d2c3..."
  },
  "l3_semantic": {
    "algorithm": "clip-vit-l-14",
    "version": "1.0",
    "value": "<base64-encoded 768-dim float32 vector>"
  },
  "image_width": 1024,
  "image_height": 1024
}
```

### 3.3 Multi-Resolution Signatures

To support partial image matching, OPP generates signatures at multiple
resolutions:

- **Full image**: Complete image signature
- **Quadrants**: Top-left, top-right, bottom-left, bottom-right (each at 50% width/height)

This enables matching when someone crops or uses a portion of an AI-generated image.

### 3.4 Algorithm Requirements

**L1 (SHA-256)**:
- Input: Raw pixel data in canonical RGB PNG format
- Output: 64-character hex string
- Deterministic: same pixels always produce same hash

**L2 (PDQ)**:
- Input: RGB image as numpy array
- Output: 64-character hex string (256 bits)
- Hamming distance ≤ 31 bits indicates likely match
- Robust to: format changes, quality reduction, resizing, light cropping,
  noise, filters, watermarks

**L3 (CLIP ViT-L/14)**:
- Input: RGB image
- Output: 768-dimensional float32 vector, L2-normalized to unit length
- Encoded as base64 for wire format
- Cosine similarity for comparison (equivalent to dot product when normalized)

---

## 4. Operations

### 4.1 Mint

Register a pre-computed provenance signature with the central index.

Generators compute the 3-layer signature on their own infrastructure
using the open-source OPP library, then submit the signature and metadata
to the central index. **No raw image data is transmitted.**

**Request**: `POST /v1/mint`
```json
{
  "signature": {
    "protocol_version": "OPP/1.0",
    "l1_exact": { "algorithm": "sha256", "version": "1.0", "value": "a1b2c3d4..." },
    "l2_perceptual": { "algorithm": "pdq", "version": "1.0", "value": "f0e1d2c3..." },
    "l3_semantic": { "algorithm": "clip-vit-l-14", "version": "1.0", "value": "<base64>" },
    "image_width": 1024,
    "image_height": 1024
  },
  "metadata": {
    "generator": "openai-dall-e-3",
    "model_version": "3.0",
    "timestamp": "2025-02-12T00:00:00Z",
    "tags": ["landscape", "photorealistic"]
  }
}
```

**Response**: `200 OK`
```json
{
  "signature_id": "uuid-v4",
  "status": "minted",
  "protocol_version": "OPP/1.0",
  "l1_hash": "a1b2c3d4...",
  "l2_hash": "f0e1d2c3...",
  "timestamp": "2025-02-12T00:00:00Z"
}
```

Generators MUST compute signatures using the canonical algorithms specified
in Section 3. The server validates that the signature conforms to the
expected format (correct hash lengths, normalized CLIP vector, etc.)
but does NOT recompute the signature.

### 4.2 Verify

Check an image against the registry for provenance matches.

**Request**: `POST /v1/verify`
```json
{
  "image_base64": "<base64-encoded image>",
  "max_results": 10,
  "min_similarity": 0.80
}
```

**Response**: `200 OK`
```json
{
  "query_l1_hash": "...",
  "query_l2_hash": "...",
  "matches": [
    {
      "signature_id": "uuid",
      "match_level": "exact|likely|possible|none",
      "cosine_similarity": 0.95,
      "pdq_hamming_distance": 12,
      "metadata": {
        "generator": "openai-dall-e-3",
        "model_version": "3.0",
        "timestamp": "2025-02-12T00:00:00Z",
        "tags": []
      }
    }
  ],
  "total_signatures_searched": 1000000,
  "protocol_version": "OPP/1.0"
}
```

### 4.3 Lookup

Retrieve a specific signature by ID.

**Request**: `GET /v1/signatures/{signature_id}`

---

## 5. Matching Pipeline

Verification uses a tiered pipeline, from fastest to most accurate:

```
Query Image
    │
    ├─── L1: SHA-256 exact match ──────── O(1) instant
    │         │
    │         └── Found? → Return EXACT match
    │
    ├─── L3: CLIP cosine similarity ───── Qdrant HNSW search
    │         │                            (sub-10ms at billions)
    │         └── Candidates ranked by cosine similarity
    │
    └─── L2: PDQ hamming distance ─────── Enrichment on candidates
              │
              └── Additional confidence signal
```

### 5.1 Match Levels

| Level | Cosine Similarity | Interpretation |
|-------|------------------|----------------|
| **exact** | ≥ 0.97 | Same image (possibly re-encoded) |
| **likely** | ≥ 0.90 | Same image with modifications (crop, filter, etc.) |
| **possible** | ≥ 0.80 | Potentially derived from the same source |
| **none** | < 0.80 | No significant match |

---

## 6. Metadata Schema

Provenance metadata is attached to each minted signature:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generator` | string | Yes | Generator identifier (e.g. "openai-dall-e-3") |
| `model_version` | string | No | Specific model version |
| `timestamp` | ISO 8601 | Yes | Generation timestamp (UTC) |
| `tags` | string[] | No | Optional classification tags |
| `extra` | object | No | Additional metadata (flexible) |

**Privacy**: OPP metadata identifies the **generator**, not the **user**.
No user-identifying information is stored in the protocol.

---

## 7. Architecture

### 7.1 Central Index Model

OPP uses a **central index** architecture. Generators compute signatures
on their own infrastructure and push them to the central search index.
All verification queries are served from this unified index.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  AI Generator   │     │  AI Generator   │     │  AI Generator   │
│  (OpenAI)       │     │  (Midjourney)   │     │  (Self-hosted)  │
│                 │     │                 │     │                 │
│  Runs OPP lib   │     │  Runs OPP lib   │     │  Runs OPP lib   │
│  on own compute │     │  on own compute │     │  on own compute │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │ push signature        │ push signature        │ push signature
         ▼                       ▼                       ▼
┌────────────────────────────────────────────────────────────────────┐
│                      OPP Central Index                            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ Qdrant       │  │ SQLite /     │  │ Node Registry           │  │
│  │ (CLIP vecs)  │  │ Postgres     │  │ (generator ↔ API keys)  │  │
│  │              │  │ (metadata,   │  │                         │  │
│  │ HNSW index   │  │  L1/L2 hash) │  │                         │  │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘  │
└────────────────────────────────┬───────────────────────────────────┘
                                │
                                ▲ verify query
                                │
              ┌─────────────────┴─────────────────┐
              │  Verifier (journalist, platform,   │
              │  fact-checker, browser extension)   │
              └────────────────────────────────────┘
```

### 7.2 Roles

- **Generators**: Run the open-source OPP library on their own compute
  to produce 3-layer signatures. Push pre-computed signatures to the
  central index via authenticated `POST /v1/mint`.

- **Central Index**: Stores all signatures in a unified search index
  (Qdrant for CLIP vectors, relational DB for metadata/hashes). Serves
  all verification queries. Operated by the OPP organization.

- **Verifiers**: Anyone who wants to check an image. Sends the image to
  the central index via `POST /v1/verify`. The server computes the query
  signature and searches the index. No special setup required.

### 7.3 Mint Flow

1. Generator creates an image
2. Generator runs OPP library locally → produces 3-layer signature
3. Generator sends `{signature, metadata}` to central index
4. Central index validates and stores the signature
5. Generator receives `signature_id` confirmation

### 7.4 Verify Flow

1. Verifier sends image to central index (`POST /v1/verify`)
2. Central index generates query signature on its own compute
3. L1: SHA-256 exact match check (O(1))
4. L3: CLIP cosine similarity via Qdrant HNSW search (sub-10ms)
5. L2: PDQ hamming distance enrichment on candidates
6. Results returned with match levels and metadata

### 7.5 Sovereign Nodes (Future)

For generators with data sovereignty requirements, OPP will support
federated nodes that sync signatures to the central index:

- Generator runs their own OPP node
- Node periodically syncs signatures to the central index
- Verification queries still go through the central index
- Central index caches synced signatures with permission

---

## 8. Protocol Versioning

All signatures include a `protocol_version` field (e.g. "OPP/1.0").
This enables:

- Forward compatibility: new layers can be added in future versions
- Backward compatibility: old signatures remain valid and searchable
- Mixed-version registries: nodes can serve multiple protocol versions

**Immutability guarantee**: Once a signature algorithm version is released,
it is never changed. New algorithms are added as new layers, not replacements.

---

## 9. Security Considerations

- **No user data**: OPP stores generator identity, not user identity
- **Rate limiting**: Implementations SHOULD rate-limit mint and verify requests
- **No image storage**: OPP nodes store only signatures and metadata, never the original images
- **Transport security**: All API communication SHOULD use HTTPS
- **Signature integrity**: Signature IDs are UUIDs; nodes MAY additionally sign records

---

## 10. References

- PDQ Hash: https://github.com/facebook/ThreatExchange/tree/main/pdq
- CLIP: https://github.com/openai/CLIP
- OpenCLIP: https://github.com/mlfoundations/open_clip
- C2PA: https://c2pa.org/
- EU AI Act, Article 50: AI-generated content labelling requirements
