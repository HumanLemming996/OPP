<p align="center">
  <h1 align="center">OPP — Open Provenance Protocol</h1>
  <p align="center">
    <strong>Shazam for AI-generated images.</strong><br>
    An open protocol for minting and verifying provenance of AI-generated content.
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#api-reference">API</a> •
    <a href="#protocol-spec">Protocol</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

## The Problem

AI generators produce billions of images. We need to know which images are AI-generated and by whom. Current solutions fall short:

| Approach | Problem |
|----------|---------|
| **C2PA / Content Credentials** | Metadata embedded in file — stripped on screenshot, re-upload, compression |
| **AI Detectors** | Probabilistic guessing — unreliable, adversarial arms race |
| **Blockchain provenance** | Over-engineered — gas fees make billion-scale minting impractical |
| **Proprietary watermarks** | No interoperability — each generator's watermark is isolated |

## The Solution

OPP is an **external, federated registry** of AI image signatures. Think DNS for AI provenance.

- **Generators mint** signatures when creating images
- **Anyone can verify** by querying the registry
- **Works after screenshots, crops, compression, filters** — no embedded metadata to strip
- **Open protocol** — any generator, any verifier, any registry node

## How It Works

### 3-Layer Signature System

Every image gets a signature with three complementary layers:

```
┌─────────────────────────────────────────────────────────┐
│  L1: SHA-256          → Exact duplicate detection       │
│  256-bit hash           O(1) instant lookup             │
├─────────────────────────────────────────────────────────┤
│  L2: PDQ (Meta)       → Near-duplicate matching         │
│  256-bit perceptual     Billion-scale, 2^256 values     │
│  hash                   Survives resize, compress, crop │
├─────────────────────────────────────────────────────────┤
│  L3: CLIP ViT-L/14    → Semantic similarity             │
│  768-dim embedding      Robust to all transformations   │
│                         Sub-10ms search at billions     │
└─────────────────────────────────────────────────────────┘
```

### Matching Pipeline

Verification uses a tiered pipeline — fast to accurate:

1. **SHA-256 exact match** → O(1), instant
2. **CLIP cosine similarity** → Qdrant HNSW vector search, sub-10ms at billions
3. **PDQ hamming distance** → additional confidence signal on candidates

### Adaptive Variant Tracking

OPP's registry **gets smarter with every verification**. When a query matches with high confidence, the variant's signature is automatically minted and linked to the original:

```
Original (minted by generator)
  ├── Screenshot (auto-minted on verify, 94% match)
  │     └── Crop of screenshot (auto-minted, 93% match)
  │           └── Re-upload of crop (auto-minted, 95% match)
  └── Compressed version (auto-minted, 96% match)
```

Each variant only needs to match its **nearest ancestor**, not the original. So even a heavily degraded image that's been screenshotted, cropped, and re-uploaded multiple times is caught — because the chain never breaks.

The most viral images (highest misuse risk) accumulate the most variants, making them the **hardest to escape detection**. The problem and the solution scale together.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  AI Generator   │     │  AI Generator   │     │  AI Generator   │
│  (OpenAI)       │     │  (Midjourney)   │     │  (Self-hosted)  │
│  Runs OPP lib   │     │  Runs OPP lib   │     │  Runs OPP lib   │
│  on own compute │     │  on own compute │     │  on own compute │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │ push signature        │ push signature        │ push signature
         ▼                       ▼                       ▼
┌────────────────────────────────────────────────────────────────────┐
│                      OPP Central Index                            │
│                 (Qdrant + metadata store)                         │
│            Unified search across ALL generators                   │
└───────────────────────────┬────────────────────────────────────────┘
                            ▲
                            │ verify query (image upload)
                     ┌──────┴──────┐
                     │  Any Client │
                     │  (verifier) │
                     └─────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/opp-protocol/opp.git
cd opp
docker compose up
```

This starts the OPP server + Qdrant vector database. Server runs on `http://localhost:8000`.

### Option 2: Local Install

```bash
pip install opp-ai-protocol
```

### Mint an Image

Generators compute signatures on their own infrastructure and push to the central index:

```bash
# Via CLI (computes signature locally, pushes to server)
opp mint my_image.png --generator "dall-e-3" --server http://localhost:8000

# Via API (pre-computed signature)
curl -X POST http://localhost:8000/v1/mint \
  -H "Content-Type: application/json" \
  -d '{
    "signature": {
      "protocol_version": "OPP/1.0",
      "l1_exact": { "algorithm": "sha256", "version": "1.0", "value": "..." },
      "l2_perceptual": { "algorithm": "pdq", "version": "1.0", "value": "..." },
      "l3_semantic": { "algorithm": "clip-vit-l-14", "version": "1.0", "value": "..." },
      "image_width": 1024,
      "image_height": 1024
    },
    "metadata": {
      "generator": "dall-e-3",
      "model_version": "3.0"
    }
  }'
```

### Verify an Image

```bash
# Via CLI
opp verify suspicious_image.png --server http://localhost:8000

# Via API
curl -X POST http://localhost:8000/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded-image>"
  }'
```

### Example Output

```
┌──────────────────────────────────────────────────┐
│                OPP Matches (2 found)             │
├────────┬────────────┬──────────┬─────────────────┤
│ Level  │ Similarity │ PDQ Dist │ Generator       │
├────────┼────────────┼──────────┼─────────────────┤
│ EXACT  │     0.9912 │        3 │ dall-e-3        │
│ LIKELY │     0.9234 │       18 │ dall-e-3        │
└────────┴────────────┴──────────┴─────────────────┘
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mint` | POST | Register a pre-computed signature |
| `/v1/verify` | POST | Verify an image against the registry |
| `/v1/signatures/{id}` | GET | Lookup a specific signature |
| `/health` | GET | Health check + stats |

Full protocol specification: [PROTOCOL.md](PROTOCOL.md)

## Why OPP?

| Feature | OPP | C2PA | AI Detectors | Blockchain |
|---------|-----|------|-------------|------------|
| Survives screenshots | ✅ | ❌ | N/A | ❌ |
| Survives compression | ✅ | ❌ | N/A | ❌ |
| Deterministic result | ✅ | ✅ | ❌ | ✅ |
| Open protocol | ✅ | ✅ | ❌ | ✅ |
| Billion-scale | ✅ | ✅ | ❌ | ❌ |
| No gas fees | ✅ | ✅ | ✅ | ❌ |
| Self-improving registry | ✅ | ❌ | ❌ | ❌ |
| Generator-agnostic | ✅ | ❌ | ✅ | ✅ |

## Tech Stack

- **Signature**: SHA-256 + PDQ (Meta) + CLIP ViT-L/14
- **Vector DB**: Qdrant (HNSW-indexed, billions of vectors)
- **Server**: FastAPI + Uvicorn
- **Storage**: SQLite (metadata) + Qdrant (embeddings)
- **CLI**: Click + Rich

## Contributing

OPP is open source under the Apache 2.0 license. Contributions welcome!

1. Fork the repo
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push and open a PR

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

<p align="center">
  <strong>Built for the age of AI. Open by design.</strong>
</p>
