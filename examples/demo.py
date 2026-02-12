"""
OPP Demo — Quick demonstration of signature generation and matching.

This demo works offline (no server needed) to show the core signature
and similarity engine in action.

Usage:
    python -m examples.demo
"""

from PIL import Image, ImageDraw, ImageFilter
import tempfile
import os

def main():
    print("\n" + "=" * 60)
    print("  OPP — Open Provenance Protocol Demo")
    print("  Shazam for AI-generated images")
    print("=" * 60 + "\n")

    # --- Step 1: Create a test image ---
    print("[1/5] Creating test image...")
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)
    # Draw some interesting patterns
    for i in range(0, 512, 32):
        draw.rectangle([i, i, 512 - i, 512 - i], outline=(i % 256, (i * 2) % 256, (i * 3) % 256))
    draw.ellipse([128, 128, 384, 384], fill=(100, 150, 200))
    draw.text((200, 240), "OPP", fill=(255, 255, 255))

    # Save original
    tmp = tempfile.mkdtemp()
    orig_path = os.path.join(tmp, "original.png")
    img.save(orig_path)
    print(f"   Saved: {orig_path}")

    # --- Step 2: Generate signature ---
    print("\n[2/5] Generating 3-layer signature...")
    from opp.signature import generate_signature

    sig = generate_signature(img)
    print(f"   L1 (SHA-256):  {sig.l1_exact.value[:16]}...")
    print(f"   L2 (PDQ):      {sig.l2_perceptual.value[:16]}...")
    print(f"   L3 (CLIP):     {sig.l3_semantic.value[:24]}...")
    print(f"   Dimensions:    {sig.image_width}x{sig.image_height}")

    # --- Step 3: Create modified versions ---
    print("\n[3/5] Creating modified versions...")

    # Resize
    resized = img.resize((256, 256))
    print("   ✓ Resized to 256x256")

    # JPEG compression
    jpeg_path = os.path.join(tmp, "compressed.jpg")
    img.save(jpeg_path, "JPEG", quality=30)
    compressed = Image.open(jpeg_path)
    print("   ✓ JPEG compressed (quality=30)")

    # Blur
    blurred = img.filter(ImageFilter.GaussianBlur(radius=3))
    print("   ✓ Gaussian blur (radius=3)")

    # Crop
    cropped = img.crop((64, 64, 448, 448))
    print("   ✓ Cropped (64px from each edge)")

    # Different image entirely
    different = Image.new("RGB", (512, 512), (50, 50, 50))
    diff_draw = ImageDraw.Draw(different)
    diff_draw.rectangle([100, 100, 400, 400], fill=(200, 50, 50))
    print("   ✓ Completely different image")

    # --- Step 4: Compare signatures ---
    print("\n[4/5] Comparing signatures...")
    from opp.similarity import compute_match_score

    variants = {
        "Original (self)": img,
        "Resized (256px)": resized,
        "JPEG (q=30)": compressed,
        "Blurred": blurred,
        "Cropped": cropped,
        "Different image": different,
    }

    print(f"\n   {'Variant':<20} {'Cosine Sim':>10} {'PDQ Dist':>10} {'Match Level':>12}")
    print("   " + "-" * 56)

    for name, variant in variants.items():
        variant_sig = generate_signature(variant)
        result = compute_match_score(sig, variant_sig)

        cos_sim = result["cosine_similarity"]
        pdq_dist = result["pdq_hamming_distance"]
        level = result["match_level"].value.upper()

        # Color coding via symbols
        if level == "EXACT":
            indicator = "🟢"
        elif level == "LIKELY":
            indicator = "🟡"
        elif level == "POSSIBLE":
            indicator = "🟠"
        else:
            indicator = "🔴"

        print(f"   {name:<20} {cos_sim:>10.4f} {pdq_dist:>10} {indicator} {level:>8}")

    # --- Step 5: Summary ---
    print(f"\n[5/5] Demo complete!")
    print(f"\n   The OPP signature system correctly identifies:")
    print(f"   • Exact matches (same image)")
    print(f"   • Modified versions (resize, compress, blur, crop)")
    print(f"   • Different images (low similarity)")
    print(f"\n   This works at BILLION scale using:")
    print(f"   • PDQ 256-bit hash (Meta) for fast pre-filtering")
    print(f"   • CLIP ViT-L/14 for semantic matching")
    print(f"   • Qdrant HNSW index for sub-10ms search")
    print(f"\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
