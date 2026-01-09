#!/usr/bin/env python3
"""
Download product images for Week 5 AI Services lab.

This script downloads 6 product images from Unsplash using specific photo IDs
for reproducibility. These images will produce interesting results when
analyzed with Amazon Rekognition (object/label detection).

Usage:
    python download_images.py

Output:
    product_images/product_001.jpg through product_006.jpg

Image Sources (Unsplash - Free for commercial use):
    All images are from Unsplash.com under their free license.
    https://unsplash.com/license
"""

import os
import urllib.request
import urllib.error
import ssl
import time

# Unsplash photo IDs and their expected Rekognition labels
# Format: (photo_id, filename, description, expected_labels)
PRODUCT_IMAGES = [
    # Laptop on desk
    (
        "XMFZqrGyV-Q",  # Photo by Kari Shea
        "product_001.jpg",
        "Laptop on wooden desk",
        ["Laptop", "Computer", "Electronics", "Desk", "Table"]
    ),
    # Cardboard box / package
    (
        "kLfkVa_4aXM",  # Photo by Kelli McClintock
        "product_002.jpg",
        "Cardboard shipping box",
        ["Box", "Cardboard", "Package", "Shipping"]
    ),
    # Sneakers / athletic shoes
    (
        "TamMbr4okv4",  # Photo by Domino
        "product_003.jpg",
        "Athletic sneakers",
        ["Shoe", "Sneaker", "Footwear", "Athletic"]
    ),
    # Kitchen appliance - blender
    (
        "MNtag_eXMKw",  # Photo by HamZa NOUASRIA
        "product_004.jpg",
        "Kitchen blender",
        ["Appliance", "Blender", "Kitchen", "Electronics"]
    ),
    # Headphones
    (
        "PDX_a_82obo",  # Photo by C D-X
        "product_005.jpg",
        "Over-ear headphones",
        ["Headphones", "Electronics", "Audio", "Music"]
    ),
    # Smartphone
    (
        "N3TVOM6NnwM",  # Photo by Jonas Leupe
        "product_006.jpg",
        "Smartphone on surface",
        ["Phone", "Smartphone", "Electronics", "Mobile Phone"]
    ),
]

# Alternative image sources (fallback)
FALLBACK_IMAGES = [
    # More reliable direct URLs if Unsplash photo IDs don't work
    (
        "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=800",
        "product_001.jpg",
        "Laptop"
    ),
    (
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80",
        "product_002.jpg",
        "Box"
    ),
    (
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800",
        "product_003.jpg",
        "Sneakers"
    ),
    (
        "https://images.unsplash.com/photo-1570222094114-d054a817e56b?w=800",
        "product_004.jpg",
        "Blender"
    ),
    (
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800",
        "product_005.jpg",
        "Headphones"
    ),
    (
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800",
        "product_006.jpg",
        "Smartphone"
    ),
]


def download_image(url: str, output_path: str, timeout: int = 30) -> bool:
    """Download an image from URL to local file."""
    try:
        # Create SSL context that doesn't verify (for some corporate networks)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Add headers to mimic browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    Error downloading: {e}")
        return False


def download_from_unsplash_id(photo_id: str, output_path: str) -> bool:
    """Download image from Unsplash using photo ID."""
    # Unsplash direct download URL format
    url = f"https://images.unsplash.com/photo-{photo_id}?w=800&q=80"

    # Try the source.unsplash.com redirect first (more reliable)
    source_url = f"https://source.unsplash.com/{photo_id}/800x600"

    print(f"    Trying source.unsplash.com...")
    if download_image(source_url, output_path):
        return True

    print(f"    Trying direct URL...")
    return download_image(url, output_path)


def main():
    """Download all product images."""
    output_dir = "product_images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {len(PRODUCT_IMAGES)} product images from Unsplash...\n")

    successful = 0
    failed = []

    for photo_id, filename, description, labels in PRODUCT_IMAGES:
        output_path = os.path.join(output_dir, filename)
        print(f"  {filename}: {description}")

        if download_from_unsplash_id(photo_id, output_path):
            # Verify file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"    ✓ Downloaded successfully")
                successful += 1
            else:
                print(f"    ✗ File too small or missing, trying fallback...")
                failed.append((filename, description))
        else:
            print(f"    ✗ Failed, trying fallback...")
            failed.append((filename, description))

        time.sleep(0.5)  # Be nice to the server

    # Try fallback URLs for failed downloads
    if failed:
        print(f"\n  Trying fallback URLs for {len(failed)} failed images...")
        for url, filename, description in FALLBACK_IMAGES:
            output_path = os.path.join(output_dir, filename)

            # Check if this file needs fallback
            needs_fallback = any(f[0] == filename for f in failed)
            if not needs_fallback:
                continue

            print(f"  {filename}: {description} (fallback)")
            if download_image(url, output_path):
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"    ✓ Downloaded successfully")
                    successful += 1
                    failed = [f for f in failed if f[0] != filename]
            time.sleep(0.5)

    # Summary
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Successful: {successful}/{len(PRODUCT_IMAGES)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for filename, desc in failed:
            print(f"    - {filename}: {desc}")
        print("\nNote: You may need to manually download missing images.")
    else:
        print(f"\nAll images downloaded to {output_dir}/")

    # List files with sizes
    print(f"\nDownloaded files:")
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".jpg"):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {filename}: {size_kb:.1f} KB")

    print(f"\nExpected Rekognition labels:")
    for _, filename, desc, labels in PRODUCT_IMAGES:
        print(f"  {filename}: {', '.join(labels[:3])}...")


if __name__ == "__main__":
    main()
