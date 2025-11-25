#!/usr/bin/env python3
"""
HTML to PNG Converter for MLOps Diagrams
Converts all HTML diagram files to high-quality PNG images
"""

import asyncio
from pathlib import Path
import sys

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Error: playwright not installed")
    print("\nInstall with:")
    print("  pip install playwright")
    print("  python -m playwright install chromium")
    sys.exit(1)


async def html_to_png(html_path: Path, output_path: Path):
    """Convert a single HTML file to PNG using Playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Load HTML file
        await page.goto(f"file://{html_path.absolute()}")

        # Wait for fonts to load
        await page.wait_for_timeout(1000)

        # Get SVG dimensions for proper screenshot
        svg_element = await page.query_selector("svg")
        if svg_element:
            bounding_box = await svg_element.bounding_box()
            if bounding_box:
                # Screenshot with high quality settings
                await page.screenshot(
                    path=str(output_path),
                    clip={
                        'x': bounding_box['x'],
                        'y': bounding_box['y'],
                        'width': bounding_box['width'],
                        'height': bounding_box['height']
                    },
                    scale='device'  # Use device scale for retina/high-DPI
                )

        await browser.close()


async def convert_all_diagrams(diagrams_dir: Path):
    """Convert all HTML diagrams to PNG"""
    html_files = sorted(diagrams_dir.glob("diagram_*.html"))

    if not html_files:
        print(f"No diagram HTML files found in {diagrams_dir}")
        return

    print(f"Found {len(html_files)} HTML diagrams to convert")
    print()

    for html_file in html_files:
        png_file = html_file.with_suffix('.png')
        print(f"Converting: {html_file.name} -> {png_file.name}")

        try:
            await html_to_png(html_file, png_file)
            print(f"  ✓ Success: {png_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print("=" * 70)
    print(f"Conversion complete! Generated {len(html_files)} PNG files")
    print(f"Location: {diagrams_dir}")
    print("=" * 70)


def main():
    """Main entry point"""
    # Get diagrams directory
    script_dir = Path(__file__).parent
    diagrams_dir = script_dir / "diagrams"

    if not diagrams_dir.exists():
        print(f"Error: Diagrams directory not found at {diagrams_dir}")
        sys.exit(1)

    # Run async conversion
    asyncio.run(convert_all_diagrams(diagrams_dir))


if __name__ == "__main__":
    main()
