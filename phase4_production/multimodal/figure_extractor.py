import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


# ─────────────────────────────────────────
# Data model for extracted figures
# ─────────────────────────────────────────
@dataclass
class ExtractedFigure:
    """
    Represents a single figure extracted from a PDF page.

    Why a dataclass?
    Clean, typed, serializable — can save to JSON,
    pass to vision model, store in ChromaDB metadata.
    """
    figure_id:    str    # unique ID: moac_p3_f1
    source_file:  str    # moac.pdf
    page_number:  int    # 3
    figure_index: int    # 1 (first figure on this page)
    image_path:   str    # data/figures/moac_p3_f1.png
    width:        int    # pixel width
    height:       int    # pixel height
    description:  Optional[str] = None  # filled by vision_chain.py later


# ─────────────────────────────────────────
# Figure Extractor
# ─────────────────────────────────────────
class FigureExtractor:
    """
    Extracts all figures from a PDF using PyMuPDF.

    Why PyMuPDF?
    - Already in your stack (used in loader.py)
    - get_images() gives direct access to embedded images
    - Faster than converting pages to screenshots
    - Preserves original image quality

    Alternative approach (not used):
    - Render each page as image, then detect figures with CV
    - Much slower, loses original image quality
    - Overkill for structured research PDFs
    """

    # Minimum figure size to extract
    # Why? PDFs contain many tiny images:
    #   - icons, bullets, decorative elements
    #   - These are noise, not meaningful figures
    # 100x100 pixels filters out decorative elements
    # while keeping all meaningful diagrams/graphs
    MIN_WIDTH  = 100   # pixels
    MIN_HEIGHT = 100   # pixels

    def __init__(self, output_dir: str = "data/figures"):
        """
        Args:
            output_dir: Where extracted figures are saved
                        Separate from raw PDFs and ChromaDB
                        Keeps data/ folder organized
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FigureExtractor] Ready — saving figures to {output_dir}")

    def _build_figure_id(
        self,
        source_file: str,
        page_number: int,
        figure_index: int
    ) -> str:
        """
        Build a unique, readable figure ID.

        Format: {stem}_p{page}_f{index}
        Example: moac_p3_f1

        Why readable IDs?
        - Easy to debug ("which figure failed?")
        - Used as ChromaDB chunk ID later
        - Shows in citations: [Figure: moac_p3_f1]
        """
        stem = Path(source_file).stem   # "moac.pdf" → "moac"
        return f"{stem}_p{page_number}_f{figure_index}"

    def _is_valid_figure(self, width: int, height: int) -> bool:
        """
        Filter out tiny decorative images.

        Why this matters:
        Research PDFs contain many small images:
        - Publisher logos (~20x20px)
        - Bullet point icons (~10x10px)
        - Section dividers (~5x5px)
        These are noise — we only want meaningful figures.
        """
        return width >= self.MIN_WIDTH and height >= self.MIN_HEIGHT

    def extract_from_pdf(
        self,
        pdf_path: str,
        save_metadata: bool = True
    ) -> List[ExtractedFigure]:
        """
        Main extraction method.

        How PyMuPDF figure extraction works:
        1. Open PDF → get page count
        2. For each page → call page.get_images()
           Returns list of image references (xref numbers)
        3. For each xref → doc.extract_image(xref)
           Returns raw image bytes + format (png/jpeg)
        4. Filter by size → save to disk → create metadata

        Why xref (cross-reference)?
        PDF stores images in a central object table
        Each image has an xref number (like a pointer)
        get_images() returns these references
        extract_image() dereferences them to get actual bytes

        Args:
            pdf_path: Path to PDF file
            save_metadata: Save figure list to JSON for reuse

        Returns:
            List of ExtractedFigure objects
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        source_file = Path(pdf_path).name   # "moac.pdf"
        figures     = []
        seen_xrefs  = set()  # avoid extracting same image twice
                             # (PDFs often reference same image
                             #  from multiple pages — e.g. logos)

        print(f"\n[FigureExtractor] Opening: {pdf_path}")
        doc = fitz.open(pdf_path)
        print(f"[FigureExtractor] Pages: {len(doc)}")

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_number = page_num + 1   # 1-indexed (matches chunker)

            # get_images() returns list of:
            # (xref, smask, width, height, bpc, colorspace,
            #  alt_colorspace, name, filter, referencer)
            image_list = page.get_images(full=True)

            if not image_list:
                continue

            figure_index = 0

            for img_info in image_list:
                xref   = img_info[0]   # cross-reference number
                width  = img_info[2]   # pixel width
                height = img_info[3]   # pixel height

                # Skip if already extracted from another page
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                # Skip tiny decorative images
                if not self._is_valid_figure(width, height):
                    continue

                figure_index += 1
                figure_id = self._build_figure_id(
                    source_file, page_number, figure_index
                )

                # Extract raw image bytes
                # extract_image returns dict:
                # {image: bytes, ext: "png"/"jpeg", ...}
                try:
                    image_data = doc.extract_image(xref)
                    image_bytes  = image_data["image"]
                    image_format = image_data["ext"]   # png, jpeg, etc.

                    # Save to disk
                    image_filename = f"{figure_id}.{image_format}"
                    image_path     = self.output_dir / image_filename

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Create figure metadata object
                    figure = ExtractedFigure(
                        figure_id    = figure_id,
                        source_file  = source_file,
                        page_number  = page_number,
                        figure_index = figure_index,
                        image_path   = str(image_path),
                        width        = width,
                        height       = height,
                        description  = None   # filled by vision_chain.py
                    )
                    figures.append(figure)

                    print(
                        f"[FigureExtractor] Page {page_number} | "
                        f"Figure {figure_index} | "
                        f"{width}x{height}px | "
                        f"Saved: {image_filename}"
                    )

                except Exception as e:
                    print(
                        f"[FigureExtractor] Failed on xref {xref} "
                        f"page {page_number}: {e}"
                    )
                    continue

        total_pages = len(doc)
        doc.close()

        # Save metadata to JSON
        # Why? vision_chain.py needs to know which figures exist
        # without re-running extraction every time
        if save_metadata and figures:
            metadata_path = self.output_dir / f"{Path(pdf_path).stem}_figures.json"
            with open(metadata_path, "w") as f:
                json.dump(
                    [asdict(fig) for fig in figures],
                    f,
                    indent=2
                )
            print(f"[FigureExtractor] Metadata saved: {metadata_path}")

        # Summary
        print(f"\n[FigureExtractor] ── Extraction Summary ──")
        print(f"  PDF           : {pdf_path}")
        print(f"  Pages scanned : {total_pages}")
        print(f"  Figures found : {len(figures)}")
        print(f"  Saved to      : {self.output_dir}")

        return figures

    def load_metadata(self, pdf_path: str) -> List[ExtractedFigure]:
        """
        Load previously extracted figure metadata from JSON.

        Why this matters for optimization:
        If PDF was already processed → skip re-extraction
        This is part of the caching strategy:
          check JSON exists → load → skip PyMuPDF entirely
        Saves ~2-5 seconds on repeated ingestion
        """
        stem          = Path(pdf_path).stem
        metadata_path = self.output_dir / f"{stem}_figures.json"

        if not metadata_path.exists():
            return []

        with open(metadata_path, "r") as f:
            data = json.load(f)

        figures = [ExtractedFigure(**fig) for fig in data]
        print(f"[FigureExtractor] Loaded {len(figures)} figures from cache")
        return figures


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python figure_extractor.py <path_to_pdf>")
        sys.exit(1)

    extractor = FigureExtractor()
    figures   = extractor.extract_from_pdf(sys.argv[1])

    print(f"\n--- Preview of extracted figures ---")
    for fig in figures[:5]:
        print(f"\nID      : {fig.figure_id}")
        print(f"Page    : {fig.page_number}")
        print(f"Size    : {fig.width}x{fig.height}px")
        print(f"Path    : {fig.image_path}")