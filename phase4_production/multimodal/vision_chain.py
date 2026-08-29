import base64
import json
import time
import requests
from pathlib import Path
from typing import List, Optional
from phase4_production.multimodal.figure_extractor import (
    ExtractedFigure,
    FigureExtractor
)


# ─────────────────────────────────────────
# Ollama LLaVA configuration
# ─────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"   # Ollama default port
LLAVA_MODEL      = "llava"                    # vision model
OLLAMA_TIMEOUT   = 120                        # seconds per image


# ─────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────
FIGURE_DESCRIPTION_PROMPT = """You are analyzing a figure from a research paper about underwater image enhancement.

Describe this figure in detail covering:
1. What TYPE of figure it is (pipeline diagram, result comparison, graph, chart, table, equation, architecture diagram)
2. What it SHOWS (key components, labels, arrows, flow)
3. What it MEANS in the context of underwater image processing
4. Any NUMBERS, METRICS, or KEY TERMS visible

Be specific and technical. Your description will be used to answer questions about this figure.
Do not say "the image shows" — directly describe what is in the figure."""


# ─────────────────────────────────────────
# Vision Chain
# ─────────────────────────────────────────
class VisionChain:
    """
    Sends extracted PDF figures to LLaVA vision model
    and generates searchable text descriptions.

    Why LLaVA over other vision models?
      GPT-4V    → expensive API, not free
      Claude    → API cost, not local
      LLaVA     → free, local, runs on M5 via Ollama
                  "Large Language and Vision Assistant"
                  Fine-tuned LLaMA with vision encoder
                  Handles diagrams, charts, text in images

    Why Ollama API over Python library?
      ollama Python library → adds dependency
      Direct HTTP to localhost:11434 → simpler
      Same result, zero extra dependencies
      Works exactly like calling any REST API

    How LLaVA processes images:
      Image → CLIP vision encoder → visual tokens
      Visual tokens + text prompt → LLaMA decoder
      LLaMA generates description token by token
      Same transformer architecture as text LLaMA
      but with visual tokens prepended to the sequence
    """

    def __init__(
        self,
        model:    str = LLAVA_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        self.model    = model
        self.base_url = base_url
        self.api_url  = f"{base_url}/api/generate"

        # Verify Ollama is running
        self._check_ollama()
        print(f"[VisionChain] Ready — model: {self.model}")

    def _check_ollama(self):
        """
        Verify Ollama server is running and LLaVA is available.

        Why check upfront?
          Fail fast — better to know immediately than
          after processing 5 figures that Ollama isn't running
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            models = [m["name"] for m in response.json().get("models", [])]

            # Check if any llava variant is available
            llava_available = any("llava" in m for m in models)

            if not llava_available:
                print(f"[VisionChain] WARNING: LLaVA not found in Ollama")
                print(f"[VisionChain] Run: ollama pull llava")
                print(f"[VisionChain] Available models: {models}")
            else:
                print(f"[VisionChain] Ollama running — LLaVA available ✅")

        except requests.exceptions.ConnectionError:
            print(f"[VisionChain] WARNING: Ollama not running!")
            print(f"[VisionChain] Start with: ollama serve")

    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 string.

        Why base64?
          Ollama API accepts images as base64 strings
          Standard way to send binary data in JSON
          Base64 encodes binary → ASCII text
          ~33% larger than original but universally compatible

        How base64 works:
          Every 3 bytes of binary → 4 ASCII characters
          Uses A-Z, a-z, 0-9, +, / (64 characters total)
          Safe to embed in JSON, URLs, HTML
        """
        with open(image_path, "rb") as f:
            image_bytes  = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")

    def _call_llava(
        self,
        image_path: str,
        prompt:     str = FIGURE_DESCRIPTION_PROMPT
    ) -> Optional[str]:
        """
        Send image to LLaVA via Ollama API and get description.

        Ollama API format:
          POST /api/generate
          {
            "model": "llava",
            "prompt": "describe this image",
            "images": ["base64_encoded_image"],
            "stream": false
          }

        Why stream=false?
          stream=true sends tokens one by one (good for UI)
          stream=false waits for complete response
          For batch processing, false is simpler

        Why timeout=120?
          LLaVA on M5 takes 10-40 seconds per image
          Complex diagrams take longer than simple graphs
          120 seconds is generous but safe
        """
        try:
            image_b64 = self._image_to_base64(image_path)

            payload = {
                "model":  self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,   # low = more factual descriptions
                    "num_predict": 500,   # max tokens in description
                }
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.Timeout:
            print(f"[VisionChain] Timeout on {image_path}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[VisionChain] Ollama not running — start with: ollama serve")
            return None
        except Exception as e:
            print(f"[VisionChain] Error on {image_path}: {e}")
            return None

    def describe_figure(self, figure: ExtractedFigure) -> ExtractedFigure:
        """
        Generate description for a single figure.

        Returns the same figure with description filled in.

        Why return the figure object?
          Immutable-style update — caller gets back
          a complete, enriched object
          Easy to chain: figure = describe_figure(figure)
        """
        print(
            f"[VisionChain] Describing {figure.figure_id} "
            f"(Page {figure.page_number}, "
            f"{figure.width}x{figure.height}px)..."
        )

        if not Path(figure.image_path).exists():
            print(f"[VisionChain] Image not found: {figure.image_path}")
            return figure

        start_time  = time.time()
        description = self._call_llava(figure.image_path)
        elapsed     = round(time.time() - start_time, 1)

        if description:
            figure.description = description
            print(f"[VisionChain] ✅ Done in {elapsed}s")
            print(f"[VisionChain] Preview: {description[:100]}...")
        else:
            print(f"[VisionChain] ❌ Failed — no description generated")

        return figure

    def describe_all(
        self,
        figures:       List[ExtractedFigure],
        metadata_path: str,
        delay:         float = 2.0
    ) -> List[ExtractedFigure]:
        """
        Generate descriptions for all extracted figures.
        Saves enriched metadata after each figure.

        Why save after each figure (not at the end)?
          If LLaVA crashes on figure 8/20 →
          figures 1-7 descriptions are not lost
          Resume from figure 8 on next run
          Production pattern: checkpoint as you go

        Args:
            figures:       list from FigureExtractor
            metadata_path: JSON file to update with descriptions
            delay:         seconds between LLaVA calls
                          Why delay? LLaVA needs time to
                          unload/reload between calls on M5
                          Prevents memory pressure buildup

        Returns:
            figures with descriptions filled in
        """
        print(f"\n[VisionChain] Describing {len(figures)} figures...")
        print(f"[VisionChain] Model: {self.model}")
        print(f"[VisionChain] Estimated time: ~{len(figures) * 20}s\n")

        described   = []
        failed      = 0

        for i, figure in enumerate(figures, 1):
            print(f"\n[VisionChain] Figure {i}/{len(figures)}")

            # Skip if already described (resume capability)
            if figure.description:
                print(f"[VisionChain] Already described — skipping")
                described.append(figure)
                continue

            enriched = self.describe_figure(figure)
            described.append(enriched)

            if not enriched.description:
                failed += 1

            # Save checkpoint after each figure
            self._save_metadata(described, metadata_path)

            # Delay between calls
            if i < len(figures):
                time.sleep(delay)

        # Final summary
        successful = len(figures) - failed
        print(f"\n[VisionChain] ── Description Summary ──")
        print(f"  Total figures    : {len(figures)}")
        print(f"  Successfully described: {successful}")
        print(f"  Failed           : {failed}")
        print(f"  Metadata saved   : {metadata_path}")

        return described

    def _save_metadata(
        self,
        figures:       List[ExtractedFigure],
        metadata_path: str
    ):
        """
        Save figure metadata with descriptions to JSON.

        Why JSON and not database?
          Figures are tied to a specific PDF
          JSON is portable — easy to inspect manually
          Database is overkill for figure metadata
          PostgreSQL will store reference to this JSON later
        """
        from dataclasses import asdict
        with open(metadata_path, "w") as f:
            json.dump(
                [asdict(fig) for fig in figures],
                f,
                indent=2
            )

    def build_text_chunks(
        self,
        figures: List[ExtractedFigure]
    ) -> List[dict]:
        """
        Convert figure descriptions into text chunks
        compatible with Phase 1/2 ChromaDB format.

        Why convert to chunks?
          ChromaDB stores text chunks + metadata
          Figure descriptions ARE text chunks
          Same embedding + retrieval pipeline works
          No new infrastructure needed — just more chunks

        Chunk format matches phase1_fundamentals/chunker.py:
          chunk_id, source_file, page_number,
          paragraph_index, text, token_count

        Special markers:
          paragraph_index = 999  → signals this is a figure chunk
          chunk_id prefix = "fig_" → easy to identify in citations
        """
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")

        chunks = []
        for figure in figures:
            if not figure.description:
                continue

            # Build rich text that includes figure context
            # Why include metadata in text?
            # Embedding captures semantic content
            # Including "Figure 1, Page 3" means queries
            # about "figure 1" or "page 3 diagram" find it
            text = (
                f"Figure {figure.figure_index} on Page {figure.page_number}: "
                f"{figure.description}\n"
                f"[Image: {figure.image_path}]"
            )

            tokens = encoder.encode(text)

            chunk = {
                "chunk_id":        f"fig_{figure.figure_id}",
                "source_file":     figure.source_file,
                "page_number":     figure.page_number,
                "paragraph_index": 999,          # figure marker
                "text":            text,
                "token_count":     len(tokens),
                "figure_id":       figure.figure_id,
                "image_path":      figure.image_path,
                "is_figure":       True
            }
            chunks.append(chunk)

        print(f"[VisionChain] Built {len(chunks)} figure text chunks")
        return chunks


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/moac.pdf"
    stem     = Path(pdf_path).stem
    figures_json = f"data/figures/{stem}_figures.json"

    # Load existing figures from extractor
    extractor = FigureExtractor()
    figures   = extractor.load_metadata(pdf_path)

    if not figures:
        print("No figures found — run figure_extractor.py first")
        sys.exit(1)

    print(f"Loaded {len(figures)} figures")

    # Describe all figures
    chain     = VisionChain()
    described = chain.describe_all(figures, figures_json)

    # Build text chunks for ChromaDB
    chunks = chain.build_text_chunks(described)

    print(f"\n--- Preview of first figure chunk ---")
    if chunks:
        print(f"ID    : {chunks[0]['chunk_id']}")
        print(f"Page  : {chunks[0]['page_number']}")
        print(f"Text  : {chunks[0]['text'][:200]}...")