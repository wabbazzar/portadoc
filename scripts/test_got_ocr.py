#!/usr/bin/env python3
"""
Test GOT-OCR2.0 on degraded PDF.

Usage:
    source .venv/bin/activate
    pip install tiktoken verovio accelerate
    python scripts/test_got_ocr.py data/input/peter_lou_50dpi.pdf

First run will download ~1.4GB model.
Expected: 3-4 minutes per page on CPU.
"""

import sys
import time
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

# Add src to path for portadoc imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portadoc.pdf import load_pdf


def load_got_model():
    """Load GOT-OCR model for CPU inference."""
    print("Loading GOT-OCR2.0 model (CPU)...")
    print("First run downloads ~1.4GB - be patient.")

    from transformers import AutoModel, AutoTokenizer

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        'srimanth-d/GOT_CPU',
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        'srimanth-d/GOT_CPU',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id
    )
    model = model.eval()

    print(f"Model loaded in {time.time() - start:.1f}s")
    return model, tokenizer


def image_to_tempfile(img_array: np.ndarray) -> str:
    """Save numpy image to temp file for GOT input."""
    img = Image.fromarray(img_array)
    tmp = NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name)
    return tmp.name


def run_got_ocr(model, tokenizer, image_path: str, ocr_type: str = 'ocr') -> str:
    """
    Run GOT-OCR on a single image.

    Args:
        model: GOT model
        tokenizer: GOT tokenizer
        image_path: Path to image file
        ocr_type: 'ocr' for plain text, 'format' for structured

    Returns:
        Extracted text
    """
    result = model.chat(tokenizer, image_path, ocr_type=ocr_type)
    return result


def run_got_ocr_with_boxes(model, tokenizer, image_path: str) -> list[dict]:
    """
    Run GOT-OCR with fine-grained bounding boxes.

    Returns list of {text, bbox} dicts.
    """
    # Fine-grained OCR returns text with box coordinates
    # Format: <ref>text</ref><box>[[x0,y0,x1,y1]]</box>
    result = model.chat(
        tokenizer,
        image_path,
        ocr_type='ocr',
        ocr_box=''  # Empty string triggers bbox output
    )

    # Parse the result - GOT returns structured format
    # This is simplified; actual parsing may need refinement
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_got_ocr.py <pdf_path> [--format] [--boxes]")
        print("\nOptions:")
        print("  --format  Use formatted OCR (preserves structure)")
        print("  --boxes   Attempt fine-grained bbox extraction")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    use_format = '--format' in sys.argv
    use_boxes = '--boxes' in sys.argv
    ocr_type = 'format' if use_format else 'ocr'

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    # Load model
    model, tokenizer = load_got_model()

    # Process PDF
    output_path = Path('data/output') / f"{pdf_path.stem}_got.csv"
    output_path.parent.mkdir(exist_ok=True)

    results = []
    total_time = 0

    print(f"\nProcessing: {pdf_path}")
    print(f"OCR type: {ocr_type}")
    print("-" * 50)

    with load_pdf(pdf_path, dpi=300) as pdf:
        for page_num, img, page_width, page_height in pdf.pages():
            print(f"\nPage {page_num + 1}/{len(pdf)}...")

            # Save image to temp file (GOT needs file path)
            tmp_path = image_to_tempfile(img)

            # Run OCR
            start = time.time()
            try:
                if use_boxes:
                    text = run_got_ocr_with_boxes(model, tokenizer, tmp_path)
                else:
                    text = run_got_ocr(model, tokenizer, tmp_path, ocr_type)
                elapsed = time.time() - start
                total_time += elapsed

                print(f"  Time: {elapsed:.1f}s")
                print(f"  Text length: {len(text)} chars")
                print(f"  Preview: {text[:200]}..." if len(text) > 200 else f"  Text: {text}")

                results.append({
                    'page': page_num,
                    'text': text,
                    'time_sec': elapsed,
                    'img_width': img.shape[1],
                    'img_height': img.shape[0],
                    'page_width_pts': page_width,
                    'page_height_pts': page_height,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'page': page_num,
                    'text': f'ERROR: {e}',
                    'time_sec': 0,
                })

            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    # Write results
    print("\n" + "=" * 50)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per page: {total_time/len(results):.1f}s")

    # Save full text output
    text_output = output_path.with_suffix('.txt')
    with open(text_output, 'w') as f:
        for r in results:
            f.write(f"=== PAGE {r['page']} ===\n")
            f.write(r['text'])
            f.write("\n\n")
    print(f"Text saved to: {text_output}")

    # Save CSV summary
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['page', 'time_sec', 'text_length', 'preview'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'page': r['page'],
                'time_sec': f"{r['time_sec']:.1f}",
                'text_length': len(r['text']),
                'preview': r['text'][:100].replace('\n', ' ')
            })
    print(f"Summary saved to: {output_path}")


if __name__ == '__main__':
    main()
