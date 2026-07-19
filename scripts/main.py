"""CLI entry point for running the Poisson-image-editing effects.

The script loads source, destination, and mask images from the data directory,
selects the requested effect, and saves the generated output under the matching
output folder. It also handles optional resizing and metadata-based offsets.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import sys
from pathlib import Path
from typing import Any, Callable

# Make `src/` importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from editing.io_utils import load_image, save_image
from editing.masks import load_mask
from editing.effects.clone import seamless_cloning
from editing.effects.mixed import mixed_gradients
from editing.effects.flatten import texture_flattening
from editing.effects.illumination import local_illumination_change
from editing.effects.color import local_color_change
from editing.effects.tile import seamless_tiling

EFFECTS: dict[str, Callable[..., Any]] = {
    "seamless_cloning": seamless_cloning,
    "mixed_gradients": mixed_gradients,
    "texture_flattening": texture_flattening,
    "local_illumination_change": local_illumination_change,
    "local_color_change": local_color_change,
    "seamless_tiling": seamless_tiling,
}


def load_metadata(input_dir: Path) -> dict[str, Any]:
    """Read optional JSON metadata from an effect input directory."""
    meta_path = input_dir / "meta.json"

    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_offset(input_dir: Path) -> tuple[int, int]:
    """Load the patch offset from an input directory's metadata file."""
    metadata = load_metadata(input_dir)
    offset = metadata.get("offset", [0, 0])

    if not isinstance(offset, list) or len(offset) != 2:
        raise ValueError(f"Invalid offset in {input_dir / 'meta.json'}")

    return int(offset[0]), int(offset[1])

def run_effect(effect_name: str, mode: str = 'subject') -> None:
    """Run one effect end to end and write its output to disk."""
    if effect_name not in EFFECTS:
        raise ValueError(f"Unknown effect: {effect_name}")

    input_dir = ROOT / "data" / "input" / effect_name
    output_dir = ROOT / "data" / "output" / effect_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = input_dir / "source.png"
    mask_path = input_dir / "mask.png"
    destination_path = input_dir / "destination.png"

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image: {source_path}")

    source = load_image(source_path)
    mask = load_mask(mask_path) if mask_path.exists() else None
    destination = load_image(destination_path) if destination_path.exists() else None

    MAX_SIZE = 1280  # Maximum dimension for processing (to avoid excessive memory usage)
    h_s, w_s = source.shape[:2]

    max_dim = max(h_s, w_s)
    if destination is not None:
        max_dim = max(max_dim, destination.shape[0], destination.shape[1])

    if max_dim > MAX_SIZE:
        from PIL import Image
        scale = MAX_SIZE / max_dim

        new_size_s = (int(w_s * scale), int(h_s * scale))
        source = np.asarray(Image.fromarray(source).resize(new_size_s, Image.Resampling.BILINEAR))

        if mask is not None:
            mask_uint8 = (mask.astype(np.uint8) * 255)
            mask = np.asarray(Image.fromarray(mask_uint8).resize(new_size_s, Image.Resampling.NEAREST)) > 128

        if destination is not None:
            h_d, w_d = destination.shape[:2]
            new_size_d = (int(w_d * scale), int(h_d * scale))
            destination = np.asarray(Image.fromarray(destination).resize(new_size_d, Image.Resampling.BILINEAR))

        raw_offset = load_offset(input_dir)
        offset = (int(raw_offset[0] * scale), int(raw_offset[1] * scale))
    else:
        offset = load_offset(input_dir)

    effect_fn = EFFECTS[effect_name]

    if effect_name in {"seamless_cloning", "mixed_gradients"}:
        if destination is None or mask is None:
            raise FileNotFoundError(f"{effect_name} requires source.png, destination.png and mask.png")
        result = effect_fn(source, destination, mask, offset)

    elif effect_name in {"texture_flattening", "local_illumination_change", "local_color_change"}:
        if mask is None:
            raise FileNotFoundError(f"{effect_name} requires source.png and mask.png")
        result = effect_fn(source, mask, mode=mode)

    elif effect_name == "seamless_tiling":
        result = effect_fn(source)

        reps = (2, 2, 1) if result.ndim == 3 else (2, 2)
        tiled_preview = np.tile(result, reps)
        save_image(tiled_preview, output_dir / "tiled_preview.png")

    else:
        raise ValueError(f"Unknown effect: {effect_name}")

    save_image(result, output_dir / "result.png")
    print(f"Saved result to {output_dir / 'result.png'}")

def main() -> None:
    """Parse command-line arguments and execute the requested effect."""
    parser = argparse.ArgumentParser(description="Run a Poisson image editing effect.")
    parser.add_argument(
        "--effect",
        required=True,
        choices=sorted(EFFECTS.keys()),
        help="Effect to execute (matches the folder name under data/input/).",
    )
    parser.add_argument(
        "--mode",
        default="subject",
        choices=["subject", "background"],
        help="Mode for local_color_change (only relevant for that effect).",
    )

    args = parser.parse_args()
    run_effect(args.effect, mode=args.mode)


if __name__ == "__main__":
    main()