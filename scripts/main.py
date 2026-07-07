from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Callable

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


EFFECTS: dict[str, Callable[..., object]] = {
	"seamless_cloning": seamless_cloning,
	"mixed_gradients": mixed_gradients,
	"texture_flattening": texture_flattening,
	"local_illumination_change": local_illumination_change,
	"local_color_change": local_color_change,
	"seamless_tiling": seamless_tiling,
}

def run_effect(effect_name: str) -> None:
    input_dir = ROOT / "data" / "input" / effect_name
    output_dir = ROOT / "data" / "output" / effect_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if effect_name not in EFFECTS:
        raise ValueError(f"Unknown effect: {effect_name}")

    source_path = input_dir / "source.png"
    mask_path = input_dir / "mask.png"
    destination_path = input_dir / "destination.png"

    if not source_path.exists(): 
        raise FileNotFoundError(f"Missing source image: {source_path}")

    source = load_image(source_path)
    mask = load_mask(mask_path) if mask_path.exists() else None
    destination = load_image(destination_path) if destination_path.exists() else None

    effect_fn = EFFECTS[effect_name]
    
    if effect_name in {"seamless_cloning", "mixed_gradients"}: 
        if destination is None or mask is None: 
            raise FileNotFoundError( f"{effect_name} requires source.png, destination.png and mask.png" ) 
        result = effect_fn(source, destination, mask)

    elif effect_name in { 
        "texture_flattening", 
        "local_illumination_change", 
        "local_color_change", 
    }: 
        if mask is None: 
            raise FileNotFoundError(f"{effect_name} requires source.png and mask.png") 
        result = effect_fn(source, mask)

    elif effect_name == "seamless_tiling": 
        result = effect_fn(source)

    else: 
        raise ValueError(f"Unknown effect: {effect_name}")

    save_image(result, output_dir / "result.png")
    print(f"Saved result to {output_dir / 'result.png'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Poisson image editing effect.")
    parser.add_argument(
        "--example",
        required=True,
        choices=sorted(EFFECTS.keys()),
        help="Effect to execute (matches the folder name under data/input/).",
    )
    args = parser.parse_args()
    run_effect(args.example)


if __name__ == "__main__":
    main()