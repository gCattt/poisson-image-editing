"""Interactive helper for placing a source patch onto a destination image.

This script loads the source image, mask, and destination image for a given
editing effect, lets the user place the patch interactively using drag-and-drop
or keyboard arrow keys, and saves the chosen offset into the effect's metadata.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make `src/` importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from editing.io_utils import load_image
from editing.masks import load_mask


class InteractiveDragger:
    """Handles mouse drag-and-drop events and keyboard fine-tuning for the patch overlay."""

    def __init__(self, ax, patch_plot, current_offset, w_crop, h_crop):
        self.ax = ax
        self.patch_plot = patch_plot
        
        # Current top-left corner coordinates of the patch
        self.offset_x = current_offset[1]
        self.offset_y = current_offset[0]
        self.w_crop = w_crop
        self.h_crop = h_crop
        
        # Tracks click states during drag operations
        self.press = None

        # Synchronize plot state and make the patch visible
        self.update_plot()
        self.patch_plot.set_visible(True)

    def connect(self):
        """Connect UI interaction callbacks to the Matplotlib canvas."""
        canvas = self.patch_plot.figure.canvas
        self.cidpress = canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkeypress = canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        """Handle mouse click events to initiate dragging or trigger teleportation."""
        if event.inaxes != self.ax:
            return

        # Teleport patch center to click position if clicked outside current patch bounds
        if not (self.offset_x <= event.xdata <= self.offset_x + self.w_crop and
                self.offset_y <= event.ydata <= self.offset_y + self.h_crop):
            self.offset_x = int(event.xdata - self.w_crop / 2)
            self.offset_y = int(event.ydata - self.h_crop / 2)
            self.update_plot()

        # Cache initial state for relative displacement tracking
        self.press = (self.offset_x, self.offset_y, event.xdata, event.ydata)

    def on_motion(self, event):
        """Update patch coordinates dynamically during an active drag operation."""
        if self.press is None or event.inaxes != self.ax:
            return
        
        ox, oy, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        
        self.offset_x = int(ox + dx)
        self.offset_y = int(oy + dy)
        self.update_plot()

    def on_release(self, event):
        """Reset drag state variables upon mouse button release."""
        self.press = None
        self.patch_plot.figure.canvas.draw()

    def on_key(self, event):
        """Handle arrow key strokes for pixel-perfect manual adjustments."""
        if event.key is None:
            return
        
        # Move by 10 pixels if Shift is held down, otherwise default to 1 pixel
        step = 10 if 'shift' in event.key else 1
        
        if 'right' in event.key:
            self.offset_x += step
        elif 'left' in event.key:
            self.offset_x -= step
        elif 'up' in event.key:
            self.offset_y -= step
        elif 'down' in event.key:
            self.offset_y += step
        else:
            return
            
        self.update_plot()

    def update_plot(self):
        """Redraw the patch graphic layer at its updated positional coordinates."""
        self.patch_plot.set_extent([self.offset_x, self.offset_x + self.w_crop,
                                    self.offset_y + self.h_crop, self.offset_y])
        self.patch_plot.figure.canvas.draw()


def interactive_placement(effect_name: str):
    """Interactively place a source patch onto the destination image."""
    base_dir = ROOT / "data" / "input" / effect_name
    if not base_dir.exists():
        print(f"Error: The folder {base_dir} does not exist.")
        return

    # Load project image resources
    dest = load_image(base_dir / "destination.png")
    source = load_image(base_dir / "source.png")
    mask = load_mask(base_dir / "mask.png")

    # Locate mask boundaries to establish crop dimensions
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        print("Error: The mask is empty.")
        return

    h_s, w_s = source.shape[:2]
    h_d, w_d = dest.shape[:2]

    # Pad bounding box borders by 1 pixel to match standard processing pipelines
    ymin = max(0, ys.min() - 1)
    ymax = min(h_s - 1, ys.max() + 1)
    xmin = max(0, xs.min() - 1)
    xmax = min(w_s - 1, xs.max() + 1)

    src_crop = source[ymin : ymax + 1, xmin : xmax + 1]
    mask_crop = mask[ymin : ymax + 1, xmin : xmax + 1]

    # Build an RGBA patch overlay: RGB channels from source, Alpha set to 50%
    # capacity (128/255) strictly inside active mask region.
    h_crop, w_crop = src_crop.shape[:2]
    patch_rgba = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
    patch_rgba[..., :3] = src_crop
    patch_rgba[..., 3] = np.where(mask_crop > 0, 128, 0)

    # Initialize the visualization window
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layer 1: Render the target canvas backdrop at full opacity
    ax.imshow(dest)

    # Enforce strict canvas view boundaries based on destination image geometry.
    # This prevents automatic zooming modifications when dragging objects outside borders.
    ax.set_xlim(0, w_d)
    ax.set_ylim(h_d, 0)
    ax.set_autoscale_on(False) 

    # Layer 2: Setup the target patch layer (initially hidden)
    patch_plot = ax.imshow(patch_rgba, extent=[0, w_crop, h_crop, 0], zorder=2)
    patch_plot.set_visible(False)

    ax.set_title(
        f"Placement for: {effect_name}\n"
        "Click to teleport. Drag to move. Arrow keys for fine-tuning. Close to save."
    )

    # Recover stored spatial offsets from meta.json if accessible
    meta_path = base_dir / "meta.json"
    current_offset = [0, 0]
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            current_offset = meta.get("offset", [0, 0])

    # Validate historical offset variables; fallback to image center if corrupted or out of bounds
    if not (0 <= current_offset[1] <= w_d and 0 <= current_offset[0] <= h_d):
        print("Previous offset out of bounds. Resetting patch position to canvas center.")
        current_offset = [max(0, h_d // 2 - h_crop // 2), max(0, w_d // 2 - w_crop // 2)]

    # Bind UI components and start interaction loop
    dragger = InteractiveDragger(ax, patch_plot, current_offset, w_crop, h_crop)
    dragger.connect()

    plt.show()  # Process blocks execution until GUI window termination

    # Commit localized adjustments back to JSON metadata
    meta["offset"] = [dragger.offset_y, dragger.offset_x]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[{effect_name}] Offset saved successfully: Y={dragger.offset_y}, X={dragger.offset_x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive tool for placing the mask.")
    parser.add_argument(
        "--effect",
        choices=["seamless_cloning", "mixed_gradients"],
        help="The effect for which you want to compute the offset.",
    )
    args = parser.parse_args()

    interactive_placement(args.effect)