#!/usr/bin/env python3
"""
Wound Annotation Tool for GC-ACT Autonomous Suturing Project.

Interactive OpenCV-based tool to annotate wound lines and insertion/exit
points on real dVRK endoscope images. Annotations are saved to JSON for
training a wound segmentation model.

Usage:
    # Annotate 20 randomly sampled episodes from tissue 7 needle throw:
    python annotate_wounds.py --image_dir ~/data/tissue_7/2_needle_throw/ --sample 20

    # Annotate specific images:
    python annotate_wounds.py --image_list img1.jpg img2.jpg img3.jpg

    # Resume from existing annotations (skip already-done images):
    python annotate_wounds.py --image_dir ~/data/tissue_7/2_needle_throw/ --sample 20 --resume

    # Specify a custom output file:
    python annotate_wounds.py --image_dir ~/data/tissue_7/2_needle_throw/ --output my_annotations.json

NOTE: This tool requires a display (X11). If running on a remote instance
(e.g., Jetstream2), use X11 forwarding:
    ssh -X user@host
or set up a VNC/noVNC session. Without a display, OpenCV windows will not open.

Controls:
    Left click  - Add a point (wound point in Phase 1, stitch point in Phase 2)
    'u'         - Undo last point
    'c'         - Clear all annotations for the current image
    'n'         - Next phase (Phase 1 -> Phase 2) or save & next image (Phase 2)
    's'         - Save current annotations and move to next image
    'q'         - Save all annotations and quit
    ESC         - Quit without saving current image (previously saved images are kept)

Author: GC-ACT Suturing Project
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np


# Colors (BGR)
COLOR_WOUND = (0, 255, 255)       # Yellow - wound curve points
COLOR_WOUND_LINE = (0, 200, 200)  # Slightly darker yellow - wound lines
COLOR_INSERTION = (255, 150, 0)   # Blue - insertion points
COLOR_EXIT = (0, 140, 255)        # Orange - exit points
COLOR_STITCH_LINE = (0, 255, 0)   # Green - insertion-exit pair lines
COLOR_TEXT_BG = (40, 40, 40)      # Dark gray - text background
COLOR_TEXT = (255, 255, 255)      # White - text
COLOR_PHASE_LABEL = (0, 200, 255) # Yellow-orange - phase label
COLOR_CROSSHAIR = (180, 180, 180) # Light gray - crosshair

POINT_RADIUS = 5
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1


class WoundAnnotator:
    """Interactive wound annotation tool using OpenCV."""

    def __init__(self, image_paths, output_path, existing_annotations=None):
        self.image_paths = image_paths
        self.output_path = output_path
        self.existing_annotations = existing_annotations or {"images": []}

        # Current image state
        self.current_idx = 0
        self.phase = 1  # 1 = wound curve, 2 = stitch points
        self.wound_points = []
        self.stitch_points = []  # flat list; pairs are (stitch_points[i], stitch_points[i+1])
        self.original_image = None
        self.display_image = None
        self.window_name = "Wound Annotation Tool"
        self.mouse_pos = (0, 0)

        # Track which images are already annotated (for resume)
        self.annotated_paths = set()
        for entry in self.existing_annotations["images"]:
            self.annotated_paths.add(entry["path"])

    def run(self):
        """Main annotation loop."""
        if not self.image_paths:
            print("No images to annotate. Exiting.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self.current_idx = 0

        # Skip already-annotated images
        while self.current_idx < len(self.image_paths):
            path = self.image_paths[self.current_idx]
            if path in self.annotated_paths:
                print(f"  Skipping (already annotated): {os.path.basename(path)}")
                self.current_idx += 1
            else:
                break

        if self.current_idx >= len(self.image_paths):
            print("All images already annotated. Nothing to do.")
            cv2.destroyAllWindows()
            return

        self._load_current_image()

        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                if key == ord('q'):
                    # Save current image if it has any annotations
                    if self.wound_points or self.stitch_points:
                        self._save_current()
                self._write_output()
                break

            elif key == ord('u'):
                self._undo()

            elif key == ord('c'):
                self._clear_current()

            elif key == ord('n'):
                if self.phase == 1:
                    self.phase = 2
                    print("  -> Phase 2: Click insertion/exit point pairs")
                else:
                    # Save and move to next image
                    self._save_current()
                    if not self._next_image():
                        self._write_output()
                        break

            elif key == ord('s'):
                self._save_current()
                if not self._next_image():
                    self._write_output()
                    break

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == 1:
                self.wound_points.append([x, y])
                print(f"    Wound point {len(self.wound_points)}: ({x}, {y})")
            elif self.phase == 2:
                self.stitch_points.append([x, y])
                n = len(self.stitch_points)
                if n % 2 == 1:
                    print(f"    Insertion point (stitch {n // 2 + 1}): ({x}, {y})")
                else:
                    print(f"    Exit point (stitch {n // 2}): ({x}, {y})")

    def _load_current_image(self):
        """Load the current image from disk."""
        path = self.image_paths[self.current_idx]
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            print(f"WARNING: Could not load image: {path}")
            # Create a placeholder
            self.original_image = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(self.original_image, "IMAGE LOAD FAILED", (200, 270),
                        FONT, 1.5, (0, 0, 255), 2)

        self.phase = 1
        self.wound_points = []
        self.stitch_points = []

        remaining = len(self.image_paths) - self.current_idx
        print(f"\nImage {self.current_idx + 1}/{len(self.image_paths)} "
              f"({remaining} remaining): {os.path.basename(path)}")
        print("  Phase 1: Click points along the wound curve (left to right)")

    def _render(self):
        """Render the current image with annotations overlaid."""
        img = self.original_image.copy()
        h, w = img.shape[:2]

        # Draw wound curve
        if len(self.wound_points) > 0:
            pts = np.array(self.wound_points, dtype=np.int32)
            # Draw connecting lines
            if len(pts) > 1:
                cv2.polylines(img, [pts], isClosed=False,
                              color=COLOR_WOUND_LINE, thickness=LINE_THICKNESS)
            # Draw individual points
            for pt in self.wound_points:
                cv2.circle(img, tuple(pt), POINT_RADIUS, COLOR_WOUND, -1)
                cv2.circle(img, tuple(pt), POINT_RADIUS + 1, (0, 0, 0), 1)

        # Draw stitch points and pairs
        for i, pt in enumerate(self.stitch_points):
            if i % 2 == 0:
                # Insertion point (blue)
                cv2.circle(img, tuple(pt), POINT_RADIUS + 1, COLOR_INSERTION, -1)
                cv2.circle(img, tuple(pt), POINT_RADIUS + 2, (0, 0, 0), 1)
                label = f"I{i // 2 + 1}"
                cv2.putText(img, label, (pt[0] + 8, pt[1] - 5),
                            FONT, 0.4, COLOR_INSERTION, 1)
            else:
                # Exit point (orange)
                cv2.circle(img, tuple(pt), POINT_RADIUS + 1, COLOR_EXIT, -1)
                cv2.circle(img, tuple(pt), POINT_RADIUS + 2, (0, 0, 0), 1)
                label = f"E{i // 2 + 1}"
                cv2.putText(img, label, (pt[0] + 8, pt[1] - 5),
                            FONT, 0.4, COLOR_EXIT, 1)

        # Draw lines connecting insertion-exit pairs
        for i in range(0, len(self.stitch_points) - 1, 2):
            ins = tuple(self.stitch_points[i])
            ext = tuple(self.stitch_points[i + 1])
            cv2.line(img, ins, ext, COLOR_STITCH_LINE, LINE_THICKNESS)

        # Draw crosshair at mouse position
        mx, my = self.mouse_pos
        if 0 <= mx < w and 0 <= my < h:
            cv2.line(img, (mx - 15, my), (mx + 15, my), COLOR_CROSSHAIR, 1)
            cv2.line(img, (mx, my - 15), (mx, my + 15), COLOR_CROSSHAIR, 1)

        # Draw info panel at top
        self._draw_info_panel(img)

        self.display_image = img
        cv2.imshow(self.window_name, img)

    def _draw_info_panel(self, img):
        """Draw the instruction and status panel at the top of the image."""
        h, w = img.shape[:2]

        # Semi-transparent background bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Image counter
        counter = f"Image {self.current_idx + 1}/{len(self.image_paths)}"
        cv2.putText(img, counter, (10, 20), FONT, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS)

        # Phase label
        if self.phase == 1:
            phase_text = "PHASE 1: WOUND CURVE"
            instructions = "Click points along wound (L-to-R). [n]=next phase  [u]=undo  [c]=clear"
            status = f"Wound points: {len(self.wound_points)}"
        else:
            phase_text = "PHASE 2: STITCH POINTS"
            n_complete = len(self.stitch_points) // 2
            waiting_exit = len(self.stitch_points) % 2 == 1
            if waiting_exit:
                instructions = "Click EXIT point for current stitch. [n/s]=save & next  [u]=undo  [c]=clear"
            else:
                instructions = "Click INSERTION point for next stitch. [n/s]=save & next  [u]=undo  [c]=clear"
            status = f"Stitches: {n_complete} complete"
            if waiting_exit:
                status += " (+1 pending exit)"

        cv2.putText(img, phase_text, (10, 42), FONT, FONT_SCALE,
                    COLOR_PHASE_LABEL, FONT_THICKNESS + 1)
        cv2.putText(img, instructions, (10, 60), FONT, 0.45, COLOR_TEXT, 1)
        cv2.putText(img, status, (10, 76), FONT, 0.45, (150, 255, 150), 1)

        # Keyboard shortcut reminder on right side
        shortcuts = "[q]=quit+save  [ESC]=quit"
        text_size = cv2.getTextSize(shortcuts, FONT, 0.4, 1)[0]
        cv2.putText(img, shortcuts, (w - text_size[0] - 10, 20),
                    FONT, 0.4, (180, 180, 180), 1)

    def _undo(self):
        """Undo the last point in the current phase."""
        if self.phase == 1:
            if self.wound_points:
                removed = self.wound_points.pop()
                print(f"    Undo wound point: ({removed[0]}, {removed[1]})")
            else:
                print("    Nothing to undo in Phase 1")
        elif self.phase == 2:
            if self.stitch_points:
                removed = self.stitch_points.pop()
                n = len(self.stitch_points)
                kind = "insertion" if n % 2 == 0 else "exit"
                print(f"    Undo {kind} point: ({removed[0]}, {removed[1]})")
            else:
                print("    Nothing to undo in Phase 2")

    def _clear_current(self):
        """Clear all annotations for the current image."""
        self.wound_points = []
        self.stitch_points = []
        self.phase = 1
        print("    Cleared all annotations for this image. Back to Phase 1.")

    def _save_current(self):
        """Save annotations for the current image into the results dict."""
        path = self.image_paths[self.current_idx]

        # Build stitch pairs (only complete pairs)
        stitches = []
        for i in range(0, len(self.stitch_points) - 1, 2):
            stitches.append({
                "insertion": self.stitch_points[i],
                "exit": self.stitch_points[i + 1]
            })

        # Warn about unpaired stitch point
        if len(self.stitch_points) % 2 == 1:
            print(f"  WARNING: Dropping unpaired insertion point "
                  f"({self.stitch_points[-1][0]}, {self.stitch_points[-1][1]})")

        entry = {
            "path": path,
            "wound_points": self.wound_points,
            "stitches": stitches
        }

        # Replace if already exists, otherwise append
        replaced = False
        for i, existing in enumerate(self.existing_annotations["images"]):
            if existing["path"] == path:
                self.existing_annotations["images"][i] = entry
                replaced = True
                break
        if not replaced:
            self.existing_annotations["images"].append(entry)

        self.annotated_paths.add(path)

        n_wound = len(self.wound_points)
        n_stitch = len(stitches)
        print(f"  Saved: {n_wound} wound points, {n_stitch} stitch pairs")

        # Write incrementally so nothing is lost on crash
        self._write_output()

    def _next_image(self):
        """Advance to the next unannotated image. Returns False if no more images."""
        self.current_idx += 1

        # Skip already-annotated
        while self.current_idx < len(self.image_paths):
            path = self.image_paths[self.current_idx]
            if path in self.annotated_paths:
                print(f"  Skipping (already annotated): {os.path.basename(path)}")
                self.current_idx += 1
            else:
                break

        if self.current_idx >= len(self.image_paths):
            print("\nAll images annotated!")
            return False

        self._load_current_image()
        return True

    def _write_output(self):
        """Write annotations to the JSON output file."""
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.existing_annotations, f, indent=2)
            n = len(self.existing_annotations["images"])
            print(f"  [Saved {n} annotations to {self.output_path}]")
        except Exception as e:
            print(f"  ERROR writing output: {e}")


def collect_images_from_dir(image_dir, sample_n=None, seed=42):
    """
    Collect frame000000_left.jpg from each episode in an image directory.

    The directory structure is expected to be:
        image_dir/
            episode_timestamp_1/
                left_img_dir/
                    frame000000_left.jpg
            episode_timestamp_2/
                ...

    If sample_n is given, randomly sample that many episodes.
    """
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: Directory not found: {image_dir}")
        return []

    # Collect all episode directories that have left_img_dir/frame000000_left.jpg
    episodes = []
    for entry in sorted(os.listdir(image_dir)):
        ep_path = os.path.join(image_dir, entry)
        if not os.path.isdir(ep_path):
            continue
        # Check for left_img_dir with at least one frame
        left_dir = os.path.join(ep_path, "left_img_dir")
        if os.path.isdir(left_dir):
            frame0 = os.path.join(left_dir, "frame000000_left.jpg")
            if os.path.isfile(frame0):
                episodes.append(frame0)
            else:
                # Try to find any frame
                frames = sorted([f for f in os.listdir(left_dir)
                                 if f.endswith('.jpg') or f.endswith('.png')])
                if frames:
                    episodes.append(os.path.join(left_dir, frames[0]))

    if not episodes:
        # Maybe the directory itself contains images directly
        direct_images = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if direct_images:
            episodes = direct_images

    if not episodes:
        print(f"WARNING: No images found in {image_dir}")
        print("  Expected structure: image_dir/episode_timestamp/left_img_dir/frameNNNNNN_left.jpg")
        print("  Or: image_dir/*.jpg")
        return []

    print(f"Found {len(episodes)} episodes in {image_dir}")

    if sample_n is not None and sample_n < len(episodes):
        random.seed(seed)
        episodes = sorted(random.sample(episodes, sample_n))
        print(f"Randomly sampled {sample_n} episodes")

    return episodes


def collect_images_from_multiple_dirs(image_dirs, sample_n=None, seed=42):
    """Collect images from multiple directories (e.g., multiple tissues)."""
    all_images = []
    for d in image_dirs:
        images = collect_images_from_dir(d, sample_n=None)
        all_images.extend(images)

    if sample_n is not None and sample_n < len(all_images):
        random.seed(seed)
        all_images = sorted(random.sample(all_images, sample_n))
        print(f"Randomly sampled {sample_n} from {len(all_images)} total episodes")

    return all_images


def load_existing_annotations(output_path):
    """Load existing annotations if the file exists."""
    if os.path.isfile(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            n = len(data.get("images", []))
            print(f"Loaded {n} existing annotations from {output_path}")
            return data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Could not parse existing annotations: {e}")
            print("  Starting fresh.")
    return {"images": []}


def main():
    parser = argparse.ArgumentParser(
        description="Wound Annotation Tool for GC-ACT Autonomous Suturing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate 20 randomly sampled episodes from tissue 7:
  python annotate_wounds.py --image_dir ~/data/tissue_7/2_needle_throw/ --sample 20

  # Annotate from multiple tissue directories:
  python annotate_wounds.py --image_dir ~/data/tissue_5/2_needle_throw/ ~/data/tissue_7/2_needle_throw/ --sample 30

  # Annotate specific image files:
  python annotate_wounds.py --image_list path/to/img1.jpg path/to/img2.jpg

  # Resume annotation (skip already-annotated images):
  python annotate_wounds.py --image_dir ~/data/tissue_7/2_needle_throw/ --sample 20 --resume

NOTE: Requires an X11 display. On remote instances (e.g., Jetstream2), use:
  ssh -X user@host    (X11 forwarding)
  or set up VNC/noVNC for a persistent remote desktop.
        """
    )
    parser.add_argument(
        '--image_dir', nargs='+',
        help='Directory(ies) containing episode folders with left_img_dir/'
    )
    parser.add_argument(
        '--image_list', nargs='+',
        help='Explicit list of image file paths to annotate'
    )
    parser.add_argument(
        '--sample', type=int, default=None,
        help='Randomly sample N episodes (for --image_dir mode)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--output', type=str, default=os.path.expanduser('~/wound_annotations.json'),
        help='Output JSON path (default: ~/wound_annotations.json)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from existing annotations (skip already-annotated images)'
    )

    args = parser.parse_args()

    # Resolve output path
    output_path = os.path.expanduser(args.output)

    # Collect image paths
    image_paths = []

    if args.image_list:
        for p in args.image_list:
            full_path = os.path.abspath(os.path.expanduser(p))
            if os.path.isfile(full_path):
                image_paths.append(full_path)
            else:
                print(f"WARNING: File not found, skipping: {full_path}")

    elif args.image_dir:
        if len(args.image_dir) == 1:
            image_paths = collect_images_from_dir(
                args.image_dir[0], sample_n=args.sample, seed=args.seed
            )
        else:
            image_paths = collect_images_from_multiple_dirs(
                args.image_dir, sample_n=args.sample, seed=args.seed
            )

    else:
        parser.print_help()
        print("\nERROR: Must specify --image_dir or --image_list")
        sys.exit(1)

    if not image_paths:
        print("No images to annotate. Exiting.")
        sys.exit(1)

    # Resolve all paths to absolute
    image_paths = [os.path.abspath(p) for p in image_paths]

    print(f"\n{'=' * 60}")
    print(f"Wound Annotation Tool")
    print(f"{'=' * 60}")
    print(f"Images to annotate: {len(image_paths)}")
    print(f"Output file: {output_path}")

    # Load existing annotations if resuming
    existing = {"images": []}
    if args.resume or os.path.isfile(output_path):
        existing = load_existing_annotations(output_path)
        if args.resume:
            n_existing = len(existing.get("images", []))
            print(f"Resume mode: {n_existing} images already done")

    print(f"\nControls:")
    print(f"  Left click  - Add point")
    print(f"  'u'         - Undo last point")
    print(f"  'c'         - Clear current image")
    print(f"  'n'         - Next phase / save & next image")
    print(f"  's'         - Save & next image")
    print(f"  'q'         - Save all & quit")
    print(f"  ESC         - Quit (current image unsaved)")
    print(f"{'=' * 60}\n")

    annotator = WoundAnnotator(image_paths, output_path, existing)
    annotator.run()

    # Final summary
    try:
        with open(output_path, 'r') as f:
            final_data = json.load(f)
        n = len(final_data["images"])
        total_wound = sum(len(img["wound_points"]) for img in final_data["images"])
        total_stitch = sum(len(img["stitches"]) for img in final_data["images"])
        print(f"\nFinal summary:")
        print(f"  Annotated images: {n}")
        print(f"  Total wound points: {total_wound}")
        print(f"  Total stitch pairs: {total_stitch}")
        print(f"  Output: {output_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
