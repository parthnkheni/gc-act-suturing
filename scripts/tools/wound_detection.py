#!/usr/bin/env python3
"""
Wound Detection Module for GC-ACT Autonomous Suturing
=====================================================

Detects the incision line on a silicone tissue pad from an endoscope camera
image and computes evenly-spaced suture insertion/exit point pairs along it.

Usage:
    python wound_detection.py --image <path>                    # detect wound
    python wound_detection.py --image <path> --num_stitches 3   # plan stitches
    python wound_detection.py --image <path> --visualize        # annotated image
    python wound_detection.py --demo                            # synthetic test
    python wound_detection.py --calibrate                       # calibration mode

Dependencies: OpenCV, NumPy (no ML frameworks)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Data structures

@dataclass
class WoundInfo:
    """Result of wound line detection."""
    endpoint_a: np.ndarray          # (x, y) pixel coords of line start
    endpoint_b: np.ndarray          # (x, y) pixel coords of line end
    center: np.ndarray              # (x, y) pixel coords of line midpoint
    angle_deg: float                # angle in degrees (0 = horizontal)
    length_px: float                # length of detected line in pixels
    mask: np.ndarray                # binary mask of wound region
    confidence: float               # 0-1 confidence score
    detection_method: str           # which strategy succeeded
    contour: Optional[np.ndarray] = None  # wound contour if available
    spine_points: Optional[np.ndarray] = None  # Nx2 ordered points along wound curve
    poly_coeffs: Optional[np.ndarray] = None   # polynomial fit coefficients (y = f(x) or x = f(y))
    poly_axis: str = "x"            # "x" means y=f(x), "y" means x=f(y)

    def direction_unit(self) -> np.ndarray:
        """Unit vector along the wound line from endpoint_a to endpoint_b."""
        d = self.endpoint_b - self.endpoint_a
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return np.array([1.0, 0.0])
        return d / norm

    def perpendicular_unit(self) -> np.ndarray:
        """Unit vector perpendicular to the wound line (rotated 90 deg CCW)."""
        d = self.direction_unit()
        return np.array([-d[1], d[0]])

    def point_on_curve(self, t: float) -> np.ndarray:
        """Get a point on the wound curve at parameter t in [0, 1].
        t=0 is endpoint_a, t=1 is endpoint_b.
        Falls back to linear interpolation if no curve fit available."""
        if self.spine_points is not None and len(self.spine_points) >= 3:
            idx = t * (len(self.spine_points) - 1)
            i = int(idx)
            frac = idx - i
            i = min(i, len(self.spine_points) - 2)
            return self.spine_points[i] * (1 - frac) + self.spine_points[i + 1] * frac
        elif self.poly_coeffs is not None:
            if self.poly_axis == "x":
                x = self.endpoint_a[0] + t * (self.endpoint_b[0] - self.endpoint_a[0])
                y = np.polyval(self.poly_coeffs, x)
                return np.array([x, y])
            else:
                y = self.endpoint_a[1] + t * (self.endpoint_b[1] - self.endpoint_a[1])
                x = np.polyval(self.poly_coeffs, y)
                return np.array([x, y])
        else:
            return self.endpoint_a + t * (self.endpoint_b - self.endpoint_a)

    def tangent_at(self, t: float) -> np.ndarray:
        """Get the tangent direction at parameter t in [0, 1].
        Returns a unit vector along the wound curve at that point."""
        dt = 0.01
        t0 = max(0, t - dt)
        t1 = min(1, t + dt)
        p0 = self.point_on_curve(t0)
        p1 = self.point_on_curve(t1)
        d = p1 - p0
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return self.direction_unit()
        return d / norm

    def normal_at(self, t: float) -> np.ndarray:
        """Get the normal (perpendicular) direction at parameter t.
        Rotated 90 deg CCW from tangent."""
        tang = self.tangent_at(t)
        return np.array([-tang[1], tang[0]])

    def arc_length(self) -> float:
        """Compute approximate arc length of the curve."""
        if self.spine_points is not None and len(self.spine_points) >= 2:
            diffs = np.diff(self.spine_points, axis=0)
            return float(np.sum(np.linalg.norm(diffs, axis=1)))
        return self.length_px


@dataclass
class StitchPlan:
    """A single planned stitch with insertion and exit points."""
    index: int                      # stitch number (0-based)
    wound_point: np.ndarray         # point on the wound line (px)
    insertion_point: np.ndarray     # point on one side of wound (px)
    exit_point: np.ndarray          # point on other side of wound (px)
    insertion_robot: Optional[np.ndarray] = None  # robot coords if calibrated
    exit_robot: Optional[np.ndarray] = None       # robot coords if calibrated


@dataclass
class CalibrationData:
    """Pixel-to-robot coordinate mapping."""
    pixel_points: np.ndarray        # Nx2 pixel coordinates
    robot_points: np.ndarray        # Nx3 robot coordinates
    transform_matrix: Optional[np.ndarray] = None  # 3x3 affine (2D) or 4x3
    residual_mm: float = 0.0        # mean residual after fitting
    pixel_to_mm: float = 0.1       # approximate mm per pixel


# Default parameters

# HSV thresholds for wound detection on silicone tissue pads.
# The wound (cut) appears darker than the surrounding pink/beige tissue.
# These are starting points; adjust per lighting conditions.
DEFAULT_HSV_LOWER_WOUND = np.array([0, 0, 0])
DEFAULT_HSV_UPPER_WOUND = np.array([180, 255, 100])

# Alternate: detect wound as dark region relative to tissue
DEFAULT_BRIGHTNESS_THRESHOLD = 80  # pixels darker than this are wound candidates

# Morphological kernel sizes
MORPH_KERNEL_SIZE = 5
MORPH_CLOSE_ITER = 3
MORPH_OPEN_ITER = 2

# Minimum wound contour area (pixels) to filter noise
MIN_WOUND_AREA_PX = 200

# dVRK workspace defaults
DEFAULT_PIXEL_TO_MM = 0.1          # rough estimate, needs calibration
DEFAULT_STITCH_SPACING_MM = 5.0
DEFAULT_MARGIN_MM = 3.0
DEFAULT_NUM_STITCHES = 3

# Image size
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480


# Detection Strategy 0: Real endoscope  -- tissue segmentation + incision finding

def _segment_tissue(image: np.ndarray):
    """Segment the tissue pad from the dark background.
    Returns a binary mask where tissue=255, background=0."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tissue is bright (high V) and pinkish/beige (low-mid S, specific H range)
    # Background is very dark (low V)
    tissue_mask = cv2.inRange(hsv, np.array([0, 10, 80]), np.array([25, 180, 255]))

    # Also include slightly desaturated tissue regions
    tissue_mask2 = cv2.inRange(hsv, np.array([0, 5, 100]), np.array([180, 80, 255]))
    tissue_mask = cv2.bitwise_or(tissue_mask, tissue_mask2)

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep only the largest connected component (the tissue pad)
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        tissue_mask = np.zeros_like(tissue_mask)
        cv2.drawContours(tissue_mask, [largest], -1, 255, cv2.FILLED)

    return tissue_mask


def _mask_instruments_and_thread(image: np.ndarray, tissue_mask: np.ndarray):
    """Mask out metallic instruments and blue suture thread.
    Returns a mask where instruments/thread=0, clean tissue=255."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Metallic instruments: high brightness with low saturation (gray/silver)
    # and often specular highlights
    metal_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 50, 255]))

    # Also detect specular highlights (very bright, any hue)
    _, highlight_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Blue/teal suture thread
    thread_mask = cv2.inRange(hsv, np.array([80, 30, 80]), np.array([130, 255, 255]))

    # Dark needle (thin, dark, curved)
    needle_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))

    # Combine all distractors
    distractor_mask = cv2.bitwise_or(metal_mask, highlight_mask)
    distractor_mask = cv2.bitwise_or(distractor_mask, thread_mask)
    distractor_mask = cv2.bitwise_or(distractor_mask, needle_mask)

    # Dilate distractors to be conservative
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    distractor_mask = cv2.dilate(distractor_mask, kernel, iterations=2)

    # Clean tissue = tissue AND NOT distractors
    clean_mask = cv2.bitwise_and(tissue_mask, cv2.bitwise_not(distractor_mask))
    return clean_mask


def _detect_real_endoscope(image: np.ndarray) -> Optional[WoundInfo]:
    """
    Detect wound line on a real endoscope image of a silicone tissue pad.

    The incision appears as a subtle shadow/crease running through the tissue.
    Strategy:
    1. Segment tissue from dark background
    2. Mask out instruments, thread, needle
    3. Within clean tissue, find the incision using gradient analysis
       (the crease creates a shadow that appears as a local brightness valley)
    4. Use CLAHE + directional filtering to enhance the subtle incision
    """
    h_img, w_img = image.shape[:2]

    # Step 1: Segment tissue
    tissue_mask = _segment_tissue(image)
    tissue_area = cv2.countNonZero(tissue_mask)
    if tissue_area < h_img * w_img * 0.1:
        return None  # tissue pad not found

    # Step 2: Mask out instruments and thread
    clean_mask = _mask_instruments_and_thread(image, tissue_mask)

    # Step 3: Enhance incision visibility with CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply tissue mask  -- set non-tissue to median value to avoid edges at boundaries
    median_val = int(np.median(enhanced[tissue_mask > 0])) if cv2.countNonZero(tissue_mask) > 0 else 128
    tissue_only = np.full_like(enhanced, median_val)
    tissue_only[clean_mask > 0] = enhanced[clean_mask > 0]

    # Step 4: Multi-scale gradient analysis for the crease
    # The incision creates a brightness valley  -- look for it with Laplacian and
    # directional Sobel filters

    # Blur to remove fine texture, keep the crease
    blurred = cv2.GaussianBlur(tissue_only, (11, 11), 3.0)

    # Method A: Local brightness valley via Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=9)
    # Positive laplacian = concave up = valley/crease
    valley = np.clip(laplacian, 0, None)
    if valley.max() > 0:
        valley_norm = (valley / valley.max() * 255).astype(np.uint8)
    else:
        valley_norm = np.zeros_like(gray)

    # Method B: Brightness is lower at the crease than neighbors
    # Compare pixel brightness to local neighborhood
    local_mean = cv2.blur(blurred, (51, 51))
    dark_relative = cv2.subtract(local_mean, blurred)  # positive where pixel is darker than surroundings
    # Boost contrast
    dark_relative = cv2.normalize(dark_relative, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine both methods
    combined = cv2.addWeighted(valley_norm, 0.5, dark_relative, 0.5, 0)

    # Apply clean tissue mask
    combined[clean_mask == 0] = 0

    # Threshold to get candidate crease pixels
    # Use Otsu on the non-zero region for adaptive threshold
    non_zero_vals = combined[combined > 0]
    if len(non_zero_vals) < 100:
        return None

    # Use a high percentile threshold (incision should be a small fraction of tissue)
    thresh_val = max(np.percentile(non_zero_vals, 85), 30)
    _, crease_mask = cv2.threshold(combined, int(thresh_val), 255, cv2.THRESH_BINARY)

    # Morphological cleanup  -- close gaps along the crease, remove noise
    # Use an elongated kernel (the crease is a line, not a blob)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    crease_mask = cv2.morphologyEx(crease_mask, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    crease_mask = cv2.morphologyEx(crease_mask, cv2.MORPH_OPEN, kernel_noise, iterations=1)

    # Also try with horizontal kernel (wound could be any orientation)
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    crease_mask_h = cv2.morphologyEx(
        cv2.threshold(combined, int(thresh_val), 255, cv2.THRESH_BINARY)[1],
        cv2.MORPH_CLOSE, kernel_line_h, iterations=2
    )
    crease_mask_h = cv2.morphologyEx(crease_mask_h, cv2.MORPH_OPEN, kernel_noise, iterations=1)

    # Use whichever gives a more elongated contour
    best_wound = None
    best_score = 0

    for mask_candidate in [crease_mask, crease_mask_h]:
        contours, _ = cv2.findContours(mask_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_r, h_r), angle = rect
            if min(w_r, h_r) < 1:
                continue
            aspect = max(w_r, h_r) / min(w_r, h_r)
            # Must be elongated (line-like) and in the center-ish region
            center_dist = abs(cx - w_img / 2) / w_img + abs(cy - h_img / 2) / h_img
            # Prefer: elongated, large, central
            score = aspect * np.sqrt(area) * (1.0 - center_dist * 0.5)
            if score > best_score and aspect > 2.0:
                best_score = score
                best_wound = cnt

    if best_wound is None:
        return None

    # Extract contour points and compute the wound spine (centerline curve)
    pts = best_wound.reshape(-1, 2).astype(np.float64)

    # Determine primary axis  -- fit along whichever axis has more spread
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()

    if x_range >= y_range:
        # Wound runs more horizontally  -- fit y = f(x)
        poly_axis = "x"
        # Sort by x for ordered spine
        sort_idx = np.argsort(pts[:, 0])
        sorted_pts = pts[sort_idx]
        # Fit polynomial (degree 3 for gentle curves, clamp to 2 if few points)
        deg = min(3, max(2, len(pts) // 20))
        poly_coeffs = np.polyfit(sorted_pts[:, 0], sorted_pts[:, 1], deg)
        # Generate evenly-spaced spine points along the curve
        x_min, x_max = sorted_pts[:, 0].min(), sorted_pts[:, 0].max()
        spine_x = np.linspace(x_min, x_max, 100)
        spine_y = np.polyval(poly_coeffs, spine_x)
        spine_points = np.column_stack([spine_x, spine_y])
    else:
        # Wound runs more vertically  -- fit x = f(y)
        poly_axis = "y"
        sort_idx = np.argsort(pts[:, 1])
        sorted_pts = pts[sort_idx]
        deg = min(3, max(2, len(pts) // 20))
        poly_coeffs = np.polyfit(sorted_pts[:, 1], sorted_pts[:, 0], deg)
        y_min, y_max = sorted_pts[:, 1].min(), sorted_pts[:, 1].max()
        spine_y = np.linspace(y_min, y_max, 100)
        spine_x = np.polyval(poly_coeffs, spine_y)
        spine_points = np.column_stack([spine_x, spine_y])

    # Endpoints and center from the spine
    endpoint_a = spine_points[0].copy()
    endpoint_b = spine_points[-1].copy()
    center = spine_points[len(spine_points) // 2].copy()

    # Arc length
    diffs = np.diff(spine_points, axis=0)
    arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Angle: overall direction from endpoint to endpoint
    d = endpoint_b - endpoint_a
    angle_deg = float(np.degrees(np.arctan2(d[1], d[0])))

    # Confidence based on elongation and area
    rect = cv2.minAreaRect(best_wound)
    (cx, cy), (w_r, h_r), angle = rect
    area = cv2.contourArea(best_wound)
    aspect = max(w_r, h_r) / max(min(w_r, h_r), 1)
    conf = min(1.0, (aspect / 6.0) * 0.5 + (area / 3000.0) * 0.3 + 0.2)

    wound_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(wound_mask, [best_wound], -1, 255, cv2.FILLED)

    return WoundInfo(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        center=center,
        angle_deg=angle_deg,
        length_px=arc_len,
        mask=wound_mask,
        confidence=conf,
        detection_method="real_endoscope",
        contour=best_wound,
        spine_points=spine_points,
        poly_coeffs=poly_coeffs,
        poly_axis=poly_axis,
    )


# Detection Strategy 1: HSV thresholding + contour fitting

def _detect_hsv_contour(image: np.ndarray) -> Optional[WoundInfo]:
    """
    Detect wound line via HSV color thresholding and contour analysis.

    The wound on a silicone pad appears as a dark line against pink/beige
    tissue. We threshold in HSV for dark pixels, find contours, pick the
    most elongated one, and fit a line through it.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Strategy: threshold on value (brightness) channel -- wound is dark
    _, dark_mask = cv2.threshold(v, DEFAULT_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Also try adaptive threshold for robustness to uneven lighting
    adaptive_mask = cv2.adaptiveThreshold(
        v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=15
    )

    # Combine: pixel is wound if flagged by either method
    combined = cv2.bitwise_or(dark_mask, adaptive_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)

    # Restrict to central region of the image (wound is typically centered)
    h_img, w_img = cleaned.shape[:2]
    roi_mask = np.zeros_like(cleaned)
    margin_x = int(w_img * 0.08)
    margin_y = int(h_img * 0.08)
    roi_mask[margin_y:h_img - margin_y, margin_x:w_img - margin_x] = 255
    cleaned = cv2.bitwise_and(cleaned, roi_mask)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area and pick the most elongated contour (highest aspect ratio)
    best_contour = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WOUND_AREA_PX:
            continue

        # Fit minimum-area bounding rectangle
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_rect, h_rect), angle = rect
        if w_rect < 1 or h_rect < 1:
            continue

        # Aspect ratio: wound line is elongated
        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
        # Score: prefer elongated + large contours
        score = aspect * np.sqrt(area)

        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return None

    # Fit a line through the best contour using least-squares
    [vx, vy, x0, y0] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    # Get bounding extent to determine line endpoints
    rect = cv2.minAreaRect(best_contour)
    (cx, cy), (w_rect, h_rect), angle = rect
    half_len = max(w_rect, h_rect) / 2.0

    endpoint_a = np.array([x0 - vx * half_len, y0 - vy * half_len])
    endpoint_b = np.array([x0 + vx * half_len, y0 + vy * half_len])
    center = np.array([x0, y0])

    angle_deg = np.degrees(np.arctan2(vy, vx))
    length_px = 2 * half_len

    # Confidence: based on aspect ratio and area
    area = cv2.contourArea(best_contour)
    aspect = max(w_rect, h_rect) / max(min(w_rect, h_rect), 1)
    conf = min(1.0, (aspect / 5.0) * 0.5 + (area / 5000.0) * 0.5)

    # Create a clean mask of just the wound
    wound_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(wound_mask, [best_contour], -1, 255, cv2.FILLED)

    return WoundInfo(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        center=center,
        angle_deg=angle_deg,
        length_px=length_px,
        mask=wound_mask,
        confidence=conf,
        detection_method="hsv_contour",
        contour=best_contour,
    )


# Detection Strategy 2: Edge detection + Hough line transform

def _detect_edge_hough(image: np.ndarray) -> Optional[WoundInfo]:
    """
    Detect wound line via Canny edge detection and probabilistic Hough
    line transform. Groups nearby collinear line segments and picks the
    dominant cluster.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Canny edge detection with automatic thresholds
    median_val = np.median(blurred)
    low_thresh = int(max(0, 0.5 * median_val))
    high_thresh = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Dilate edges slightly to connect broken segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Restrict to central region
    h_img, w_img = edges.shape[:2]
    margin_x = int(w_img * 0.08)
    margin_y = int(h_img * 0.08)
    roi_mask = np.zeros_like(edges)
    roi_mask[margin_y:h_img - margin_y, margin_x:w_img - margin_x] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    # Probabilistic Hough line transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return None

    # Collect all line segments
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        segments.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "length": length, "angle": angle,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        })

    if not segments:
        return None

    # Cluster segments by angle (within 15 degrees) and proximity (within 50px)
    segments.sort(key=lambda s: -s["length"])  # longest first

    best_cluster = [segments[0]]
    best_angle = segments[0]["angle"]

    for seg in segments[1:]:
        angle_diff = min(
            abs(seg["angle"] - best_angle),
            180 - abs(seg["angle"] - best_angle)
        )
        # Check if collinear: similar angle and close to the cluster centroid
        cluster_cx = np.mean([s["cx"] for s in best_cluster])
        cluster_cy = np.mean([s["cy"] for s in best_cluster])
        dist = np.sqrt((seg["cx"] - cluster_cx) ** 2 + (seg["cy"] - cluster_cy) ** 2)

        if angle_diff < 15 and dist < 150:
            best_cluster.append(seg)

    # Collect all points from the best cluster
    all_points = []
    for seg in best_cluster:
        all_points.append([seg["x1"], seg["y1"]])
        all_points.append([seg["x2"], seg["y2"]])
    all_points = np.array(all_points, dtype=np.float32)

    if len(all_points) < 4:
        return None

    # Fit a line through all cluster points
    [vx, vy, x0, y0] = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    # Project all points onto the fitted line to find endpoints
    projections = []
    for pt in all_points:
        t = (pt[0] - x0) * vx + (pt[1] - y0) * vy
        projections.append(t)

    t_min = min(projections)
    t_max = max(projections)

    endpoint_a = np.array([x0 + vx * t_min, y0 + vy * t_min])
    endpoint_b = np.array([x0 + vx * t_max, y0 + vy * t_max])
    center = np.array([x0, y0])
    length_px = t_max - t_min

    angle_deg = np.degrees(np.arctan2(vy, vx))

    # Confidence based on cluster size and total length
    total_seg_len = sum(s["length"] for s in best_cluster)
    conf = min(1.0, (len(best_cluster) / 5.0) * 0.3 + (total_seg_len / 300.0) * 0.7)

    # Build a mask from the Hough line region
    wound_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Draw a thick line along the detected wound
    pt_a = tuple(endpoint_a.astype(int))
    pt_b = tuple(endpoint_b.astype(int))
    cv2.line(wound_mask, pt_a, pt_b, 255, thickness=8)

    return WoundInfo(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        center=center,
        angle_deg=angle_deg,
        length_px=length_px,
        mask=wound_mask,
        confidence=conf,
        detection_method="edge_hough",
    )


# Detection Strategy 3: Gradient-based ridge detection

def _detect_gradient_ridge(image: np.ndarray) -> Optional[WoundInfo]:
    """
    Detect wound as a dark ridge using second-order gradient (Laplacian)
    analysis. The wound line creates a valley in brightness that produces
    strong second-derivative response.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)

    # Laplacian for ridge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=7)

    # Positive Laplacian = dark valley (wound)
    ridge = np.clip(laplacian, 0, None)

    # Normalize to 0-255
    if ridge.max() > 0:
        ridge_norm = (ridge / ridge.max() * 255).astype(np.uint8)
    else:
        return None

    # Threshold the ridge response
    _, ridge_mask = cv2.threshold(ridge_norm, 80, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close gaps, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Restrict to central ROI
    h_img, w_img = ridge_mask.shape[:2]
    margin_x = int(w_img * 0.10)
    margin_y = int(h_img * 0.10)
    roi_mask = np.zeros_like(ridge_mask)
    roi_mask[margin_y:h_img - margin_y, margin_x:w_img - margin_x] = 255
    ridge_mask = cv2.bitwise_and(ridge_mask, roi_mask)

    # Skeletonize to get thin line
    skeleton = cv2.ximgproc.thinning(ridge_mask) if hasattr(cv2, "ximgproc") else ridge_mask

    # Find contours on the ridge mask
    contours, _ = cv2.findContours(ridge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the most elongated contour
    best_contour = None
    best_aspect = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_WOUND_AREA_PX * 0.5:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_r, h_r), angle = rect
        if min(w_r, h_r) < 1:
            continue
        aspect = max(w_r, h_r) / min(w_r, h_r)
        if aspect > best_aspect:
            best_aspect = aspect
            best_contour = cnt

    if best_contour is None or best_aspect < 2.0:
        return None

    # Fit line
    [vx, vy, x0, y0] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    rect = cv2.minAreaRect(best_contour)
    (cx, cy), (w_r, h_r), angle = rect
    half_len = max(w_r, h_r) / 2.0

    endpoint_a = np.array([x0 - vx * half_len, y0 - vy * half_len])
    endpoint_b = np.array([x0 + vx * half_len, y0 + vy * half_len])
    center = np.array([x0, y0])
    angle_deg = np.degrees(np.arctan2(vy, vx))
    length_px = 2 * half_len

    conf = min(1.0, (best_aspect / 8.0) * 0.6 + 0.3)

    wound_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(wound_mask, [best_contour], -1, 255, cv2.FILLED)

    return WoundInfo(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        center=center,
        angle_deg=angle_deg,
        length_px=length_px,
        mask=wound_mask,
        confidence=conf,
        detection_method="gradient_ridge",
        contour=best_contour,
    )


# Main detection: run all strategies, pick best

def detect_wound_line(image: np.ndarray, verbose: bool = False) -> Optional[WoundInfo]:
    """
    Detect the wound/incision line on a silicone tissue pad.

    Runs three independent detection strategies and picks the result with
    the highest confidence score.

    Parameters
    ----------
    image : np.ndarray
        BGR image from the endoscope camera (640x480 expected).
    verbose : bool
        If True, print detection details to stdout.

    Returns
    -------
    WoundInfo or None
        Detected wound line information, or None if detection failed.
    """
    if image is None or image.size == 0:
        if verbose:
            print("[wound_detection] ERROR: empty or None image")
        return None

    # Ensure BGR 3-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    results = []

    # Strategy 0: Real endoscope (tissue segmentation + crease finding)
    # This is the primary strategy for actual dVRK camera images
    try:
        r0 = _detect_real_endoscope(image)
        if r0 is not None:
            results.append(r0)
            if verbose:
                print(f"  [real_endoscope] conf={r0.confidence:.3f}  "
                      f"len={r0.length_px:.1f}px  angle={r0.angle_deg:.1f} deg")
    except Exception as e:
        if verbose:
            print(f"  [real_endoscope] FAILED: {e}")

    # Strategy 1: HSV contour (works best on synthetic/clean images)
    try:
        r1 = _detect_hsv_contour(image)
        if r1 is not None:
            results.append(r1)
            if verbose:
                print(f"  [hsv_contour]    conf={r1.confidence:.3f}  "
                      f"len={r1.length_px:.1f}px  angle={r1.angle_deg:.1f} deg")
    except Exception as e:
        if verbose:
            print(f"  [hsv_contour]    FAILED: {e}")

    # Strategy 2: Edge + Hough
    try:
        r2 = _detect_edge_hough(image)
        if r2 is not None:
            results.append(r2)
            if verbose:
                print(f"  [edge_hough]     conf={r2.confidence:.3f}  "
                      f"len={r2.length_px:.1f}px  angle={r2.angle_deg:.1f} deg")
    except Exception as e:
        if verbose:
            print(f"  [edge_hough]     FAILED: {e}")

    # Strategy 3: Gradient ridge
    try:
        r3 = _detect_gradient_ridge(image)
        if r3 is not None:
            results.append(r3)
            if verbose:
                print(f"  [gradient_ridge] conf={r3.confidence:.3f}  "
                      f"len={r3.length_px:.1f}px  angle={r3.angle_deg:.1f} deg")
    except Exception as e:
        if verbose:
            print(f"  [gradient_ridge] FAILED: {e}")

    if not results:
        if verbose:
            print("[wound_detection] All detection strategies failed.")
            print("  Suggestion: try hardcoding wound coordinates if the tissue")
            print("  pad is in a known fixed position. Example:")
            print("    wound_info = WoundInfo(")
            print("        endpoint_a=np.array([160, 240]),")
            print("        endpoint_b=np.array([480, 240]),")
            print("        center=np.array([320, 240]),")
            print("        angle_deg=0.0, length_px=320,")
            print("        mask=np.zeros((480,640), dtype=np.uint8),")
            print("        confidence=1.0, detection_method='hardcoded')")
        return None

    # If multiple strategies agree (similar angle), boost confidence
    if len(results) >= 2:
        angles = [r.angle_deg for r in results]
        for i, r in enumerate(results):
            agreements = 0
            for j, other_angle in enumerate(angles):
                if i == j:
                    continue
                diff = abs(r.angle_deg - other_angle)
                diff = min(diff, 180 - diff)
                if diff < 15:
                    agreements += 1
            if agreements > 0:
                r.confidence = min(1.0, r.confidence + 0.15 * agreements)

    # Pick the best result
    best = max(results, key=lambda r: r.confidence)

    # Ensure endpoint_a is to the left of endpoint_b (or above if vertical)
    if best.endpoint_a[0] > best.endpoint_b[0]:
        best.endpoint_a, best.endpoint_b = best.endpoint_b.copy(), best.endpoint_a.copy()

    if verbose:
        print(f"[wound_detection] Selected: {best.detection_method} "
              f"(conf={best.confidence:.3f})")
        print(f"  Line: ({best.endpoint_a[0]:.0f},{best.endpoint_a[1]:.0f}) -> "
              f"({best.endpoint_b[0]:.0f},{best.endpoint_b[1]:.0f})")
        print(f"  Length: {best.length_px:.1f}px  Angle: {best.angle_deg:.1f} deg")

    return best


# Insertion point planning

def plan_insertion_points(
    wound_info: WoundInfo,
    num_stitches: int = DEFAULT_NUM_STITCHES,
    spacing_mm: float = DEFAULT_STITCH_SPACING_MM,
    margin_mm: float = DEFAULT_MARGIN_MM,
    pixel_to_mm: Optional[float] = None,
) -> List[StitchPlan]:
    """
    Compute evenly-spaced suture insertion/exit point pairs along the wound.

    Points are placed along the wound line with a perpendicular offset on
    each side of the wound for insertion (one side) and exit (other side).

    Parameters
    ----------
    wound_info : WoundInfo
        Detected wound line from detect_wound_line().
    num_stitches : int
        Number of stitches to plan.
    spacing_mm : float
        Desired spacing between stitches in millimeters.
    margin_mm : float
        Distance from wound edge to insertion/exit points in millimeters.
    pixel_to_mm : float or None
        Conversion factor (mm per pixel). If None, uses DEFAULT_PIXEL_TO_MM.

    Returns
    -------
    list of StitchPlan
        Planned stitch positions in pixel coordinates.
    """
    if pixel_to_mm is None:
        pixel_to_mm = DEFAULT_PIXEL_TO_MM

    spacing_px = spacing_mm / pixel_to_mm
    margin_px = margin_mm / pixel_to_mm

    # Use arc length for curve-aware spacing
    arc_len = wound_info.arc_length()

    # Total length needed for all stitches
    total_needed_px = (num_stitches - 1) * spacing_px if num_stitches > 1 else 0

    # Check if wound is long enough
    if total_needed_px > arc_len:
        if num_stitches > 1:
            spacing_px = arc_len * 0.8 / (num_stitches - 1)
            total_needed_px = (num_stitches - 1) * spacing_px
            print(f"[plan] Warning: wound too short for requested spacing. "
                  f"Reduced to {spacing_px * pixel_to_mm:.1f}mm")

    # Center the stitch pattern along the wound curve
    start_offset = (arc_len - total_needed_px) / 2.0

    stitches = []
    for i in range(num_stitches):
        # Parameter t along the curve [0, 1]
        dist_along = start_offset + i * spacing_px
        t = dist_along / arc_len if arc_len > 0 else 0.5

        # Get point on the curve and local perpendicular direction
        wound_point = wound_info.point_on_curve(t)
        normal = wound_info.normal_at(t)

        # Insertion and exit points perpendicular to the LOCAL curve direction
        insertion_point = wound_point + normal * margin_px
        exit_point = wound_point - normal * margin_px

        stitches.append(StitchPlan(
            index=i,
            wound_point=wound_point,
            insertion_point=insertion_point,
            exit_point=exit_point,
        ))

    return stitches


# Coordinate calibration

def calibrate(
    pixel_points: np.ndarray,
    robot_points: np.ndarray,
) -> CalibrationData:
    """
    Compute a pixel-to-robot coordinate transform from known correspondences.

    Fits an affine transform mapping 2D pixel coordinates to 3D robot
    coordinates. Requires at least 3 correspondences (4+ recommended).

    Parameters
    ----------
    pixel_points : np.ndarray
        Nx2 array of pixel coordinates (x, y).
    robot_points : np.ndarray
        Nx3 array of robot coordinates (x, y, z) in robot base frame.

    Returns
    -------
    CalibrationData
        Calibration data including the fitted affine transform.
    """
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    robot_points = np.asarray(robot_points, dtype=np.float64)

    n = pixel_points.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 calibration points, got {n}")

    # Build the design matrix for affine transform:
    # robot_xyz = A @ [px, py, 1]^T
    # Where A is 3x3: [[a11 a12 a13], [a21 a22 a23], [a31 a32 a33]]
    # Solve via least squares: robot = pixel_h @ A^T

    pixel_h = np.hstack([pixel_points, np.ones((n, 1))])  # Nx3 homogeneous

    # Solve for each robot dimension independently
    # A^T = (pixel_h^T pixel_h)^{-1} pixel_h^T robot
    A_T, residuals, rank, sv = np.linalg.lstsq(pixel_h, robot_points, rcond=None)
    # A_T is 3x3: each column maps [px, py, 1] -> one robot dimension

    transform = A_T.T  # 3x3 matrix: robot_xyz = transform @ [px, py, 1]

    # Compute residuals
    predicted = pixel_h @ A_T
    errors = np.linalg.norm(predicted - robot_points, axis=1)
    mean_residual = np.mean(errors) * 1000  # convert to mm (robot coords in meters)

    # Estimate pixel_to_mm from the transform
    # The scale is approximately the norm of the first two columns of transform
    scale_x = np.linalg.norm(transform[:2, 0])
    scale_y = np.linalg.norm(transform[:2, 1])
    pixel_to_mm = ((scale_x + scale_y) / 2) * 1000  # meters -> mm

    return CalibrationData(
        pixel_points=pixel_points,
        robot_points=robot_points,
        transform_matrix=transform,
        residual_mm=mean_residual,
        pixel_to_mm=pixel_to_mm,
    )


def pixel_to_robot_coords(
    pixel_xy: np.ndarray,
    calibration: Optional[CalibrationData] = None,
    camera_matrix: Optional[np.ndarray] = None,
    depth: Optional[float] = None,
) -> np.ndarray:
    """
    Convert pixel coordinates to robot workspace coordinates.

    Uses the calibration affine transform if available. Falls back to a
    simple scaling with assumed depth if no calibration is provided.

    Parameters
    ----------
    pixel_xy : np.ndarray
        (x, y) pixel coordinates, or Nx2 array.
    calibration : CalibrationData or None
        Calibration data from calibrate(). Preferred.
    camera_matrix : np.ndarray or None
        3x3 camera intrinsic matrix (alternative to calibration).
    depth : float or None
        Distance from camera to tissue surface in meters.

    Returns
    -------
    np.ndarray
        (x, y, z) robot coordinates, or Nx3 array.
    """
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    single = pixel_xy.ndim == 1
    if single:
        pixel_xy = pixel_xy.reshape(1, -1)

    n = pixel_xy.shape[0]

    if calibration is not None and calibration.transform_matrix is not None:
        # Use affine transform
        pixel_h = np.hstack([pixel_xy, np.ones((n, 1))])
        robot_xyz = (calibration.transform_matrix @ pixel_h.T).T
    elif camera_matrix is not None and depth is not None:
        # Back-project using camera intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        x = (pixel_xy[:, 0] - cx) / fx * depth
        y = (pixel_xy[:, 1] - cy) / fy * depth
        z = np.full(n, depth)
        robot_xyz = np.column_stack([x, y, z])
    else:
        # Fallback: simple scaling centered on image center
        # Map pixel coords to robot frame using default pixel_to_mm
        # This is a rough approximation; proper calibration is strongly recommended
        cx, cy = EXPECTED_WIDTH / 2, EXPECTED_HEIGHT / 2
        scale = DEFAULT_PIXEL_TO_MM / 1000.0  # mm -> meters
        x = (pixel_xy[:, 0] - cx) * scale
        y = (pixel_xy[:, 1] - cy) * scale
        z = np.full(n, -0.098)  # default dVRK EE z in base frame
        robot_xyz = np.column_stack([x, y, z])
        print("[pixel_to_robot_coords] WARNING: No calibration provided. "
              "Using rough default transform. Results will be inaccurate.")

    if single:
        return robot_xyz[0]
    return robot_xyz


def apply_calibration_to_plan(
    stitches: List[StitchPlan],
    calibration: CalibrationData,
) -> List[StitchPlan]:
    """
    Convert planned stitch pixel coordinates to robot coordinates.

    Parameters
    ----------
    stitches : list of StitchPlan
        Output from plan_insertion_points().
    calibration : CalibrationData
        Calibration from calibrate().

    Returns
    -------
    list of StitchPlan
        Same list with insertion_robot and exit_robot fields populated.
    """
    for stitch in stitches:
        stitch.insertion_robot = pixel_to_robot_coords(
            stitch.insertion_point, calibration=calibration
        )
        stitch.exit_robot = pixel_to_robot_coords(
            stitch.exit_point, calibration=calibration
        )
    return stitches


# Visualization

def visualize_plan(
    image: np.ndarray,
    wound_info: WoundInfo,
    stitches: Optional[List[StitchPlan]] = None,
    show_mask: bool = False,
) -> np.ndarray:
    """
    Draw the detected wound line and planned suture points on the image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    wound_info : WoundInfo
        Detected wound information.
    stitches : list of StitchPlan or None
        Planned stitch positions. If None, only the wound line is drawn.
    show_mask : bool
        If True, overlay the wound detection mask in semi-transparent red.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    vis = image.copy()

    # Overlay wound mask
    if show_mask and wound_info.mask is not None:
        overlay = vis.copy()
        overlay[wound_info.mask > 0] = [0, 0, 200]  # red tint
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    # Draw wound curve or line (yellow, thick)
    pt_a = tuple(wound_info.endpoint_a.astype(int))
    pt_b = tuple(wound_info.endpoint_b.astype(int))

    if wound_info.spine_points is not None and len(wound_info.spine_points) >= 3:
        # Draw the curve using polylines
        curve_pts = wound_info.spine_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [curve_pts], isClosed=False, color=(0, 255, 255),
                       thickness=2, lineType=cv2.LINE_AA)
    else:
        # Fallback: straight line
        cv2.line(vis, pt_a, pt_b, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw wound endpoints (yellow circles)
    cv2.circle(vis, pt_a, 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, pt_b, 6, (0, 255, 255), -1, cv2.LINE_AA)

    # Draw wound center (white cross)
    center = tuple(wound_info.center.astype(int))
    cv2.drawMarker(vis, center, (255, 255, 255), cv2.MARKER_CROSS, 12, 2)

    # Label wound info
    info_text = (f"Wound: {wound_info.length_px:.0f}px  "
                 f"{wound_info.angle_deg:.1f} deg  "
                 f"[{wound_info.detection_method}]  "
                 f"conf={wound_info.confidence:.2f}")
    cv2.putText(vis, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw stitch plan
    if stitches:
        # Colors
        insertion_color = (255, 100, 0)   # blue-ish: insertion side
        exit_color = (0, 100, 255)        # orange-ish: exit side
        path_color = (0, 255, 0)          # green: suture path
        label_color = (255, 255, 255)     # white: text

        for stitch in stitches:
            ins_pt = tuple(stitch.insertion_point.astype(int))
            ext_pt = tuple(stitch.exit_point.astype(int))
            wound_pt = tuple(stitch.wound_point.astype(int))

            # Draw suture path (insertion -> exit through wound)
            cv2.line(vis, ins_pt, ext_pt, path_color, 1, cv2.LINE_AA)

            # Draw insertion point (filled circle)
            cv2.circle(vis, ins_pt, 5, insertion_color, -1, cv2.LINE_AA)
            cv2.circle(vis, ins_pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw exit point (filled circle)
            cv2.circle(vis, ext_pt, 5, exit_color, -1, cv2.LINE_AA)
            cv2.circle(vis, ext_pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw wound crossing point (small dot)
            cv2.circle(vis, wound_pt, 2, (0, 255, 255), -1, cv2.LINE_AA)

            # Stitch number label
            label_pos = (ins_pt[0] - 15, ins_pt[1] - 10)
            cv2.putText(vis, f"#{stitch.index + 1}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)

        # Legend
        legend_y = vis.shape[0] - 50
        cv2.circle(vis, (15, legend_y), 5, insertion_color, -1)
        cv2.putText(vis, "Insertion", (25, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
        cv2.circle(vis, (115, legend_y), 5, exit_color, -1)
        cv2.putText(vis, "Exit", (125, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
        cv2.putText(vis, f"{len(stitches)} stitches", (175, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

    return vis


# Synthetic demo image generation

def generate_synthetic_tissue(
    width: int = EXPECTED_WIDTH,
    height: int = EXPECTED_HEIGHT,
    wound_angle_deg: float = 10.0,
    wound_length_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a synthetic silicone tissue pad image with a wound line.

    Creates a pink/beige background with texture and a dark line representing
    the incision. Useful for testing the detection pipeline without a real
    camera.

    Parameters
    ----------
    width, height : int
        Image dimensions.
    wound_angle_deg : float
        Angle of the wound line in degrees.
    wound_length_frac : float
        Fraction of image width occupied by the wound.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (image, ground_truth) : (np.ndarray, dict)
        Synthetic BGR image and dict with ground-truth wound parameters.
    """
    rng = np.random.RandomState(seed)

    # Pink/beige tissue background
    # Base color: pinkish silicone (BGR)
    base_color = np.array([180, 190, 220], dtype=np.uint8)  # light pinkish
    image = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add texture noise
    noise = rng.normal(0, 8, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add some slight color gradient (simulates uneven lighting)
    for c in range(3):
        gradient = np.linspace(-15, 15, width).reshape(1, -1)
        gradient = np.repeat(gradient, height, axis=0).astype(np.int16)
        image[:, :, c] = np.clip(image[:, :, c].astype(np.int16) + gradient, 0, 255).astype(np.uint8)

    # Add Gaussian blur for realistic soft-tissue look
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Draw wound line
    cx, cy = width // 2, height // 2
    angle_rad = np.radians(wound_angle_deg)
    half_len = int(width * wound_length_frac / 2)

    x1 = int(cx - half_len * np.cos(angle_rad))
    y1 = int(cy - half_len * np.sin(angle_rad))
    x2 = int(cx + half_len * np.cos(angle_rad))
    y2 = int(cy + half_len * np.sin(angle_rad))

    # Draw wound as a dark line with slight randomness
    wound_color = (40, 30, 35)  # dark brownish
    # Main wound line (thick)
    cv2.line(image, (x1, y1), (x2, y2), wound_color, thickness=4, lineType=cv2.LINE_AA)

    # Add some variation along the wound (slightly wavy)
    num_segments = 20
    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 1) / num_segments
        px1 = int(x1 + (x2 - x1) * t1 + rng.normal(0, 1))
        py1 = int(y1 + (y2 - y1) * t1 + rng.normal(0, 1.5))
        px2 = int(x1 + (x2 - x1) * t2 + rng.normal(0, 1))
        py2 = int(y1 + (y2 - y1) * t2 + rng.normal(0, 1.5))
        thickness = rng.randint(2, 6)
        shade = tuple(int(c + rng.randint(-10, 10)) for c in wound_color)
        shade = tuple(max(0, min(255, c)) for c in shade)
        cv2.line(image, (px1, py1), (px2, py2), shade, thickness, cv2.LINE_AA)

    # Add slight shadow around wound
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(shadow_mask, (x1, y1), (x2, y2), 255, thickness=12, lineType=cv2.LINE_AA)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 5)
    for c in range(3):
        image[:, :, c] = np.clip(
            image[:, :, c].astype(np.int16) - (shadow_mask.astype(np.int16) * 0.15).astype(np.int16),
            0, 255
        ).astype(np.uint8)

    # Add a few reference marks (dots) on the tissue (common on suture pads)
    for _ in range(3):
        rx = rng.randint(int(width * 0.2), int(width * 0.8))
        ry = rng.randint(int(height * 0.15), int(height * 0.85))
        cv2.circle(image, (rx, ry), 2, (100, 100, 120), -1)

    ground_truth = {
        "endpoint_a": np.array([x1, y1]),
        "endpoint_b": np.array([x2, y2]),
        "center": np.array([cx, cy]),
        "angle_deg": wound_angle_deg,
        "length_px": 2 * half_len,
    }

    return image, ground_truth


# Interactive calibration mode

def interactive_calibration() -> CalibrationData:
    """
    Run interactive calibration: user provides pixel-robot point correspondences.

    Prompts the user to enter at least 3 pairs of (pixel_x, pixel_y) and
    (robot_x, robot_y, robot_z) coordinates. Computes and saves the
    calibration transform.

    Returns
    -------
    CalibrationData
        Fitted calibration data.
    """
    print("=" * 60)
    print("Wound Detection - Interactive Calibration")
    print("=" * 60)
    print()
    print("You need at least 3 known pixel-to-robot coordinate pairs.")
    print("Move the robot end-effector to visible positions in the camera")
    print("and record the pixel coordinates and robot coordinates.")
    print()
    print("Robot coordinates should be in the robot BASE frame (meters).")
    print("For dVRK PSM1, the base is at world [0.2, 0.0, 0.15].")
    print()

    pixel_points = []
    robot_points = []

    while True:
        idx = len(pixel_points) + 1
        print(f"--- Point {idx} ---")

        try:
            px_str = input(f"  Pixel X, Y (comma-separated, or 'done'): ").strip()
            if px_str.lower() == "done":
                if len(pixel_points) < 3:
                    print(f"  Need at least 3 points, have {len(pixel_points)}. "
                          "Keep going.")
                    continue
                break

            px, py = [float(v.strip()) for v in px_str.split(",")]

            rx_str = input(f"  Robot X, Y, Z (comma-separated, meters): ").strip()
            rx, ry, rz = [float(v.strip()) for v in rx_str.split(",")]

            pixel_points.append([px, py])
            robot_points.append([rx, ry, rz])
            print(f"  Recorded: pixel ({px:.0f}, {py:.0f}) -> "
                  f"robot ({rx:.4f}, {ry:.4f}, {rz:.4f})")

        except (ValueError, KeyboardInterrupt):
            print("  Invalid input. Try again or type 'done'.")
            continue

    pixel_arr = np.array(pixel_points)
    robot_arr = np.array(robot_points)

    cal = calibrate(pixel_arr, robot_arr)

    print()
    print(f"Calibration complete!")
    print(f"  Points used: {len(pixel_points)}")
    print(f"  Mean residual: {cal.residual_mm:.3f} mm")
    print(f"  Pixel-to-mm scale: {cal.pixel_to_mm:.4f} mm/px")
    print()
    print("Transform matrix (3x3):")
    print(cal.transform_matrix)

    # Save calibration
    cal_path = os.path.expanduser("~/wound_calibration.json")
    cal_dict = {
        "pixel_points": pixel_arr.tolist(),
        "robot_points": robot_arr.tolist(),
        "transform_matrix": cal.transform_matrix.tolist(),
        "residual_mm": cal.residual_mm,
        "pixel_to_mm": cal.pixel_to_mm,
    }
    with open(cal_path, "w") as f:
        json.dump(cal_dict, f, indent=2)
    print(f"\nCalibration saved to {cal_path}")

    return cal


def load_calibration(path: Optional[str] = None) -> Optional[CalibrationData]:
    """
    Load previously saved calibration from JSON file.

    Parameters
    ----------
    path : str or None
        Path to calibration JSON. Defaults to ~/wound_calibration.json.

    Returns
    -------
    CalibrationData or None
        Loaded calibration, or None if file not found.
    """
    if path is None:
        path = os.path.expanduser("~/wound_calibration.json")

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return CalibrationData(
        pixel_points=np.array(data["pixel_points"]),
        robot_points=np.array(data["robot_points"]),
        transform_matrix=np.array(data["transform_matrix"]),
        residual_mm=data["residual_mm"],
        pixel_to_mm=data["pixel_to_mm"],
    )


# CLI entry point

def main():
    parser = argparse.ArgumentParser(
        description="Wound Detection Module for GC-ACT Autonomous Suturing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wound_detection.py --demo
  python wound_detection.py --demo --num_stitches 5 --visualize
  python wound_detection.py --image tissue_photo.jpg --visualize --output plan.png
  python wound_detection.py --calibrate
        """,
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", type=str, help="Path to input endoscope image"
    )
    input_group.add_argument(
        "--camera", type=str, default=None,
        help="ROS camera topic (placeholder, not implemented)"
    )
    input_group.add_argument(
        "--demo", action="store_true",
        help="Generate and process a synthetic tissue pad image"
    )
    input_group.add_argument(
        "--calibrate", action="store_true",
        help="Enter interactive calibration mode"
    )

    parser.add_argument(
        "--num_stitches", type=int, default=DEFAULT_NUM_STITCHES,
        help=f"Number of stitches to plan (default: {DEFAULT_NUM_STITCHES})"
    )
    parser.add_argument(
        "--spacing", type=float, default=DEFAULT_STITCH_SPACING_MM,
        help=f"Stitch spacing in mm (default: {DEFAULT_STITCH_SPACING_MM})"
    )
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_MARGIN_MM,
        help=f"Distance from wound edge in mm (default: {DEFAULT_MARGIN_MM})"
    )
    parser.add_argument(
        "--pixel_to_mm", type=float, default=None,
        help=f"Pixel-to-mm conversion (default: {DEFAULT_PIXEL_TO_MM})"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate annotated visualization image"
    )
    parser.add_argument(
        "--show_mask", action="store_true",
        help="Overlay wound detection mask on visualization"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save annotated image (default: auto-named)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed detection info"
    )
    parser.add_argument(
        "--wound_angle", type=float, default=10.0,
        help="Wound angle for --demo mode (degrees, default: 10)"
    )
    parser.add_argument(
        "--calibration_file", type=str, default=None,
        help="Path to calibration JSON file"
    )

    args = parser.parse_args()

    # Calibration mode
    if args.calibrate:
        interactive_calibration()
        return

    # Camera mode (placeholder)
    if args.camera:
        print(f"[wound_detection] Camera mode not yet implemented.")
        print(f"  Topic: {args.camera}")
        print(f"  To implement: subscribe to the ROS topic, grab a frame,")
        print(f"  and pass it to detect_wound_line().")
        print(f"  For now, save a frame and use --image instead.")
        return

    # Load or generate image
    if args.demo:
        print("[wound_detection] Generating synthetic tissue pad image...")
        image, ground_truth = generate_synthetic_tissue(
            wound_angle_deg=args.wound_angle
        )
        print(f"  Ground truth: angle={ground_truth['angle_deg']:.1f} deg  "
              f"length={ground_truth['length_px']}px")
        print(f"  Wound: ({ground_truth['endpoint_a'][0]},{ground_truth['endpoint_a'][1]}) -> "
              f"({ground_truth['endpoint_b'][0]},{ground_truth['endpoint_b'][1]})")
        # Save the synthetic image
        synth_path = os.path.expanduser("~/synthetic_tissue.png")
        cv2.imwrite(synth_path, image)
        print(f"  Saved synthetic image to {synth_path}")
    elif args.image:
        image_path = os.path.expanduser(args.image)
        if not os.path.exists(image_path):
            print(f"[wound_detection] ERROR: Image not found: {image_path}")
            sys.exit(1)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[wound_detection] ERROR: Could not read image: {image_path}")
            sys.exit(1)
        print(f"[wound_detection] Loaded image: {image_path} "
              f"({image.shape[1]}x{image.shape[0]})")
    else:
        parser.print_help()
        return

    # Detect wound
    print("\n[wound_detection] Running detection...")
    wound_info = detect_wound_line(image, verbose=True)

    if wound_info is None:
        print("\n[wound_detection] DETECTION FAILED")
        print("  Could not find a wound line in the image.")
        print("  Possible causes:")
        print("    - Image too bright/dark (adjust lighting)")
        print("    - Wound is outside the expected region")
        print("    - Wound is too small or too faint")
        print()
        print("  Fallback: hardcode wound coordinates in your script.")
        print("  Example for a horizontal wound centered at (320, 240):")
        print("    wound = WoundInfo(")
        print("        endpoint_a=np.array([160, 240]),")
        print("        endpoint_b=np.array([480, 240]),")
        print("        center=np.array([320, 240]),")
        print("        angle_deg=0.0, length_px=320,")
        print("        mask=np.zeros((480,640), dtype=np.uint8),")
        print("        confidence=1.0, detection_method='hardcoded')")
        sys.exit(1)

    # Compare with ground truth if demo
    if args.demo:
        gt = ground_truth
        angle_err = abs(wound_info.angle_deg - gt["angle_deg"])
        angle_err = min(angle_err, 180 - angle_err)
        center_err = np.linalg.norm(wound_info.center - gt["center"])
        length_err = abs(wound_info.length_px - gt["length_px"])
        print(f"\n  Demo accuracy check:")
        print(f"    Angle error:  {angle_err:.2f} deg")
        print(f"    Center error: {center_err:.1f} px")
        print(f"    Length error:  {length_err:.1f} px")

    # Plan insertion points
    print(f"\n[wound_detection] Planning {args.num_stitches} stitches "
          f"(spacing={args.spacing}mm, margin={args.margin}mm)...")

    ptm = args.pixel_to_mm
    if ptm is None:
        # Try loading from calibration
        cal = load_calibration(args.calibration_file)
        if cal is not None:
            ptm = cal.pixel_to_mm
            print(f"  Using calibrated pixel_to_mm: {ptm:.4f}")
        else:
            ptm = DEFAULT_PIXEL_TO_MM
            print(f"  Using default pixel_to_mm: {ptm} (calibrate for accuracy)")

    stitches = plan_insertion_points(
        wound_info,
        num_stitches=args.num_stitches,
        spacing_mm=args.spacing,
        margin_mm=args.margin,
        pixel_to_mm=ptm,
    )

    # Try to add robot coordinates if calibration exists
    cal = load_calibration(args.calibration_file)
    if cal is not None:
        stitches = apply_calibration_to_plan(stitches, cal)
        print(f"  Applied calibration (residual: {cal.residual_mm:.3f}mm)")

    # Print stitch plan
    print(f"\n{'='*60}")
    print(f"SUTURE PLAN: {len(stitches)} stitches")
    print(f"{'='*60}")
    for s in stitches:
        print(f"  Stitch #{s.index + 1}:")
        print(f"    Wound point:     ({s.wound_point[0]:.1f}, {s.wound_point[1]:.1f}) px")
        print(f"    Insertion point: ({s.insertion_point[0]:.1f}, {s.insertion_point[1]:.1f}) px")
        print(f"    Exit point:      ({s.exit_point[0]:.1f}, {s.exit_point[1]:.1f}) px")
        if s.insertion_robot is not None:
            print(f"    Insertion robot: ({s.insertion_robot[0]:.5f}, "
                  f"{s.insertion_robot[1]:.5f}, {s.insertion_robot[2]:.5f}) m")
            print(f"    Exit robot:      ({s.exit_robot[0]:.5f}, "
                  f"{s.exit_robot[1]:.5f}, {s.exit_robot[2]:.5f}) m")

    # Visualization
    if args.visualize or args.output:
        vis = visualize_plan(image, wound_info, stitches, show_mask=args.show_mask)

        if args.output:
            out_path = os.path.expanduser(args.output)
        else:
            out_path = os.path.expanduser("~/wound_plan.png")

        cv2.imwrite(out_path, vis)
        print(f"\n[wound_detection] Annotated image saved to {out_path}")

    print("\n[wound_detection] Done.")


# Convenience API for programmatic use

def detect_and_plan(
    image: np.ndarray,
    num_stitches: int = DEFAULT_NUM_STITCHES,
    spacing_mm: float = DEFAULT_STITCH_SPACING_MM,
    margin_mm: float = DEFAULT_MARGIN_MM,
    pixel_to_mm: Optional[float] = None,
    calibration: Optional[CalibrationData] = None,
    verbose: bool = False,
) -> Tuple[Optional[WoundInfo], List[StitchPlan]]:
    """
    Full pipeline: detect wound and plan insertion points in one call.

    Parameters
    ----------
    image : np.ndarray
        BGR endoscope image.
    num_stitches : int
        Number of stitches.
    spacing_mm : float
        Stitch spacing in mm.
    margin_mm : float
        Insertion point offset from wound in mm.
    pixel_to_mm : float or None
        Scale factor.
    calibration : CalibrationData or None
        If provided, adds robot coordinates.
    verbose : bool
        Print details.

    Returns
    -------
    (WoundInfo or None, list of StitchPlan)
        Wound detection result and stitch plan. If detection fails, returns
        (None, []).
    """
    wound_info = detect_wound_line(image, verbose=verbose)
    if wound_info is None:
        return None, []

    if pixel_to_mm is None and calibration is not None:
        pixel_to_mm = calibration.pixel_to_mm

    stitches = plan_insertion_points(
        wound_info,
        num_stitches=num_stitches,
        spacing_mm=spacing_mm,
        margin_mm=margin_mm,
        pixel_to_mm=pixel_to_mm,
    )

    if calibration is not None:
        stitches = apply_calibration_to_plan(stitches, calibration)

    return wound_info, stitches


if __name__ == "__main__":
    main()
