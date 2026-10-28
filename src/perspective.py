"""
2D Planar Homography and Perspective Rectification Module
Computes robust geometric corner ordering, the 3x3 projective transformation matrix,
and warps perspective-distorted documents into upright canonical Euclidean planes.
"""

import cv2
import numpy as np

def order_points_clockwise(pts):
    """
    Robust geometric corner ordering (TL, TR, BR, BL) using centroid polar angles.
    Unlike naive sum/diff methods, this remains stable across 360-degree rotations.
    """
    pts = np.array(pts, dtype="float32")
    
    # 1. Compute geometric centroid of the quadrilateral
    centroid = np.mean(pts, axis=0)
    
    # 2. Compute polar angles of each point relative to the centroid
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    
    # 3. Sort points in clockwise order based on angle
    sort_idx = np.argsort(angles)
    sorted_pts = pts[sort_idx]
    
    # 4. Identify Top-Left corner: the point in the upper-left quadrant (smallest x + y)
    # Roll the array so that Top-Left is at index 0
    s = sorted_pts[:, 0] + sorted_pts[:, 1]
    tl_idx = np.argmin(s)
    ordered = np.roll(sorted_pts, -tl_idx, axis=0)
    
    # Verify clockwise orientation: cross product between (p1 - p0) and (p3 - p0)
    v1 = ordered[1] - ordered[0]
    v2 = ordered[3] - ordered[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        # Counter-clockwise; swap index 1 and 3
        ordered[[1, 3]] = ordered[[3, 1]]
        
    return ordered

def compute_homography_and_rectify(frame, corners, enforce_a4_ratio=True, target_width=None, max_output_width=None):
    """
    Computes the 3x3 projective homography transformation matrix and warps
    the detected document quadrilateral into an upright Euclidean rectangular plane.
    
    Args:
        frame: Original high-resolution BGR input frame.
        corners: 4 corner coordinates (float32 array).
        enforce_a4_ratio: When True, adjusts foreshortened height to standard A4 (1:1.414)
                          aspect ratio, eliminating perspective squashing.
        target_width: Explicit output width (e.g. 1400 for high-resolution static scans).
        max_output_width: Maximum width cap for real-time video streaming (e.g. 800px).
    
    Returns:
        warped: Perspective-corrected rectified BGR image.
        H_matrix: The 3x3 projective homography transformation matrix.
        ordered_corners: Ordered (TL, TR, BR, BL) corners in source frame coordinates.
    """
    if corners is None or len(corners) != 4:
        return None, None, None
        
    ordered_corners = order_points_clockwise(corners)
    (tl, tr, br, bl) = ordered_corners
    
    if target_width is not None and target_width > 0:
        max_width = int(target_width)
        max_height = int(max_width * 1.414) if enforce_a4_ratio else int(np.linalg.norm(tl - bl) * (target_width / max(1.0, np.linalg.norm(tr - tl))))
    else:
        # Compute measured Euclidean dimensions of the document
        width_bottom = np.linalg.norm(br - bl)
        width_top = np.linalg.norm(tr - tl)
        max_width = int(max(width_bottom, width_top))
        
        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        max_height = int(max(height_right, height_left))
        
        # Perspective foreshortening compensation (Standard A4 ratio 1:1.414)
        if enforce_a4_ratio and max_width > 100:
            measured_ratio = max_height / float(max_width)
            if measured_ratio < 1.414:
                # Document is tilted backwards; restore canonical A4 height
                max_height = int(max_width * 1.414)
                
        # Scale dimensions for real-time video processing if capped
        if max_output_width is not None and max_width > max_output_width:
            scale = max_output_width / float(max_width)
            max_width = int(max_output_width)
            max_height = int(max_height * scale)
    
    # Sanity bounds to prevent degenerate warps
    max_width = int(np.clip(max_width, 150, 4000))
    max_height = int(np.clip(max_height, 150, 4000))
    
    # Destination canonical Euclidean rectangle
    dst_rect = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Calculate 3x3 Projective Homography Matrix (Direct Linear Transform)
    H_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_rect)
    
    # For high-fidelity static scans use Lanczos-4; for live video streaming use bilinear (< 1 ms)
    warp_flags = cv2.INTER_LANCZOS4 if (target_width is not None) else cv2.INTER_LINEAR
    warped = cv2.warpPerspective(
        frame,
        H_matrix,
        (max_width, max_height),
        flags=warp_flags,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped, H_matrix, ordered_corners
