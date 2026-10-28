"""
Hybrid Document Boundary Extraction Module
Implements Canny gradient contour tracking, convex hull occlusion recovery
(for hand/thumb grips), and multi-scale Douglas-Peucker polygon approximation.
"""

import cv2
import numpy as np

def calculate_quad_angles(pts):
    """Calculates the 4 interior angles of a quadrilateral."""
    angles = []
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]
        
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    return angles

def is_valid_quadrilateral(approx, frame_shape, min_area_ratio=0.08, max_area_ratio=0.98):
    """
    Validates whether a 4-point polygon is a plausible physical document:
    1. Must be convex.
    2. Area must be within acceptable bounds (not tiny clutter).
    3. Interior angles must be within plausible perspective range (35° to 145°).
    """
    if len(approx) != 4:
        return False
        
    pts = approx.reshape(4, 2)
    
    # 1. Strict convexity check
    if not cv2.isContourConvex(approx):
        return False
        
    # 2. Area ratio check
    h, w = frame_shape[:2]
    frame_area = h * w
    area = cv2.contourArea(approx)
    if area < (min_area_ratio * frame_area) or area > (max_area_ratio * frame_area):
        return False
        
    # 3. Angle sanity check (angles shouldn't be degenerate needles or flat lines)
    angles = calculate_quad_angles(pts)
    for ang in angles:
        if ang < 35.0 or ang > 145.0:
            return False
            
    return True

def detect_document(edges, frame_shape, scale_to_orig=1.0):
    """
    Detects the 4-corner boundary of a target document from edge maps:
    1. Extracts top contour candidates by area.
    2. Applies Convex Hull to eliminate boundary indentations from hand/thumb occlusions.
    3. Sweeps Douglas-Peucker epsilon parameter dynamically to find exactly 4 stable vertices.
    4. Validates geometric plausibility.
    5. Returns coordinates scaled back to the original full-resolution frame.
    """
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    # Sort contours by area descending, examine top 10 candidates
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        if peri < 100:
            continue
            
        # Strategy A: Multi-scale Douglas-Peucker directly on contour
        # Sweep epsilon from fine (0.015) to coarse (0.045)
        quad_candidate = None
        for eps_ratio in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]:
            approx = cv2.approxPolyDP(c, eps_ratio * peri, True)
            if is_valid_quadrilateral(approx, frame_shape):
                quad_candidate = approx.reshape(4, 2)
                break
                
        # Strategy B: If finger/hand occlusion caused bumps/notches, wrap in Convex Hull
        # This bridges finger grips on the edge of the paper
        if quad_candidate is None:
            hull = cv2.convexHull(c)
            hull_peri = cv2.arcLength(hull, True)
            for eps_ratio in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
                approx = cv2.approxPolyDP(hull, eps_ratio * hull_peri, True)
                if is_valid_quadrilateral(approx, frame_shape):
                    quad_candidate = approx.reshape(4, 2)
                    break
                    
        if quad_candidate is not None:
            # Map coordinates back to original full resolution
            if scale_to_orig != 1.0:
                scaled_corners = (quad_candidate.astype(np.float32) * scale_to_orig)
                return scaled_corners
            return quad_candidate.astype(np.float32)
            
    return None
