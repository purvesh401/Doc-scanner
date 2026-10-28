"""
Chromatic Verification Mark Extraction Module
Operates in the HSV cylindrical color space using selective hue-saturation bandpass
filters, morphological clustering, and connected-component analysis to isolate
blue handwritten signatures and red official validation stamps from printed text.
"""

import cv2
import numpy as np

def filter_noise_components(mask, min_area=20, cluster_kernel_size=19):
    """
    Applies fast morphological opening to eliminate isolated dust and noise speckles,
    then clusters adjacent ink strokes/letters into cohesive bounding boxes for
    entire stamps and signatures using 2x downscaled morphology (< 1ms).
    """
    # 1. Fast sub-millisecond morphological opening to eliminate paper noise
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)

    # 2. Fast 2x downscaled morphological closure for sub-millisecond clustering
    h, w = clean_mask.shape[:2]
    small_mask = cv2.resize(clean_mask, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_NEAREST)
    k_size = max(3, cluster_kernel_size // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for c in cnts:
        area = cv2.contourArea(c) * 4.0
        if area >= 120:
            sbx, sby, sbw, sbh = cv2.boundingRect(c)
            bx, by, bw, bh = sbx * 2, sby * 2, sbw * 2, sbh * 2
            bx = int(np.clip(bx, 0, w - 1))
            by = int(np.clip(by, 0, h - 1))
            bw = int(np.clip(bw, 1, w - bx))
            bh = int(np.clip(bh, 1, h - by))
            
            # Ignore border artifacts touching the very edge
            if (bx == 0 or by == 0 or bx + bw >= w or by + bh >= h) and (bw < 15 or bh < 15):
                continue
                
            ink_px = cv2.countNonZero(clean_mask[by:by+bh, bx:bx+bw])
            if ink_px >= 35:
                bounding_boxes.append((bx, by, bw, bh, ink_px))
                
    # Sort bounding boxes by ink area descending
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[4], reverse=True)
    return clean_mask, bounding_boxes

def extract_color_marks(warped_bgr, blue_hue_range=(95, 138), red_hue_low=(0, 12), red_hue_high=(165, 180)):
    """
    Extracts blue signatures and red stamps from a rectified color document:
    1. Converts BGR to HSV cylindrical space.
    2. Applies selective Hue-Saturation-Value bandpass filters (calibrated for phone photos & compression).
    3. Clears outer edge margin to reject desk bleed.
    4. Executes morphological clustering and noise filtering.
    5. Produces isolated marks on clean white backgrounds and labeled overlays.
    """
    if warped_bgr is None:
        return {
            'blue_mask': None,
            'red_mask': None,
            'blue_marks': None,
            'red_marks': None,
            'composite_overlay': None,
            'blue_boxes': [],
            'red_boxes': []
        }
        
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    h, w = warped_bgr.shape[:2]
    
    # 1. Blue Pen Ink Bandpass Filter (S >= 25 accommodates phone cameras and indoor lighting)
    lower_blue = np.array([blue_hue_range[0], 25, 40], dtype=np.uint8)
    upper_blue = np.array([blue_hue_range[1], 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 2. Red Stamp Bandpass Filter (Hue wraps around 0 and 180)
    lower_red1 = np.array([red_hue_low[0], 25, 40], dtype=np.uint8)
    upper_red1 = np.array([red_hue_low[1], 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([red_hue_high[0], 25, 40], dtype=np.uint8)
    upper_red2 = np.array([red_hue_high[1], 255, 255], dtype=np.uint8)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 3. Suppress 2.5% outer perimeter margin to eliminate dark desk bleeding
    by = max(3, int(h * 0.025))
    bx = max(3, int(w * 0.025))
    mask_blue[:by, :] = 0; mask_blue[h-by:, :] = 0; mask_blue[:, :bx] = 0; mask_blue[:, w-bx:] = 0
    mask_red[:by, :] = 0; mask_red[h-by:, :] = 0; mask_red[:, :bx] = 0; mask_red[:, w-bx:] = 0
    
    # 4. Morphological Cleanup & Clustering
    clean_mask_blue, blue_boxes = filter_noise_components(mask_blue, min_area=15, cluster_kernel_size=17)
    clean_mask_red, red_boxes = filter_noise_components(mask_red, min_area=20, cluster_kernel_size=21)
    
    # 5. Render Isolated Marks on Clean White Background (for forensic inspection)
    white_bg = np.full_like(warped_bgr, 255)
    
    blue_marks = white_bg.copy()
    np.copyto(blue_marks, warped_bgr, where=(clean_mask_blue[:, :, np.newaxis] > 0))
    
    red_marks = white_bg.copy()
    np.copyto(red_marks, warped_bgr, where=(clean_mask_red[:, :, np.newaxis] > 0))
    
    # 6. Composite Overlay with Labeled Bounding Boxes
    overlay = warped_bgr.copy()
    for (x, y, bw, bh, _) in blue_boxes:
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 120, 0), 2)
        cv2.putText(overlay, "SIGNATURE", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1, cv2.LINE_AA)
        
    for (x, y, bw, bh, _) in red_boxes:
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(overlay, "OFFICIAL STAMP", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
    return {
        'blue_mask': clean_mask_blue,
        'red_mask': clean_mask_red,
        'blue_marks': blue_marks,
        'red_marks': red_marks,
        'composite_overlay': overlay,
        'blue_boxes': blue_boxes,
        'red_boxes': red_boxes
    }
