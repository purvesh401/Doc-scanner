"""
Adaptive Spatial Enhancement Module
Implements background illumination normalization, CLAHE local contrast stretching,
unsharp mask edge peaking, and CamScanner-style Magic Color reconstruction to
eliminate shadows, creases, and lighting gradients while producing razor-sharp text.
"""

import cv2
import numpy as np

def normalize_illumination(gray_img):
    """
    Estimates the low-frequency background illumination plane using morphological
    dilation and Gaussian smoothing, dividing the image against it to completely
    neutralize hand shadows, creases, and non-uniform room lighting.
    """
    h, w = gray_img.shape[:2]
    ksize = max(21, int(w / 35))
    if ksize % 2 == 0:
        ksize += 1
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(gray_img, kernel)
    bg = cv2.GaussianBlur(dilated, (ksize, ksize), 0)
    bg = np.maximum(bg, 1)
    
    normalized = cv2.divide(gray_img, bg, scale=255)
    
    # Fast percentile contrast stretch via convertScaleAbs (< 0.1 ms)
    p_low, p_high = np.percentile(normalized[::4, ::4], (2, 96))
    scale = 255.0 / max(1.0, p_high - p_low)
    stretched = cv2.convertScaleAbs(normalized, alpha=scale, beta=-p_low * scale)
    return stretched

def enhance_document(warped_bgr):
    """
    Multi-branch document enhancement pipeline:
    1. Shadow & crease elimination via divisive background normalization on Luminance.
    2. Adaptive local contrast enhancement (CLAHE) for deep black character strokes.
    3. Unsharp masking for razor-sharp letter edge definition.
    4. CamScanner-style Magic Color reconstruction (vivid red stamp & blue signature on pure white paper).
    5. Clean photocopy binary thresholding without character stroke erosion.
    
    Returns dictionary with:
      - 'gray': Raw grayscale warped frame.
      - 'normalized': Razor-sharp shadow-free grayscale scan (clean paper, dark text).
      - 'magic_color': Full-color shadow-free magic scan.
      - 'binary': High-contrast clean photocopy binary.
    """
    if warped_bgr is None:
        return {'gray': None, 'normalized': None, 'magic_color': None, 'binary': None}
        
    h, w = warped_bgr.shape[:2]
    
    # Convert to LAB color space to isolate illumination from chrominance
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Estimate background illumination plane: dilation fills dark text with surrounding paper
    ksize = max(21, int(w / 35))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(l, kernel)
    bg = cv2.GaussianBlur(dilated, (ksize, ksize), 0)
    bg = np.maximum(bg, 1)
    
    # 1. Divisive illumination normalization to remove desk lighting & hand shadows
    norm_l = cv2.divide(l, bg, scale=255)
    
    # 2. CLAHE for boosting local character stroke contrast
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(norm_l)
    
    # 3. Unsharp Masking for razor-sharp character contours (eliminates fuzzy edges)
    blurred_l = cv2.GaussianBlur(enhanced_l, (0, 0), sigmaX=1.2)
    sharp_l = cv2.addWeighted(enhanced_l, 1.7, blurred_l, -0.7, 0)
    
    # 4. Fast percentile contrast stretching: forces paper to pure white (255) and ink to solid dark
    p_low, p_high = np.percentile(sharp_l[::4, ::4], (2, 95))
    scale = 255.0 / max(1.0, p_high - p_low)
    clean_l = cv2.convertScaleAbs(sharp_l, alpha=scale, beta=-p_low * scale)
    
    # 5. Magic Color scan: recombine sharp luminance with original chromatic channels
    magic_color = cv2.cvtColor(cv2.merge([clean_l, a, b]), cv2.COLOR_LAB2BGR)
    
    # 6. Photocopy Binary: Otsu optimal global thresholding preserving complete stroke shapes
    _, binary = cv2.threshold(clean_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    
    return {
        'gray': gray,
        'normalized': clean_l,
        'magic_color': magic_color,
        'binary': binary
    }
