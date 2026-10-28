"""
Preprocessing and Edge Extraction Module
Implements spatial normalization, bilateral and Gaussian filtering,
proxy downscaling for 30+ FPS real-time tracking, and Auto-Canny edge detection.
"""

import cv2
import numpy as np

def resize_to_proxy(frame, target_width=500):
    """
    Downscales a high-resolution frame (e.g. 1280x720) to a lightweight proxy
    (e.g. width=500) for real-time edge detection and contour approximation (<5ms).
    Returns the proxy frame and the inverse scale factor to map coordinates back.
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame.copy(), 1.0
    scale = target_width / float(w)
    target_height = int(h * scale)
    proxy = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return proxy, 1.0 / scale

def preprocess(frame, apply_bilateral=True, sigma_color=75, sigma_space=75):
    """
    Preprocesses the input frame:
    1. Grayscale conversion.
    2. Spatial smoothing via Gaussian Blur (eliminates high-frequency noise).
    3. Bilateral Filtering (edge-preserving smoothing that suppresses paper/desk
       texture while keeping document perimeter edges sharp).
    4. Auto-Canny edge detection based on median brightness.
    5. Morphological dilation/closing to seal edge discontinuities.
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
        
    # Standard Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Bilateral Filter for edge-preserving denoising (as specified in MSU Abstract)
    if apply_bilateral:
        smoothed = cv2.bilateralFilter(blurred, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    else:
        smoothed = blurred

    # Auto-Canny Edge Detection based on median pixel intensity
    v = np.median(smoothed)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    
    # Ensure minimum spread between thresholds
    if upper_thresh - lower_thresh < 30:
        lower_thresh = max(0, lower_thresh - 15)
        upper_thresh = min(255, upper_thresh + 15)
        
    edges = cv2.Canny(smoothed, lower_thresh, upper_thresh)
    
    # Morphological dilation to bridge micro-gaps in the document perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return gray, smoothed, edges
