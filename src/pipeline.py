"""
End-to-End Computer Vision Pipeline Orchestrator
Coordinates preprocessing, boundary detection, corner stabilization,
homography rectification, dual-branch enhancement, chromatic segmentation,
and automated verification with complete telemetry.
"""

import time
import cv2
import numpy as np

from src.preprocessing import resize_to_proxy, preprocess
from src.document_detector import detect_document
from src.stabilizer import CornerStabilizer
from src.perspective import compute_homography_and_rectify
from src.enhancement import enhance_document
from src.color_extractor import extract_color_marks
from src.verifier import VerificationEngine

class DocumentScannerPipeline:
    def __init__(self, templates_dir="templates"):
        self.stabilizer = CornerStabilizer()
        self.verifier = VerificationEngine(templates_dir=templates_dir)
        self.last_process_time = time.time()
        self.fps = 0.0
        
    def reset_stabilizer(self):
        """Resets tracking history."""
        self.stabilizer.reset()

    def process_frame(self, frame, run_verification=True, is_static_image=False):
        """
        Executes the full pipeline on a single frame.
        Args:
            frame: Input image (live video frame or uploaded smartphone photo).
            run_verification: Whether to run automated stamp/signature matching.
            is_static_image: True for uploaded photos (uses full high-resolution 1200px),
                             False for live video (uses 600px for 35+ FPS performance).
        """
        start_time = time.perf_counter()
        
        # Border padding ensures closed contours even if document touches image borders
        pad = 16
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[35, 35, 35])
        
        # 1. Multi-scale downscale proxy for ultra-fast boundary detection (< 5ms)
        proxy_frame, scale_factor = resize_to_proxy(padded, target_width=500)
        
        # 2. Spatial Normalization & Denoising (Bilateral + Gaussian + Canny)
        proxy_gray, proxy_smoothed, proxy_edges = preprocess(proxy_frame, apply_bilateral=True)
        
        # 3. Hybrid Edge-and-Contour Boundary Extraction
        detected_corners = detect_document(proxy_edges, proxy_frame.shape, scale_to_orig=scale_factor)
        if detected_corners is not None:
            # Unpad coordinates and clamp to frame boundaries
            detected_corners = detected_corners - pad
            detected_corners[:, 0] = np.clip(detected_corners[:, 0], 0, frame.shape[1] - 1)
            detected_corners[:, 1] = np.clip(detected_corners[:, 1], 0, frame.shape[0] - 1)
            
        # 4. Temporal Corner Stabilization (EMA motion gating)
        stabilized_corners = self.stabilizer.update(detected_corners)
        
        # Initialize result payload
        result = {
            'frame': frame,
            'proxy_edges': proxy_edges,
            'detected_corners': detected_corners,
            'corners': stabilized_corners,
            'homography_matrix': None,
            'warped': None,
            'gray': None,
            'normalized': None,
            'binary': None,
            'blue_mask': None,
            'red_mask': None,
            'blue_marks': None,
            'red_marks': None,
            'composite_overlay': None,
            'blue_boxes': [],
            'red_boxes': [],
            'verification': None,
            'latency_ms': 0.0,
            'fps': 0.0,
            'status': "SEARCHING FOR DOCUMENT"
        }
        
        # If document is detected and stabilized, run rectification and dual-branch pipelines
        if stabilized_corners is not None:
            result['status'] = "DOCUMENT LOCKED"
            
            # 5. 2D Planar Homography Rectification (High-res 1400px Lanczos-4 for static photos, fast 480px for 35+ FPS video)
            if is_static_image:
                warped, H_matrix, ordered_corners = compute_homography_and_rectify(
                    frame, stabilized_corners, target_width=1400
                )
            else:
                warped, H_matrix, ordered_corners = compute_homography_and_rectify(
                    frame, stabilized_corners, max_output_width=480
                )
            result['warped'] = warped
            result['homography_matrix'] = H_matrix
            result['ordered_corners'] = ordered_corners
            
            if warped is not None:
                # 6. Branch A: Adaptive Spatial Enhancement (Shadow-free scan + magic color + unsharp masking)
                enhanced_data = enhance_document(warped)
                result['gray'] = enhanced_data['gray']
                result['normalized'] = enhanced_data['normalized']
                result['magic_color'] = enhanced_data['magic_color']
                result['binary'] = enhanced_data['binary']
                
                # 7. Branch B: HSV Chromatic Segmentation
                color_data = extract_color_marks(warped)
                result['blue_mask'] = color_data['blue_mask']
                result['red_mask'] = color_data['red_mask']
                result['blue_marks'] = color_data['blue_marks']
                result['red_marks'] = color_data['red_marks']
                result['composite_overlay'] = color_data['composite_overlay']
                result['blue_boxes'] = color_data['blue_boxes']
                result['red_boxes'] = color_data['red_boxes']
                
                # 8. Automated Verification (Feature matching against templates)
                if run_verification:
                    v_res = self.verifier.verify_marks(color_data)
                    result['verification'] = v_res

        # Calculate latency and FPS
        elapsed = time.perf_counter() - start_time
        result['latency_ms'] = round(elapsed * 1000.0, 1)
        
        now = time.time()
        fps_instant = 1.0 / max(1e-4, now - self.last_process_time)
        self.last_process_time = now
        self.fps = 0.9 * self.fps + 0.1 * fps_instant
        result['fps'] = round(self.fps, 1)
        
        return result
