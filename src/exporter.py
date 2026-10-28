"""
Comprehensive Document & Verification Mark Exporter
Saves rectified scans, high-contrast binarized documents, isolated marks,
verification match plots, and structured metadata.json records.
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

def save_capture(capture_data, base_output_dir="output"):
    """
    Exports all scan and verification artifacts into a timestamped directory:
    output/scan_YYYYMMDD_HHMMSS/
    """
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"scan_{timestamp_str}"
    target_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    saved_files = []
    
    def write_img(filename, img):
        if img is not None:
            path = os.path.join(target_dir, filename)
            cv2.imwrite(path, img)
            saved_files.append(path)
            
    # Check if document was locked
    if capture_data.get('warped') is not None:
        # High-clarity shadow-free scans
        write_img("clean_magic_color_scan.png", capture_data.get('magic_color'))
        write_img("clean_shadowfree_gray.png", capture_data.get('normalized'))
        write_img("high_contrast_photocopy_bw.png", capture_data.get('binary'))
        # Standard enhanced_scan points to the crisp shadow-free magic scan
        write_img("enhanced_scan.png", capture_data.get('magic_color') if capture_data.get('magic_color') is not None else capture_data.get('normalized'))
        
        # Raw rectified and chromatic extractions
        write_img("rectified_color.png", capture_data.get('warped'))
        write_img("gray.png", capture_data.get('gray'))
        write_img("blue_marks_isolated.png", capture_data.get('blue_marks'))
        write_img("red_marks_isolated.png", capture_data.get('red_marks'))
        write_img("composite_verification.png", capture_data.get('composite_overlay'))
        
        # Save verification match visualizations for both stamp and signature
        v_data = capture_data.get('verification')
        if v_data:
            if v_data.get('stamp') and v_data['stamp'].get('vis') is not None:
                write_img("stamp_match_analysis.png", v_data['stamp']['vis'])
            if v_data.get('signature') and v_data['signature'].get('vis') is not None:
                write_img("signature_match_analysis.png", v_data['signature']['vis'])
            
        # Compile formal metadata JSON
        h_matrix = capture_data.get('homography_matrix')
        h_list = h_matrix.tolist() if h_matrix is not None else []
        
        corners = capture_data.get('ordered_corners')
        corners_list = corners.tolist() if corners is not None else []
        
        stamp_meta = v_data.get('stamp', {}) if v_data else {}
        sig_meta = v_data.get('signature', {}) if v_data else {}
        
        metadata = {
            "timestamp": timestamp_str,
            "document_status": capture_data.get('status'),
            "resolution": {
                "input_width": capture_data['frame'].shape[1],
                "input_height": capture_data['frame'].shape[0],
                "warped_width": capture_data['warped'].shape[1],
                "warped_height": capture_data['warped'].shape[0]
            },
            "source_quad_corners": corners_list,
            "homography_3x3_matrix": h_list,
            "verification": {
                "stamp_status": stamp_meta.get('status', 'N/A'),
                "stamp_confidence_score": stamp_meta.get('score', 0.0),
                "orb_stamp_matches": stamp_meta.get('good_matches', 0),
                "signature_status": sig_meta.get('status', 'N/A'),
                "signature_confidence_score": sig_meta.get('score', 0.0),
                "orb_signature_matches": sig_meta.get('good_matches', 0)
            },
            "performance": {
                "latency_ms": capture_data.get('latency_ms', 0.0),
                "fps": capture_data.get('fps', 0.0)
            }
        }
        
        meta_path = os.path.join(target_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        saved_files.append(meta_path)
    else:
        # Fallback raw photo if no document detected
        write_img("raw_photo.png", capture_data.get('frame'))
        
    return target_dir, saved_files
