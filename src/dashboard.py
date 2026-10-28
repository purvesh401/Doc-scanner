"""
Multi-Viewport Diagnostic HUD Composite Generator
Assembles the 4-pane diagnostic display with razor-sharp document rendering,
dedicated side-by-side stamp/signature template matching, and clean telemetry.
"""

import os
import cv2
import numpy as np

def create_labeled_viewport(img, title, border_color=(60, 60, 60), target_size=(570, 365)):
    """
    Resizes an image into a uniform viewport pane with a styled dark header and border.
    Uses high-quality INTER_AREA interpolation to preserve text sharpness.
    """
    tw, th = target_size
    
    if img is None:
        pane = np.full((th, tw, 3), 22, dtype=np.uint8)
        cv2.putText(pane, "NO ACTIVE SIGNAL", (tw // 2 - 100, th // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    else:
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
            
        # Aspect-ratio preserving fit with letterboxing
        ih, iw = img_bgr.shape[:2]
        header_h = 32
        avail_h = th - header_h
        
        scale = min(tw / float(iw), avail_h / float(ih))
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        
        pane = np.full((th, tw, 3), 22, dtype=np.uint8)
        x_off = (tw - nw) // 2
        y_off = header_h + (avail_h - nh) // 2
        pane[y_off:y_off+nh, x_off:x_off+nw] = resized
        
    # Draw dark header bar
    header_h = 32
    cv2.rectangle(pane, (0, 0), (tw, header_h), (32, 32, 32), -1)
    cv2.line(pane, (0, header_h), (tw, header_h), border_color, 1)
    
    # Title text
    cv2.putText(pane, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Outer border
    cv2.rectangle(pane, (0, 0), (tw - 1, th - 1), border_color, 1)
    return pane

def render_forensic_verification_viewport(state, target_size=(570, 365), templates_dir="templates"):
    """
    Dedicated Viewport 4: Displays side-by-side template matching for official stamps
    (with ORB keypoint correspondence lines) and isolated signatures on white background.
    """
    tw, th = target_size
    pane = np.full((th, tw, 3), 22, dtype=np.uint8)
    header_h = 32
    
    # Header bar
    cv2.rectangle(pane, (0, 0), (tw, header_h), (32, 32, 32), -1)
    cv2.line(pane, (0, header_h), (tw, header_h), (220, 60, 60), 1)
    cv2.putText(pane, "VIEWPORT 4: FORENSIC VERIFICATION & MATCHING", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(pane, (0, 0), (tw - 1, th - 1), (220, 60, 60), 1)
    
    v_data = state.get('verification')
    has_stamp_match = False
    
    # --- Top Section: Official Stamp Verification Match ---
    if v_data and v_data.get('stamp') and v_data['stamp'].get('vis') is not None:
        vis = v_data['stamp']['vis']
        vh, vw = vis.shape[:2]
        max_vh = 150
        scale_v = min((tw - 24) / float(vw), max_vh / float(vh))
        rw, rh = int(vw * scale_v), int(vh * scale_v)
        resized_vis = cv2.resize(vis, (rw, rh), interpolation=cv2.INTER_AREA)
        
        x_off = (tw - rw) // 2
        y_off = header_h + 8
        pane[y_off:y_off+rh, x_off:x_off+rw] = resized_vis
        cv2.rectangle(pane, (x_off - 1, y_off - 1), (x_off + rw, y_off + rh), (100, 100, 100), 1)
        
        # Decision Badge
        score = v_data['stamp'].get('score', 0.0)
        status = v_data['stamp'].get('status', 'UNVERIFIED')
        good_m = v_data['stamp'].get('good_matches', 0)
        badge_col = (0, 230, 80) if status == "AUTHENTICATED" else (0, 80, 230)
        
        badge_txt = f"STAMP: [{status}]  Match: {score:.1f}%  ({good_m} Keypoint Pairs)"
        cv2.putText(pane, badge_txt, (16, y_off + rh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_col, 1, cv2.LINE_AA)
        has_stamp_match = True
    else:
        # Placeholder if no stamp is visible
        cv2.putText(pane, "[ OFFICIAL STAMP: Awaiting Document Detection ]", 
                    (tw // 2 - 190, header_h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
    # Divider line
    div_y = header_h + 185
    cv2.line(pane, (10, div_y), (tw - 10, div_y), (50, 50, 50), 1)
    
    # --- Bottom Section: Signature Isolation & Verification ---
    sig_boxes = state.get('blue_boxes', [])
    blue_marks = state.get('blue_marks')
    
    y_sig_base = div_y + 10
    if sig_boxes and blue_marks is not None:
        bx, by, bw, bh, _ = sig_boxes[0]
        sig_crop = blue_marks[by:by+bh, bx:bx+bw]
        
        # Display Extracted Signature on left
        box_w, box_h = 240, 68
        sh, sw = sig_crop.shape[:2]
        scale_s = min(box_w / float(sw), box_h / float(sh))
        cw, ch = int(sw * scale_s), int(sh * scale_s)
        resized_sig = cv2.resize(sig_crop, (cw, ch), interpolation=cv2.INTER_AREA)
        
        x_sig = 20 + (box_w - cw) // 2
        y_sig = y_sig_base + (box_h - ch) // 2
        pane[y_sig:y_sig+ch, x_sig:x_sig+cw] = resized_sig
        cv2.rectangle(pane, (20, y_sig_base), (20 + box_w, y_sig_base + box_h), (255, 120, 0), 1)
        cv2.putText(pane, "EXTRACTED BLUE SIGNATURE", (22, y_sig_base + box_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 150, 50), 1, cv2.LINE_AA)
                    
        # Display Authorized Signature Template on right
        tmpl_sig_path = os.path.join(templates_dir, "authorized_signature_template.png")
        if os.path.exists(tmpl_sig_path):
            tmpl_sig = cv2.imread(tmpl_sig_path)
            if tmpl_sig is not None:
                t_box_w = 240
                t_sh, t_sw = tmpl_sig.shape[:2]
                t_scale = min(t_box_w / float(t_sw), box_h / float(t_sh))
                tcw, tch = int(t_sw * t_scale), int(t_sh * t_scale)
                resized_tmpl = cv2.resize(tmpl_sig, (tcw, tch), interpolation=cv2.INTER_AREA)
                
                x_tmpl = tw - 20 - t_box_w + (t_box_w - tcw) // 2
                y_tmpl = y_sig_base + (box_h - tch) // 2
                pane[y_tmpl:y_tmpl+tch, x_tmpl:x_tmpl+tcw] = resized_tmpl
                cv2.rectangle(pane, (tw - 20 - t_box_w, y_sig_base), (tw - 20, y_sig_base + box_h), (80, 180, 80), 1)
                cv2.putText(pane, "AUTHORIZED REFERENCE TEMPLATE", (tw - 20 - t_box_w + 2, y_sig_base + box_h + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 100), 1, cv2.LINE_AA)
                
        # Signature Decision Badge
        sig_meta = v_data.get('signature', {}) if v_data else {}
        sig_score = sig_meta.get('score', 0.0)
        sig_status = sig_meta.get('status', 'UNVERIFIED')
        sig_m = sig_meta.get('good_matches', 0)
        sig_badge_col = (0, 230, 80) if sig_status == "AUTHENTICATED" else (0, 165, 255)
        
        sig_badge_txt = f"SIGNATURE: [{sig_status}]  Match: {sig_score:.1f}%  ({sig_m} Keypoints)"
        cv2.putText(pane, sig_badge_txt, (16, y_sig_base + box_h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, sig_badge_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(pane, "[ BLUE SIGNATURE: No Signature Detected in Frame ]", 
                    (tw // 2 - 180, y_sig_base + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                    
    return pane

def render_multi_viewport_hud(state, target_size=(1160, 780), scan_mode=0, templates_dir="templates"):
    """
    Composites the 4-viewport diagnostic HUD:
    - Viewport 1 (Top-Left)    : Live Camera Tracker with stabilized boundaries
    - Viewport 2 (Top-Right)   : Canny Edge & Gradient Map
    - Viewport 3 (Bottom-Left) : Rectified Razor-Sharp Scan (Magic Color / Clean Gray / Photocopy B&W)
    - Viewport 4 (Bottom-Right): Dedicated Forensic Verification & Template Matching
    """
    vw = target_size[0] // 2
    vh = (target_size[1] - 50) // 2  # reserve 50px for bottom telemetry bar
    
    # --- Viewport 1: Live Tracker Overlay ---
    tracker_frame = state['frame'].copy()
    corners = state.get('ordered_corners')
    if corners is None:
        corners = state.get('corners')
    if corners is not None:
        pts = corners.astype(np.int32)
        cv2.polylines(tracker_frame, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        
        labels = ["TL (0)", "TR (1)", "BR (2)", "BL (3)"]
        colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0)]
        for i, (pt, label, col) in enumerate(zip(pts, labels, colors)):
            cv2.circle(tracker_frame, tuple(pt), 7, col, -1, cv2.LINE_AA)
            cv2.circle(tracker_frame, tuple(pt), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tracker_frame, label, (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        
    vp1 = create_labeled_viewport(tracker_frame, "VIEWPORT 1: LIVE BOUNDARY TRACKER", 
                                  border_color=(0, 200, 100), target_size=(vw, vh))
    
    # --- Viewport 2: Canny Edge Map ---
    edges = state.get('proxy_edges')
    vp2 = create_labeled_viewport(edges, "VIEWPORT 2: CANNY GRADIENT EDGE MAP", 
                                  border_color=(180, 180, 50), target_size=(vw, vh))
    
    # --- Viewport 3: Rectified Razor-Sharp Scan ---
    # Cycle between 0: Magic Color, 1: Clean Gray, 2: Photocopy B&W, 3: Raw Rectified
    if scan_mode == 0 and state.get('magic_color') is not None:
        scan_img = state.get('magic_color')
        vp3_title = "VIEWPORT 3: SHADOW-FREE SCAN (MAGIC COLOR)"
    elif scan_mode == 1 and state.get('normalized') is not None:
        scan_img = state.get('normalized')
        vp3_title = "VIEWPORT 3: SHADOW-FREE SCAN (CLEAN GRAY)"
    elif scan_mode == 2 and state.get('binary') is not None:
        scan_img = state.get('binary')
        vp3_title = "VIEWPORT 3: PHOTOCOPY SCAN (HIGH-CONTRAST B&W)"
    elif state.get('magic_color') is not None:
        scan_img = state.get('magic_color')
        vp3_title = "VIEWPORT 3: SHADOW-FREE SCAN (MAGIC COLOR)"
    elif state.get('normalized') is not None:
        scan_img = state.get('normalized')
        vp3_title = "VIEWPORT 3: SHADOW-FREE SCAN (CLEAN GRAY)"
    elif state.get('binary') is not None:
        scan_img = state.get('binary')
        vp3_title = "VIEWPORT 3: RECTIFIED SCAN (HIGH-CONTRAST B&W)"
    else:
        scan_img = state.get('warped')
        vp3_title = "VIEWPORT 3: RECTIFIED ADAPTIVE SCAN"
        
    vp3 = create_labeled_viewport(scan_img, vp3_title, 
                                  border_color=(50, 150, 255), target_size=(vw, vh))
    
    # --- Viewport 4: Dedicated Forensic Verification Viewport ---
    vp4 = render_forensic_verification_viewport(state, target_size=(vw, vh), templates_dir=templates_dir)
    
    # Assemble 2x2 grid
    top_row = np.hstack((vp1, vp2))
    bottom_row = np.hstack((vp3, vp4))
    grid = np.vstack((top_row, bottom_row))
    
    # --- Bottom Telemetry & Navigation Bar ---
    telemetry_bar = np.full((50, target_size[0], 3), 20, dtype=np.uint8)
    cv2.line(telemetry_bar, (0, 0), (target_size[0], 0), (60, 60, 60), 1)
    
    fps_val = state.get('fps', 0.0)
    lat_val = state.get('latency_ms', 0.0)
    doc_status = state.get('status', 'SEARCHING')
    
    status_color = (0, 255, 0) if "LOCKED" in doc_status else (0, 165, 255)
    
    # Clean non-overlapping telemetry coordinates
    cv2.putText(telemetry_bar, f"STATUS: {doc_status}", (15, 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
                
    cv2.putText(telemetry_bar, f"PERF: {fps_val:.1f} FPS ({lat_val:.1f}ms)", 
                (360, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
                
    cv2.putText(telemetry_bar, "[O] Open Photo  |  [E] Toggle Scan Mode  |  [S] Save  |  [Q] Exit", 
                (590, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
                
    hud = np.vstack((grid, telemetry_bar))
    return hud
