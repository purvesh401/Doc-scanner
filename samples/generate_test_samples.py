"""
Synthetic Sample Document and Template Generator
Creates realistic academic and official document images with:
- Formal printed text
- Blue handwritten signature
- Official red circular university stamp/seal
- Perspective tilt, desk background, and uneven shadow gradient
"""

import os
import cv2
import numpy as np

def create_official_stamp(size=300):
    """Generates an official circular university seal/stamp in red ink."""
    stamp = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA
    center = (size // 2, size // 2)
    radius = size // 2 - 10
    
    # Outer concentric circles
    red_color = (30, 30, 200, 240)  # BGR + Alpha
    cv2.circle(stamp, center, radius, red_color, 4, cv2.LINE_AA)
    cv2.circle(stamp, center, radius - 12, red_color, 2, cv2.LINE_AA)
    cv2.circle(stamp, center, radius - 35, red_color, 2, cv2.LINE_AA)
    
    # Text around circle / inside
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(stamp, "M. S. UNIVERSITY", (center[0] - 90, center[1] - 45), font, 0.65, red_color, 2, cv2.LINE_AA)
    cv2.putText(stamp, "* BARODA *", (center[0] - 60, center[1] - 15), font, 0.6, red_color, 2, cv2.LINE_AA)
    cv2.putText(stamp, "FACULTY OF TECH", (center[0] - 88, center[1] + 18), font, 0.58, red_color, 2, cv2.LINE_AA)
    cv2.putText(stamp, "OFFICIAL SEAL", (center[0] - 70, center[1] + 48), font, 0.55, red_color, 2, cv2.LINE_AA)
    
    # Add star symbols
    cv2.putText(stamp, "[ VERIFIED ]", (center[0] - 65, center[1] + 80), font, 0.5, red_color, 1, cv2.LINE_AA)
    
    # Add slight stamp texture / irregularity (rubber stamp imperfection)
    noise = np.random.normal(0, 15, (size, size)).astype(np.int16)
    for c in range(3):
        channel = stamp[:, :, c].astype(np.int16) + noise
        stamp[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
    # Mask out outside circle
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, center, radius + 2, 255, -1)
    stamp[:, :, 3] = cv2.bitwise_and(stamp[:, :, 3], mask)
    
    return stamp

def create_signature(width=320, height=140):
    """Generates a realistic cursive handwritten blue pen signature."""
    sig = np.zeros((height, width, 4), dtype=np.uint8)
    blue_color = (200, 80, 20, 240)  # BGR + Alpha (blue ink)
    
    # Draw simulated cursive signature curves using polylines
    points = []
    # Starting stroke
    t = np.linspace(0, 4 * np.pi, 120)
    x = np.linspace(20, width - 30, 120)
    y = height // 2 + 25 * np.sin(t) - 15 * np.cos(2 * t) + np.sin(5 * t) * 8
    
    # Smooth points
    pts = np.vstack((x, y)).T.astype(np.int32)
    cv2.polylines(sig, [pts], False, blue_color, 3, cv2.LINE_AA)
    
    # Add capital starting flourish
    cv2.ellipse(sig, (55, 60), (35, 45), -20, 0, 360, blue_color, 3, cv2.LINE_AA)
    
    # Add underline flourish
    under_pts = np.array([[30, height - 30], [width // 2, height - 25], [width - 20, height - 35]], dtype=np.int32)
    cv2.polylines(sig, [under_pts], False, blue_color, 2, cv2.LINE_AA)
    
    # Add two small dots
    cv2.circle(sig, (width - 45, height - 18), 3, blue_color, -1)
    cv2.circle(sig, (width - 30, height - 19), 3, blue_color, -1)
    
    return sig

def overlay_alpha(bg, fg, x, y):
    """Blends an RGBA image onto a BGR background at position (x, y)."""
    h, w = fg.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg
    
    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1.0 - alpha) * bg[y:y+h, x:x+w, c] + alpha * fg[:, :, c]
    return bg

def generate_academic_document(width=1000, height=1400):
    """Generates an upright canonical A4-proportion academic document."""
    # Light paper texture (off-white)
    doc = np.full((height, width, 3), 250, dtype=np.uint8)
    # Add subtle paper grain
    paper_noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
    doc = np.clip(doc.astype(np.int16) + paper_noise, 0, 255).astype(np.uint8)
    
    # Outer border
    cv2.rectangle(doc, (40, 40), (width - 40, height - 40), (80, 80, 80), 2)
    cv2.rectangle(doc, (46, 46), (width - 46, height - 46), (150, 150, 150), 1)
    
    # Document Header
    font_bold = cv2.FONT_HERSHEY_SIMPLEX
    font_regular = cv2.FONT_HERSHEY_DUPLEX
    
    cv2.putText(doc, "THE MAHARAJA SAYAJIRAO UNIVERSITY OF BARODA", (120, 120), font_bold, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(doc, "FACULTY OF TECHNOLOGY AND ENGINEERING", (180, 160), font_bold, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(doc, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", (140, 195), font_regular, 0.58, (50, 50, 50), 1, cv2.LINE_AA)
    
    cv2.line(doc, (60, 220), (width - 60, 220), (0, 0, 0), 2)
    cv2.line(doc, (60, 225), (width - 60, 225), (120, 120, 120), 1)
    
    cv2.putText(doc, "ACADEMIC PROJECT VERIFICATION CERTIFICATE", (190, 280), font_bold, 0.75, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(doc, "B.Tech. Seventh Semester Examination - Computer Vision", (200, 315), font_regular, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    
    # Body Text Paragraphs
    paragraphs = [
        "This is to certify that the project entitled 'Real-Time Document Scanner and",
        "Chromatic Verification Mark Extractor Using Planar Homography and Adaptive Image",
        "Segmentation' has been successfully submitted and verified as part of the academic",
        "curriculum requirements for the Degree of Bachelor of Technology in Computer Science.",
        "",
        "The project demonstrates advanced deterministic classical Computer Vision pipelines",
        "incorporating Gaussian and bilateral filtering, Canny edge gradient extraction,",
        "Douglas-Peucker polygon approximation, 3x3 planar homography perspective rectification,",
        "and dual-branch HSV chromatic segmentation for verification mark isolation.",
        "",
        "All experimental benchmarks confirm real-time execution exceeding 30+ frames per",
        "second on commodity CPU hardware without dependency on deep learning APIs."
    ]
    
    y_start = 390
    for line in paragraphs:
        if line:
            cv2.putText(doc, line, (80, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y_start += 36
        
    # Table of specifications
    cv2.rectangle(doc, (80, 840), (width - 80, 1060), (100, 100, 100), 2)
    cv2.line(doc, (80, 885), (width - 80, 885), (100, 100, 100), 2)
    cv2.line(doc, (450, 840), (450, 1060), (100, 100, 100), 2)
    
    cv2.putText(doc, "PIPELINE PARAMETER", (100, 870), font_bold, 0.55, (20, 20, 20), 2)
    cv2.putText(doc, "SPECIFIED SPECIFICATION", (480, 870), font_bold, 0.55, (20, 20, 20), 2)
    
    specs = [
        ("Denoising Filter", "Bilateral (d=9, sigmaColor=75, sigmaSpace=75)"),
        ("Boundary Approximation", "Douglas-Peucker (epsilon = 0.02 * perimeter)"),
        ("Homography Transformation", "Direct Linear Transform (3x3 Matrix, 8 DoF)"),
        ("Chromatic Bandpass", "HSV Hue: Blue [95-135], Red [0-10 & 165-180]")
    ]
    
    row_y = 925
    for k, v in specs:
        cv2.putText(doc, k, (95, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
        cv2.putText(doc, v, (465, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 80), 1)
        row_y += 38

    # Sign and Stamp areas
    cv2.putText(doc, "Authorized Signatory:", (100, 1260), font_bold, 0.55, (20, 20, 20), 1)
    cv2.putText(doc, "Department Stamp & Seal:", (580, 1260), font_bold, 0.55, (20, 20, 20), 1)
    
    # Overlay signature (blue) and stamp (red)
    stamp = create_official_stamp(220)
    sig = create_signature(280, 110)
    
    doc = overlay_alpha(doc, sig, 110, 1140)
    doc = overlay_alpha(doc, stamp, 600, 1100)
    
    return doc, stamp, sig

def create_perspective_scene(doc, scene_w=1280, scene_h=720):
    """Places the document onto a realistic desk scene with perspective distortion & shadows."""
    # Create dark textured wooden desk background
    desk = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)
    for y in range(scene_h):
        desk[y, :, 0] = int(35 + 15 * np.sin(y / 20.0))  # B
        desk[y, :, 1] = int(50 + 20 * np.sin(y / 20.0))  # G
        desk[y, :, 2] = int(75 + 25 * np.sin(y / 20.0))  # R
        
    desk_noise = np.random.normal(0, 10, (scene_h, scene_w, 3)).astype(np.int16)
    desk = np.clip(desk.astype(np.int16) + desk_noise, 0, 255).astype(np.uint8)
    
    # Source corners of the document (top-left, top-right, bottom-right, bottom-left)
    doc_h, doc_w = doc.shape[:2]
    src_pts = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])
    
    # Skewed perspective destination quad on desk (simulating webcam looking down at an angle)
    # TL, TR, BR, BL
    dst_pts = np.float32([
        [240, 75],    # TL
        [980, 110],   # TR
        [1060, 660],  # BR
        [160, 620]    # BL
    ])
    
    # Calculate homography matrix for warping document into scene
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc, H, (scene_w, scene_h), flags=cv2.INTER_LINEAR)
    
    # Create mask for warped document
    mask = np.zeros((doc_h, doc_w), dtype=np.uint8)
    mask[:] = 255
    warped_mask = cv2.warpPerspective(mask, H, (scene_w, scene_h), flags=cv2.INTER_NEAREST)
    
    # Add non-uniform lighting shadow (simulating desk lamp / hand shadow gradient)
    shadow = np.ones((scene_h, scene_w), dtype=np.float32)
    for x in range(scene_w):
        for y in range(scene_h):
            # Gradient shadow across the top-left
            shadow[y, x] = 0.65 + 0.35 * (x / scene_w) * (y / scene_h)
            
    shadow_3ch = np.repeat(shadow[:, :, np.newaxis], 3, axis=2)
    
    # Composite document onto desk
    mask_inv = cv2.bitwise_not(warped_mask)
    bg_part = cv2.bitwise_and(desk, desk, mask=mask_inv)
    doc_part = cv2.bitwise_and(warped_doc, warped_doc, mask=warped_mask)
    
    # Apply shadow only to document area
    doc_part_shaded = np.clip(doc_part.astype(np.float32) * shadow_3ch, 0, 255).astype(np.uint8)
    doc_part_shaded = cv2.bitwise_and(doc_part_shaded, doc_part_shaded, mask=warped_mask)
    
    scene = cv2.add(bg_part, doc_part_shaded)
    
    return scene, dst_pts

def main():
    os.makedirs("samples", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("[*] Generating synthetic academic document...")
    doc, stamp, sig = generate_academic_document()
    
    # Save canonical document
    cv2.imwrite("samples/canonical_document.png", doc)
    
    # Save reference templates for automated verification
    # Convert RGBA stamp/sig to BGR with white background for template matching
    def rgba_to_white_bgr(rgba):
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3] / 255.0
        bg = np.full_like(rgb, 255)
        out = (1.0 - alpha[:, :, np.newaxis]) * bg + alpha[:, :, np.newaxis] * rgb
        return out.astype(np.uint8)
        
    stamp_bgr = rgba_to_white_bgr(stamp)
    sig_bgr = rgba_to_white_bgr(sig)
    
    cv2.imwrite("templates/official_stamp_template.png", stamp_bgr)
    cv2.imwrite("templates/authorized_signature_template.png", sig_bgr)
    print("    -> Saved templates/official_stamp_template.png")
    print("    -> Saved templates/authorized_signature_template.png")
    
    # Generate perspective scenes (tilted webcam view)
    print("[*] Generating perspective desk scenes...")
    scene_normal, quad_pts = create_perspective_scene(doc)
    cv2.imwrite("samples/sample_webcam_scene.png", scene_normal)
    print("    -> Saved samples/sample_webcam_scene.png")
    
    # Generate second scene with severe angle
    doc_h, doc_w = doc.shape[:2]
    src_pts = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])
    dst_pts_angled = np.float32([[320, 110], [900, 80], [1020, 680], [140, 610]])
    desk = np.full((720, 1280, 3), 40, dtype=np.uint8)
    H, _ = cv2.findHomography(src_pts, dst_pts_angled)
    warped = cv2.warpPerspective(doc, H, (1280, 720))
    mask = cv2.warpPerspective(np.full((doc_h, doc_w), 255, dtype=np.uint8), H, (1280, 720))
    scene_angled = np.where(mask[:, :, np.newaxis] > 0, warped, desk)
    cv2.imwrite("samples/sample_angled_scene.png", scene_angled)
    print("    -> Saved samples/sample_angled_scene.png")
    
    # Generate third scene with straight front-facing view (top-down / bird's-eye)
    dw, dh = 460, int(460 * 1.414)
    x0, y0 = (1280 - dw) // 2, (720 - dh) // 2
    dst_pts_straight = np.float32([[x0, y0], [x0 + dw, y0], [x0 + dw, y0 + dh], [x0, y0 + dh]])
    H_straight, _ = cv2.findHomography(src_pts, dst_pts_straight)
    warped_straight = cv2.warpPerspective(doc, H_straight, (1280, 720), flags=cv2.INTER_LANCZOS4)
    mask_straight = cv2.warpPerspective(np.full((doc_h, doc_w), 255, dtype=np.uint8), H_straight, (1280, 720))
    scene_straight = np.where(mask_straight[:, :, np.newaxis] > 0, warped_straight, desk)
    cv2.imwrite("samples/sample_straight_scene.png", scene_straight)
    print("    -> Saved samples/sample_straight_scene.png")
    
    print("[+] Synthetic samples and verification templates successfully generated!")

if __name__ == "__main__":
    main()
