"""
Real-Time Document Scanner and Chromatic Verification Mark Extractor
MSU Baroda - 7th Semester B.Tech CSE Computer Vision Project

Dual-Mode Operation:
1. Live Camera Stream Mode (Webcam, DroidCam/Iriun phone camera, or IP stream at 30+ FPS)
2. Uploaded Phone Image Mode (Opens any pre-taken smartphone JPG/PNG from WhatsApp/Gallery)
"""

import sys
import os
import glob
import argparse
import cv2
import numpy as np

# Native file dialog for picking uploaded images with one click
import tkinter as tk
from tkinter import filedialog

from src.camera import Camera
from src.pipeline import DocumentScannerPipeline
from src.dashboard import render_multi_viewport_hud
from src.exporter import save_capture

def on_trackbar_change(val):
    """Callback for trackbar events."""
    pass

def setup_calibration_window():
    """Creates an interactive GUI panel for live parameter tuning during professor demonstrations."""
    win_name = "Interactive CV Calibrator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 450, 280)
    
    cv2.createTrackbar("Bilateral Filter", win_name, 1, 1, on_trackbar_change)
    cv2.createTrackbar("Blue Hue Min", win_name, 95, 180, on_trackbar_change)
    cv2.createTrackbar("Blue Hue Max", win_name, 138, 180, on_trackbar_change)
    cv2.createTrackbar("Red Hue Wrap", win_name, 10, 30, on_trackbar_change)
    cv2.createTrackbar("Min Ink Sat", win_name, 25, 255, on_trackbar_change)
    cv2.createTrackbar("Adaptive C", win_name, 10, 25, on_trackbar_change)
    return win_name

def select_image_file():
    """Opens a native Windows File Dialog to select any phone photo or WhatsApp image."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Phone Document Image (JPG / PNG)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[!] File dialog error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="MSU Real-Time Document Scanner & Mark Extractor")
    parser.add_argument("--image", default=None, help="Path to an uploaded mobile phone document photo (JPG/PNG)")
    parser.add_argument("--source", default="0", help="Camera index (0, 1, 2), IP URL, or image file path")
    args = parser.parse_args()

    print("=" * 70)
    print("  REAL-TIME DOCUMENT SCANNER & CHROMATIC VERIFICATION MARK EXTRACTOR")
    print("  The Maharaja Sayajirao University of Baroda - Department of CSE")
    print("=" * 70)
    
    pipeline = DocumentScannerPipeline(templates_dir="templates")
    os.makedirs("uploads", exist_ok=True)
    
    # Determine initial operating mode: Image File vs. Camera Stream
    target_image_path = None
    if args.image and os.path.isfile(args.image):
        target_image_path = args.image
    elif os.path.isfile(args.source):
        target_image_path = args.source
        
    camera = None
    using_camera = False
    current_image = None
    mode_name = "CAMERA"
    scan_mode = 0  # 0: Magic Color, 1: Clean Gray, 2: Photocopy B&W
    
    if target_image_path is not None:
        current_image = cv2.imread(target_image_path)
        if current_image is not None:
            print(f"[+] Loaded Uploaded Phone Image: {target_image_path} (Shape: {current_image.shape})")
            mode_name = "UPLOADED_IMAGE"
        else:
            print(f"[!] Failed to decode image: {target_image_path}")
            
    if current_image is None:
        # Camera Mode
        initial_source = int(args.source) if args.source.isdigit() else args.source
        camera = Camera(initial_source)
        test_frame = camera.read_frame()
        
        # If specified camera index returned no frame, auto-scan for working cameras
        if test_frame is None and isinstance(initial_source, int):
            from src.camera import find_available_cameras
            avail = find_available_cameras()
            if avail:
                camera = Camera(avail[0])
                test_frame = camera.read_frame()
                print(f"[+] Auto-detected active hardware camera at index {avail[0]}.")
        
        sample_paths = [
            "samples/sample_webcam_scene.png",
            "samples/sample_angled_scene.png",
            "samples/canonical_document.png"
        ]
        sample_images = [cv2.imread(p) for p in sample_paths if os.path.exists(p)]
        
        if test_frame is not None:
            using_camera = True
            print(f"[+] Live Camera source active.")
        else:
            print(f"[!] Camera source unavailable. Launching in Offline Demo Mode.")
            current_image = sample_images[0]
            mode_name = "SAMPLE_DEMO"
            
    hud_window = "MSU Real-Time Document Scanner - Multi-Viewport Diagnostic HUD"
    cv2.namedWindow(hud_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hud_window, 1200, 820)
    
    calib_window = None
    show_calibration = False
    is_frozen = False
    frozen_state = None
    show_verification_plot = False
    
    print("\n[CONTROLS]")
    print("  [O]           : Open Image File (Native Windows File Picker for Phone/WhatsApp Photos)")
    print("  [E]           : Toggle Viewport 3 Mode (Clean Razor-Sharp Scan vs. High-Contrast B&W)")
    print("  [U]           : Load Next Image from uploads/ folder")
    print("  [SPACE] / [C] : Freeze / Unfreeze Capture")
    print("  [S]           : Save Rectified Scans & Verification Artifacts to output/")
    print("  [T]           : Toggle Interactive Calibration Trackbars")
    print("  [V]           : Toggle Detailed Stamp Feature Matching Plot")
    print("  [A]           : Enroll / Register Current Physical Marks as Authorized Templates")
    print("  [D]           : Restore Default MSU Reference Templates")
    print("  [0] / [1] / [2]: Switch Scenes (0: Straight Desk, 1: Angled Desk, 2: Severe Perspective)")
    print("  [9] / [K]     : Load Native Canonical High-Res Document (samples/canonical_document.png)")
    print("  [P]           : Connect Phone Camera over Wi-Fi (Enter DroidCam IP)")
    print("  [3]           : Switch to Built-in Laptop Webcam")
    print("  [4]           : Switch to External USB / Virtual Camera")
    print("  [R]           : Reset Stabilizer Tracking History")
    print("  [Q] / [ESC]   : Exit Application\n")
    
    while True:
        # 1. Acquire Frame
        if not is_frozen:
            if using_camera and camera is not None:
                frame = camera.read_frame()
                if frame is None:
                    print("[!] Camera read error. Falling back to uploaded/sample image.")
                    using_camera = False
                    frame = current_image if current_image is not None else cv2.imread("samples/sample_webcam_scene.png")
            else:
                frame = current_image.copy()
                
            # 2. Execute Computer Vision Pipeline (High-res 1200px for uploaded photos, 35+ FPS for video)
            is_static = (mode_name == "UPLOADED_IMAGE" or not using_camera)
            current_state = pipeline.process_frame(frame, run_verification=True, is_static_image=is_static)
            if mode_name == "UPLOADED_IMAGE":
                current_state['status'] = f"UPLOADED: {current_state['status']}"
            frozen_state = current_state
        else:
            current_state = frozen_state
            
        # 3. Render 4-Viewport Diagnostic HUD
        hud_display = render_multi_viewport_hud(
            current_state,
            target_size=(1160, 780),
            scan_mode=scan_mode,
            templates_dir="templates"
        )
        
        # Overlay banner if Frozen
        if is_frozen:
            cv2.rectangle(hud_display, (420, 15), (740, 55), (0, 0, 180), -1)
            cv2.putText(hud_display, "[CAPTURE FROZEN - PRESS 'S' TO SAVE]", (435, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                        
        cv2.imshow(hud_window, hud_display)
        
        # 4. Show Verification Match Plot if requested
        if show_verification_plot:
            v_data = current_state.get('verification')
            if v_data and v_data.get('stamp') and v_data['stamp'].get('vis') is not None:
                cv2.imshow("ORB Stamp Feature Matching Analysis", v_data['stamp']['vis'])
            else:
                blank = np.zeros((300, 500, 3), dtype=np.uint8)
                cv2.putText(blank, "No Stamp Extracted For Matching", (60, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                cv2.imshow("ORB Stamp Feature Matching Analysis", blank)
                
        # 5. Handle Keyboard Controls
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            break
            
        elif key == ord('o'):  # Open Image File via Windows Dialog
            chosen_file = select_image_file()
            if chosen_file:
                loaded = cv2.imread(chosen_file)
                if loaded is not None:
                    current_image = loaded
                    using_camera = False
                    is_frozen = False
                    mode_name = "UPLOADED_IMAGE"
                    pipeline.reset_stabilizer()
                    print(f"\n[+] Successfully opened phone photo: {chosen_file}")
                    print("    -> Detecting paper boundaries, unwarping perspective, cleaning shadows, and verifying seal...")
                else:
                    print(f"[!] Could not decode selected file: {chosen_file}")
                    
        elif key == ord('e'):  # Cycle Viewport 3 scan mode
            scan_mode = (scan_mode + 1) % 3
            modes_labels = ["Magic Color (Shadow-Free Full Color)", "Clean Grayscale (Unsharp Edge Sharpened)", "Photocopy B&W Binary"]
            print(f"[*] Viewport 3 Mode: {modes_labels[scan_mode]}")
            
        elif key == ord('u'):  # Load Next Image from uploads/ directory
            upload_files = glob.glob("uploads/*.jpg") + glob.glob("uploads/*.png") + glob.glob("uploads/*.jpeg")
            if upload_files:
                chosen = upload_files[0]
                current_image = cv2.imread(chosen)
                using_camera = False
                is_frozen = False
                mode_name = "UPLOADED_IMAGE"
                pipeline.reset_stabilizer()
                print(f"[+] Loaded from uploads/: {chosen}")
            else:
                print("[!] No images found in uploads/ folder. Place your phone photos in the uploads/ directory.")
                
        elif key == ord(' ') or key == ord('c'):  # Freeze / Capture
            is_frozen = not is_frozen
            print(f"[*] Capture state: {'FROZEN' if is_frozen else 'LIVE TRACKING'}")
            
        elif key == ord('s'):  # Save Outputs
            if frozen_state is not None:
                target_dir, files = save_capture(frozen_state)
                print(f"[+] Successfully exported {len(files)} scan artifacts to: {target_dir}")
                cv2.putText(hud_display, "SAVED TO OUTPUT!", (50, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(hud_window, hud_display)
                cv2.waitKey(500)
                
        elif key == ord('t'):  # Toggle Calibration Window
            show_calibration = not show_calibration
            if show_calibration:
                calib_window = setup_calibration_window()
            else:
                if calib_window:
                    cv2.destroyWindow(calib_window)
                    calib_window = None
                    
        elif key == ord('v'):  # Toggle Detailed ORB Plot
            show_verification_plot = not show_verification_plot
            if not show_verification_plot:
                cv2.destroyWindow("ORB Stamp Feature Matching Analysis")
                
        elif key == ord('a'):  # Enroll Current Extracted Marks as Reference Templates
            state_to_use = frozen_state if is_frozen else current_state
            if state_to_use and state_to_use.get('color_data'):
                cd = state_to_use['color_data']
                stamp_crop = None
                sig_crop = None
                if cd.get('red_boxes') and cd.get('red_marks') is not None:
                    largest_red = max(cd['red_boxes'], key=lambda b: b[4])
                    x, y, w, h, _ = largest_red
                    pad = 10
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(cd['red_marks'].shape[1], x + w + pad), min(cd['red_marks'].shape[0], y + h + pad)
                    stamp_crop = cd['red_marks'][y1:y2, x1:x2]
                if cd.get('blue_boxes') and cd.get('blue_marks') is not None:
                    largest_blue = max(cd['blue_boxes'], key=lambda b: b[4])
                    x, y, w, h, _ = largest_blue
                    pad = 10
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(cd['blue_marks'].shape[1], x + w + pad), min(cd['blue_marks'].shape[0], y + h + pad)
                    sig_crop = cd['blue_marks'][y1:y2, x1:x2]
                    
                if stamp_crop is not None or sig_crop is not None:
                    pipeline.verifier.register_template_from_crop(stamp_crop, sig_crop)
                    print("[+] ENROLLED: Current physical marks registered as Authorized Reference Specimens!")
                else:
                    print("[!] No blue signature or red stamp detected in current view to enroll.")
            else:
                print("[!] No rectified document available to enroll templates.")

        elif key == ord('d'):  # Restore Default MSU Reference Templates
            pipeline.verifier.restore_default_templates()
            print("[*] Restored default MSU Department reference templates.")
                
        elif key == ord('0'):  # Sample 0 (Straight Front-Facing Document)
            using_camera = False
            current_image = cv2.imread("samples/sample_straight_scene.png")
            is_frozen = False
            mode_name = "SAMPLE_DEMO"
            pipeline.reset_stabilizer()
            print("[*] Switched to Sample Scene 0 (Straight / Front-Facing Bird's-Eye View)")
            
        elif key == ord('1'):  # Sample 1
            using_camera = False
            current_image = cv2.imread("samples/sample_webcam_scene.png")
            is_frozen = False
            mode_name = "SAMPLE_DEMO"
            pipeline.reset_stabilizer()
            print("[*] Switched to Sample Scene 1 (Normal Perspective)")
            
        elif key == ord('2'):  # Sample 2
            using_camera = False
            current_image = cv2.imread("samples/sample_angled_scene.png")
            is_frozen = False
            mode_name = "SAMPLE_DEMO"
            pipeline.reset_stabilizer()
            print("[*] Switched to Sample Scene 2 (Severe Angled Perspective)")
            
        elif key == ord('9') or key == ord('k'):  # Canonical High-Res Document
            using_camera = False
            current_image = cv2.imread("samples/canonical_document.png")
            is_frozen = False
            mode_name = "UPLOADED_IMAGE"
            pipeline.reset_stabilizer()
            print("[*] Switched to Canonical High-Res Academic Document (samples/canonical_document.png)")
            
        elif key == ord('3'):  # Auto-Detect / Switch to Laptop Webcam
            from src.camera import find_available_cameras
            avail = find_available_cameras()
            if avail:
                cam_idx = avail[0]
                if camera is None:
                    camera = Camera(cam_idx)
                else:
                    camera.open_camera(cam_idx)
                using_camera = True
                is_frozen = False
                mode_name = "CAMERA"
                pipeline.reset_stabilizer()
                print(f"[*] Switched to Laptop Webcam (Camera Index {cam_idx})")
            else:
                print("[!] No active hardware webcam found.")
                
        elif key == ord('4'):  # Switch to External / Phone Camera (DroidCam / Iriun)
            from src.camera import find_available_cameras
            avail = find_available_cameras()
            cam_idx = avail[1] if len(avail) > 1 else (avail[0] if avail else 1)
            if camera is None:
                camera = Camera(cam_idx)
            else:
                camera.open_camera(cam_idx)
            using_camera = True
            is_frozen = False
            mode_name = "CAMERA"
            pipeline.reset_stabilizer()
            print(f"[*] Switched to Camera Index {cam_idx} (DroidCam / Phone / External)")
                
        elif key == ord('p'):  # Connect Phone Camera via DroidCam Wi-Fi IP
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            phone_ip = simpledialog.askstring("DroidCam Connection", "Enter Phone IP from DroidCam screen (e.g. 192.168.1.15):", parent=root)
            root.destroy()
            if phone_ip:
                phone_ip = phone_ip.strip()
                if not phone_ip.startswith("http"):
                    stream_url = f"http://{phone_ip}:4747/video" if ":4747" not in phone_ip else f"http://{phone_ip}/video"
                else:
                    stream_url = phone_ip
                if camera is None:
                    camera = Camera(stream_url)
                else:
                    camera.open_camera(stream_url)
                if camera.read_frame() is not None:
                    using_camera = True
                    is_frozen = False
                    mode_name = "CAMERA"
                    pipeline.reset_stabilizer()
                    print(f"[+] Successfully connected to Phone: {stream_url}")
                else:
                    print(f"[!] Could not connect to {stream_url}. Ensure phone and laptop are on same Wi-Fi.")
                    
        elif key == ord('r'):  # Reset Stabilizer
            pipeline.reset_stabilizer()
            print("[*] Stabilizer history reset.")
            
    # Cleanup
    if using_camera and camera is not None:
        camera.release()
    cv2.destroyAllWindows()
    print("[+] Application terminated cleanly.")

if __name__ == "__main__":
    main()
