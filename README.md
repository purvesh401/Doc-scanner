# Real-Time Document Scanner and Chromatic Verification Mark Extractor

### Academic Computer Vision Project
* **Subject:** Computer Vision (7th Semester B.Tech Computer Science & Engineering)  
* **Institution:** The Maharaja Sayajirao University of Baroda (MSU), Faculty of Technology & Engineering  
* **Technology Stack:** Python 3.10+, OpenCV 5.0+, NumPy 2.0+, ReportLab 4.0+  
* **Execution Paradigm:** 100% Deterministic Classical Computer Vision (Zero Cloud / Zero Deep Learning / Zero GPU Requirements)  
* **Live Performance:** **37.4 – 58.6 sustained FPS** on commodity CPU hardware.

---

## 1. Abstract & System Overview

This project presents an efficient, real-time Computer Vision document digitization and forensic verification pipeline executing on live camera video streams and uploaded smartphone photos at **35+ FPS**. The system addresses the real-world challenges of 3D camera tilt, hand shadows, sensor grain, and automated forensic authentication of university ink seals and blue signatures.

The pipeline processes input through seven deterministic stages:
1. **Spatial Normalization & Denoising:** Dual-stage Gaussian and **Bilateral filtering** ($d=9, \sigma_c=75, \sigma_s=75$) to suppress camera sensor noise while maintaining sharp boundary step-edge gradients.
2. **Hybrid Boundary & Quad Extraction:** Auto-threshold Canny gradient tracking combined with **Convex Hull occlusion recovery** (handling finger/hand grips) and multi-scale **Douglas-Peucker polygon approximation** (`cv2.approxPolyDP`) to continuously detect the four document corners.
3. **Temporal Corner Stabilization:** Motion-gated Exponential Moving Average (EMA) corner stabilizer that freezes sensor jitter on stationary documents ($\alpha=0.15$) and smoothly tracks moving documents ($\alpha=0.65$).
4. **2D Planar Homography Rectification:** Geometric polar-angle corner sorting (`TL`, `TR`, `BR`, `BL`) and computation of the $3 \times 3$ projective homography matrix ($H \in \mathbb{R}^{3 \times 3}$) via Direct Linear Transform (DLT) to warp skewed quadrilaterals into upright Euclidean planes with canonical A4 aspect ratio ($1:\sqrt{2}$) restoration and **Lanczos-4 8-tap sinc interpolation**.
5. **Adaptive Spatial Enhancement (Branch A):** Background illumination normalization (morphological rectangular dilation $31 \times 31$ division) combined with CLAHE local contrast and unsharp mask edge peaking ($1.7 \times I - 0.7 \times \text{Gauss}$), completely eliminating hand shadows, creases, and lighting gradients without eroding fine font strokes.
6. **Chromatic Mark Segmentation (Branch B):** HSV cylindrical color space bandpass filtering to isolate handwritten blue pen ink ($H \in [95, 138]$) and official red validation stamps ($H \in [0, 12] \cup [165, 180]$) from printed black toner.
7. **Automated Forensic Verification Engine:** Automated pattern matching of extracted red stamps and blue signatures against reference authorized templates using **ORB (Oriented FAST and Rotated BRIEF)** feature descriptors and Normalized Cross-Correlation (NCC), outputting an authenticity decision and confidence percentage.

---

## 2. Multi-Viewport Diagnostic HUD

The system displays processing states through an interactive, multi-viewport diagnostic HUD composed of four synchronized real-time viewports:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ VIEWPORT 1: LIVE BOUNDARY TRACKER    │ VIEWPORT 2: CANNY GRADIENT EDGE MAP  │
│ - Colored quad overlay (TL/TR/BR/BL) │ - High-frequency structural gradients│
│ - EMA temporal jitter damping        │ - Real-time edge continuity display  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ VIEWPORT 3: SHADOW-FREE CLEAN SCAN   │ VIEWPORT 4: FORENSIC VERIFICATION    │
│ - Magic Color / Clean Gray / B&W     │ - Official Stamp match lines (ORB)   │
│ - Pure white background (255)        │ - Signature template side-by-side    │
├──────────────────────────────────────┴──────────────────────────────────────┤
│ BOTTOM TELEMETRY BAR: Status | Live FPS (37+ FPS) | Latency (ms) | Controls │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quickstart & Setup for Teammates (Git Clone)

### Step 1: Clone Repository & Install Dependencies
Only standard Python packages are required:
```powershell
git clone https://github.com/purvesh401/Doc-scanner.git
cd Doc-scanner
pip install opencv-python numpy reportlab
```

### Step 2: Run Verification Test Suite (Validates Pipeline & Camera Hardware)
Before running the interactive application, verify that all 7 classical CV stages pass:
```powershell
python tests/test_cv_pipeline.py
```
*Expected Output:* `ALL 7 TESTS PASSED SUCCESSFULLY! 100% SPEC COMPLIANCE (Sustained FPS >= 35.0 FPS)`.

### Step 3: Launch the Application
Choose your preferred operating mode:

```powershell
# Mode A: Standard Launch (Starts in Demo Mode or Live Camera)
python main.py

# Mode B: Direct Phone Camera Stream (via DroidCam Wi-Fi IP)
python main.py --source http://<PHONE-IP>:4747/video

# Mode C: Direct Phone Photo Launch (WhatsApp / Gallery Image)
python main.py --image uploads/test_phone_document.jpg
```

---

## 4. Interactive Keyboard Controls

| Key | Action | What It Does |
| :---: | :--- | :--- |
| **`[P]`** | **Connect Phone Camera** | Prompts for DroidCam IP (e.g. `192.168.1.15`) and connects wirelessly |
| **`[3]`** | **Laptop Built-in Webcam** | Auto-detects and switches to built-in laptop camera |
| **`[4]`** | **External / Virtual Camera** | Switches to external USB or virtual camera safely |
| **`[0]`** | **Straight Top-Down Scene** | Simulates phone pointed straight down at paper on a wooden desk |
| **`[1]`** | **Normal Angled Scene** | Handheld perspective capture ($15^\circ$ tilt) |
| **`[2]`** | **Severe Tilted Scene** | Extreme perspective foreshortening ($40^\circ$ tilt) |
| **`[9]`** / **`[K]`** | **Canonical Document** | Loads native $1400 \times 1000$ academic certificate with pin-sharp text |
| **`[E]`** | **Cycle Scan Modes** | Toggles: Magic Color (vivid ink) $\rightarrow$ Clean Gray $\rightarrow$ Photocopy B&W |
| **`[O]`** | **Open File Dialog** | Opens native Windows file picker to load any phone/WhatsApp photo |
| **`[U]`** | **Cycle Uploads** | Batch cycles through images in `uploads/` folder |
| **`[SPACE]`** / **`[C]`** | **Freeze / Unfreeze** | Freezes the frame for detailed close-up examination |
| **`[S]`** | **Save High-Res Scan** | Exports print-ready $1400 \times 1979$ artifacts + `metadata.json` to `output/` |
| **`[T]`** | **Calibration Trackbars** | Interactive sliders for real-time Canny and HSV tuning |
| **`[V]`** | **Feature Match Plot** | Dedicated window showing ORB keypoint correspondence lines |
| **`[A]`** | **Enroll Live Mark** | Enrolls current document's blue signature & red stamp as authorized specimen |
| **`[D]`** | **Restore Defaults** | Restores default MSU Department reference templates |
| **`[R]`** | **Reset Stabilizer** | Clears EMA tracking memory for instant re-detection |
| **`[Q]`** / **`[ESC]`** | **Quit Application** | Gracefully releases camera and closes all windows |

---

## 5. Connecting Your Android Phone Camera (DroidCam)

You **do not** need to hold physical documents up to your laptop webcam. Pointing your Android phone at a document on your desk is the cleanest way to demonstrate:

1. Install **DroidCam** from Google Play Store on your phone.
2. Connect your phone and laptop to the **same Wi-Fi** (or phone mobile hotspot).
3. Open DroidCam on your phone. Look at the screen:
   ```text
   WiFi IP: 192.168.X.Y       (Example: 192.168.1.15)
   DroidCam Port: 4747
   ```
4. Run `python main.py`, press **`[P]`**, type the IP address, and hit Enter. The phone camera activates immediately!

---

## 6. Exported Forensic Artifacts Schema

When pressing **`[S]`**, the system exports 8 high-resolution image artifacts and structured telemetry to `output/scan_YYYYMMDD_HHMMSS/`:

* `clean_magic_color_scan.png`: CamScanner-style full-color scan ($1400 \times 1979$, pure white paper, vivid ink).
* `clean_shadowfree_gray.png`: Shadow-free grayscale scan with unsharp-masked text edges.
* `high_contrast_photocopy_bw.png`: Clean binarized photocopy with solid character strokes.
* `rectified_color.png`: Unwarped perspective color document.
* `stamp_match_analysis.png`: ORB keypoint correspondence lines against official university seal template.
* `signature_match_analysis.png`: ORB keypoint correspondence lines against authorized blue signature template.
* `blue_marks_isolated.png` & `red_marks_isolated.png`: Isolated ink strokes on pure white background.
* `composite_verification.png`: Document with color-coded bounding boxes.
* `metadata.json`: Complete forensic JSON record containing the $3 \times 3$ homography matrix, detected quad corner coordinates, verification scores, and timestamps.

---

## 7. Automated Test Suite Benchmark Results

```text
=================================================================
  COMPUTER VISION PROJECT VERIFICATION & BENCHMARK SUITE
  MSU Baroda - 7th Sem B.Tech CSE
=================================================================
[TEST 1/7] Testing Preprocessing (Bilateral + Gaussian + Canny)...
    -> Passed! Bilateral filtering and Canny edge map generated successfully.
[TEST 2/7] Testing Boundary Detection & Douglas-Peucker Polygon Approximation...
    -> Passed! Successfully detected 4 document corners.
[TEST 3/7] Testing Corner Stabilizer (EMA motion gating)...
    -> Passed! Jitter damped from 2.0px to 0.12px.
[TEST 4/7] Testing 2D Planar Homography Perspective Rectification...
    -> Passed! Rectified document shape: (1272, 900, 3). Homography matrix computed.
[TEST 5/7] Testing Shadow Elimination and HSV Chromatic Segmentation...
    -> Passed! Found 1 blue signature(s) and 6 red stamp(s).
[TEST 6/7] Testing Automated Stamp Verification (ORB Feature Matcher)...
    -> Stamp Match Decision: AUTHENTICATED | Confidence: 78.0% | Good Matches: 173
    -> Passed! Automated verification successfully authenticated official university seal.
[TEST 7/7] Running 50-Frame Real-Time Benchmark for 30+ FPS Validation...
    ---------------------------------------------
    Average Latency : 17.06 ms
    95th Percentile : 18.48 ms
    Sustained FPS   : 58.6 FPS
    ---------------------------------------------
    -> Passed! Pipeline exceeds 30+ FPS real-time execution requirement.
=================================================================
  ALL 7 TESTS PASSED SUCCESSFULLY! 100% SPEC COMPLIANCE.
=================================================================
```

---

## 8. Repository Structure

```
Doc-scanner/
├── .gitignore                                      # Standard Git ignore file
├── CV_Project_Definition_MSU.pdf                   # Official MSU Baroda project definition
├── main.py                                         # High-performance desktop application with 4-viewport HUD
├── README.md                                       # Complete project documentation
├── output/                                         # Exported scan artifacts and metadata.json
├── samples/                                        # Canonical document and perspective test scenes
│   ├── canonical_document.png                      # Vector-grade 1400x1000 academic document
│   ├── sample_straight_scene.png                   # Top-down desk scene (Key [0])
│   ├── sample_webcam_scene.png                     # Angled desk scene (Key [1])
│   └── sample_angled_scene.png                     # Severe perspective scene (Key [2])
├── src/                                            # Production-grade classical CV modules
│   ├── camera.py                                   # Safe video capture with DroidCam IP stream support
│   ├── color_extractor.py                          # Dual-branch HSV bandpass filtering & clustering
│   ├── dashboard.py                                # Synchronized 4-viewport diagnostic HUD
│   ├── document_detector.py                        # Canny + Douglas-Peucker + Convex Hull
│   ├── enhancement.py                              # Divisive normalization + CLAHE + Unsharp mask
│   ├── exporter.py                                 # High-resolution print-ready artifact exporter
│   ├── perspective.py                              # Polar-angle corner ordering & 3x3 DLT Homography
│   ├── pipeline.py                                 # Pipeline orchestrator
│   ├── preprocessing.py                            # Bilateral filtering & Gaussian blur
│   ├── stabilizer.py                               # Motion-gated temporal EMA stabilizer
│   └── verifier.py                                 # ORB + NCC seal and signature verification
├── templates/                                      # Authorized reference templates
│   ├── authorized_signature_template.png
│   └── official_stamp_template.png
├── tests/
│   └── test_cv_pipeline.py                         # Automated 7-test verification benchmark
└── uploads/                                        # Staging folder for phone document photos
```
