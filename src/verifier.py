"""
Automated Verification Engine (Zero-Cost Classical Pattern Matcher)
Performs automated verification of extracted stamps and signatures against
authorized reference templates using ORB feature keypoint matching and
multi-scale Normalized Cross-Correlation (NCC).
"""

import os
import cv2
import numpy as np

class VerificationEngine:
    def __init__(self, templates_dir="templates"):
        self.templates_dir = templates_dir
        self.orb = cv2.ORB_create(nfeatures=500, fastThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.stamp_template = None
        self.stamp_gray = None
        self.stamp_kp = None
        self.stamp_des = None
        
        self.sig_template = None
        self.sig_gray = None
        self.sig_kp = None
        self.sig_des = None
        
        self._load_templates()

    def _load_templates(self):
        """Loads and precomputes ORB keypoint descriptors for reference templates."""
        stamp_path = os.path.join(self.templates_dir, "official_stamp_template.png")
        sig_path = os.path.join(self.templates_dir, "authorized_signature_template.png")
        
        if os.path.exists(stamp_path):
            self.stamp_template = cv2.imread(stamp_path)
            self.stamp_gray = cv2.cvtColor(self.stamp_template, cv2.COLOR_BGR2GRAY)
            self.stamp_kp, self.stamp_des = self.orb.detectAndCompute(self.stamp_gray, None)
            
        if os.path.exists(sig_path):
            self.sig_template = cv2.imread(sig_path)
            self.sig_gray = cv2.cvtColor(self.sig_template, cv2.COLOR_BGR2GRAY)
            self.sig_kp, self.sig_des = self.orb.detectAndCompute(self.sig_gray, None)

    def register_template_from_crop(self, stamp_crop=None, sig_crop=None):
        """
        Dynamically enrolls a new stamp/signature as the reference template
        directly from live physical document crops.
        """
        if stamp_crop is not None:
            stamp_path = os.path.join(self.templates_dir, "official_stamp_template.png")
            cv2.imwrite(stamp_path, stamp_crop)
        if sig_crop is not None:
            sig_path = os.path.join(self.templates_dir, "authorized_signature_template.png")
            cv2.imwrite(sig_path, sig_crop)
        self._load_templates()

    def restore_default_templates(self):
        """Restores the default university specimen templates from default_msu/."""
        import shutil
        default_dir = os.path.join(self.templates_dir, "default_msu")
        if os.path.exists(default_dir):
            for fname in ["official_stamp_template.png", "authorized_signature_template.png"]:
                src_f = os.path.join(default_dir, fname)
                dst_f = os.path.join(self.templates_dir, fname)
                if os.path.exists(src_f):
                    shutil.copy2(src_f, dst_f)
        self._load_templates()

    def verify_stamp(self, extracted_stamp_bgr):
        """
        Verifies extracted red stamp against reference template:
        1. Computes ORB keypoints and descriptors.
        2. Matches with Brute-Force Matcher.
        3. Computes normalized cross-correlation peak score.
        4. Returns confidence percentage, decision, and match visualization image.
        """
        if self.stamp_des is None or extracted_stamp_bgr is None:
            return {'verified': False, 'score': 0.0, 'good_matches': 0, 'ncc_score': 0.0, 'vis': None}
            
        stamp_gray = cv2.cvtColor(extracted_stamp_bgr, cv2.COLOR_BGR2GRAY)
        
        # Scale-normalization to reference template canonical dimensions for scale-invariance
        h_t, w_t = self.stamp_gray.shape[:2]
        stamp_normalized = cv2.resize(stamp_gray, (w_t, h_t), interpolation=cv2.INTER_CUBIC)
        
        kp, des = self.orb.detectAndCompute(stamp_normalized, None)
        
        if des is None or len(des) < 5:
            return {'verified': False, 'score': 0.0, 'good_matches': 0, 'ncc_score': 0.0, 'vis': None}
            
        # 1. Feature matching via ORB
        matches = self.matcher.match(self.stamp_des, des)
        # Sort matches by Hamming distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Good matches have low Hamming distance (< 52)
        good_matches = [m for m in matches if m.distance < 52]
        
        # 2. Template cross-correlation score (using canonical normalized size)
        res = cv2.matchTemplate(stamp_normalized, self.stamp_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        ncc_score = max(0.0, float(max_val))
        
        # Combined confidence metric
        match_score = min(1.0, len(good_matches) / 22.0)
        confidence = float(np.clip((match_score * 0.70 + ncc_score * 0.30) * 100.0, 0.0, 99.9))
        is_verified = bool(len(good_matches) >= 10 or confidence >= 50.0)
        
        # 3. Create visual matching plot
        vis = cv2.drawMatches(
            self.stamp_template, self.stamp_kp,
            extracted_stamp_bgr, kp,
            good_matches[:25], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return {
            'verified': is_verified,
            'score': round(confidence, 1),
            'good_matches': len(good_matches),
            'ncc_score': round(ncc_score, 2),
            'vis': vis
        }

    def verify_signature(self, extracted_sig_bgr):
        """
        Verifies extracted blue signature against authorized reference template:
        1. Computes ORB keypoints and descriptors on scale-normalized signature.
        2. Matches with Brute-Force Hamming distance matcher.
        3. Computes normalized cross-correlation peak score.
        4. Returns confidence percentage, decision, and match visualization plot.
        """
        if self.sig_des is None or extracted_sig_bgr is None:
            return {'verified': False, 'score': 0.0, 'good_matches': 0, 'ncc_score': 0.0, 'vis': None}
            
        sig_gray = cv2.cvtColor(extracted_sig_bgr, cv2.COLOR_BGR2GRAY) if len(extracted_sig_bgr.shape) == 3 else extracted_sig_bgr
        
        # Scale-normalization to reference template canonical dimensions
        h_t, w_t = self.sig_gray.shape[:2]
        sig_normalized = cv2.resize(sig_gray, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        
        kp, des = self.orb.detectAndCompute(sig_normalized, None)
        if des is None or len(des) < 5:
            return {'verified': False, 'score': 0.0, 'good_matches': 0, 'ncc_score': 0.0, 'vis': None}
            
        # 1. Feature matching via ORB
        matches = self.matcher.match(self.sig_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 55]
        
        # 2. Template cross-correlation score
        res = cv2.matchTemplate(sig_normalized, self.sig_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        ncc_score = max(0.0, float(max_val))
        
        # Combined confidence metric
        match_score = min(1.0, len(good_matches) / 18.0)
        confidence = float(np.clip((match_score * 0.70 + ncc_score * 0.30) * 100.0, 0.0, 99.9))
        is_verified = bool(len(good_matches) >= 8 or confidence >= 45.0)
        
        # 3. Create visual matching correspondence plot
        vis = cv2.drawMatches(
            self.sig_template, self.sig_kp,
            cv2.cvtColor(sig_normalized, cv2.COLOR_GRAY2BGR), kp,
            good_matches[:25], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return {
            'verified': is_verified,
            'score': round(confidence, 1),
            'good_matches': len(good_matches),
            'ncc_score': round(ncc_score, 2),
            'vis': vis
        }

    def verify_marks(self, color_data):
        """
        Runs automated verification across detected red stamp and blue signature regions.
        """
        results = {
            'stamp': {'verified': False, 'score': 0.0, 'status': 'No Stamp Found', 'vis': None},
            'signature': {'verified': False, 'score': 0.0, 'status': 'No Signature Found', 'vis': None}
        }
        
        # 1. Process red stamp if detected
        if color_data.get('red_boxes') and color_data.get('red_marks') is not None:
            largest_red = max(color_data['red_boxes'], key=lambda b: b[4])
            x, y, w, h, _ = largest_red
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(color_data['red_marks'].shape[1], x + w + pad)
            y2 = min(color_data['red_marks'].shape[0], y + h + pad)
            
            crop = color_data['red_marks'][y1:y2, x1:x2]
            stamp_res = self.verify_stamp(crop)
            stamp_res['status'] = "AUTHENTICATED" if stamp_res['verified'] else "UNVERIFIED"
            results['stamp'] = stamp_res
            
        # 2. Process blue signature if detected
        if color_data.get('blue_boxes') and color_data.get('blue_marks') is not None:
            largest_blue = max(color_data['blue_boxes'], key=lambda b: b[4])
            x, y, w, h, _ = largest_blue
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(color_data['blue_marks'].shape[1], x + w + pad)
            y2 = min(color_data['blue_marks'].shape[0], y + h + pad)
            
            crop_sig = color_data['blue_marks'][y1:y2, x1:x2]
            sig_res = self.verify_signature(crop_sig)
            sig_res['status'] = "AUTHENTICATED" if sig_res['verified'] else "UNVERIFIED"
            results['signature'] = sig_res
            
        return results
