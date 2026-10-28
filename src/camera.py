"""
Camera Interface Module
Supports physical webcams, virtual cameras (DroidCam, Iriun, Camo),
and mobile phone IP camera streams (e.g., http://192.168.x.x:8080/video).
"""

import cv2

def find_available_cameras(max_check=6):
    """Scans and returns all working camera indices on the system safely without crashing."""
    available = []
    for idx in range(max_check):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(idx)
                cap.release()
        except Exception:
            pass
    return available

class Camera:
    def __init__(self, camera_source=0):
        self.cap = None
        self.camera_source = camera_source
        self.open_camera(camera_source)
        
    def open_camera(self, camera_source):
        """
        Opens a video source:
        - int or digit string (e.g. 0, 1, 2): Hardware or virtual webcam (DroidCam/Iriun).
        - URL string (e.g. http://192.168.1.5:8080/video or http://192.168.1.5:4747/video).
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.camera_source = camera_source
        
        # Check if source is a network IP camera URL (DroidCam Wi-Fi stream)
        if isinstance(camera_source, str) and (camera_source.startswith("http://") or 
                                               camera_source.startswith("https://") or 
                                               camera_source.startswith("rtsp://")):
            try:
                self.cap = cv2.VideoCapture(camera_source)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"[+] Connected to Mobile Phone Stream: {camera_source}")
                        return True
            except Exception as e:
                print(f"[!] Stream error: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print(f"[!] Could not open stream: {camera_source}")
            return False
                
        # Hardware / Virtual camera index
        try:
            cam_idx = int(camera_source)
        except ValueError:
            cam_idx = 0
            
        try:
            self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return True
                self.cap.release()
                self.cap = None
        except Exception:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
        return False
        
    def read_frame(self):
        """Safely reads a frame from the capture source, guarding against driver exceptions."""
        if self.cap is None or not self.cap.isOpened():
            return None
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0 and len(frame.shape) == 3:
                return frame
        except Exception as e:
            # Silently catch driver/codec assertion errors (e.g. NVIDIA Broadcast or disconnected virtual cams)
            return None
        return None
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
