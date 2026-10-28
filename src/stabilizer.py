"""
Temporal Corner Stabilizer Module
Implements motion-gated Exponential Moving Average (EMA) and temporal persistence
to eliminate corner jitter on stationary documents and survive transient frame drops.
"""

import numpy as np

class CornerStabilizer:
    def __init__(self, alpha_smooth=0.25, jitter_threshold=12.0, max_persistence_frames=6):
        """
        Args:
            alpha_smooth: Weight of new frame in EMA when moving (0.0 to 1.0).
            jitter_threshold: Distance in pixels below which movement is treated as micro-jitter.
            max_persistence_frames: Number of frames to hold last known position on dropped detection.
        """
        self.alpha_smooth = alpha_smooth
        self.jitter_threshold = jitter_threshold
        self.max_persistence_frames = max_persistence_frames
        
        self.current_corners = None
        self.dropped_frame_count = 0
        self.is_locked = False
        
    def _align_corners_to_reference(self, new_pts, ref_pts):
        """
        Finds the cyclic shift or permutation of new_pts that minimizes
        the sum of squared distances to ref_pts, ensuring stable point-to-point correspondence.
        """
        best_pts = new_pts
        min_dist = float('inf')
        
        # Test 4 possible cyclic rotations of the 4 points
        for shift in range(4):
            rolled = np.roll(new_pts, shift, axis=0)
            dist = np.sum((rolled - ref_pts) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_pts = rolled
        return best_pts

    def update(self, detected_corners):
        """
        Updates the stabilizer with newly detected corner coordinates.
        Returns stabilized corners (4, 2) or None if no document is tracked.
        """
        if detected_corners is None:
            self.dropped_frame_count += 1
            if self.dropped_frame_count <= self.max_persistence_frames and self.current_corners is not None:
                # Retain previous position through brief blinks/occlusions
                return self.current_corners.copy()
            else:
                self.current_corners = None
                self.is_locked = False
                return None
                
        # Detection succeeded; reset dropped count
        self.dropped_frame_count = 0
        detected_corners = np.array(detected_corners, dtype=np.float32)
        
        if self.current_corners is None:
            # First lock
            self.current_corners = detected_corners.copy()
            self.is_locked = True
            return self.current_corners.copy()
            
        # Align new points with existing tracking reference
        aligned_new = self._align_corners_to_reference(detected_corners, self.current_corners)
        
        # Calculate maximum per-corner displacement
        corner_shifts = np.linalg.norm(aligned_new - self.current_corners, axis=1)
        max_shift = np.max(corner_shifts)
        
        if max_shift < self.jitter_threshold:
            # Stationary document: Heavy temporal damping to kill pixel jitter
            weight_new = 0.08
            self.current_corners = (1.0 - weight_new) * self.current_corners + weight_new * aligned_new
        elif max_shift > 150.0:
            # Large abrupt change / new document placement: Snap immediately
            self.current_corners = aligned_new.copy()
        else:
            # Active movement: Smooth tracking without lag
            weight_new = self.alpha_smooth
            self.current_corners = (1.0 - weight_new) * self.current_corners + weight_new * aligned_new
            
        return self.current_corners.copy()

    def reset(self):
        """Clears tracking history."""
        self.current_corners = None
        self.dropped_frame_count = 0
        self.is_locked = False
