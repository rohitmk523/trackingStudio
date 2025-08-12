"""
Shot Detection Module - Basketball shot tracking and analysis

This module provides shot detection capabilities for basketball videos using
trajectory analysis and linear regression to determine made/missed shots.
Based on AI-Basketball-Shot-Detection-Tracker repository.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ShotEvent:
    """Represents a basketball shot event."""
    
    def __init__(self, timestamp: float, frame_number: int, is_made: bool, 
                 position: Optional[Tuple[int, int]] = None, confidence: float = 0.0):
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.is_made = is_made
        self.position = position
        self.confidence = confidence
        self.shot_type = "unknown"  # Will be determined later
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "is_made": self.is_made,
            "position": self.position,
            "confidence": self.confidence,
            "shot_type": self.shot_type
        }


class ShotDetector:
    """
    Basketball shot detection system using trajectory analysis.
    
    This class tracks basketball shots and determines whether they are made or missed
    by analyzing ball trajectory relative to hoop positions.
    """
    
    def __init__(self, court_boundaries=None):
        """Initialize shot detector with court boundaries."""
        self.court_boundaries = court_boundaries
        self.shot_threshold = 0.7
        
        # Shot tracking state
        self.ball_positions = []
        self.hoop_positions = []
        self.shot_attempts = 0
        self.shots_made = 0
        self.shots_missed = 0
        self.shot_events = []
        
        # Detection state
        self.in_shot_sequence = False
        self.detected_up = False
        self.detected_down = False
        self.last_ball_y = None
        
        # Fixed hoop positions (will be updated if detected)
        self.fixed_hoop_positions = self._get_fixed_hoop_positions()
        
    def _get_fixed_hoop_positions(self) -> List[Dict]:
        """Get fixed hoop positions based on court screenshots."""
        # Based on the court screenshots, approximate hoop positions
        # These are fallback positions if hoop detection fails
        return [
            {"center": (320, 180), "width": 80, "height": 40},  # Left hoop (approximate)
            {"center": (960, 180), "width": 80, "height": 40}   # Right hoop (approximate)
        ]
    
    def clean_ball_positions(self, ball_pos: List, frame_count: int) -> List:
        """
        Remove inaccurate ball tracking data points.
        
        Args:
            ball_pos: List of ball position data
            frame_count: Current frame number
            
        Returns:
            Cleaned list of ball positions
        """
        if len(ball_pos) > 1:
            # Validate ball size and movement consistency
            w1, h1 = ball_pos[-2][2]  # Size is stored as (width, height) tuple
            w2, h2 = ball_pos[-1][2]
            
            # Remove if ball size changes dramatically (likely detection error)
            if abs(w1 - w2) > 50 or abs(h1 - h2) > 50:
                ball_pos.pop()
                return ball_pos
            
            # Remove if ball jumps too far between frames (likely detection error)
            x1, y1 = ball_pos[-2][0]  # Position is stored as (x, y) tuple
            x2, y2 = ball_pos[-1][0]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance > 200:  # Threshold for maximum ball movement per frame
                ball_pos.pop()
                
        # Limit tracking history to prevent memory issues
        if len(ball_pos) > 30:
            ball_pos.pop(0)
            
        return ball_pos
    
    def score_shot(self, ball_pos: List, hoop_pos: List) -> bool:
        """
        Determine if a basketball shot will score by analyzing ball trajectory.
        
        Args:
            ball_pos: List of ball positions [(center, frame, size, confidence)]
            hoop_pos: List of hoop positions [(center, frame, size, confidence)]
            
        Returns:
            True if shot is made, False if missed
        """
        if not ball_pos or not hoop_pos:
            return False
            
        try:
            x, y = [], []
            
            # Get rim height from latest hoop detection
            hoop_center = hoop_pos[-1][0]  # (x, y) tuple
            hoop_size = hoop_pos[-1][2]    # (width, height) tuple
            rim_height = hoop_center[1] - 0.5 * hoop_size[1]
            
            # Collect trajectory points below rim height
            for i in reversed(range(len(ball_pos))):
                if ball_pos[i][0][1] < rim_height:
                    x.append(ball_pos[i][0][0])
                    y.append(ball_pos[i][0][1])
                    if i + 1 < len(ball_pos):
                        x.append(ball_pos[i + 1][0][0])
                        y.append(ball_pos[i + 1][0][1])
                    break
            
            # Perform linear regression if we have enough points
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                
                # Predict where ball trajectory crosses rim height
                predicted_x = (rim_height - b) / m
                
                # Define rim boundaries using consistent data structure
                rim_x1 = hoop_center[0] - 0.4 * hoop_size[0]
                rim_x2 = hoop_center[0] + 0.4 * hoop_size[0]
                
                # Check if trajectory intersects rim
                if rim_x1 < predicted_x < rim_x2:
                    return True
                
                # Check rebound zone (near misses that might bounce in)
                hoop_rebound_zone = 10
                if rim_x1 - hoop_rebound_zone < predicted_x < rim_x2 + hoop_rebound_zone:
                    return True
                    
        except (np.linalg.LinAlgError, ValueError, IndexError) as e:
            logger.warning(f"Error in trajectory analysis: {e}")
            
        return False
    
    def detect_up(self, ball_pos: List, hoop_pos: List) -> bool:
        """Detect if ball is in 'up' trajectory near hoop."""
        if not ball_pos or not hoop_pos:
            return False
            
        ball_center = ball_pos[-1][0]
        hoop_center = hoop_pos[-1][0]
        
        # Check if ball is in upper region near hoop
        upper_region_y = hoop_center[1] - 50
        hoop_vicinity_x = abs(ball_center[0] - hoop_center[0]) < 100
        
        return ball_center[1] < upper_region_y and hoop_vicinity_x
    
    def detect_down(self, ball_pos: List, hoop_pos: List) -> bool:
        """Detect if ball is in 'down' trajectory near hoop."""
        if not ball_pos or not hoop_pos:
            return False
            
        ball_center = ball_pos[-1][0]
        hoop_center = hoop_pos[-1][0]
        
        # Check if ball is in lower region near hoop
        lower_region_y = hoop_center[1] + 20
        hoop_vicinity_x = abs(ball_center[0] - hoop_center[0]) < 100
        
        return ball_center[1] > lower_region_y and hoop_vicinity_x
    
    def process_frame(self, frame: np.ndarray, ball_detection: Optional[Dict] = None,
                     hoop_detections: Optional[List[Dict]] = None, 
                     frame_number: int = 0, timestamp: float = 0.0) -> Optional[ShotEvent]:
        """
        Process a video frame for shot detection.
        
        Args:
            frame: Current video frame
            ball_detection: Ball detection data {center, confidence, bbox}
            hoop_detections: List of hoop detection data
            frame_number: Current frame number
            timestamp: Current timestamp in seconds
            
        Returns:
            ShotEvent if a shot is detected and completed, None otherwise
        """
        shot_event = None
        
        # Update ball positions if ball is detected
        if ball_detection and ball_detection['confidence'] > self.shot_threshold:
            ball_center = ball_detection['center']
            ball_size = ball_detection.get('size', (20, 20))
            
            ball_data = (ball_center, frame_number, ball_size, ball_detection['confidence'])
            self.ball_positions.append(ball_data)
            self.ball_positions = self.clean_ball_positions(self.ball_positions, frame_number)
        
        # Update hoop positions (use detected hoops or fall back to fixed positions)
        if hoop_detections:
            for hoop in hoop_detections:
                if hoop['confidence'] > self.shot_threshold:
                    hoop_center = hoop['center']
                    hoop_size = hoop.get('size', (80, 40))
                    hoop_data = (hoop_center, frame_number, hoop_size, hoop['confidence'])
                    self.hoop_positions.append(hoop_data)
        else:
            # Use fixed hoop positions as fallback
            for fixed_hoop in self.fixed_hoop_positions:
                hoop_data = (fixed_hoop['center'], frame_number, 
                           (fixed_hoop['width'], fixed_hoop['height']), 1.0)
                self.hoop_positions.append(hoop_data)
        
        # Limit hoop position history
        if len(self.hoop_positions) > 10:
            self.hoop_positions = self.hoop_positions[-10:]
        
        # Shot detection logic
        if len(self.ball_positions) > 5 and len(self.hoop_positions) > 0:
            # Detect upward trajectory
            if not self.detected_up and self.detect_up(self.ball_positions, self.hoop_positions):
                self.detected_up = True
                self.in_shot_sequence = True
                logger.debug("Detected upward ball trajectory")
            
            # Detect downward trajectory after upward
            if self.detected_up and not self.detected_down and \
               self.detect_down(self.ball_positions, self.hoop_positions):
                self.detected_down = True
                logger.debug("Detected downward ball trajectory")
                
                # Complete shot sequence - analyze if made or missed
                is_made = self.score_shot(self.ball_positions, self.hoop_positions)
                
                self.shot_attempts += 1
                if is_made:
                    self.shots_made += 1
                    logger.info(f"Shot MADE at frame {frame_number}")
                else:
                    self.shots_missed += 1
                    logger.info(f"Shot MISSED at frame {frame_number}")
                
                # Create shot event
                shot_event = ShotEvent(
                    timestamp=timestamp,
                    frame_number=frame_number,
                    is_made=is_made,
                    position=ball_detection['center'] if ball_detection else None,
                    confidence=ball_detection['confidence'] if ball_detection else 0.5
                )
                
                self.shot_events.append(shot_event)
                
                # Reset detection state
                self.detected_up = False
                self.detected_down = False
                self.in_shot_sequence = False
                self.ball_positions = []  # Clear trajectory for next shot
        
        return shot_event
    
    def get_shot_statistics(self) -> Dict:
        """Get shot statistics summary."""
        return {
            "total_shots": self.shot_attempts,
            "shots_made": self.shots_made,
            "shots_missed": self.shots_missed,
            "shooting_percentage": (self.shots_made / max(1, self.shot_attempts)) * 100,
            "shot_events": [event.to_dict() for event in self.shot_events]
        }