"""
DeepSORT tracker implementation for basketball player tracking
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

logger = logging.getLogger(__name__)

class BasketballDeepSortTracker:
    """Enhanced basketball player tracking using DeepSORT algorithm."""
    
    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep alive a track without detection
            n_init: Number of consecutive detections before track is confirmed
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.3,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Track management
        self.player_tracks = {}
        self.track_colors = {}
        self.next_color_idx = 0
        
        # Define colors for different tracks
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
    
    def update_tracks(self, detections: List[Dict], frame: np.ndarray, 
                     frame_number: int, timestamp: float, camera_id: int = 1) -> List[Dict]:
        """
        Update tracks with new detections using DeepSORT.
        
        Args:
            detections: List of detection dictionaries with bbox, confidence, class
            frame: Current video frame
            frame_number: Current frame number
            timestamp: Current timestamp
            camera_id: Camera identifier
            
        Returns:
            List of tracked objects with IDs
        """
        
        if not detections:
            # Update tracker with empty detections to age existing tracks
            tracks = self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert detections to DeepSORT format
        # deep-sort-realtime expects detections as tuples: ([left, top, width, height], confidence, class)
        detection_list = []
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            # Convert from [x1, y1, x2, y2] to [left, top, width, height]
            left = float(bbox[0])
            top = float(bbox[1]) 
            width = float(bbox[2]) - float(bbox[0])
            height = float(bbox[3]) - float(bbox[1])
            
            # Create detection tuple: ([left, top, width, height], confidence, class)
            detection_tuple = ([left, top, width, height], float(det['confidence']), det['class_name'])
            detection_list.append(detection_tuple)
        
        # Update tracker with correct format
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Process confirmed tracks
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            try:
                ltrb = track.to_ltrb()
                
                # Ensure ltrb is a proper list/array with 4 elements
                if hasattr(ltrb, 'tolist'):
                    ltrb = ltrb.tolist()
                
                # Validate ltrb format
                if not isinstance(ltrb, (list, tuple)) or len(ltrb) != 4:
                    logger.warning(f"Invalid track bbox format: {ltrb}")
                    continue
                    
                # Ensure all elements are numbers
                ltrb = [float(x) for x in ltrb]
                
            except Exception as e:
                logger.warning(f"Error getting track bbox: {e}")
                continue
            
            # Get corresponding detection info
            det_idx = self._find_best_detection_match(ltrb, detections)
            
            if det_idx is not None:
                detection = detections[det_idx]
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Create unique player ID for basketball context
                if class_name == 'person':
                    player_id = f"Player_{camera_id}_{track_id}"
                    
                    # Assign color for visualization
                    if track_id not in self.track_colors:
                        self.track_colors[track_id] = self.colors[self.next_color_idx % len(self.colors)]
                        self.next_color_idx += 1
                    
                    # Store track history
                    if player_id not in self.player_tracks:
                        self.player_tracks[player_id] = []
                    
                    center_x = (ltrb[0] + ltrb[2]) / 2
                    center_y = (ltrb[1] + ltrb[3]) / 2
                    
                    self.player_tracks[player_id].append({
                        'frame': frame_number,
                        'timestamp': timestamp,
                        'position': (center_x, center_y),
                        'bbox': ltrb,
                        'confidence': confidence,
                        'track_id': track_id
                    })
                    
                    # Keep only recent history (last 300 frames ~ 10 seconds at 30fps)
                    if len(self.player_tracks[player_id]) > 300:
                        self.player_tracks[player_id] = self.player_tracks[player_id][-300:]
                    
                    tracked_objects.append({
                        'player_id': player_id,
                        'track_id': track_id,
                        'bbox': ltrb,
                        'center': (center_x, center_y),
                        'confidence': confidence,
                        'class_name': class_name,
                        'color': self.track_colors[track_id]
                    })
                
                elif class_name == 'sports ball':
                    # Ball tracking (simpler, no need for persistent ID)
                    center_x = (ltrb[0] + ltrb[2]) / 2
                    center_y = (ltrb[1] + ltrb[3]) / 2
                    
                    tracked_objects.append({
                        'player_id': 'ball',
                        'track_id': track_id,
                        'bbox': ltrb,
                        'center': (center_x, center_y),
                        'confidence': confidence,
                        'class_name': class_name,
                        'color': (0, 255, 255)  # Yellow for ball
                    })
        
        return tracked_objects
    
    def _find_best_detection_match(self, track_bbox: List[float], 
                                  detections: List[Dict]) -> Optional[int]:
        """Find the detection that best matches a track bbox."""
        
        best_iou = 0.0
        best_idx = None
        
        # Ensure track_bbox is a proper list
        if not isinstance(track_bbox, (list, tuple)):
            if hasattr(track_bbox, 'tolist'):
                track_bbox = track_bbox.tolist()
            else:
                return None
        
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            
            # Ensure det_bbox is a proper list
            if not isinstance(det_bbox, (list, tuple)):
                if hasattr(det_bbox, 'tolist'):
                    det_bbox = det_bbox.tolist()
                else:
                    continue
            
            try:
                iou = self._calculate_iou(track_bbox, det_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            except Exception as e:
                logger.warning(f"Error calculating IoU: {e}")
                continue
        
        # Only return match if IoU is reasonable
        return best_idx if best_iou > 0.1 else None
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        
        # Ensure both boxes have 4 elements
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
        
        try:
            # Convert to float lists
            box1 = [float(x) for x in box1]
            box2 = [float(x) for x in box2]
            
            # Calculate intersection coordinates
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Check if there's an intersection
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            # Calculate intersection area
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union area
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in IoU calculation: {e}")
            return 0.0
    
    def get_track_history(self, player_id: str, frames_back: int = 30) -> List[Dict]:
        """Get recent track history for a player."""
        
        if player_id not in self.player_tracks:
            return []
        
        history = self.player_tracks[player_id]
        return history[-frames_back:] if len(history) > frames_back else history
    
    def visualize_tracks(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """Visualize tracks on frame with trail history."""
        
        vis_frame = frame.copy()
        
        for obj in tracked_objects:
            bbox = obj['bbox']
            player_id = obj['player_id']
            color = obj['color']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID
            label = f"{player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw track trail for players (not ball)
            if obj['class_name'] == 'person' and player_id in self.player_tracks:
                history = self.get_track_history(player_id, frames_back=15)
                
                if len(history) > 1:
                    points = []
                    for track in history:
                        center = track['position']
                        points.append((int(center[0]), int(center[1])))
                    
                    # Draw trail
                    for i in range(1, len(points)):
                        alpha = i / len(points)  # Fade effect
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(vis_frame, points[i-1], points[i], trail_color, 2)
        
        return vis_frame
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.player_tracks.clear()
        self.track_colors.clear()
        self.next_color_idx = 0