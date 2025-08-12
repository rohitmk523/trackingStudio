import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import logging
from .deepsort_tracker import BasketballDeepSortTracker
from .cross_camera_merger import CrossCameraMerger
from ..utils.shot_detector import ShotDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasketballEvent:
    def __init__(self, event_type: str, timestamp: float, frame_number: int, 
                 player_id: str = None, position: Tuple[float, float] = None,
                 confidence: float = None):
        self.event_type = event_type
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.player_id = player_id
        self.position = position
        self.confidence = confidence
    
    def to_dict(self):
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "player_id": self.player_id,
            "position": self.position,
            "confidence": self.confidence
        }

class VideoProcessor:
    def __init__(self, job_id: str, status_tracker: dict):
        self.job_id = job_id
        self.status_tracker = status_tracker
        self.output_dir = Path("outputs") / job_id
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize YOLO model using latest ultralytics approach
        # Use YOLO11 medium model for better accuracy
        try:
            self.model = YOLO('../models/yolo11m.pt')  # YOLO11 medium - better accuracy
            logger.info("Loaded YOLO11 medium model successfully")
        except Exception as e:
            logger.warning(f"Failed to load YOLO11 medium, trying smaller models: {e}")
            try:
                self.model = YOLO('../models/yolo11n.pt')  # Fallback to nano
                logger.info("Loaded YOLO11 small model successfully")
            except Exception as e2:
                logger.warning(f"Failed to load YOLO11 small, trying YOLOv8: {e2}")
                try:
                    self.model = YOLO('../models/yolov8n.pt')  # YOLOv8 nano fallback
                    logger.info("Loaded YOLOv8 medium model successfully")
                except Exception as e3:
                    logger.error(f"Failed to load any YOLO model: {e3}")
                    raise ValueError("Could not initialize YOLO model")
        
        # Basketball court dimensions (in feet, will convert to meters)
        self.court_length = 94  # feet
        self.court_width = 50   # feet
        
        # Event tracking
        self.events: List[BasketballEvent] = []
        self.ball_tracks = {}
        self.last_ball_position = None
        self.possession_player = None
        
        # Court transformation matrix
        self.homography_matrix = None
        
        # Initialize DeepSORT trackers for both cameras
        self.tracker_camera1 = BasketballDeepSortTracker(max_age=30, n_init=3)
        self.tracker_camera2 = BasketballDeepSortTracker(max_age=30, n_init=3)
        
        # Initialize Cross-Camera Merger for unified player IDs (optimized for overlapping views)
        self.cross_camera_merger = CrossCameraMerger(
            max_player_distance_bev=1.5,  # 1.5m max distance for overlapping camera views
            appearance_similarity_threshold=0.5,  # Lower threshold for different angles
            max_frames_missing=60  # 2 seconds at 30fps for overlapping coverage
        )
        
        # Initialize Shot Detectors for both cameras
        self.shot_detector_camera1 = ShotDetector()
        self.shot_detector_camera2 = ShotDetector()
        
    async def process_dual_videos(self, video1_path: str, video2_path: str, court_points: dict = None) -> Tuple[str, str]:
        """Main dual camera video processing pipeline with BEV transformation."""
        
        try:
            self._update_status("loading", 15.0, "Loading dual camera videos and initializing models...")
            
            # Load both videos
            cap1 = cv2.VideoCapture(video1_path)
            cap2 = cv2.VideoCapture(video2_path)
            
            if not cap1.isOpened():
                raise ValueError(f"Could not open video file: {video1_path}")
            if not cap2.isOpened():
                raise ValueError(f"Could not open video file: {video2_path}")
            
            # Get video properties (assume both videos have same properties)
            fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
            fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use first video properties for output
            fps = fps1
            frame_count = min(frame_count1, frame_count2)  # Use shorter video length
            
            # Setup output videos
            output_video_path = str(self.output_dir / "annotated_video.mp4")
            output_bev_path = str(self.output_dir / "bev_video.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Side-by-side output
            combined_width = width1 + width2
            combined_height = max(height1, height2)
            out_combined = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_width, combined_height))
            
            # BEV output (court dimensions)
            bev_width, bev_height = 800, 600  # Standard BEV output size
            out_bev = cv2.VideoWriter(output_bev_path, fourcc, fps, (bev_width, bev_height))
            
            self._update_status("processing", 20.0, "Setting up homography transformations...")
            
            # Setup homography for both cameras
            self.homography_matrix_1 = None
            self.homography_matrix_2 = None
            
            if court_points and 'camera1' in court_points:
                self._setup_homography_camera(court_points['camera1'], width1, height1, camera_id=1)
            if court_points and 'camera2' in court_points:
                self._setup_homography_camera(court_points['camera2'], width2, height2, camera_id=2)
            
            frame_number = 0
            processed_frames = 0
            
            self._update_status("processing", 25.0, "Processing synchronized dual camera frames...")
            
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                # Process both frames individually with DeepSORT
                tracked_objects1 = await self._process_frame_for_tracking(
                    frame1, frame_number, fps, camera_id=1
                )
                tracked_objects2 = await self._process_frame_for_tracking(
                    frame2, frame_number, fps, camera_id=2
                )
                
                # Perform cross-camera merging to get global player IDs
                global_players = self.cross_camera_merger.merge_camera_detections(
                    tracked_objects1, tracked_objects2, frame_number, frame_number / fps,
                    self.homography_matrix_1, self.homography_matrix_2
                )
                
                # Create annotated frames with global player information
                annotated_frame1 = self._create_annotated_frame(frame1, tracked_objects1, global_players, camera_id=1)
                annotated_frame2 = self._create_annotated_frame(frame2, tracked_objects2, global_players, camera_id=2)
                
                # Detect basketball events using global player information
                timestamp = frame_number / fps
                self._detect_basketball_events_with_global_players(
                    global_players, tracked_objects1 + tracked_objects2, timestamp, frame_number
                )
                
                # Create combined side-by-side view
                combined_frame = self._create_combined_frame(
                    annotated_frame1, annotated_frame2, combined_width, combined_height
                )
                
                # Create enhanced BEV frame with global player information
                bev_frame = self._create_enhanced_bev_frame(
                    frame1, frame2, bev_width, bev_height, frame_number, fps, global_players
                )
                
                # Write frames
                out_combined.write(combined_frame)
                out_bev.write(bev_frame)
                
                frame_number += 1
                processed_frames += 1
                
                # Update progress
                progress = 25.0 + (processed_frames / frame_count) * 55.0
                if processed_frames % 30 == 0:  # Update every 30 frames
                    self._update_status("processing", progress, 
                                      f"Processed {processed_frames}/{frame_count} dual camera frames...")
            
            cap1.release()
            cap2.release()
            out_combined.release()
            out_bev.release()
            
            self._update_status("finalizing", 85.0, "Generating analysis JSON...")
            
            # Generate JSON output
            output_json_path = await self._generate_json_output(fps, frame_count)
            
            self._update_status("completed", 100.0, "Video processing completed!")
            
            return output_video_path, output_json_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            self._update_status("failed", 0.0, f"Processing failed: {str(e)}")
            raise
    
    async def _process_frame_for_tracking(self, frame: np.ndarray, frame_number: int, fps: int, camera_id: int = 1) -> List[Dict]:
        """Process a single frame for object detection and tracking, returning tracked objects."""
        
        timestamp = frame_number / fps
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Convert YOLO results to detection format for DeepSORT
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Filter detections by confidence and include potential hoop objects
                    if ((class_name == 'person' and conf > 0.5) or 
                        (class_name == 'sports ball' and conf > 0.3) or
                        (conf > 0.4 and class_name in ['frisbee', 'umbrella', 'kite', 'chair'])):  # Potential hoop objects
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name
                        })
        
        # Update DeepSORT tracker
        tracker = self.tracker_camera1 if camera_id == 1 else self.tracker_camera2
        tracked_objects = tracker.update_tracks(detections, frame, frame_number, timestamp, camera_id)
        
        # Process shot detection
        shot_detector = self.shot_detector_camera1 if camera_id == 1 else self.shot_detector_camera2
        
        # Extract ball detection and potential hoop detections for shot analysis
        ball_detection = None
        hoop_detections = []
        
        for detection in detections:
            if detection['class_name'] == 'sports ball':
                ball_detection = {
                    'center': ((detection['bbox'][0] + detection['bbox'][2]) / 2, 
                             (detection['bbox'][1] + detection['bbox'][3]) / 2),
                    'confidence': detection['confidence'],
                    'size': (detection['bbox'][2] - detection['bbox'][0], 
                           detection['bbox'][3] - detection['bbox'][1])
                }
            elif detection['class_name'] in ['frisbee', 'umbrella', 'kite', 'chair']:
                # Potential hoop detection
                hoop_detections.append({
                    'center': ((detection['bbox'][0] + detection['bbox'][2]) / 2, 
                             (detection['bbox'][1] + detection['bbox'][3]) / 2),
                    'confidence': detection['confidence'],
                    'size': (detection['bbox'][2] - detection['bbox'][0], 
                           detection['bbox'][3] - detection['bbox'][1])
                })
        
        # Process frame for shot detection
        shot_event = shot_detector.process_frame(
            frame, ball_detection, hoop_detections, frame_number, timestamp
        )
        
        # Store shot event if detected
        if shot_event:
            self.events.append(BasketballEvent(
                event_type="shot_made" if shot_event.is_made else "shot_missed",
                timestamp=shot_event.timestamp,
                frame_number=shot_event.frame_number,
                position=shot_event.position,
                confidence=shot_event.confidence
            ))
            logger.info(f"Shot {'MADE' if shot_event.is_made else 'MISSED'} detected at {timestamp:.2f}s")
        
        return tracked_objects
    
    def _create_annotated_frame(self, frame: np.ndarray, tracked_objects: List[Dict], 
                               global_players: List, camera_id: int) -> np.ndarray:
        """Create annotated frame with global player IDs."""
        
        annotated_frame = frame.copy()
        
        # Create mapping from local to global IDs
        local_to_global = {}
        for global_player in global_players:
            if camera_id in global_player.camera_detections:
                detection = global_player.camera_detections[camera_id]
                local_to_global[detection.local_player_id] = global_player.global_id
        
        # Draw annotations
        for obj in tracked_objects:
            bbox = obj['bbox']
            player_id = obj['player_id']
            color = obj['color']
            
            # Use global ID if available
            display_id = local_to_global.get(player_id, player_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID (global if available)
            if display_id.startswith('GlobalPlayer_'):
                label = f"GP_{display_id.split('_')[-1]}"  # Shorten global ID
            else:
                label = player_id
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add camera label
        cv2.putText(annotated_frame, f'Camera {camera_id}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _detect_basketball_events_with_global_players(self, global_players: List, 
                                                     all_tracked_objects: List[Dict],
                                                     timestamp: float, frame_number: int):
        """Detect basketball events using global player information for more accurate attribution."""
        
        # Find ball objects
        ball_objects = [obj for obj in all_tracked_objects if obj.get('class_name') == 'sports ball']
        
        if not ball_objects:
            return
        
        # Use the most confident ball detection
        ball_obj = max(ball_objects, key=lambda x: x['confidence'])
        ball_center = ball_obj['center']
        
        # Update ball tracking
        self.last_ball_position = ball_center
        self.ball_tracks[frame_number] = {
            'timestamp': timestamp,
            'position': ball_center,
            'confidence': ball_obj['confidence']
        }
        
        # Find closest global player to ball in BEV space
        closest_global_player = None
        min_distance = float('inf')
        
        for global_player in global_players:
            # Get the most recent BEV position
            recent_position = global_player.get_average_bev_position()
            if recent_position != (0.0, 0.0):
                # For ball, use any available homography to get BEV position
                ball_bev_pos = None
                
                # Try to get ball BEV position using available homography
                if hasattr(self, 'homography_matrix_1') and self.homography_matrix_1 is not None:
                    ball_bev_pos = self._transform_to_bev_coordinates_new(
                        ball_center[0], ball_center[1], self.homography_matrix_1
                    )
                elif hasattr(self, 'homography_matrix_2') and self.homography_matrix_2 is not None:
                    ball_bev_pos = self._transform_to_bev_coordinates_new(
                        ball_center[0], ball_center[1], self.homography_matrix_2
                    )
                
                if ball_bev_pos:
                    distance = np.sqrt(
                        (ball_bev_pos[0] - recent_position[0])**2 + 
                        (ball_bev_pos[1] - recent_position[1])**2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_global_player = global_player
        
        # Detect possession and events
        if closest_global_player and min_distance < 2.0:  # 2 meters threshold in BEV
            current_possessor = closest_global_player.global_id
            
            # Check for possession change (assist detection)
            if current_possessor != self.possession_player and self.possession_player:
                assist_event = BasketballEvent(
                    "assist", timestamp, frame_number,
                    self.possession_player, ball_center, 0.6
                )
                self.events.append(assist_event)
                logger.info(f"Assist detected: {self.possession_player} -> {current_possessor}")
            
            self.possession_player = current_possessor
            
            # Check for shot attempts based on BEV position
            player_bev_pos = closest_global_player.get_average_bev_position()
            if player_bev_pos != (0.0, 0.0):
                if self._is_shot_attempt_bev(player_bev_pos[0], player_bev_pos[1]):
                    if self._is_three_point_shot_bev(player_bev_pos[0], player_bev_pos[1]):
                        shot_event = BasketballEvent(
                            "3_point_attempt", timestamp, frame_number,
                            current_possessor, ball_center, 0.7
                        )
                        self.events.append(shot_event)
                        logger.info(f"3-point attempt: {current_possessor}")
                    else:
                        shot_event = BasketballEvent(
                            "2_point_attempt", timestamp, frame_number,
                            current_possessor, ball_center, 0.6
                        )
                        self.events.append(shot_event)
                        logger.info(f"2-point attempt: {current_possessor}")
    
    def _transform_to_bev_coordinates_new(self, x: float, y: float, homography_matrix: np.ndarray) -> Optional[Tuple[float, float]]:
        """Transform image coordinates to BEV coordinates."""
        try:
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), homography_matrix)
            return tuple(transformed[0][0])
        except:
            return None
    
    def _is_shot_attempt_bev(self, bev_x: float, bev_y: float) -> bool:
        """Check if position indicates a shot attempt in BEV coordinates."""
        court_length_m = self.court_length * 0.3048
        court_width_m = self.court_width * 0.3048
        
        # Distance from both baskets (assuming baskets at ends of court)
        dist_to_basket1 = np.sqrt(bev_x**2 + (bev_y - court_width_m/2)**2)
        dist_to_basket2 = np.sqrt((bev_x - court_length_m)**2 + (bev_y - court_width_m/2)**2)
        
        min_dist_to_basket = min(dist_to_basket1, dist_to_basket2)
        return min_dist_to_basket < 8.0  # Within 8 meters of basket
    
    def _is_three_point_shot_bev(self, bev_x: float, bev_y: float) -> bool:
        """Check if position is 3-point range in BEV coordinates."""
        court_length_m = self.court_length * 0.3048
        court_width_m = self.court_width * 0.3048
        
        # Distance from both baskets
        dist_to_basket1 = np.sqrt(bev_x**2 + (bev_y - court_width_m/2)**2)
        dist_to_basket2 = np.sqrt((bev_x - court_length_m)**2 + (bev_y - court_width_m/2)**2)
        
        min_dist_to_basket = min(dist_to_basket1, dist_to_basket2)
        return min_dist_to_basket > 6.7  # Beyond 3-point line (approximately 6.7m)
    
    async def _process_frame(self, frame: np.ndarray, frame_number: int, fps: int, camera_id: int = 1) -> np.ndarray:
        """Process a single frame for object detection and tracking with DeepSORT."""
        
        timestamp = frame_number / fps
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Convert YOLO results to detection format for DeepSORT
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Filter detections by confidence
                    if (class_name == 'person' and conf > 0.5) or (class_name == 'sports ball' and conf > 0.3):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': class_name
                        })
        
        # Update DeepSORT tracker
        tracker = self.tracker_camera1 if camera_id == 1 else self.tracker_camera2
        tracked_objects = tracker.update_tracks(detections, frame, frame_number, timestamp, camera_id)
        
        # Process tracked objects for basketball events
        for obj in tracked_objects:
            if obj['class_name'] == 'person':
                center_x, center_y = obj['center']
                self._check_basketball_events_deepsort(
                    obj['player_id'], center_x, center_y, timestamp, frame_number
                )
            elif obj['class_name'] == 'sports ball':
                center_x, center_y = obj['center']
                self._track_ball_deepsort(center_x, center_y, frame_number, timestamp)
        
        # Annotate frame with DeepSORT tracking
        annotated_frame = tracker.visualize_tracks(frame, tracked_objects)
        
        # Add timestamp
        cv2.putText(annotated_frame, f'Camera {camera_id} - Time: {timestamp:.2f}s', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _track_player(self, box, frame_number: int, timestamp: float):
        """Track player movement and assign IDs."""
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Simple player tracking - assign to nearest existing track or create new
        player_id = self._assign_player_id(center_x, center_y)
        
        # Update player track
        if player_id not in self.player_tracks:
            self.player_tracks[player_id] = []
        
        self.player_tracks[player_id].append({
            'frame': frame_number,
            'timestamp': timestamp,
            'position': (center_x, center_y),
            'bbox': (x1, y1, x2, y2)
        })
        
        # Check for basketball events
        self._check_basketball_events(player_id, center_x, center_y, timestamp, frame_number)
    
    def _track_ball(self, box, frame_number: int, timestamp: float):
        """Track basketball movement."""
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Update ball track
        self.ball_tracks[frame_number] = {
            'timestamp': timestamp,
            'position': (center_x, center_y),
            'bbox': (x1, y1, x2, y2)
        }
        
        self.last_ball_position = (center_x, center_y)
        
        # Detect possession changes
        self._detect_possession_change(center_x, center_y, timestamp, frame_number)
    
    def _assign_player_id(self, x: float, y: float) -> str:
        """Assign player ID based on position tracking."""
        
        # Simple distance-based assignment
        min_distance = float('inf')
        assigned_id = None
        
        for player_id, tracks in self.player_tracks.items():
            if tracks:
                last_pos = tracks[-1]['position']
                distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                
                if distance < min_distance and distance < 100:  # Max distance threshold
                    min_distance = distance
                    assigned_id = player_id
        
        # Create new player if no match found
        if assigned_id is None:
            assigned_id = f"Player_{len(self.player_tracks) + 1}"
        
        return assigned_id
    
    def _check_basketball_events(self, player_id: str, x: float, y: float, 
                                timestamp: float, frame_number: int):
        """Check for basketball events (shots, assists)."""
        
        if not self.last_ball_position:
            return
        
        ball_x, ball_y = self.last_ball_position
        distance_to_ball = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
        
        # If player is close to ball, they likely have possession
        if distance_to_ball < 50:
            self.possession_player = player_id
        
        # Simple shot detection based on ball trajectory and court position
        if self.homography_matrix is not None:
            court_pos = self._transform_to_court_coordinates(x, y)
            if court_pos:
                court_x, court_y = court_pos
                
                # Check if near 3-point line or 2-point area
                if self._is_three_point_shot(court_x, court_y):
                    event = BasketballEvent(
                        "3_point_attempt", timestamp, frame_number,
                        player_id, (x, y), 0.7
                    )
                    self.events.append(event)
                elif self._is_two_point_shot(court_x, court_y):
                    event = BasketballEvent(
                        "2_point_attempt", timestamp, frame_number,
                        player_id, (x, y), 0.6
                    )
                    self.events.append(event)
    
    def _detect_possession_change(self, ball_x: float, ball_y: float, 
                                 timestamp: float, frame_number: int):
        """Detect ball possession changes."""
        
        # Find closest player to ball
        min_distance = float('inf')
        closest_player = None
        
        for player_id, tracks in self.player_tracks.items():
            if tracks:
                last_track = tracks[-1]
                if last_track['frame'] >= frame_number - 5:  # Recent track
                    px, py = last_track['position']
                    distance = np.sqrt((ball_x - px)**2 + (ball_y - py)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_player = player_id
        
        # Check for possession change
        if (closest_player and closest_player != self.possession_player 
            and min_distance < 50):
            
            # Record assist if previous player passed the ball
            if self.possession_player:
                assist_event = BasketballEvent(
                    "assist", timestamp, frame_number,
                    self.possession_player, (ball_x, ball_y), 0.5
                )
                self.events.append(assist_event)
            
            self.possession_player = closest_player
    
    def _compile_all_player_tracks(self) -> Dict:
        """Compile player tracks from both DeepSORT trackers."""
        all_tracks = {}
        
        # Combine tracks from both cameras
        for tracker in [self.tracker_camera1, self.tracker_camera2]:
            for player_id, tracks in tracker.player_tracks.items():
                if tracks:
                    all_tracks[player_id] = {
                        "total_detections": len(tracks),
                        "first_appearance": tracks[0]["timestamp"] if tracks else None,
                        "last_appearance": tracks[-1]["timestamp"] if tracks else None,
                        "track_id": tracks[0].get("track_id", "unknown") if tracks else "unknown",
                        "positions": [
                            {
                                "timestamp": track["timestamp"],
                                "frame": track["frame"],
                                "position": track["position"],
                                "confidence": track.get("confidence", 0.0)
                            }
                            for track in tracks[::30]  # Sample every 30 frames
                        ]
                    }
        
        return all_tracks
    
    def _annotate_frame(self, frame: np.ndarray, results, frame_number: int, timestamp: float) -> np.ndarray:
        """Add annotations to frame."""
        
        annotated = frame.copy()
        
        # Draw YOLO detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    if class_name == 'person' and conf > 0.5:
                        # Draw player bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Get player ID
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        player_id = self._assign_player_id(center_x, center_y)
                        
                        cv2.putText(annotated, f'{player_id}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    elif class_name == 'sports ball' and conf > 0.3:
                        # Draw ball bounding box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, 'Ball', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw timestamp
        cv2.putText(annotated, f'Time: {timestamp:.2f}s', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated
    
    def _setup_homography_camera(self, court_points: dict, width: int, height: int, camera_id: int):
        """Setup homography transformation for specific camera."""
        
        # Court corners in real coordinates (in feet, converted to meters)
        court_corners_real = np.array([
            [0, 0],
            [self.court_length * 0.3048, 0],
            [self.court_length * 0.3048, self.court_width * 0.3048],
            [0, self.court_width * 0.3048]
        ], dtype=np.float32)
        
        # Image corners from user input
        if 'corners' in court_points:
            image_corners = np.array(court_points['corners'], dtype=np.float32)
            
            if len(image_corners) == 4:
                homography_matrix, _ = cv2.findHomography(
                    image_corners, court_corners_real
                )
                
                if camera_id == 1:
                    self.homography_matrix_1 = homography_matrix
                else:
                    self.homography_matrix_2 = homography_matrix

    def _setup_homography(self, court_points: dict, width: int, height: int):
        """Setup homography transformation for court view (legacy method)."""
        self._setup_homography_camera(court_points, width, height, camera_id=1)
        self.homography_matrix = self.homography_matrix_1
    
    def _transform_to_court_coordinates(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Transform image coordinates to court coordinates."""
        
        if self.homography_matrix is None:
            return None
        
        point = np.array([[x, y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), self.homography_matrix)
        
        return tuple(transformed[0][0])
    
    def _is_three_point_shot(self, court_x: float, court_y: float) -> bool:
        """Check if position is in 3-point shot range."""
        # Simplified 3-point detection
        center_x = self.court_length * 0.3048 / 2
        distance_from_basket = np.sqrt((court_x - center_x)**2 + court_y**2)
        return distance_from_basket > 6.7  # 3-point line is ~6.7m from basket
    
    def _is_two_point_shot(self, court_x: float, court_y: float) -> bool:
        """Check if position is in 2-point shot range."""
        center_x = self.court_length * 0.3048 / 2
        distance_from_basket = np.sqrt((court_x - center_x)**2 + court_y**2)
        return distance_from_basket <= 6.7 and distance_from_basket > 1.0
    
    async def _generate_json_output(self, fps: int, total_frames: int) -> str:
        """Generate structured JSON output with analysis results."""
        
        output_json_path = str(self.output_dir / "analysis.json")
        
        # Compile analysis data
        analysis_data = {
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": total_frames / fps,
                "processing_timestamp": datetime.now().isoformat()
            },
            "events": [event.to_dict() for event in self.events],
            "player_tracks": self._compile_all_player_tracks(),
            "global_player_tracks": self.cross_camera_merger.get_merged_player_tracks(),
            "cross_camera_statistics": self.cross_camera_merger.get_statistics(),
            "statistics": {
                "total_players_detected": len(self._compile_all_player_tracks()),
                "total_events": len(self.events),
                "event_breakdown": self._get_event_breakdown()
            },
            "shot_statistics": {
                "camera1": self.shot_detector_camera1.get_shot_statistics(),
                "camera2": self.shot_detector_camera2.get_shot_statistics(),
                "combined": self._get_combined_shot_statistics()
            }
        }
        
        # Save JSON
        with open(output_json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        return output_json_path
    
    def _get_event_breakdown(self) -> Dict[str, int]:
        """Get breakdown of events by type."""
        breakdown = {}
        for event in self.events:
            event_type = event.event_type
            breakdown[event_type] = breakdown.get(event_type, 0) + 1
        return breakdown
    
    def _get_combined_shot_statistics(self) -> Dict:
        """Get combined shot statistics from both cameras."""
        camera1_stats = self.shot_detector_camera1.get_shot_statistics()
        camera2_stats = self.shot_detector_camera2.get_shot_statistics()
        
        # Combine statistics
        total_shots = camera1_stats["total_shots"] + camera2_stats["total_shots"]
        total_made = camera1_stats["shots_made"] + camera2_stats["shots_made"]
        total_missed = camera1_stats["shots_missed"] + camera2_stats["shots_missed"]
        
        # Combine shot events from both cameras and sort by timestamp
        all_shot_events = camera1_stats["shot_events"] + camera2_stats["shot_events"]
        all_shot_events.sort(key=lambda x: x["timestamp"])
        
        return {
            "total_shots": total_shots,
            "shots_made": total_made,
            "shots_missed": total_missed,
            "shooting_percentage": (total_made / max(1, total_shots)) * 100,
            "shot_timeline": all_shot_events
        }
    
    def _update_status(self, status: str, progress: float, message: str):
        """Update processing status."""
        self.status_tracker[self.job_id].update({
            "status": status,
            "progress": progress,
            "message": message
        })
        logger.info(f"Job {self.job_id}: {message} ({progress:.1f}%)")
    
    def _create_combined_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                              combined_width: int, combined_height: int) -> np.ndarray:
        """Create side-by-side combined frame from both cameras."""
        
        # Create blank combined frame
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Place first frame on the left
        combined[0:h1, 0:w1] = frame1
        
        # Place second frame on the right
        combined[0:h2, w1:w1+w2] = frame2
        
        # Add separator line
        cv2.line(combined, (w1, 0), (w1, combined_height), (255, 255, 255), 2)
        
        # Add camera labels
        cv2.putText(combined, 'Camera 1', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Camera 2', (w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def _create_bev_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                         bev_width: int, bev_height: int, 
                         frame_number: int, fps: int) -> np.ndarray:
        """Create Bird's Eye View frame by combining both camera perspectives."""
        
        # Create blank BEV frame (basketball court background)
        bev_frame = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        # Draw basketball court outline
        bev_frame = self._draw_court_outline(bev_frame, bev_width, bev_height)
        
        # Transform detections from both cameras to BEV
        if hasattr(self, 'homography_matrix_1') and self.homography_matrix_1 is not None:
            bev_frame = self._add_camera_detections_to_bev(
                bev_frame, frame1, self.homography_matrix_1, bev_width, bev_height, camera_id=1
            )
        
        if hasattr(self, 'homography_matrix_2') and self.homography_matrix_2 is not None:
            bev_frame = self._add_camera_detections_to_bev(
                bev_frame, frame2, self.homography_matrix_2, bev_width, bev_height, camera_id=2
            )
        
        # Add timestamp
        timestamp = frame_number / fps
        cv2.putText(bev_frame, f'BEV Time: {timestamp:.2f}s', 
                   (10, bev_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return bev_frame
    
    def _draw_court_outline(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Draw basketball court outline on BEV frame."""
        
        # Court proportions (94 x 50 feet)
        court_color = (50, 50, 50)  # Dark gray for court
        line_color = (255, 255, 255)  # White lines
        
        # Fill court background
        cv2.rectangle(frame, (50, 50), (width-50, height-50), court_color, -1)
        
        # Court boundary
        cv2.rectangle(frame, (50, 50), (width-50, height-50), line_color, 2)
        
        # Center circle
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), 50, line_color, 2)
        
        # 3-point arcs (simplified)
        cv2.ellipse(frame, (100, center_y), (80, 120), 0, -90, 90, line_color, 2)
        cv2.ellipse(frame, (width-100, center_y), (80, 120), 0, 90, 270, line_color, 2)
        
        # Free throw circles
        cv2.circle(frame, (150, center_y), 40, line_color, 2)
        cv2.circle(frame, (width-150, center_y), 40, line_color, 2)
        
        return frame
    
    def _add_camera_detections_to_bev(self, bev_frame: np.ndarray, camera_frame: np.ndarray,
                                     homography_matrix: np.ndarray, bev_width: int, bev_height: int,
                                     camera_id: int) -> np.ndarray:
        """Add detections from a camera to the BEV frame."""
        
        # Run YOLO detection on camera frame
        results = self.model(camera_frame, verbose=False)
        
        # Transform and draw detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    if class_name == 'person' and conf > 0.5:
                        # Get center point
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = y2  # Use bottom of bounding box as ground point
                        
                        # Transform to BEV coordinates
                        bev_point = self._transform_point_to_bev(
                            center_x, center_y, homography_matrix, bev_width, bev_height
                        )
                        
                        if bev_point:
                            bev_x, bev_y = bev_point
                            # Draw player as circle
                            color = (0, 255, 0) if camera_id == 1 else (0, 0, 255)
                            cv2.circle(bev_frame, (int(bev_x), int(bev_y)), 8, color, -1)
                            cv2.putText(bev_frame, f'P{camera_id}', 
                                      (int(bev_x)-10, int(bev_y)-15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    elif class_name == 'sports ball' and conf > 0.3:
                        # Get center point
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Transform to BEV coordinates
                        bev_point = self._transform_point_to_bev(
                            center_x, center_y, homography_matrix, bev_width, bev_height
                        )
                        
                        if bev_point:
                            bev_x, bev_y = bev_point
                            # Draw ball as small filled circle
                            cv2.circle(bev_frame, (int(bev_x), int(bev_y)), 5, (0, 255, 255), -1)
        
        return bev_frame
    
    def _create_enhanced_bev_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                                  bev_width: int, bev_height: int, 
                                  frame_number: int, fps: int, global_players: List) -> np.ndarray:
        """Create enhanced Bird's Eye View frame with global player tracking."""
        
        # Create blank BEV frame (basketball court background)
        bev_frame = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        
        # Draw basketball court outline
        bev_frame = self._draw_court_outline(bev_frame, bev_width, bev_height)
        
        # Use cross-camera merger to visualize global tracks
        bev_frame = self.cross_camera_merger.visualize_global_tracks_bev(bev_frame)
        
        # Add merger statistics
        stats = self.cross_camera_merger.get_statistics()
        cv2.putText(bev_frame, f'Global Players: {stats["active_global_players"]}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bev_frame, f'Associations: {stats["total_associations"]}', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = frame_number / fps
        cv2.putText(bev_frame, f'BEV Time: {timestamp:.2f}s', 
                   (10, bev_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return bev_frame
    
    def _transform_point_to_bev(self, x: float, y: float, homography_matrix: np.ndarray,
                               bev_width: int, bev_height: int) -> Optional[Tuple[float, float]]:
        """Transform a point from camera view to BEV coordinates."""
        
        try:
            # Transform to court coordinates
            point = np.array([[x, y]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point.reshape(1, 1, 2), homography_matrix)
            court_x, court_y = transformed[0][0]
            
            # Map court coordinates to BEV frame coordinates
            # Court is 94x50 feet, map to BEV frame size
            court_width_m = self.court_length * 0.3048  # 94 feet in meters
            court_height_m = self.court_width * 0.3048  # 50 feet in meters
            
            # Map to BEV frame (with padding)
            padding = 50
            usable_width = bev_width - 2 * padding
            usable_height = bev_height - 2 * padding
            
            bev_x = padding + (court_x / court_width_m) * usable_width
            bev_y = padding + (court_y / court_height_m) * usable_height
            
            # Check bounds
            if 0 <= bev_x < bev_width and 0 <= bev_y < bev_height:
                return (bev_x, bev_y)
            
        except:
            pass
        
        return None
    
    def _check_basketball_events_deepsort(self, player_id: str, x: float, y: float, 
                                         timestamp: float, frame_number: int):
        """Check for basketball events using DeepSORT tracked player IDs."""
        
        if not self.last_ball_position:
            return
        
        ball_x, ball_y = self.last_ball_position
        distance_to_ball = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
        
        # If player is close to ball, they likely have possession
        if distance_to_ball < 50:
            self.possession_player = player_id
        
        # Simple shot detection based on ball trajectory and court position
        if self.homography_matrix is not None:
            court_pos = self._transform_to_court_coordinates(x, y)
            if court_pos:
                court_x, court_y = court_pos
                
                # Check if near 3-point line or 2-point area
                if self._is_three_point_shot(court_x, court_y):
                    event = BasketballEvent(
                        "3_point_attempt", timestamp, frame_number,
                        player_id, (x, y), 0.7
                    )
                    self.events.append(event)
                elif self._is_two_point_shot(court_x, court_y):
                    event = BasketballEvent(
                        "2_point_attempt", timestamp, frame_number,
                        player_id, (x, y), 0.6
                    )
                    self.events.append(event)
    
    def _track_ball_deepsort(self, ball_x: float, ball_y: float, 
                            frame_number: int, timestamp: float):
        """Track basketball movement with DeepSORT integration."""
        
        # Update ball track
        self.ball_tracks[frame_number] = {
            'timestamp': timestamp,
            'position': (ball_x, ball_y),
            'bbox': None  # Will be filled if needed
        }
        
        self.last_ball_position = (ball_x, ball_y)
        
        # Detect possession changes
        self._detect_possession_change_deepsort(ball_x, ball_y, timestamp, frame_number)
    
    def _detect_possession_change_deepsort(self, ball_x: float, ball_y: float, 
                                          timestamp: float, frame_number: int):
        """Detect ball possession changes using DeepSORT tracked players."""
        
        # Find closest tracked player to ball from both cameras
        min_distance = float('inf')
        closest_player = None
        
        # Check players from both cameras
        for tracker in [self.tracker_camera1, self.tracker_camera2]:
            for player_id, tracks in tracker.player_tracks.items():
                if tracks:
                    last_track = tracks[-1]
                    if last_track['frame'] >= frame_number - 5:  # Recent track
                        px, py = last_track['position']
                        distance = np.sqrt((ball_x - px)**2 + (ball_y - py)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_player = player_id
        
        # Check for possession change
        if (closest_player and closest_player != self.possession_player 
            and min_distance < 50):
            
            # Record assist if previous player passed the ball
            if self.possession_player:
                assist_event = BasketballEvent(
                    "assist", timestamp, frame_number,
                    self.possession_player, (ball_x, ball_y), 0.5
                )
                self.events.append(assist_event)
            
            self.possession_player = closest_player
    
    def _compile_all_player_tracks(self) -> Dict:
        """Compile player tracks from both DeepSORT trackers."""
        all_tracks = {}
        
        # Combine tracks from both cameras
        for tracker in [self.tracker_camera1, self.tracker_camera2]:
            for player_id, tracks in tracker.player_tracks.items():
                if tracks:
                    all_tracks[player_id] = {
                        "total_detections": len(tracks),
                        "first_appearance": tracks[0]["timestamp"] if tracks else None,
                        "last_appearance": tracks[-1]["timestamp"] if tracks else None,
                        "track_id": tracks[0].get("track_id", "unknown") if tracks else "unknown",
                        "positions": [
                            {
                                "timestamp": track["timestamp"],
                                "frame": track["frame"],
                                "position": track["position"],
                                "confidence": track.get("confidence", 0.0)
                            }
                            for track in tracks[::30]  # Sample every 30 frames
                        ]
                    }
        
        return all_tracks