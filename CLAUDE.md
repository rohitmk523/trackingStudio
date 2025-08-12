# Basketball Video Analysis Project with AI Shot Detection

## Project Overview
Create a comprehensive basketball video analysis system that processes local video files to track players and ball, detect made/missed shots with 97% accuracy, and generate annotated video output with structured JSON data for AI model analysis like Gemini.

## Source Repositories
- **Base project**: https://github.com/playbox-dev/trackstudio
- **Shot detection**: https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker

## Project Goals

### Input Requirements
- Process local video files (MP4) showing basketball gameplay
- Handle videos that may be stitched from multiple camera angles
- Work with pre-recorded content instead of live streams

### Desired Outputs
1. **Tracked Video**: Annotated video file with bounding boxes around players and ball
2. **Bird's Eye View Video**: Top-down court visualization with player tracking
3. **Structured JSON**: Event log containing timestamps, player positions, detected basketball events, and shot statistics
4. **Shot Analytics**: Made/missed shot detection with 97% accuracy and timeline tracking

## Core Components Needed

### From TrackStudio Repository
- Object detection models for players and ball identification
- Multi-object tracking algorithms
- Homography transformation for court view normalization
- Basketball action recognition capabilities
- Cross-camera player merging and global ID assignment

### From AI-Basketball-Shot-Detection-Tracker Repository
- AI-powered trajectory analysis using linear regression
- Made/missed shot determination with 97% accuracy
- Sequential shot detection (up/down ball movement)
- Data cleaning algorithms for trajectory accuracy

### Components to Exclude
- Frontend interfaces and dashboards
- Real-time streaming components
- WebSocket implementations
- Web-based visualization tools

## Basketball Events to Track
- Shot attempts (2-point vs 3-point classification)
- **Shot outcomes (made vs missed) with 97% accuracy**
- Ball possession changes
- Player positions and movements
- Assists
- Steals
- Blocks

## Expected Workflow
1. Input dual camera basketball video files
2. Apply homography transformation to normalize court view
3. Detect and track players and ball across frames using YOLO11 + DeepSORT
4. Analyze ball trajectory for shot detection (made/missed)
5. Merge cross-camera tracking for global player IDs
6. Recognize basketball-specific actions and events
7. Output annotated video with tracking visualizations
8. Generate Bird's Eye View video with court overlay
9. Generate structured JSON log with shot statistics and event timeline

## Success Criteria
- Successfully processes dual camera MP4 basketball videos
- Maintains accurate tracking of players and ball across frames with DeepSORT
- Correctly identifies and classifies basketball events
- **Achieves 97% accuracy in shot detection (made vs missed)**
- Outputs clean annotated video file with dual camera views
- Generates Bird's Eye View video with court transformation
- Generates structured JSON data with shot statistics suitable for AI analysis
- Handles cross-camera player merging for global IDs

## Final Use Case
The generated JSON data with comprehensive shot statistics will be fed to a multimodal AI model (like Gemini 2.5 Pro) along with the original video for advanced basketball game analysis, including:
- Detailed shooting performance analytics
- Real-time made/missed shot tracking
- Player performance evaluation with shooting percentages
- Timeline-based game analysis with shot events
- Statistical summaries and play-by-play commentary

## Development Commands

### Installation
```bash
# Install all dependencies with Poetry (includes DeepSORT tracking and shot detection)
poetry install
```

### Running the API Server
```bash
poetry run dev
```
Server will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

### Testing the API
```bash
poetry run python trackstudio/client_example.py
```

## API Architecture

### Core Components
- **FastAPI Server** (`api/main.py`): REST API with async processing
- **VideoProcessor** (`processors/video_processor.py`): Main video analysis pipeline
- **ShotDetector** (`utils/shot_detector.py`): AI-powered shot detection with 97% accuracy
- **YOLO11 Medium Model**: High-accuracy object detection for players and basketball
- **DeepSORT Tracking** (`processors/deepsort_tracker.py`): Advanced multi-object tracking with appearance-based re-identification
- **Cross-Camera Merger** (`processors/cross_camera_merger.py`): Global player ID assignment across cameras
- **Homography Transformation**: Court view normalization using manual court points

### Processing Pipeline
1. **Video Upload**: Accept 1080p 30fps MP4 files from dual cameras
2. **Object Detection**: YOLO11 medium model detection of players and basketball  
3. **DeepSORT Tracking**: Advanced tracking with appearance-based re-identification
4. **Shot Detection**: AI trajectory analysis for made/missed shots with 97% accuracy
5. **Cross-Camera Merging**: Assign global player IDs across both camera views
6. **Court Transformation**: Apply homography using manually provided court boundary points
7. **Event Recognition**: Detect 2-point shots, 3-point shots, assists, and shot outcomes
8. **Output Generation**: Create annotated video, BEV video, and structured JSON with shot statistics

### API Endpoints
- `POST /upload`: Upload video with optional court boundary points
- `GET /status/{job_id}`: Check processing status and progress
- `GET /download/video/{job_id}`: Download annotated video
- `GET /download/json/{job_id}`: Download analysis JSON

### File Structure
```
trackingStudio/
├── trackstudio/                    # Main application package
│   ├── api/                       # FastAPI endpoints
│   │   ├── __init__.py
│   │   └── main.py               # REST API server
│   ├── processors/               # Video analysis modules
│   │   ├── __init__.py
│   │   ├── video_processor.py    # Main video pipeline
│   │   ├── deepsort_tracker.py   # DeepSORT tracking
│   │   └── cross_camera_merger.py # Cross-camera fusion
│   ├── utils/                    # Utility modules
│   │   ├── __init__.py
│   │   └── shot_detector.py      # AI shot detection
│   ├── models/                   # Model weights
│   │   ├── yolo11m.pt           # YOLO11 medium model
│   │   ├── yolo11n.pt           # YOLO11 nano model
│   │   └── yolov8n.pt           # YOLOv8 fallback
│   ├── __init__.py              # Package initialization
│   ├── run.py                   # Server startup script
│   ├── client_example.py        # Example API client
│   ├── uploads/                 # Uploaded videos
│   └── outputs/                 # Processed results
│       └── {job_id}/
│           ├── annotated_video.mp4
│           ├── bev_video.mp4
│           └── analysis.json
├── context/                     # Documentation
│   ├── IMPROVEMENTS.md
│   ├── PROJECT_PROGRESS.md
│   ├── SHOT_DETECTION_IMPLEMENTATION.md
│   └── PROJECT_STRUCTURE_UPDATE.md
└── README.md                    # User documentation
```

### Court Boundary Setup
Videos require manual court boundary points for homography transformation:
```json
{
  "corners": [
    [x1, y1],  // Top-left court corner
    [x2, y2],  // Top-right court corner  
    [x3, y3],  // Bottom-right court corner
    [x4, y4]   // Bottom-left court corner
  ]
}
```

### Basketball Events Detected
- **2-point shot attempts**: Shots inside 3-point line
- **3-point shot attempts**: Shots beyond 3-point line
- **Assists**: Ball possession changes leading to shots
- **Shot outcomes**: Made vs missed shots with 97% accuracy using AI trajectory analysis
- **Shot timeline**: Real-time tracking of all shot events with timestamps

### Output JSON Structure
```json
{
  "video_info": {
    "fps": 30,
    "total_frames": 9000,
    "duration_seconds": 300,
    "processing_timestamp": "2025-08-12T10:30:00"
  },
  "events": [
    {
      "event_type": "2_point_attempt",
      "timestamp": 45.6,
      "frame_number": 1368,
      "player_id": "Player_1",
      "position": [640, 360],
      "confidence": 0.7
    },
    {
      "event_type": "shot_made",
      "timestamp": 67.2,
      "frame_number": 2016,
      "position": [580, 320],
      "confidence": 0.85
    }
  ],
  "player_tracks": {
    "Player_1_42": {
      "total_detections": 150,
      "track_id": 42,
      "first_appearance": 1.5,
      "last_appearance": 298.7,
      "positions": [...]
    }
  },
  "global_player_tracks": {
    "Global_Player_1": {
      "camera1_track_id": 42,
      "camera2_track_id": 15,
      "total_detections": 295,
      "unified_positions": [...]
    }
  },
  "statistics": {
    "total_players_detected": 8,
    "total_events": 25,
    "event_breakdown": {
      "2_point_attempt": 12,
      "3_point_attempt": 8,
      "assist": 5,
      "shot_made": 8,
      "shot_missed": 12
    }
  },
  "shot_statistics": {
    "camera1": {
      "total_shots": 5,
      "shots_made": 3,
      "shots_missed": 2,
      "shooting_percentage": 60.0,
      "shot_events": [...]
    },
    "camera2": {
      "total_shots": 3,
      "shots_made": 2,
      "shots_missed": 1,
      "shooting_percentage": 66.7,
      "shot_events": [...]
    },
    "combined": {
      "total_shots": 8,
      "shots_made": 5,
      "shots_missed": 3,
      "shooting_percentage": 62.5,
      "shot_timeline": [
        {
          "timestamp": 45.6,
          "frame_number": 1368,
          "is_made": true,
          "position": [640, 360],
          "confidence": 0.85,
          "shot_type": "unknown"
        }
      ]
    }
  }
}
```

## 🎯 Key Features Implemented

### Shot Detection System
- **97% Accuracy**: Based on AI-Basketball-Shot-Detection-Tracker repository
- **Linear Regression**: Advanced trajectory analysis to predict ball path
- **Sequential Detection**: Ball must move through "up" and "down" phases near hoop
- **Data Cleaning**: Filters inaccurate tracking points for better accuracy
- **Real-time Processing**: Integrates seamlessly with existing video pipeline

### Cross-Camera Tracking
- **Global Player IDs**: Unified player identification across both cameras
- **Appearance Matching**: DeepSORT-based re-identification across camera views  
- **BEV Fusion**: Merged detections in Bird's Eye View coordinate space
- **Enhanced Analytics**: Combined statistics from multiple viewpoints

### Professional Output
- **Dual Video Formats**: Side-by-side annotated + Bird's Eye View videos
- **Comprehensive JSON**: Shot statistics, player tracks, event timeline
- **AI-Ready Data**: Structured for multimodal AI model consumption
- **Timeline Analysis**: Precise timestamps for all basketball events

## 📚 Documentation Structure

### Context Files
- **`context/SHOT_DETECTION_IMPLEMENTATION.md`**: Technical implementation details
- **`context/PROJECT_STRUCTURE_UPDATE.md`**: Reorganization documentation
- **`context/IMPROVEMENTS.md`**: DeepSORT tracking enhancements
- **`context/PROJECT_PROGRESS.md`**: Development history and milestones

### User Documentation  
- **`README.md`**: Complete user guide with API endpoints
- **`CLAUDE.md`**: This development specification document