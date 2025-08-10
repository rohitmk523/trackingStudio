# Basketball Video Analysis Project

## Project Overview
Create a basketball video analysis system that processes local video files to track players and ball, generating both annotated video output and structured JSON data for further analysis with AI models like Gemini.

## Source Repository
Base project: https://github.com/playbox-dev/trackstudio

## Project Goals

### Input Requirements
- Process local video files (MP4) showing basketball gameplay
- Handle videos that may be stitched from multiple camera angles
- Work with pre-recorded content instead of live streams

### Desired Outputs
1. **Tracked Video**: Annotated video file with bounding boxes around players and ball
2. **Structured JSON**: Event log containing timestamps, player positions, and detected basketball events

## Core Components Needed

### From TrackStudio Repository
- Object detection models for players and ball identification
- Multi-object tracking algorithms
- Homography transformation for court view normalization
- Basketball action recognition capabilities

### Components to Exclude
- Frontend interfaces and dashboards
- Real-time streaming components
- WebSocket implementations
- Web-based visualization tools

## Basketball Events to Track
- Shot attempts (2-point vs 3-point classification)
- Ball possession changes
- Player positions and movements
- Assists
- Steals
- Blocks

## Expected Workflow
1. Input local basketball video file
2. Apply homography transformation to normalize court view
3. Detect and track players and ball across frames
4. Recognize basketball-specific actions and events
5. Output annotated video with tracking visualizations
6. Generate structured JSON log of all detected events

## Success Criteria
- Successfully processes local MP4 basketball videos
- Maintains accurate tracking of players and ball across frames
- Correctly identifies and classifies basketball events
- Outputs clean annotated video file
- Generates structured JSON data suitable for AI analysis
- Handles multi-angle video stitching effectively

## Final Use Case
The generated JSON data will be fed to a multimodal AI model (like Gemini 2.5 Pro) along with the original video for comprehensive basketball game analysis, including statistics generation and play-by-play commentary.

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the API Server
```bash
python run.py
```
Server will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

### Testing the API
```bash
python client_example.py
```

## API Architecture

### Core Components
- **FastAPI Server** (`main.py`): REST API with async processing
- **VideoProcessor** (`video_processor.py`): Main video analysis pipeline
- **YOLO11 Medium Model**: High-accuracy object detection for players and basketball
- **DeepSORT Tracking**: Advanced multi-object tracking with appearance-based re-identification
- **Homography Transformation**: Court view normalization using manual court points

### Processing Pipeline
1. **Video Upload**: Accept 1080p 30fps MP4 files from dual cameras
2. **Object Detection**: YOLO11 medium model detection of players and basketball  
3. **DeepSORT Tracking**: Advanced tracking with appearance-based re-identification
4. **Court Transformation**: Apply homography using manually provided court boundary points
5. **Event Recognition**: Detect 2-point shots, 3-point shots, and assists with stable player IDs
6. **Output Generation**: Create annotated video, BEV video, and structured JSON analysis

### API Endpoints
- `POST /upload`: Upload video with optional court boundary points
- `GET /status/{job_id}`: Check processing status and progress
- `GET /download/video/{job_id}`: Download annotated video
- `GET /download/json/{job_id}`: Download analysis JSON

### File Structure
```
trackingStudio/
├── main.py              # FastAPI application
├── video_processor.py   # Video analysis pipeline
├── run.py              # Server startup script
├── client_example.py   # Example API client
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded videos
└── outputs/           # Processed results
    └── {job_id}/
        ├── annotated_video.mp4
        └── analysis.json
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

### Output JSON Structure
```json
{
  "video_info": {
    "fps": 30,
    "total_frames": 9000,
    "duration_seconds": 300
  },
  "events": [
    {
      "event_type": "2_point_attempt",
      "timestamp": 45.6,
      "frame_number": 1368,
      "player_id": "Player_1",
      "position": [640, 360],
      "confidence": 0.7
    }
  ],
  "player_tracks": {
    "Player_1": {
      "total_detections": 150,
      "positions": [...]
    }
  },
  "statistics": {
    "total_players_detected": 8,
    "event_breakdown": {
      "2_point_attempt": 12,
      "3_point_attempt": 8,
      "assist": 5
    }
  }
}
```