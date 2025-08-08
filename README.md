# 🏀 Basketball Video Analysis Studio

A FastAPI-based system for analyzing basketball videos from dual camera angles, featuring Bird's Eye View (BEV) transformation, player/ball tracking, and basketball event detection.

## 🎯 Features

- **Dual Camera Processing**: Upload videos from two camera angles simultaneously
- **Bird's Eye View Generation**: Transform dual camera feeds into a top-down court view
- **Player & Ball Tracking**: YOLO-based detection and tracking across frames
- **Basketball Event Recognition**: Detect 2-point shots, 3-point shots, and assists
- **Multiple Output Formats**: 
  - Side-by-side annotated video (both cameras)
  - Bird's Eye View video with court overlay
  - Structured JSON data for AI analysis
- **Async Processing**: Non-blocking video processing with status tracking
- **RESTful API**: Easy integration with web interfaces or other applications

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd trackingStudio

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python run.py
```

The API will be available at:
- **Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive Testing**: http://localhost:8000/docs

### 3. Upload Videos

#### Using the Web Interface
1. Go to http://localhost:8000/docs
2. Find the `/upload` endpoint
3. Upload your two basketball video files
4. Optionally provide court boundary points for better accuracy

#### Using Python Client
```python
python client_example.py
```

#### Using curl
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "video1=@camera1.mp4" \
  -F "video2=@camera2.mp4" \
  -F "court_points_1={\"corners\": [[100,100], [1800,100], [1800,900], [100,900]]}" \
  -F "court_points_2={\"corners\": [[150,120], [1750,150], [1700,950], [120,920]]}"
```

## 📋 API Usage

### 1. Upload Videos
**POST** `/upload`

Upload two basketball videos from different camera angles.

**Parameters:**
- `video1`: First camera angle video file (MP4)
- `video2`: Second camera angle video file (MP4)
- `court_points_1` (optional): JSON string with court boundary points for camera 1
- `court_points_2` (optional): JSON string with court boundary points for camera 2

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Videos uploaded successfully. Processing started."
}
```

### 2. Check Processing Status
**GET** `/status/{job_id}`

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "progress": 45.2,
  "message": "Processed 1350/3000 dual camera frames...",
  "output_video_path": null,
  "output_json_path": null
}
```

Status values: `queued`, `processing`, `completed`, `failed`

### 3. Download Results
**GET** `/download/video/{job_id}` - Download annotated video
**GET** `/download/json/{job_id}` - Download analysis JSON

## 🎬 Video Requirements

### Supported Formats
- **Format**: MP4 (recommended)
- **Resolution**: 1080p (1920x1080)
- **Frame Rate**: 30fps
- **Duration**: Any length (both videos should be similar duration)

### Camera Setup
The system is designed for elevated camera angles as shown in the example screenshots:

![Camera Angle 1](Screenshot%202025-08-06%20at%209.20.54%20PM.png)
![Camera Angle 2](Screenshot%202025-08-06%20at%209.20.17%20PM.png)

**Best Practices:**
- Use two cameras positioned at different corners of the court
- Ensure both cameras capture the entire basketball court
- Maintain consistent lighting across both views
- Synchronize video recording start times

## 🎯 Court Boundary Setup

For accurate Bird's Eye View transformation, provide court boundary points for each camera:

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

### How to Find Court Points
1. Open your video in any video player
2. Pause on a clear frame showing the court
3. Note the pixel coordinates of the four court corners
4. Order them: top-left, top-right, bottom-right, bottom-left

**Example for 1920x1080 video:**
```json
{
  "corners": [
    [245, 185],   // Top-left court corner
    [1675, 185],  // Top-right court corner
    [1890, 895],  // Bottom-right court corner
    [30, 895]     // Bottom-left court corner
  ]
}
```

## 📊 Output Files

### 1. Annotated Video (`annotated_video.mp4`)
- Side-by-side view of both cameras
- Bounding boxes around detected players and ball
- Player IDs and confidence scores
- Timestamp overlay

### 2. Bird's Eye View Video (`bev_video.mp4`)
- Top-down basketball court visualization
- Player positions from both cameras merged
- Basketball court lines and markings
- Color-coded players by camera source

### 3. Analysis JSON (`analysis.json`)
Structure optimized for AI analysis:

```json
{
  "video_info": {
    "fps": 30,
    "total_frames": 9000,
    "duration_seconds": 300,
    "processing_timestamp": "2025-08-08T10:30:00"
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
      "first_appearance": 0.5,
      "last_appearance": 299.8,
      "positions": [...]
    }
  },
  "statistics": {
    "total_players_detected": 8,
    "total_events": 25,
    "event_breakdown": {
      "2_point_attempt": 12,
      "3_point_attempt": 8,
      "assist": 5
    }
  }
}
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `UPLOAD_DIR`: Upload directory (default: uploads/)
- `OUTPUT_DIR`: Output directory (default: outputs/)

### Processing Settings
Modify these in `video_processor.py`:
- `PLAYER_CONFIDENCE_THRESHOLD`: 0.5 (minimum confidence for player detection)
- `BALL_CONFIDENCE_THRESHOLD`: 0.3 (minimum confidence for ball detection)
- `BEV_RESOLUTION`: (800, 600) (Bird's Eye View output size)

## 🚀 Advanced Usage

### Batch Processing
```python
import requests
import time

def process_multiple_games(game_videos):
    job_ids = []
    
    for video_pair in game_videos:
        result = upload_videos(video_pair['camera1'], video_pair['camera2'])
        if result:
            job_ids.append(result['job_id'])
    
    # Monitor all jobs
    completed_jobs = []
    while len(completed_jobs) < len(job_ids):
        for job_id in job_ids:
            if job_id not in completed_jobs:
                status = check_status(job_id)
                if status and status['status'] == 'completed':
                    completed_jobs.append(job_id)
                    download_results(job_id)
        
        time.sleep(10)  # Check every 10 seconds
```

### Custom Event Detection
Extend the `VideoProcessor` class to detect custom basketball events:

```python
def _detect_custom_event(self, player_id, x, y, timestamp, frame_number):
    # Your custom detection logic here
    if self._is_dunk_attempt(x, y):
        event = BasketballEvent(
            "dunk_attempt", timestamp, frame_number,
            player_id, (x, y), 0.8
        )
        self.events.append(event)
```

## 📝 File Structure

```
trackingStudio/
├── main.py                 # FastAPI application
├── video_processor.py      # Video analysis pipeline
├── run.py                 # Server startup script
├── client_example.py      # Example API client
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── CLAUDE.md             # Development documentation
├── README.md             # This file
├── uploads/              # Uploaded videos (gitignored)
├── outputs/              # Processing results (gitignored)
│   └── {job_id}/
│       ├── annotated_video.mp4
│       ├── bev_video.mp4
│       └── analysis.json
└── downloads/            # Client downloads (gitignored)
```

## 🎯 Basketball Events Detected

### 2-Point Shots
- Detected when player is inside 3-point line
- Based on court position analysis via homography
- Confidence scoring based on player-ball proximity

### 3-Point Shots  
- Detected when player is beyond 3-point line
- Higher confidence for clear positioning
- Considers court geometry and player stance

### Assists
- Detected on ball possession changes
- Tracks ball movement between players
- Temporal analysis for pass-to-shot sequences

## 🔍 Troubleshooting

### Common Issues

**1. "Could not open video file"**
- Ensure video files are valid MP4 format
- Check file paths are correct
- Verify files aren't corrupted

**2. "Processing failed with homography error"**
- Check court boundary points are valid
- Ensure points form a proper quadrilateral
- Verify coordinate values are within video dimensions

**3. "Low detection confidence"**
- Improve lighting in source videos
- Ensure basketball court is clearly visible
- Consider adjusting confidence thresholds

**4. "Memory issues during processing"**
- Reduce video resolution if possible
- Process shorter video segments
- Monitor system RAM usage

### Performance Optimization

**For faster processing:**
- Use smaller input videos (720p instead of 1080p)
- Reduce frame rate if timing precision isn't critical
- Process shorter video clips for testing

**For better accuracy:**
- Provide accurate court boundary points
- Use high-quality source videos
- Ensure good lighting and contrast

## 📚 API Documentation

Once the server is running, visit http://localhost:8000/docs for complete interactive API documentation with:
- Request/response schemas
- Try-it-now functionality
- Parameter descriptions
- Example requests and responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Next Steps for AI Analysis

The generated JSON output is specifically structured for feeding into multimodal AI models like Gemini 2.5 Pro:

1. **Upload the original videos and JSON** to your AI model
2. **Use the structured event data** for statistical analysis
3. **Generate play-by-play commentary** using event timestamps
4. **Create highlight reels** based on detected events
5. **Analyze player performance** using tracking data

**Example AI prompt:**
```
Analyze this basketball game using the provided video files and JSON analysis data. 
Generate a comprehensive game summary including:
- Key plays and highlights
- Player performance statistics  
- Team strategy analysis
- Game flow and momentum shifts
```