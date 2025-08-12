from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import json
from typing import Optional
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from ..processors.video_processor import VideoProcessor

app = FastAPI(title="Basketball Video Analysis API", version="1.0.0")

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global storage for processing status
processing_status = {}

class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    message: str
    output_video_path: Optional[str] = None
    output_json_path: Optional[str] = None

@app.post("/upload", response_model=ProcessingResponse)
async def upload_videos(
    background_tasks: BackgroundTasks,
    video1: UploadFile = File(..., description="First camera angle video"),
    video2: UploadFile = File(..., description="Second camera angle video"),
    court_points_1: str = None,
    court_points_2: str = None
):
    """
    Upload two basketball videos from different camera angles for BEV analysis.
    court_points_1: JSON string with court boundary points for first video
    court_points_2: JSON string with court boundary points for second video
    """
    
    # Validate files
    if not video1.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="video1 must be a video file")
    if not video2.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="video2 must be a video file")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    video1_path = UPLOAD_DIR / f"{job_id}_camera1.mp4"
    video2_path = UPLOAD_DIR / f"{job_id}_camera2.mp4"
    
    with open(video1_path, "wb") as buffer:
        content = await video1.read()
        buffer.write(content)
        
    with open(video2_path, "wb") as buffer:
        content = await video2.read()
        buffer.write(content)
    
    # Parse court points if provided
    court_points_data = {}
    if court_points_1:
        try:
            court_points_data['camera1'] = json.loads(court_points_1)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid court_points_1 JSON")
    
    if court_points_2:
        try:
            court_points_data['camera2'] = json.loads(court_points_2)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid court_points_2 JSON")
    
    # Initialize processing status
    processing_status[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Video uploaded and queued for processing"
    }
    
    # Start background processing
    background_tasks.add_task(
        process_videos_async, 
        job_id, 
        str(video1_path),
        str(video2_path), 
        court_points_data
    )
    
    return ProcessingResponse(
        job_id=job_id,
        status="queued",
        message="Video uploaded successfully. Processing started."
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_processing_status(job_id: str):
    """Get the processing status of a video analysis job."""
    
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    status_data = processing_status[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=status_data["status"],
        progress=status_data.get("progress"),
        message=status_data["message"],
        output_video_path=status_data.get("output_video_path"),
        output_json_path=status_data.get("output_json_path")
    )

@app.get("/download/video/{job_id}")
async def download_processed_video(job_id: str):
    """Download the processed video with annotations."""
    
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    status_data = processing_status[job_id]
    
    if status_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    video_path = status_data.get("output_video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(
        path=video_path,
        filename=f"basketball_analysis_{job_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/download/json/{job_id}")
async def download_analysis_json(job_id: str):
    """Download the analysis results as JSON."""
    
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    status_data = processing_status[job_id]
    
    if status_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    json_path = status_data.get("output_json_path")
    if not json_path or not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Analysis JSON not found")
    
    return FileResponse(
        path=json_path,
        filename=f"basketball_analysis_{job_id}.json",
        media_type="application/json"
    )

async def process_videos_async(job_id: str, video1_path: str, video2_path: str, court_points: dict = None):
    """Background task to process dual camera videos."""
    
    try:
        # Update status to processing
        processing_status[job_id].update({
            "status": "processing",
            "progress": 10.0,
            "message": "Starting dual camera video analysis..."
        })
        
        # Initialize video processor
        processor = VideoProcessor(job_id, processing_status)
        
        # Process the videos
        output_video_path, output_json_path = await processor.process_dual_videos(
            video1_path, video2_path, court_points
        )
        
        # Update status to completed
        processing_status[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Video analysis completed successfully",
            "output_video_path": output_video_path,
            "output_json_path": output_json_path
        })
        
    except Exception as e:
        # Update status to failed
        processing_status[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Processing failed: {str(e)}"
        })

@app.get("/")
async def root():
    return {"message": "Basketball Video Analysis API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)