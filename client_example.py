#!/usr/bin/env python3
"""
Example client for testing the Basketball Video Analysis API
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

def upload_videos(video1_path: str, video2_path: str, 
                 court_points_1: dict = None, court_points_2: dict = None):
    """Upload two videos for dual camera processing."""
    
    url = f"{API_BASE}/upload"
    
    files = {
        'video1': ('video1.mp4', open(video1_path, 'rb'), 'video/mp4'),
        'video2': ('video2.mp4', open(video2_path, 'rb'), 'video/mp4')
    }
    
    data = {}
    if court_points_1:
        data['court_points_1'] = json.dumps(court_points_1)
    if court_points_2:
        data['court_points_2'] = json.dumps(court_points_2)
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Upload failed: {response.status_code}")
        print(response.text)
        return None

def check_status(job_id: str):
    """Check processing status."""
    
    url = f"{API_BASE}/status/{job_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Status check failed: {response.status_code}")
        return None

def download_results(job_id: str, download_dir: str = "downloads"):
    """Download processed video and JSON results."""
    
    Path(download_dir).mkdir(exist_ok=True)
    
    # Download video
    video_url = f"{API_BASE}/download/video/{job_id}"
    response = requests.get(video_url)
    
    if response.status_code == 200:
        video_path = Path(download_dir) / f"processed_video_{job_id}.mp4"
        with open(video_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded video: {video_path}")
    else:
        print(f"❌ Video download failed: {response.status_code}")
    
    # Download JSON
    json_url = f"{API_BASE}/download/json/{job_id}"
    response = requests.get(json_url)
    
    if response.status_code == 200:
        json_path = Path(download_dir) / f"analysis_{job_id}.json"
        with open(json_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded JSON: {json_path}")
    else:
        print(f"❌ JSON download failed: {response.status_code}")

def main():
    """Example usage of the API."""
    
    # Example court points for both cameras (you'll need to adjust these for your videos)
    court_points_1 = {
        "corners": [
            [100, 100],  # Top-left court corner
            [1800, 100], # Top-right court corner
            [1800, 900], # Bottom-right court corner
            [100, 900]   # Bottom-left court corner
        ]
    }
    
    court_points_2 = {
        "corners": [
            [150, 120],  # Top-left court corner (different angle)
            [1750, 150], # Top-right court corner
            [1700, 950], # Bottom-right court corner
            [120, 920]   # Bottom-left court corner
        ]
    }
    
    # Upload videos (replace with your video paths)
    video1_path = "basketball_camera1.mp4"
    video2_path = "basketball_camera2.mp4"
    
    if not Path(video1_path).exists():
        print(f"❌ Video file not found: {video1_path}")
        print("Please update video1_path with your first camera video file")
        return
    
    if not Path(video2_path).exists():
        print(f"❌ Video file not found: {video2_path}")
        print("Please update video2_path with your second camera video file")
        return
    
    print(f"🎬 Uploading videos:")
    print(f"  Camera 1: {video1_path}")
    print(f"  Camera 2: {video2_path}")
    
    result = upload_videos(video1_path, video2_path, court_points_1, court_points_2)
    
    if not result:
        return
    
    job_id = result['job_id']
    print(f"📝 Job ID: {job_id}")
    print(f"🎯 Status: {result['status']}")
    
    # Poll for completion
    print("⏳ Waiting for processing to complete...")
    
    while True:
        status = check_status(job_id)
        
        if not status:
            break
        
        print(f"📊 Progress: {status['progress']:.1f}% - {status['message']}")
        
        if status['status'] == 'completed':
            print("🎉 Processing completed!")
            break
        elif status['status'] == 'failed':
            print(f"❌ Processing failed: {status['message']}")
            return
        
        time.sleep(5)  # Wait 5 seconds before next check
    
    # Download results
    print("📥 Downloading results...")
    download_results(job_id)
    print("✅ All done!")

if __name__ == "__main__":
    main()