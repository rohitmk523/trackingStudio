#!/usr/bin/env python3

import uvicorn
import sys
import os

if __name__ == "__main__":
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("🏀 Starting Basketball Video Analysis API...")
    print("📁 Upload directory: uploads/")
    print("📁 Output directory: outputs/")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("\n" + "="*50)
    
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)