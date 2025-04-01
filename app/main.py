from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from app.routers import image_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Fashion Vision API",
    description="API for fashion image analysis using Qwen2VL and U-2-Net models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(image_analysis.router, prefix="/api/image", tags=["Image Analysis"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Fashion Vision API",
        "documentation": "/docs",
        "available_endpoints": [
            "/api/image/upload",
            "/api/image/analyze"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# If running as main script
if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    
    # Run the server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)