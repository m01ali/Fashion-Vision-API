from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Optional
import os
import logging
import json
from app.models.qwen_model import Qwen2VLModel
from app.models.u2net_model import U2NetModel
from app.utils.file_utils import save_upload_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize models
try:
    qwen_model = Qwen2VLModel()
    u2net_model = U2NetModel()
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    # We'll handle this in the routes

class AnalysisRequest(BaseModel):
    image_path: str
    analyze_logo: bool = False
    analyze_garments: bool = False
    analyze_patterns: bool = False
    generate_silhouette: bool = False

class AnalysisResponse(BaseModel):
    image_path: str
    logo: Optional[str] = None
    garments: Optional[Dict] = None
    patterns: Optional[Dict] = None
    silhouette_path: Optional[str] = None

@router.post("/upload", response_model=AnalysisResponse)
async def upload_image(
    file: UploadFile = File(...),
    analyze_logo: bool = False,
    analyze_garments: bool = False,
    analyze_patterns: bool = False,
    generate_silhouette: bool = False
):
    """
    Upload an image for fashion analysis.
    
    - analyze_logo: Whether to detect logo in the image
    - analyze_garments: Whether to analyze garments and colors
    - analyze_patterns: Whether to analyze fashion patterns
    - generate_silhouette: Whether to generate a silhouette of the image
    """
    try:
        # Save the uploaded file
        image_path = await save_upload_file(file)
        
        # Initialize response
        response = AnalysisResponse(image_path=image_path)
        
        # Perform requested analyses
        if analyze_logo:
            response.logo = qwen_model.get_logo_name(image_path)
        
        if analyze_garments:
            garments_json = qwen_model.analyze_garments(image_path)
            try:
                response.garments = json.loads(garments_json)
            except json.JSONDecodeError:
                # Handle case where model didn't return valid JSON
                response.garments = {"error": "Unable to parse garments data"}
        
        if analyze_patterns:
            patterns_json = qwen_model.analyze_fashion_patterns(image_path)
            try:
                response.patterns = json.loads(patterns_json)
            except json.JSONDecodeError:
                # Handle case where model didn't return valid JSON
                response.patterns = {"error": "Unable to parse patterns data"}
        
        if generate_silhouette:
            response.silhouette_path = u2net_model.generate_silhouette(image_path)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(request: AnalysisRequest):
    """
    Analyze an already uploaded image.
    """
    try:
        # Verify the image exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Initialize response
        response = AnalysisResponse(image_path=request.image_path)
        
        # Perform requested analyses
        if request.analyze_logo:
            response.logo = qwen_model.get_logo_name(request.image_path)
        
        if request.analyze_garments:
            garments_json = qwen_model.analyze_garments(request.image_path)
            try:
                response.garments = json.loads(garments_json)
            except json.JSONDecodeError:
                response.garments = {"error": "Unable to parse garments data"}
        
        if request.analyze_patterns:
            patterns_json = qwen_model.analyze_fashion_patterns(request.image_path)
            try:
                response.patterns = json.loads(patterns_json)
            except json.JSONDecodeError:
                response.patterns = {"error": "Unable to parse patterns data"}
        
        if request.generate_silhouette:
            response.silhouette_path = u2net_model.generate_silhouette(request.image_path)
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))