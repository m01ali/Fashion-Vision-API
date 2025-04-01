import os
import uuid
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, folder: str = "static/uploads") -> str:
    """
    Save an uploaded file to the specified folder with a unique filename.
    
    Args:
        upload_file: The uploaded file.
        folder: The folder to save the file to.
        
    Returns:
        The path to the saved file.
    """
    try:
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4().hex}{os.path.splitext(upload_file.filename)[1]}"
        file_path = os.path.join(folder, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)
        
        logger.info(f"File saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise