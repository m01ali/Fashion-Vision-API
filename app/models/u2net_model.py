import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

class U2NetModel:
    def __init__(self):
        # Initialize the model path
        self.model_path = self._download_model()
        
        # Add the U-2-Net repository to the Python path if it exists
        if os.path.exists('./U-2-Net'):
            sys.path.append('./U-2-Net')
        else:
            # Clone the repository if it doesn't exist
            import subprocess
            logger.info("Cloning U-2-Net repository...")
            subprocess.run(["git", "clone", "https://github.com/xuebinqin/U-2-Net.git"])
            sys.path.append('./U-2-Net')
        
        # Import the U2NET model from the repository
        from model import U2NET  # This might need adjustment based on the actual path
        
        # Initialize the model
        self.model = U2NET(3, 1)
        
        # Load the model weights
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_path))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        
        self.model.eval()
        logger.info("U-2-Net model loaded successfully")
    
    def _download_model(self):
        """Download the U2NET model weights from Hugging Face."""
        try:
            model_path = hf_hub_download(repo_id="lilpotat/pytorch3d", filename="u2net.pth")
            logger.info(f"U2NET model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def _norm_pred(self, d):
        """Normalize the prediction map to [0, 1]."""
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn
    
    def generate_silhouette(self, image_path, output_dir="static/results"):
        """Generate a silhouette for the given image and save it to the output directory."""
        try:
            # Open image
            image = Image.open(image_path).convert('RGB')
            
            # Transform the image
            transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
            image_tensor = Variable(image_tensor)
            
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            # Run inference
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = self.model(image_tensor)
            
            # Get the primary saliency map
            pred = d1[:, 0, :, :]
            pred = self._norm_pred(pred)
            pred = pred.squeeze().cpu().data.numpy()
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output filename
            basename = os.path.basename(image_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_silhouette.png")
            
            # Save the binary mask
            image = np.array(image)
            pred = (pred * 255).astype(np.uint8)
            pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
            ret, mask = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(output_path, mask)
            
            logger.info(f"Silhouette saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating silhouette: {str(e)}")
            raise