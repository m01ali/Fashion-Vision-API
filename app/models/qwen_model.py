import torch
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import logging

logger = logging.getLogger(__name__)

# Define helper function for image to data URL conversion
def image_to_data_url(image: Image.Image, fmt: str = None):
    if fmt is None:
        if image.format and image.format.upper() == "AVIF":
            fmt = "PNG"
        else:
            fmt = "PNG"
    if image.format and image.format.upper() == "AVIF":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_str}"

# Helper function to process vision information
def process_vision_info(messages):
    image_inputs, video_inputs = [], []
    for message in messages:
        if message["role"] != "assistant":
            for content in message["content"]:
                if content["type"] == "image":
                    image_url = content.get("image_url", "")
                    
                    # Handle data URL format
                    if image_url.startswith("data:"):
                        image_data = image_url.split(",")[1]
                        image = Image.open(BytesIO(base64.b64decode(image_data)))
                        image_inputs.append(image)
    return image_inputs, video_inputs

class Qwen2VLModel:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        """Initialize the Qwen2VL model."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_id = model_id
            logger.info(f"Loading Qwen2VL model {model_id} on {self.device}...")
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",
                offload_state_dict=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Qwen2VL model loaded successfully")
            
            # Define the allowed fashion patterns
            self.allowed_patterns = [
                "Houndstooth", "Herringbone", "Chevron", "Paisley", "Plaid (Tartan)", "Gingham", "Argyle",
                "Polka Dots", "Stripes (Vertical, Horizontal, Diagonal)", "Pin Stripes", "Windowpane", "Tattersall",
                "Buffalo Check", "Madras", "Seersucker", "Moroccan Tile Patterns", "Greek Key", "Zigzag",
                "Diamond Patterns", "Kaleidoscope Patterns", "Leopard Print", "Zebra Print", "Tiger Print", "Snake Print",
                "Giraffe Print", "Cheetah Print", "Aztec Patterns", "Graffiti Prints", "Cubist Patterns", "Camouflage"
            ]
            self.patterns_str = ", ".join(self.allowed_patterns)
            
        except Exception as e:
            logger.error(f"Error initializing Qwen2VL model: {str(e)}")
            raise
    
    def run_example(self, task_prompt: str, image: Image.Image, max_new_tokens=256, **kwargs):
        """Run a single example with the model."""
        try:
            data_url = image_to_data_url(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_url": data_url},
                        {"type": "text", "text": task_prompt}
                    ]
                }
            ]
            
            input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[input_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            logger.error(f"Error running example: {str(e)}")
            raise
    
    def get_logo_name(self, image_path):
        """Detect and return the logo name from an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = (
                "Detect the logo in the clothing based on any text or famous symbol you find on any piece of garment. "
                "What is the full brand name of this logo? "
                "Only give the brand name (no special characters or extra words)"
            )
            return self.run_example(prompt, image=image)
        except Exception as e:
            logger.error(f"Error detecting logo: {str(e)}")
            raise
    
    def analyze_garments(self, image_path):
        """Analyze garments and their colors from an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = (
                "Analyze the image and list every garment worn by the model, along with its predominant color(s). "
                "Return response in json format. "
                "Example format: {\"shirt\": \"white\", \"jeans\": \"grey\",...} "
                "Strictly don't return any other information other than format defined! "
                "Give the response like example defined"
            )
            return self.run_example(prompt, image=image, max_new_tokens=100)
        except Exception as e:
            logger.error(f"Error analyzing garments: {str(e)}")
            raise
    
    def analyze_fashion_patterns(self, image_path):
        """Analyze fashion patterns in an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = (
                "Analyze the image and ONLY list garments worn by the model having fashion pattern. "
                f"Identify the fashion pattern on the outfit from the following list: {self.patterns_str}. "
                "Return response in json format. "
                "Example format: {\"shirt\": \"Chevron\", \"skirt\": \"plaid\",...} "
                "Return {\"pattern\": \"None\"} if you can't find any pattern. "
                "Strictly don't return any other information other than format defined! "
                "Give the response like example defined"
            )
            return self.run_example(prompt, image=image, max_new_tokens=100)
        except Exception as e:
            logger.error(f"Error analyzing fashion patterns: {str(e)}")
            raise