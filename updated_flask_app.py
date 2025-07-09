from flask import Flask, request, render_template_string, jsonify, send_file
import os
import io
import sys
import base64
import numpy as np
from PIL import Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import threading
import uuid
from datetime import datetime, timedelta
import zipfile
import math
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

@app.route("/ping")
def ping():
    return "OK", 200

# Global variables
upsampler = None
processing_status = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get absolute directory of the current file (safe for local + deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative-safe paths for your models
MODEL_PATHS = {
    "RealESRGAN_x4plus": os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth"),
    "NAFNet-GoPro-width64": os.path.join(BASE_DIR, "weights", "NAFNet-GoPro-width64.pth"),
    # Add more if needed
}

for model_name, model_path in MODEL_PATHS.items():
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

class CustomRealESRGANer(RealESRGANer):
    """Custom RealESRGANer with tile progress tracking"""
    def __init__(self, scale, model_path, model=None, tile=0, tile_pad=10, pre_pad=0, half=False, device=None, gpu_id=None):
        # MODIFICATION: Call parent with explicit parameters instead of *args, **kwargs
        super().__init__(
            scale=scale,
            model_path=model_path, 
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            device=device,
            gpu_id=gpu_id
        )
        
        # MODIFICATION: Explicitly store tile parameters as instance attributes
        self.tile = tile
        self.tile_pad = tile_pad
        self.scale = scale
        
        # Initialize custom tracking attributes
        self.current_task_id = None
        self.total_tiles = 0
        self.processed_tiles = 0
    
    # MODIFICATION: Ensure set_task_id method is properly defined
    def set_task_id(self, task_id):
        """Set the current task ID for progress tracking"""
        self.current_task_id = task_id
        self.processed_tiles = 0
    
    def _update_tile_progress(self):
        """Update tile progress in processing status"""
        if self.current_task_id and self.current_task_id in processing_status:
            tile_progress = (self.processed_tiles / self.total_tiles) * 100 if self.total_tiles > 0 else 0
            processing_status[self.current_task_id].update({
                "tiles_processed": self.processed_tiles,
                "total_tiles": self.total_tiles,
                "tile_progress": tile_progress
            })
    
    def _log_to_cmd(self, message):
        """Log message to CMD/terminal"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        sys.stdout.flush()
        
        # Also update processing status with latest log
        if self.current_task_id and self.current_task_id in processing_status:
            if "logs" not in processing_status[self.current_task_id]:
                processing_status[self.current_task_id]["logs"] = []
            processing_status[self.current_task_id]["logs"].append(formatted_message)
            # Keep only last 10 logs
            processing_status[self.current_task_id]["logs"] = processing_status[self.current_task_id]["logs"][-10:]

    def _enhance_single(self, img):
        """Process single image without tiling"""
        self._log_to_cmd("🎯 Processing single tile (no tiling needed)")
        
        # MODIFICATION: Use the parent class's enhance method for single image processing
        # This avoids duplicating the complex image processing logic
        try:
            # Temporarily disable tiling for single image processing
            original_tile = self.tile
            self.tile = 0  # Disable tiling
            
            # Call parent's enhance method
            output, img_mode = super().enhance(img)
            
            # Restore original tile setting
            self.tile = original_tile
            
            return output
        except Exception as e:
            # Restore original tile setting in case of error
            self.tile = original_tile
            raise e

    def _process_with_tiling(self, img, tile_size, overlap):
        """Process image with tiling and progress tracking"""
        # MODIFICATION: Use parent class's enhance method with proper tiling
        # Set the tile size temporarily
        original_tile = self.tile
        self.tile = tile_size
        
        try:
            # Call parent's enhance method which handles tiling
            output, img_mode = super().enhance(img)
            
            # Update progress as we go
            self.processed_tiles = self.total_tiles
            self._update_tile_progress()
            
            # Restore original tile setting
            self.tile = original_tile
            
            return output
        except Exception as e:
            # Restore original tile setting in case of error
            self.tile = original_tile
            raise e
        
    def enhance(self, img, outscale=None, alpha_upsampler=None):
        """Enhanced method with tile progress tracking"""
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        h, w = img.shape[0:2]
        # MODIFICATION: Use self.tile instead of max(self.tile, 512)
        tile_threshold = max(self.tile, 512) if self.tile > 0 else 512
        
        if max(h, w) <= tile_threshold:
            # No tiling needed
            self.total_tiles = 1
            self.processed_tiles = 0
            self._update_tile_progress()
            
            output = self._enhance_single(img)
            
            self.processed_tiles = 1
            self._update_tile_progress()
            
            return output, img_mode
        else:
            # Calculate tiles
            # MODIFICATION: Ensure tile_size is properly set
            tile_size = self.tile if self.tile > 0 else 256
            overlap = self.tile_pad
            
            tiles_x = math.ceil(w / (tile_size - overlap))
            tiles_y = math.ceil(h / (tile_size - overlap))
            self.total_tiles = tiles_x * tiles_y
            self.processed_tiles = 0
            
            self._log_to_cmd(f"🔢 Processing {self.total_tiles} tiles ({tiles_x}x{tiles_y})")
            self._update_tile_progress()
            
            # Process with tiling
            output = self._process_with_tiling(img, tile_size, overlap)
            
            return output, img_mode

    
    def _enhance_single(self, img):
        """Process single image without tiling"""
        self._log_to_cmd("🎯 Processing single tile (no tiling needed)")
        
        img = img.astype(np.float32)
        if np.max(img) > 256:
            max_range = 65535
            img /= 65535
        else:
            max_range = 255
            img /= 255

        if len(img.shape) == 2:
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img

        # Convert to tensor and process
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()

        with torch.no_grad():
            output = self.model(img)
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        
        if max_range == 65535:
            output = (output * 65535.0).round().astype(np.uint16)
        else:
            output = (output * 255.0).round().astype(np.uint8)
        
        return output
    
    def _process_with_tiling(self, img, tile_size, overlap):
        """Process image with tiling and progress tracking"""
        h, w, c = img.shape
        output_h = h * self.scale
        output_w = w * self.scale
        output = np.zeros((output_h, output_w, c), dtype=img.dtype)
        
        # Calculate step size
        step_h = tile_size - overlap
        step_w = tile_size - overlap
        
        tile_count = 0
        
        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                tile_count += 1
                
                # Extract tile
                tile_h = min(tile_size, h - y)
                tile_w = min(tile_size, w - x)
                
                tile = img[y:y+tile_h, x:x+tile_w, :]
                
                self._log_to_cmd(f"🔄 Processing tile {tile_count}/{self.total_tiles} at ({x}, {y}) size {tile_w}x{tile_h}")
                
                # Process tile
                enhanced_tile = self._enhance_single(tile)
                
                # Place in output
                out_y = y * self.scale
                out_x = x * self.scale
                out_h = enhanced_tile.shape[0]
                out_w = enhanced_tile.shape[1]
                
                output[out_y:out_y+out_h, out_x:out_x+out_w, :] = enhanced_tile
                
                self.processed_tiles += 1
                self._update_tile_progress()
                
                self._log_to_cmd(f"✅ Completed tile {tile_count}/{self.total_tiles}")
        
        return output
    
    def _update_tile_progress(self):
        """Update tile progress in processing status"""
        if self.current_task_id and self.current_task_id in processing_status:
            tile_progress = (self.processed_tiles / self.total_tiles) * 100 if self.total_tiles > 0 else 0
            processing_status[self.current_task_id].update({
                "tiles_processed": self.processed_tiles,
                "total_tiles": self.total_tiles,
                "tile_progress": tile_progress
            })
    
    def _log_to_cmd(self, message):
        """Log message to CMD/terminal"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        sys.stdout.flush()
        
        # Also update processing status with latest log
        if self.current_task_id and self.current_task_id in processing_status:
            if "logs" not in processing_status[self.current_task_id]:
                processing_status[self.current_task_id]["logs"] = []
            processing_status[self.current_task_id]["logs"].append(formatted_message)
            # Keep only last 10 logs
            processing_status[self.current_task_id]["logs"] = processing_status[self.current_task_id]["logs"][-10:]

def setup_model(model_name="RealESRGAN_x4plus", scale=4, tile_size=256, tile_pad=10, half_precision=False):
    """Setup the RealESRGAN model with your weights"""
    global upsampler
    
    try:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🔧 Loading model: {model_name}")
        print(f"📁 Model path: {model_path}")
        print(f"⚙️  Scale: {scale}x, Tile: {tile_size}, Pad: {tile_pad}")
        
        # Load RRDBNet model architecture
        if "anime" in model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6,
                num_grow_ch=32, scale=scale
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=scale
            )
        
        # MODIFICATION: Ensure tile_size is passed correctly and handle zero values
        actual_tile_size = tile_size if tile_size > 0 else 0
        
        # MODIFICATION: Create CustomRealESRGANer with explicit parameters
        upsampler = CustomRealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=actual_tile_size,        # Make sure tile_size is passed as 'tile'
            tile_pad=tile_pad,
            pre_pad=0,
            half=half_precision and torch.cuda.is_available(),
            device=device,
            gpu_id=None           # Add explicit gpu_id parameter
        )
        
        print(f"✅ Model loaded successfully on {device}")
        return True, f"Model loaded successfully: {model_name} on {device}"
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        return False, error_msg

def process_image_thread(task_id, image_data, settings):
    """Process image in a separate thread"""
    global processing_status, upsampler
    
    try:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING IMAGE ENHANCEMENT - Task ID: {task_id}")
        print(f"{'='*60}")
        
        processing_status[task_id] = {
            "progress": 10, 
            "status": "Initializing...", 
            "tiles_processed": 0, 
            "total_tiles": 0,
            "tile_progress": 0,
            "logs": []
        }
        
        # Convert base64 to PIL Image
        print("📥 Decoding input image...")
        image_bytes = base64.b64decode(image_data.split(',')[1])
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(input_image)
        
        print(f"📊 Input image size: {input_image.width}x{input_image.height}")
        
        processing_status[task_id].update({"progress": 30, "status": "Setting up model..."})
        
        # Setup model with new settings if needed
        print("🔧 Setting up model with new settings...")
        setup_success, setup_message = setup_model(
            model_name=settings.get('model', 'RealESRGAN_x4plus'),
            scale=settings.get('scale', 4),
            tile_size=settings.get('tile_size', 256),
            tile_pad=settings.get('tile_pad', 10),
            half_precision=settings.get('half_precision', False)
        )
        
        if not setup_success:
            processing_status[task_id] = {"progress": 0, "status": f"Error: {setup_message}", "error": True}
            return
        
        processing_status[task_id].update({"progress": 50, "status": "Enhancing image..."})
        
        # Set task ID for tile progress tracking
        upsampler.set_task_id(task_id)
        
        print("✨ Starting image enhancement...")
        
        # Enhance the image using your weights
        output, _ = upsampler.enhance(img_np)
        
        processing_status[task_id].update({"progress": 80, "status": "Converting result..."})
        
        print("🔄 Converting result to base64...")
        
        # Convert result to base64
        output_image = Image.fromarray(output)
        buffer = io.BytesIO()
        output_image.save(buffer, format='PNG')
        output_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        print(f"📊 Output image size: {output_image.width}x{output_image.height}")
        print("🎉 Enhancement completed successfully!")
        
        processing_status[task_id].update({
            "progress": 100, 
            "status": "Complete!", 
            "result": f"data:image/png;base64,{output_base64}",
            "input_size": f"{input_image.width}x{input_image.height}",
            "output_size": f"{output_image.width}x{output_image.height}"
        })
        
        print(f"{'='*60}")
        print(f"✅ ENHANCEMENT COMPLETED - Task ID: {task_id}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        print(error_msg)
        processing_status[task_id] = {"progress": 0, "status": error_msg, "error": True}

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Real-ESRGAN Pro Server</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --secondary-color: #06b6d4;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --glass-bg: rgba(30, 41, 59, 0.8);
            --glass-border: rgba(148, 163, 184, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
        }
        
        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
            z-index: -1;
            animation: backgroundShift 20s ease-in-out infinite;
        }
        
        @keyframes backgroundShift {
            0%, 100% { transform: translateX(0) translateY(0); }
            33% { transform: translateX(-20px) translateY(-20px); }
            66% { transform: translateX(20px) translateY(-10px); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
        }
        
        .header h1 {
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
            animation: titleGlow 3s ease-in-out infinite alternate;
        }
        
        @keyframes titleGlow {
            from { filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3)); }
            to { filter: drop-shadow(0 0 30px rgba(6, 182, 212, 0.5)); }
        }
        
        .header p {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 300;
        }
        
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .status-card {
            background: linear-gradient(135deg, var(--success-color), #059669);
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        }
        
        .status-card.error {
            background: linear-gradient(135deg, var(--danger-color), #dc2626);
            box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
        }
        
        .image-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .image-container {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 2px dashed var(--border-color);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            min-height: 350px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .image-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(99, 102, 241, 0.05), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .image-container:hover::before {
            opacity: 1;
        }
        
        .image-container h3 {
            font-size: 1.3rem;
            margin-bottom: 20px;
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 250px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease;
        }
        
        .image-container img:hover {
            transform: scale(1.05);
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .form-group label {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .form-group select, .form-group input {
            padding: 12px 16px;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            background: var(--card-bg);
            color: var(--text-primary);
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-group select:focus, .form-group input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .file-input {
            margin-bottom: 25px;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .file-input input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-input-label {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 20px;
            border: 2px dashed var(--primary-color);
            border-radius: 15px;
            background: rgba(99, 102, 241, 0.05);
            color: var(--text-primary);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-input-label:hover {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary-dark);
        }
        
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
            margin: 8px;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, var(--secondary-color), #0891b2);
            color: white;
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3);
        }
        
        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(6, 182, 212, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .progress-section {
            display: none;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .progress-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: var(--card-bg);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .stat-number {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            width: 0%;
            transition: width 0.3s ease;
            position: relative;
        }
        
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .tile-progress-bar {
            height: 6px;
            background: linear-gradient(90deg, var(--success-color), #059669);
        }
        
        .logs-section {
            background: var(--dark-bg);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
        }
        
        .log-entry {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            padding: 2px 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .log-entry:last-child {
            border-bottom: none;
        }
        
        .button-group {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }
        
        @media (max-width: 768px) {
            .image-section {
                grid-template-columns: 1fr;
            }
            
            .controls-grid {
                grid-template-columns: 1fr;
            }
            
            .progress-stats {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .button-group {
                flex-direction: column;
            }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--card-bg);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-dark);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Real-ESRGAN Pro Server</h1>
            <p>Professional AI Image Enhancement with Real-Time Tile Progress</p>
        </div>
        
        <div class="status-card" id="server-status">
            🔄 Checking server status...
        </div>
        
        <div class="image-section">
            <div class="image-container">
                <h3>📥 Input Image</h3>
                <div id="input-preview">
                    <p>Drag & drop or click to select an image</p>
                </div>
            </div>
            
            <div class="image-container">
                <h3>✨ Enhanced Image</h3>
                <div id="output-preview">
                    <p>Enhanced image will appear here</p>
                </div>
            </div>
        </div>
        
        <div class="glass-card">
            <div class="file-input">
                <div class="file-input-wrapper">
                    <input type="file" id="imageInput" accept="image/*">
                    <label for="imageInput" class="file-input-label">
                        📁 Choose Image File
                    </label>
                </div>
            </div>
            
            <div class="controls-grid">
                <div class="form-group">
                    <label>🤖 AI Model</label>
                    <select id="model">
                        <option value="RealESRGAN_x4plus">RealESRGAN x4plus (General)</option>
                        <option value="RealESRGAN_x4plus_anime">RealESRGAN x4plus (Anime)</option>
                        <option value="NAFNet-GoPro-width64">NAFNet GoPro (Deblur)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>🔍 Scale Factor</label>
                    <select id="scale">
                        <option value="2">2x Enhancement</option>
                        <option value="4" selected>4x Enhancement</option>
                        <option value="8">8x Enhancement</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>🧩 Tile Size</label>
                    <input type="number" id="tileSize" value="256" min="128" max="1024" step="64">
                </div>
                
                <div class="form-group">
                    <label>📏 Tile Padding</label>
                    <input type="number" id="tilePad" value="10" min="0" max="50">
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" id="processBtn" disabled>
                    🚀 Enhance Image
                </button>
                <button class="btn btn-secondary" id="downloadBtn" disabled>
                    💾 Download Result
                </button>
            </div>
        </div>
        
        <div class="glass-card progress-section" id="progressSection">
            <div class="progress-header">
                <h3>🔄 Processing Status</h3>
                <div id="progressText">0%</div>
            </div>
            
            <div class="progress-stats">
                <div class="stat-card">
                    <div class="stat-number" id="overallProgress">0%</div>
                    <div class="stat-label">Overall Progress</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="tilesProcessed">0/0</div>
                    <div class="stat-label">Tiles Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="tileProgress">0%</div>
                    <div class="stat-label">Tile Progress</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="estimatedTime">--</div>
                    <div class="stat-label">Est. Time</div>
                </div>
            </div>
            
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Overall Progress</span>
                    <span id="statusText">Initializing...</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <div style="margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Tile Processing</span>
                    <span id="tileStatusText">Waiting...</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill tile-progress-bar" id="tileProgressFill"></div>
                </div>
            </div>
            
            <div class="logs-section" id="logsSection">
                <div style="margin-bottom: 10px; font-weight: 600; color: var(--text-primary);">
                    📋 Processing Logs
                </div>
                <div id="logEntries">
                    <div class="log-entry">Waiting for processing to start...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let resultImage = null;
        let startTime = null;
        let progressInterval = null;
        
        // Check server status
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                const statusCard = document.getElementById('server-status');
                statusCard.innerHTML = `
                    ✅ Server Ready | 
                    Device: ${data.device.toUpperCase()} | 
                    GPU: ${data.gpu_available ? '🟢 Available' : '🔴 Not Available'} |
                    Models: ${data.models_available.length} Loaded
                `;
                statusCard.classList.remove('error');
            })
            .catch(() => {
                const statusCard = document.getElementById('server-status');
                statusCard.innerHTML = '❌ Server Connection Error - Please check if the server is running';
                statusCard.classList.add('error');
            });
        
        // Handle file input with drag & drop
        const imageInput = document.getElementById('imageInput');
        const inputPreview = document.getElementById('input-preview');
        const fileLabel = document.querySelector('.file-input-label');
        
        // Drag and drop handlers
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            inputPreview.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            inputPreview.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            inputPreview.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            inputPreview.style.background = 'rgba(99, 102, 241, 0.1)';
            inputPreview.style.borderColor = 'var(--primary-color)';
        }
        
        function unhighlight(e) {
            inputPreview.style.background = '';
            inputPreview.style.borderColor = '';
        }
        
        inputPreview.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleImageFile(files[0]);
            }
        }
        
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleImageFile(file);
            }
        });
        
        function handleImageFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file');
                return;
            }
            
            if (file.size > 50 * 1024 * 1024) { // 50MB
                alert('File size too large. Maximum 50MB allowed.');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.style.maxWidth = '100%';
                img.style.maxHeight = '250px';
                img.style.borderRadius = '15px';
                img.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.5)';
                
                inputPreview.innerHTML = '';
                inputPreview.appendChild(img);
                
                // Update file label
                fileLabel.innerHTML = `✅ ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                fileLabel.style.background = 'rgba(16, 185, 129, 0.1)';
                fileLabel.style.borderColor = 'var(--success-color)';
                
                document.getElementById('processBtn').disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        // Process image
        document.getElementById('processBtn').addEventListener('click', function() {
            if (!imageInput.files[0]) {
                alert('Please select an image first');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const settings = {
                    model: document.getElementById('model').value,
                    scale: parseInt(document.getElementById('scale').value),
                    tile_size: parseInt(document.getElementById('tileSize').value),
                    tile_pad: parseInt(document.getElementById('tilePad').value),
                    half_precision: false
                };
                
                // Show progress section
                document.getElementById('progressSection').style.display = 'block';
                document.getElementById('processBtn').disabled = true;
                document.getElementById('downloadBtn').disabled = true;
                
                // Reset progress
                resetProgress();
                startTime = Date.now();
                
                console.log('🚀 Starting image enhancement...');
                console.log('📊 Settings:', settings);
                
                fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: e.target.result,
                        settings: settings
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.task_id) {
                        currentTaskId = data.task_id;
                        console.log('✅ Task started with ID:', currentTaskId);
                        checkProgress();
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                })
                .catch(error => {
                    console.error('❌ Error starting processing:', error);
                    alert('Error starting processing: ' + error);
                    resetUI();
                });
            };
            reader.readAsDataURL(imageInput.files[0]);
        });
        
        function resetProgress() {
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('tileProgressFill').style.width = '0%';
            document.getElementById('progressText').textContent = '0%';
            document.getElementById('overallProgress').textContent = '0%';
            document.getElementById('tilesProcessed').textContent = '0/0';
            document.getElementById('tileProgress').textContent = '0%';
            document.getElementById('estimatedTime').textContent = '--';
            document.getElementById('statusText').textContent = 'Initializing...';
            document.getElementById('tileStatusText').textContent = 'Waiting...';
            document.getElementById('logEntries').innerHTML = '<div class="log-entry">Starting processing...</div>';
        }
        
        function resetUI() {
            document.getElementById('progressSection').style.display = 'none';
            document.getElementById('processBtn').disabled = false;
            currentTaskId = null;
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
        }
        
        // Check processing progress
        function checkProgress() {
            if (!currentTaskId) return;
            
            fetch(`/progress/${currentTaskId}`)
                .then(response => response.json())
                .then(data => {
                    updateProgress(data);
                    
                    if (data.result) {
                        // Processing complete
                        handleCompletedProcessing(data);
                    } else if (data.error) {
                        // Error occurred
                        console.error('❌ Processing error:', data.status);
                        alert('Processing error: ' + data.status);
                        resetUI();
                    } else if (data.progress < 100) {
                        // Still processing
                        setTimeout(checkProgress, 1000);
                    }
                })
                .catch(error => {
                    console.error('❌ Error checking progress:', error);
                    setTimeout(checkProgress, 2000);
                });
        }
        
        function updateProgress(data) {
            // Update overall progress
            const progress = data.progress || 0;
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = progress + '%';
            document.getElementById('overallProgress').textContent = progress + '%';
            document.getElementById('statusText').textContent = data.status || 'Processing...';
            
            // Update tile progress
            const tilesProcessed = data.tiles_processed || 0;
            const totalTiles = data.total_tiles || 0;
            const tileProgress = data.tile_progress || 0;
            
            document.getElementById('tilesProcessed').textContent = `${tilesProcessed}/${totalTiles}`;
            document.getElementById('tileProgress').textContent = Math.round(tileProgress) + '%';
            document.getElementById('tileProgressFill').style.width = tileProgress + '%';
            
            if (totalTiles > 0) {
                document.getElementById('tileStatusText').textContent = 
                    `Processing tile ${tilesProcessed}/${totalTiles}`;
            }
            
            // Update estimated time
            if (startTime && progress > 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const estimated = (elapsed / progress) * (100 - progress);
                const estimatedText = estimated > 60 ? 
                    `${Math.round(estimated / 60)}m ${Math.round(estimated % 60)}s` : 
                    `${Math.round(estimated)}s`;
                document.getElementById('estimatedTime').textContent = estimatedText;
            }
            
            // Update logs
            if (data.logs && data.logs.length > 0) {
                const logEntries = document.getElementById('logEntries');
                logEntries.innerHTML = '';
                data.logs.forEach(log => {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    logEntry.textContent = log;
                    logEntries.appendChild(logEntry);
                });
                logEntries.scrollTop = logEntries.scrollHeight;
            }
        }
        
        function handleCompletedProcessing(data) {
            console.log('🎉 Enhancement completed!');
            console.log('📊 Input size:', data.input_size);
            console.log('📊 Output size:', data.output_size);
            
            // Display result image
            const img = document.createElement('img');
            img.src = data.result;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '250px';
            img.style.borderRadius = '15px';
            img.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.5)';
            
            const outputPreview = document.getElementById('output-preview');
            outputPreview.innerHTML = '';
            outputPreview.appendChild(img);
            
            // Add size info
            const sizeInfo = document.createElement('p');
            sizeInfo.style.marginTop = '10px';
            sizeInfo.style.fontSize = '0.9rem';
            sizeInfo.style.color = 'var(--text-secondary)';
            sizeInfo.textContent = `${data.input_size} → ${data.output_size}`;
            outputPreview.appendChild(sizeInfo);
            
            document.getElementById('downloadBtn').disabled = false;
            resultImage = data.result;
            
            // Hide progress section after a delay
            setTimeout(() => {
                document.getElementById('progressSection').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
            }, 3000);
        }
        
        // Download result
        document.getElementById('downloadBtn').addEventListener('click', function() {
            if (!resultImage) {
                alert('No result available to download');
                return;
            }
            
            const link = document.createElement('a');
            link.href = resultImage;
            link.download = `enhanced_${Date.now()}.png`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log('💾 Image downloaded successfully');
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'o':
                        e.preventDefault();
                        imageInput.click();
                        break;
                    case 'Enter':
                        e.preventDefault();
                        if (!document.getElementById('processBtn').disabled) {
                            document.getElementById('processBtn').click();
                        }
                        break;
                    case 's':
                        e.preventDefault();
                        if (!document.getElementById('downloadBtn').disabled) {
                            document.getElementById('downloadBtn').click();
                        }
                        break;
                }
            }
        });
        
        // Add tooltip for keyboard shortcuts
        const processBtn = document.getElementById('processBtn');
        processBtn.title = 'Ctrl+Enter to process';
        
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.title = 'Ctrl+S to download';
        
        console.log('🚀 Real-ESRGAN Pro Server initialized');
        console.log('⌨️  Keyboard shortcuts:');
        console.log('   Ctrl+O: Open file');
        console.log('   Ctrl+Enter: Process image');
        console.log('   Ctrl+S: Download result');
    </script>
</body>
</html>
    """)

@app.route('/status')
def status():
    """Get server status"""
    return jsonify({
        "status": "ready",
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "models_available": list(MODEL_PATHS.keys())
    })

@app.route('/process', methods=['POST'])
def process():
    """Start image processing"""
    try:
        data = request.json
        image_data = data.get('image')
        settings = data.get('settings', {})
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        print(f"\n🆔 New processing task: {task_id}")
        print(f"⚙️  Settings: {settings}")
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_image_thread,
            args=(task_id, image_data, settings)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"task_id": task_id})
        
    except Exception as e:
        error_msg = f"❌ Error starting processing: {str(e)}"
        print(error_msg)
        return jsonify({"error": str(e)}), 500

@app.route('/progress/<task_id>')
def progress(task_id):
    """Get processing progress"""
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    else:
        return jsonify({"error": "Task not found"}), 404

@app.route('/logs/<task_id>')
def get_logs(task_id):
    """Get processing logs for a specific task"""
    if task_id in processing_status and "logs" in processing_status[task_id]:
        return jsonify({"logs": processing_status[task_id]["logs"]})
    else:
        return jsonify({"logs": []})

if __name__ == '__main__':
    print("🚀 Starting Real-ESRGAN Pro Server...")
    print("=" * 60)
    print(f"📦 Device: {device}")
    print(f"🎯 GPU Available: {torch.cuda.is_available()}")
    print(f"🔧 Models Available: {len(MODEL_PATHS)}")
    
    for model_name in MODEL_PATHS.keys():
        model_exists = os.path.exists(MODEL_PATHS[model_name])
        status_icon = "✅" if model_exists else "❌"
        print(f"   {status_icon} {model_name}")
    
    print("=" * 60)
    
    # Initialize default model
    print("🔧 Initializing default model...")
    success, message = setup_model()
    if success:
        print(f"✅ {message}")
    else:
        print(f"⚠️ {message}")
    
    # Clean up old processing status every hour
    def cleanup_old_tasks():
        threading.Timer(3600.0, cleanup_old_tasks).start()
        cutoff = datetime.now() - timedelta(hours=1)
        # Simple cleanup - in production, you'd want to track task timestamps
        if len(processing_status) > 100:
            old_count = len(processing_status)
            processing_status.clear()
            print(f"🧹 Cleaned up {old_count} old processing tasks")
    
    cleanup_old_tasks()
    
    print("\n🌐 Server starting on http://localhost:5000")
    print("🎮 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        print("=" * 60)
