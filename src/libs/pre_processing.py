import numpy as np
import cv2
import os

class PreProcessor:
  config = {
    'grayscale': False,
    'downscale': False,
    'clahe': False,
    'gaussian_blur': False
  }

  params = { 
    'downscale': {'factor': 0.5},
    'clahe': {'clip_limit': 2.0, 'grid_size': (8, 8)},
    'gaussian_blur': {'kernel_size': (3, 3), 'sigma': 1.0}
  }

  frame_folder = None
  output_folder = None

  def __init__(self, frame_folder, output_folder, config = None, params = None):
    if config:
      self.config.update(config)
    if params:
      self.params.update(params)  # Fix: was updating config instead of params
    self.frame_folder = frame_folder
    self.output_folder = output_folder
  
  def toGrayScale(self, image) :
    if len(image.shape) == 3:
      return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

  def toDownscale(self, image):
    factor = self.params['downscale']['factor']
    height, width = image.shape[:2]
    new_height, new_width = int(height * factor), int(width * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

  def applyClahe(self, image):
    clip_limit = self.params['clahe']['clip_limit']
    grid_size = self.params['clahe']['grid_size']

    if len(image.shape) == 3:
      image = self.toGrayScale(image)

    # Fix: Use correct parameter names
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

  def applyGaussianBlur(self,image):
    kernel_size = self.params['gaussian_blur']['kernel_size']
    sigma = self.params['gaussian_blur']['sigma']
    return cv2.GaussianBlur(image, kernel_size, sigma)
  
  def processImages(self):
    os.makedirs(self.output_folder, exist_ok=True)  # Add exist_ok=True
    image_files = [i for i in os.listdir(self.frame_folder) if i.lower().endswith('png')]  # Fix: use self.frame_folder
    
    if not image_files:
      print(f"No images found in {self.frame_folder}")  # Fix: use self.frame_folder
      return

    print(f"Processing {len(image_files)} images...")

    for i, filename in enumerate(image_files, 1):
      input_path = os.path.join(self.frame_folder, filename)  # Fix: use self.frame_folder
      output_path = os.path.join(self.output_folder, filename)
      try:
        image = cv2.imread(input_path)
        if image is None:
          print(f"Warning: Could not load {filename}")
          continue
        
        processed_image = self._applyProcessing(image)  
        cv2.imwrite(output_path, processed_image)
        print(f"Processed {i}/{len(image_files)}: {filename}")
            
      except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
    
    print(f"Processing complete! Images saved to: {self.output_folder}")
  
  def _applyProcessing(self, image):
    processed_image = image.copy()
    
    if self.config['grayscale']:
        processed_image = self.toGrayScale(processed_image)
    
    if self.config['downscale']:
        processed_image = self.toDownscale(processed_image)
    
    if self.config['clahe']:
        processed_image = self.applyClahe(processed_image)
    
    if self.config['gaussian_blur']:
        processed_image = self.applyGaussianBlur(processed_image)
    
    return processed_image