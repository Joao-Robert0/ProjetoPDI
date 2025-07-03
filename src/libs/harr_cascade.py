import numpy as np
import cv2
import os

class HarrCascade:

  params = {
    'scaleFactor': 1.2,
    'minNeighbors': 6,
    'minSize': (120, 120),
    'flags': cv2.CASCADE_SCALE_IMAGE
  }

  frame_folder = None
  output_folder = None
    
  def __init__(self, input_folder=None, output_folder=None, params=None):   
    self.input_folder = input_folder
    self.output_folder = output_folder
    
    # Atualiza com parâmetros do usuário
    if params:
        self.params.update(params)
    
    # Carrega o classificador Haar Cascade
    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Contador para rostos detectados
    self.face_count = 0
    
  def detect_faces(self, image):     
    faces = self.face_cascade.detectMultiScale(
      image,
      scaleFactor=self.params['scaleFactor'],
      minNeighbors=self.params['minNeighbors'],
      minSize=self.params['minSize'],
      flags=self.params['flags']
    )
    return faces
    
  def crop_and_save_face(self, image, face, face_id=None, filename=None):
    x, y, w, h = face
    face_img = image[y:y+h, x:x+w]
    
    if face_id is None:
      face_id = self.face_count
      self.face_count += 1
        
    if filename is None:
      filename = f"rosto_{face_id:06d}.png"
    else:
      base_name = os.path.splitext(os.path.basename(filename))[0]
      filename = f"rosto_{base_name}_{face_id}.png"
        
    # Cria a pasta de saída se não existir
    if not os.path.exists(self.output_folder):
      os.makedirs(self.output_folder)
        
    output_path = os.path.join(self.output_folder, filename)
    cv2.imwrite(output_path, face_img)
    
    return output_path
    
  def process_image(self, image_path):
    image = cv2.imread(image_path)
    if image is None:
      print(f"Não foi possível carregar a imagem: {image_path}")
      return 0
        
    faces = self.detect_faces(image)
    
    for i, face in enumerate(faces):
      filename = os.path.basename(image_path)
      self.crop_and_save_face(image, face, face_id=i, filename=filename)
        
    return len(faces)
    
  def process_frames(self):
      if not self.input_folder:
          raise ValueError("Pasta de entrada não especificada")
          
      if not self.output_folder:
          raise ValueError("Pasta de saída não especificada")
          
      # Verifica se a pasta existe
      if not os.path.exists(self.input_folder):
          raise FileNotFoundError(f"Pasta não encontrada: {self.input_folder}")
          
      # Cria pasta de saída se não existir
      if not os.path.exists(self.output_folder):
          os.makedirs(self.output_folder)
          
      # Lista todas as imagens na pasta
      image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
      image_files = [f for f in os.listdir(self.input_folder) 
                    if os.path.splitext(f.lower())[1] in image_extensions]
                    
      if not image_files:
          print(f"Nenhuma imagem encontrada na pasta {self.input_folder}")
          return 0, 0
          
      print(f"Processando {len(image_files)} imagens...")
      
      total_faces = 0
      processed_images = 0
      
      # Processa cada imagem
      for img_file in image_files:
          img_path = os.path.join(self.input_folder, img_file)
          image = cv2.imread(img_path)
          
          if image is None:
              print(f"Não foi possível carregar a imagem: {img_path}")
              continue
              
          faces = self.detect_faces(image)
          
          # Recorta e salva cada face
          for i, face in enumerate(faces):
              self.crop_and_save_face(image, face, filename=img_file)
              
          num_faces = len(faces)
          total_faces += num_faces
          processed_images += 1
          
          print(f"Processada imagem {img_file}: {num_faces} rostos encontrados")
          
      print(f"Processamento concluído. Total de {total_faces} rostos detectados em {processed_images} imagens.")
      return total_faces, processed_images


