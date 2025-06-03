import cv2
import os

class extractor:
  total_frames = 0
  video = None
  video_path = None
  interval = 0 #intervalo de tempo em que pegaremos os prints

  def __init__(self, video_path = "./videos/ep1.mkv", interval = 3):
    self.video_path = video_path
    self.interval = interval
  
  def openVideo(self):
    self.video = cv2.VideoCapture(self.video_path)
    if not self.video.isOpened():
      raise ValueError(f"Não foi possível abrir o vídeo: {self.video_path}")
    self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = self.video.get(cv2.CAP_PROP_FPS)
  
  def extractFrames(self, output_path = "./frames/", format = 'png'):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
        
    if self.video is None:
      self.openVideo()
    
    frames_per_interval = int(self.fps * self.interval)
        
    frame_count = 0
    saved_count = 0
    
    while True:
      ret, frame = self.video.read()
      if not ret:
        break
      
      # Só salva se for no intervalo correto
      if frame_count % frames_per_interval == 0:
        file_name = f"frame_{saved_count:06d}.{format}"
        full_path = os.path.join(output_path, file_name)
        cv2.imwrite(full_path, frame)
        saved_count += 1
        print(f"Frame salvo: {file_name} (tempo: {frame_count/self.fps:.2f}s)")
      
      frame_count += 1
    
    print(f"Extraídos {saved_count} frames de {frame_count} total")
    print(f"Intervalo usado: {self.interval} segundos")


  def closeVideo(self):
    if self.video:
      self.video.release()
    
    def __del__(self):
      self.closeVideo()

