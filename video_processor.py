import cv2
import numpy as np
import base64
import json
import asyncio

class VideoProcessor:
    def __init__(self):
        self.initialized = True
    
    async def process_frame(self, frame_data: str, timestamp: str = None) -> dict:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._process_frame_sync, 
                frame_data
            )
            return result
        except Exception as e:
            return {"error": f"Error en procesamiento: {str(e)}"}
    
    def _process_frame_sync(self, frame_data: str) -> dict:
        try:
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "No se pudo decodificar la imagen"}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            mean_intensity = np.mean(gray)
            edge_intensity = np.mean(edges)
            
            height, width = img.shape[:2]
            
            return {
                "mean_intensity": float(mean_intensity),
                "edge_intensity": float(edge_intensity),
                "image_dimensions": {"width": width, "height": height},
                "processing_time": "realtime",
                "status": "processed"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def set_processing_parameters(self, **kwargs):
        if 'canny_threshold1' in kwargs and 'canny_threshold2' in kwargs:
            self.canny_threshold1 = kwargs['canny_threshold1']
            self.canny_threshold2 = kwargs['canny_threshold2']
    
    async def cleanup(self):
        pass
