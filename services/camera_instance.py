import cv2
import base64
import asyncio
import time
from typing import List, Optional
import numpy as np
from fastapi import WebSocket
from config.settings import logger, VIDEO_CONFIG
from models.sign_language_translator import SignLanguageTranslator

class VideoProcessor:
    def __init__(self, websocket: WebSocket, translator: SignLanguageTranslator, session_id: str):
        self.websocket = websocket
        self.translator = translator
        self.session_id = session_id
        self.is_running = False
        self.frame_sequence: List[np.ndarray] = []
    
    async def process_video_stream(self):
        """Procesar stream de video desde el cliente"""
        self.is_running = True
        logger.info(f"Procesador de video iniciado para sesión {self.session_id}")
        
        try:
            while self.is_running:
                # Recibir frame del cliente via WebSocket
                data = await self.websocket.receive_json()
                
                if data.get("type") == "video_frame":
                    frame_data = data.get("frame", "")
                    
                    # Decodificar frame base64
                    try:
                        frame_base64 = frame_data.split(",")[1] if "," in frame_data else frame_data
                        frame_bytes = base64.b64decode(frame_base64)
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Procesar frame
                            prediction_result, self.frame_sequence = self.translator.process_frame(
                                frame, self.frame_sequence
                            )
                            
                            # Enviar resultado de vuelta al cliente
                            message = {
                                "type": "prediction_update",
                                "prediction": prediction_result["prediction"],
                                "confidence": prediction_result["confidence"],
                                "all_predictions": prediction_result["all_predictions"],
                                "has_hand_detection": prediction_result["has_hand_detection"],
                                "sequence_ready": prediction_result["sequence_ready"],
                                "timestamp": time.time()
                            }
                            
                            await self.websocket.send_json(message)
                        else:
                            await self.send_error("Error decodificando frame")
                            
                    except Exception as e:
                        logger.error(f"Error procesando frame: {e}")
                        await self.send_error(f"Error procesando frame: {str(e)}")
                
                elif data.get("action") == "stop_video":
                    break
                    
        except Exception as e:
            logger.error(f"Error en procesador de video: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Detener procesador"""
        self.is_running = False
        self.frame_sequence.clear()
        logger.info(f"Procesador de video detenido para sesión {self.session_id}")
    
    async def send_error(self, error_message: str):
        """Enviar mensaje de error"""
        message = {
            "type": "error",
            "message": error_message
        }
        try:
            await self.websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error enviando mensaje de error: {e}")