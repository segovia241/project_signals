import cv2
import base64
import asyncio
import time
from typing import List
import numpy as np
from fastapi import WebSocket
from config.settings import logger, CAMERA_CONFIG
from models.sign_language_translator import SignLanguageTranslator

class CameraInstance:
    def __init__(self, websocket: WebSocket, translator: SignLanguageTranslator, session_id: str):
        self.websocket = websocket
        self.translator = translator
        self.session_id = session_id
        self.is_running = False
        self.cap = None
        self.frame_sequence: List[np.ndarray] = []
    
    async def start_camera(self):
        """Iniciar cámara y procesamiento"""
        self.is_running = True
        self.cap = cv2.VideoCapture(CAMERA_CONFIG["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["frame_height"])
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
        
        if not self.cap.isOpened():
            await self.send_error("No se pudo acceder a la cámara")
            return
        
        logger.info(f"Cámara iniciada para sesión {self.session_id}")
        
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Procesar frame
            prediction_result, self.frame_sequence = self.translator.process_frame(
                frame, self.frame_sequence
            )
            
            # Codificar frame para enviar
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Enviar resultado
            message = {
                "type": "prediction_update",
                "frame": f"data:image/jpeg;base64,{frame_base64}",
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "all_predictions": prediction_result["all_predictions"],
                "has_hand_detection": prediction_result["has_hand_detection"],
                "sequence_ready": prediction_result["sequence_ready"],
                "timestamp": time.time()
            }
            
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje WebSocket: {e}")
                break
            
            # Pequeña pausa para controlar FPS
            await asyncio.sleep(CAMERA_CONFIG["websocket_delay"])
        
        self.stop()
    
    def stop(self):
        """Detener cámara"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_sequence.clear()
        logger.info(f"Cámara detenida para sesión {self.session_id}")
    
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
    
    def is_active(self) -> bool:
        return self.is_running and self.cap is not None
