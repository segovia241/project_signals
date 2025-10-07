import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import threading
import time
import base64
import logging
from typing import Dict, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageTranslator:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.holistic = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.initialize_model()
        
    def initialize_model(self):
        """Inicializar el modelo de TensorFlow y MediaPipe"""
        try:
            # Cargar modelo
            self.model = load_model('sign_language_model.h5')
            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            # Configuración
            self.NUM_FRAMES = self.metadata['num_frames']
            self.CLASSES = self.metadata['classes']
            self.FACE_KEYPOINTS_INDICES = self.metadata['face_keypoints_indices']
            
            # Inicializar MediaPipe
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False, 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            )
            
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise e
    
    def extract_keypoints(self, frame):
        """Extraer puntos clave del frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        keypoints = []
        has_hand_detection = False

        # Manos
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                has_hand_detection = True
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0.0] * (21 * 3))

        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (33 * 3))

        # Cara
        if results.face_landmarks:
            for idx in self.FACE_KEYPOINTS_INDICES:
                lm = results.face_landmarks.landmark[idx]
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (20 * 3))

        return np.array(keypoints), has_hand_detection, results
    
    def process_frame(self, frame, frame_sequence):
        """Procesar un frame y hacer predicción"""
        keypoints, has_hand_detection, results = self.extract_keypoints(frame)
        frame_sequence.append(keypoints)
        
        # Mantener solo los últimos NUM_FRAMES
        if len(frame_sequence) > self.NUM_FRAMES:
            frame_sequence.pop(0)
        
        prediction_result = {
            "prediction": "---",
            "confidence": 0,
            "all_predictions": {},
            "has_hand_detection": has_hand_detection,
            "sequence_ready": len(frame_sequence) == self.NUM_FRAMES
        }
        
        # Hacer predicción si tenemos suficientes frames
        if len(frame_sequence) == self.NUM_FRAMES:
            hand_detection_count = sum([1 for _ in frame_sequence])  # Simplificado
            keypoint_sum = np.sum(np.abs(frame_sequence))
            
            if hand_detection_count >= self.NUM_FRAMES * 0.5 and keypoint_sum > 1.0:
                keypoints_sequence = np.array([frame_sequence])
                prediction = self.model.predict(keypoints_sequence, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                predicted_class = self.CLASSES[predicted_class_idx]
                
                prediction_result["prediction"] = predicted_class
                prediction_result["confidence"] = float(confidence)
                
                # Todas las predicciones
                for i, class_name in enumerate(self.CLASSES):
                    prediction_result["all_predictions"][class_name] = float(prediction[0][i])
        
        return prediction_result, frame_sequence

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.camera_instances: Dict[str, CameraInstance] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Cliente conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Detener cámara si existe para este websocket
        for session_id, instance in list(self.camera_instances.items()):
            if instance.websocket == websocket:
                instance.stop()
                del self.camera_instances[session_id]
        logger.info(f"Cliente desconectado. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error enviando mensaje: {e}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error en broadcast: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

class CameraInstance:
    def __init__(self, websocket: WebSocket, translator: SignLanguageTranslator, session_id: str):
        self.websocket = websocket
        self.translator = translator
        self.session_id = session_id
        self.is_running = False
        self.cap = None
        self.frame_sequence = []
    
    async def start_camera(self):
        """Iniciar cámara y procesamiento"""
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
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
            await asyncio.sleep(0.033)  # ~30 FPS
        
        self.stop()
    
    def stop(self):
        """Detener cámara"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
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

# Inicializar FastAPI
app = FastAPI(title="Sign Language Translator API", version="1.0.0")

# Configurar CORS para Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
translator = SignLanguageTranslator()
manager = ConnectionManager()

# Rutas de la API
@app.get("/")
async def root():
    return {"message": "Sign Language Translator API", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": translator.model is not None}

@app.get("/api/classes")
async def get_classes():
    return {"classes": translator.CLASSES}

@app.post("/api/predict")
async def predict_from_frame(frame_data: dict):
    """
    Endpoint para predecir desde un frame base64
    """
    try:
        # Decodificar frame base64
        frame_base64 = frame_data.get("frame", "").split(",")[1]
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Procesar frame
        prediction_result, _ = translator.process_frame(frame, [])
        
        return {
            "success": True,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "all_predictions": prediction_result["all_predictions"],
            "has_hand_detection": prediction_result["has_hand_detection"]
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return {"success": False, "error": str(e)}

# WebSocket para transmisión en tiempo real
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    camera_instance = CameraInstance(websocket, translator, session_id)
    manager.camera_instances[session_id] = camera_instance
    
    try:
        # Esperar mensaje de inicio
        data = await websocket.receive_json()
        if data.get("action") == "start_camera":
            await camera_instance.start_camera()
        elif data.get("action") == "stop_camera":
            camera_instance.stop()
        
        # Mantener conexión activa
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "stop_camera":
                camera_instance.stop()
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket desconectado para sesión {session_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        camera_instance.stop()
        if session_id in manager.camera_instances:
            del manager.camera_instances[session_id]
        manager.disconnect(websocket)

# Endpoint para detener cámara específica
@app.post("/api/camera/{session_id}/stop")
async def stop_camera(session_id: str):
    if session_id in manager.camera_instances:
        manager.camera_instances[session_id].stop()
        del manager.camera_instances[session_id]
        return {"success": True, "message": "Cámara detenida"}
    return {"success": False, "message": "Sesión no encontrada"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )