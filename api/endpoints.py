import cv2
import numpy as np
import base64
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from config.settings import logger, APP_CONFIG, CORS_CONFIG
from models.sign_language_translator import SignLanguageTranslator
from services.connection_manager import ConnectionManager
from services.camera_instance import VideoProcessor

# Inicializar componentes
translator = SignLanguageTranslator()
manager = ConnectionManager()

def create_app() -> FastAPI:
    """Factory function para crear la aplicación FastAPI"""
    app = FastAPI(
        title=APP_CONFIG["app_name"],
        version=APP_CONFIG["version"]
    )

    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_CONFIG["allow_origins"],
        allow_credentials=CORS_CONFIG["allow_credentials"],
        allow_methods=CORS_CONFIG["allow_methods"],
        allow_headers=CORS_CONFIG["allow_headers"],
    )

    # Rutas de la API (mantener las mismas que antes)
    @app.get("/")
    async def root():
        return {
            "message": "Sign Language Translator API", 
            "status": "running",
            "sessions_active": manager.get_active_sessions_count(),
            "mode": "video_streaming"  # Indicar el modo de operación
        }

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy", 
            "model_loaded": translator.model is not None,
            "timestamp": time.time(),
            "mode": "video_streaming"
        }

    @app.get("/api/classes")
    async def get_classes():
        return {"classes": translator.CLASSES}

    @app.get("/api/status")
    async def get_status():
        return {
            "sessions_active": manager.get_active_sessions_count(),
            "connections_active": len(manager.active_connections),
            "model_loaded": translator.model is not None,
            "mode": "video_streaming"
        }

    @app.post("/api/predict")
    async def predict_from_frame(frame_data: dict):
        """
        Endpoint para predecir desde un frame base64
        """
        try:
            # Decodificar frame base64
            frame_base64 = frame_data.get("frame", "").split(",")[1] if "," in frame_data.get("frame", "") else frame_data.get("frame", "")
            frame_bytes = base64.b64decode(frame_base64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"success": False, "error": "No se pudo decodificar el frame"}
            
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

    # WebSocket para transmisión en tiempo real - MODIFICADO
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await manager.connect(websocket)
        video_processor = VideoProcessor(websocket, translator, session_id)
        manager.camera_instances[session_id] = video_processor
        
        try:
            # Iniciar procesamiento de video stream
            await video_processor.process_video_stream()
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket desconectado para sesión {session_id}")
        except Exception as e:
            logger.error(f"Error en WebSocket: {e}")
        finally:
            video_processor.stop()
            if session_id in manager.camera_instances:
                del manager.camera_instances[session_id]
            manager.disconnect(websocket)

    # Endpoints de gestión (mantener igual)
    @app.post("/api/camera/{session_id}/stop")
    async def stop_camera(session_id: str):
        if session_id in manager.camera_instances:
            manager.camera_instances[session_id].stop()
            del manager.camera_instances[session_id]
            return {"success": True, "message": "Procesador de video detenido"}
        return {"success": False, "message": "Sesión no encontrada"}

    @app.post("/api/camera/stop-all")
    async def stop_all_cameras():
        manager.stop_all_cameras()
        return {"success": True, "message": "Todos los procesadores detenidos"}

    return app