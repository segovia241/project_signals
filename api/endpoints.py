import cv2
import numpy as np
import base64
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from config.settings import logger, APP_CONFIG, CORS_CONFIG
from models.sign_language_translator import SignLanguageTranslator
from services.connection_manager import ConnectionManager
from services.camera_instance import CameraInstance

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

    # Rutas de la API
    @app.get("/")
    async def root():
        return {
            "message": "Sign Language Translator API", 
            "status": "running",
            "sessions_active": manager.get_active_sessions_count()
        }

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy", 
            "model_loaded": translator.model is not None,
            "timestamp": time.time()
        }

    @app.get("/api/classes")
    async def get_classes():
        return {"classes": translator.CLASSES}

    @app.get("/api/status")
    async def get_status():
        return {
            "sessions_active": manager.get_active_sessions_count(),
            "connections_active": len(manager.active_connections),
            "model_loaded": translator.model is not None
        }

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

    @app.post("/api/camera/stop-all")
    async def stop_all_cameras():
        manager.stop_all_cameras()
        return {"success": True, "message": "Todas las cámaras detenidas"}

    return app
