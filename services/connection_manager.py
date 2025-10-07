from typing import Dict, List
from fastapi import WebSocket
from config.settings import logger

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.camera_instances: Dict[str, 'VideoProcessor'] = {}  # Cambiar a VideoProcessor
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Cliente conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Detener procesador si existe para este websocket
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
    
    def get_active_sessions_count(self) -> int:
        return len(self.camera_instances)
    
    def stop_all_cameras(self):
        """Detener todas las cámaras activas"""
        for session_id, instance in list(self.camera_instances.items()):
            instance.stop()
            del self.camera_instances[session_id]
        logger.info("Todas las cámaras detenidas")
