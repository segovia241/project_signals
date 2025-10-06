from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from video_processor import VideoProcessor

app = FastAPI(title="Sign Language Recognition WebSocket Server")

# ✅ Habilitar CORS público
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.video_processor = None
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Inicializar el procesador de video con manejo de errores"""
        try:
            self.video_processor = VideoProcessor()
            if self.video_processor.initialized:
                print("✅ VideoProcessor inicializado correctamente")
                print(f"✅ Clases disponibles: {self.video_processor.CLASSES}")
            else:
                print("❌ VideoProcessor no se pudo inicializar")
                self.video_processor = None
        except Exception as e:
            print(f"❌ Error crítico inicializando VideoProcessor: {e}")
            self.video_processor = None
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🟢 Cliente conectado ({len(self.active_connections)} total)")
        
        # Informar al cliente sobre el estado del procesador
        if self.video_processor and self.video_processor.initialized:
            status_msg = {
                "type": "system_status",
                "status": "ready",
                "message": "Sistema de reconocimiento de señas listo",
                "available_classes": self.video_processor.CLASSES
            }
        else:
            status_msg = {
                "type": "system_status", 
                "status": "error",
                "message": "Sistema de reconocimiento no disponible"
            }
        
        await self.send_message(json.dumps(status_msg), websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"🔴 Cliente desconectado ({len(self.active_connections)} total)")
    
    async def send_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
            # Log opcional para debugging
            try:
                parsed = json.loads(message)
                if parsed.get("type") != "analysis":  # No loguear cada frame para evitar spam
                    print("📤 Enviando:", json.dumps(parsed, indent=2, ensure_ascii=False))
            except:
                print(f"📤 Enviando texto plano: {message}")
        except Exception as e:
            print(f"❌ Error enviando mensaje: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # 📥 Log de mensaje recibido
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                ts = message.get("timestamp")
                if msg_type == "frame":
                    print(f"📩 Frame recibido - timestamp: {ts}")
                else:
                    print(f"📩 Mensaje tipo '{msg_type}' recibido")
            except Exception:
                print(f"⚠️ No se pudo parsear mensaje: {data[:100]}...")
                continue

            if message.get("type") == "frame":
                try:
                    # Verificar que el procesador esté disponible
                    if not manager.video_processor or not manager.video_processor.initialized:
                        error_response = {
                            "type": "analysis",
                            "data": {"error": "Sistema de reconocimiento no disponible"},
                            "timestamp": message.get("timestamp")
                        }
                        await manager.send_message(json.dumps(error_response), websocket)
                        continue

                    # 🧠 Procesar frame
                    result = await manager.video_processor.process_frame(
                        message["data"],
                        message.get("timestamp")
                    )

                    # 📤 Preparar respuesta
                    response = {
                        "type": "analysis",
                        "data": result,
                        "timestamp": message.get("timestamp")
                    }

                    # 💬 Enviar respuesta al cliente
                    await manager.send_message(json.dumps(response), websocket)

                except Exception as e:
                    print(f"❌ Error procesando frame: {e}")
                    error_response = {
                        "type": "analysis", 
                        "data": {"error": f"Error interno: {str(e)}"},
                        "timestamp": message.get("timestamp")
                    }
                    await manager.send_message(json.dumps(error_response), websocket)

            elif message.get("type") == "clear_sentence":
                # Comando para limpiar la oración acumulada
                if manager.video_processor:
                    manager.video_processor.clear_sentence()
                    response = {
                        "type": "system",
                        "data": {"message": "Oración limpiada"},
                        "timestamp": message.get("timestamp")
                    }
                    await manager.send_message(json.dumps(response), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ Error en conexión WebSocket: {e}")
        manager.disconnect(websocket)

# 🌐 Endpoints REST
@app.get("/")
async def root():
    return {
        "message": "Servidor de reconocimiento de lenguaje de señas activo",
        "service": "sign-language-recognition"
    }

@app.get("/health")
async def health_check():
    processor_status = "ready" if manager.video_processor and manager.video_processor.initialized else "error"
    
    return {
        "status": "healthy",
        "service": "sign-language-recognition",
        "processor_status": processor_status,
        "active_connections": len(manager.active_connections),
        "available_classes": manager.video_processor.CLASSES if manager.video_processor else []
    }

@app.post("/clear-sentence")
async def clear_sentence():
    """Endpoint para limpiar la oración acumulada"""
    if manager.video_processor:
        manager.video_processor.clear_sentence()
        return {"message": "Oración limpiada"}
    return {"error": "Processor no disponible"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)