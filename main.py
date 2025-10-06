from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from video_processor import VideoProcessor

app = FastAPI(title="Video Processing WebSocket Server")

# ✅ Habilitar CORS público
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.video_processor = VideoProcessor()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🟢 Cliente conectado ({len(self.active_connections)} total)")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"🔴 Cliente desconectado ({len(self.active_connections)} total)")
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        # 🧾 Log del mensaje enviado
        try:
            parsed = json.loads(message)
            print("📤 Enviando respuesta al cliente:")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            print(f"📤 Enviando texto plano: {message}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # 📥 Log de frame recibido
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                ts = message.get("timestamp")
                print(f"\n📩 Mensaje recibido: type={msg_type}, timestamp={ts}")
            except Exception:
                print(f"\n⚠️ No se pudo parsear mensaje: {data}")
                continue

            if message.get("type") == "frame":
                try:
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

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ Error en conexión WebSocket: {e}")
        manager.disconnect(websocket)

# 🌐 Endpoints REST normales
@app.get("/")
async def root():
    return {"message": "Servidor de procesamiento de video activo"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "video-processing-server",
        "active_connections": len(manager.active_connections),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
