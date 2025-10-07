from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from video_processor import VideoProcessor
import uuid

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
        self.active_connections: dict[str, WebSocket] = {}
        self.video_processor = None
        self.recording_buffers: dict[str, list] = {}
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Inicializar el procesador de video con manejo de errores"""
        try:
            print("🔄 Intentando inicializar VideoProcessor...")
            self.video_processor = VideoProcessor()
            if hasattr(self.video_processor, 'initialized') and self.video_processor.initialized:
                print("✅ VideoProcessor inicializado correctamente")
                print(f"✅ Clases disponibles: {self.video_processor.CLASSES}")
            else:
                print("❌ VideoProcessor no disponible")
                self.video_processor = None
        except Exception as e:
            print(f"❌ Error crítico inicializando VideoProcessor: {e}")
            import traceback
            traceback.print_exc()
            self.video_processor = None
    
    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        self.recording_buffers[client_id] = []
        print(f"🟢 Cliente conectado ({len(self.active_connections)} total) - ID: {client_id}")

        # Informar al cliente sobre el estado del procesador
        status_msg = {
            "type": "system_status",
            "status": "ready" if self.video_processor else "error",
            "message": "Sistema de reconocimiento de señas listo" if self.video_processor else "Sistema de reconocimiento no disponible",
            "available_classes": self.video_processor.CLASSES if self.video_processor else []
        }
        await self.send_message(json.dumps(status_msg), websocket)
        return client_id
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            if client_id in self.recording_buffers:
                del self.recording_buffers[client_id]
            print(f"🔴 Cliente desconectado ({len(self.active_connections)} total) - ID: {client_id}")
    
    async def send_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
            try:
                parsed = json.loads(message)
                if parsed.get("type") != "analysis":
                    print("📤 Enviando:", json.dumps(parsed, indent=2, ensure_ascii=False))
            except:
                print(f"📤 Enviando texto plano: {message}")
        except Exception as e:
            print(f"❌ Error enviando mensaje: {e}")

    async def process_recording(self, client_id: str):
        """Procesar la grabación completa del cliente"""
        try:
            if client_id not in self.recording_buffers:
                return {"error": "No hay grabación para procesar"}
            
            frames = self.recording_buffers[client_id]  # 🔥 CORREGIDO: Usar la variable frames
            print(f"🎬 INICIANDO PROCESAMIENTO - {len(frames)} frames para cliente {client_id}")
            
            if len(frames) == 0:
                return {"error": "Grabación vacía"}
            
            if not self.video_processor:
                return {"error": "Procesador de video no disponible"}
            
            # 🔥 PROCESAR GRABACIÓN COMPLETA
            result = await self.video_processor.process_recording_frames(frames)
            return result
            
        except Exception as e:
            print(f"❌ Error procesando grabación: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error procesando grabación: {str(e)}"}

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                ts = message.get("timestamp")
                print(f"📩 Mensaje tipo '{msg_type}' recibido - timestamp: {ts}")
            except Exception:
                print(f"⚠️ No se pudo parsear mensaje: {data[:100]}...")
                continue

            # ==========================
            # ✅ Nuevo manejo de recording_frame
            # ==========================
            if msg_type == "recording_frame":
                try:
                    if not manager.video_processor or not manager.video_processor.initialized:
                        error_response = {
                            "type": "analysis",
                            "data": {"error": "Sistema de reconocimiento no disponible"},
                            "timestamp": ts
                        }
                        await manager.send_message(json.dumps(error_response), websocket)
                        continue

                    frame_data = message["data"]
                    frame_index = message.get("frame_index", 0)
                    total_frames = message.get("total_frames", 0)
                    is_complete = message.get("is_complete", False)
                    
                    print(f"🎥 RECORDING_FRAME - Index: {frame_index}, Total: {total_frames}, Complete: {is_complete}")

                    # Agregar frame al buffer del cliente
                    manager.recording_buffers[client_id].append(frame_data)
                    print(f"   📊 Buffer actual: {len(manager.recording_buffers[client_id])} frames")

                    if is_complete:
                        print(f"🎬 🎬 🎬 GRABACIÓN MARCADA COMO COMPLETA! Procesando...")
                        processing_msg = {
                            "type": "processing_status",
                            "data": {
                                "status": "processing",
                                "message": f"Analizando {len(manager.recording_buffers[client_id])} frames...",
                                "frames_count": len(manager.recording_buffers[client_id])
                            },
                            "timestamp": ts
                        }
                        await manager.send_message(json.dumps(processing_msg), websocket)

                        try:
                            # 🔥 PROCESAR GRABACIÓN COMPLETA
                            result = await manager.process_recording(client_id)
                            print(f"🎯 RESULTADO OBTENIDO: {result}")

                            # 🔥 CONSTRUIR Y ENVIAR RESPUESTA
                            response = {
                                "type": "recording_result",
                                "data": result,
                                "timestamp": ts
                            }
                            
                            print(f"📤 PREPARANDO PARA ENVIAR RESPUESTA: {response['type']}")
                            
                            # 🔥 ENVIAR RESPUESTA FINAL
                            await manager.send_message(json.dumps(response), websocket)
                            print(f"✅ RESPUESTA ENVIADA EXITOSAMENTE AL CLIENTE")

                        except Exception as e:
                            print(f"❌ ERROR ENVIANDO RESPUESTA: {e}")
                            error_response = {
                                "type": "recording_result",
                                "data": {"error": f"Error enviando resultado: {str(e)}"},
                                "timestamp": ts
                            }
                            await manager.send_message(json.dumps(error_response), websocket)

                        # Limpiar buffer
                        manager.recording_buffers[client_id] = []
                        print(f"🧹 Buffer limpiado para cliente {client_id}")
                    else:
                        # Confirmación de frame recibido
                        ack_response = {
                            "type": "frame_ack",
                            "data": {
                                "frame_index": frame_index,
                                "buffer_size": len(manager.recording_buffers[client_id]),
                                "status": "frame_received"
                            },
                            "timestamp": ts
                        }
                        await manager.send_message(json.dumps(ack_response), websocket)

                except Exception as e:
                    print(f"❌ Error procesando recording_frame: {e}")
                    import traceback
                    traceback.print_exc()
                    error_response = {
                        "type": "analysis",
                        "data": {"error": f"Error interno: {str(e)}"},
                        "timestamp": ts
                    }
                    await manager.send_message(json.dumps(error_response), websocket)

            # ==========================
            # 🕒 Mantener compatibilidad con frame en tiempo real
            # ==========================
            elif msg_type == "frame":
                try:
                    if not manager.video_processor or not manager.video_processor.initialized:
                        error_response = {
                            "type": "analysis",
                            "data": {"error": "Sistema de reconocimiento no disponible"},
                            "timestamp": ts
                        }
                        await manager.send_message(json.dumps(error_response), websocket)
                        continue

                    # Procesar frame individual (modo tiempo real)
                    result = await manager.video_processor.process_frame(
                        message["data"],
                        ts
                    )

                    response = {
                        "type": "analysis",
                        "data": result,
                        "timestamp": ts
                    }

                    await manager.send_message(json.dumps(response), websocket)

                except Exception as e:
                    print(f"❌ Error procesando frame: {e}")
                    error_response = {
                        "type": "analysis", 
                        "data": {"error": f"Error interno: {str(e)}"},
                        "timestamp": ts
                    }
                    await manager.send_message(json.dumps(error_response), websocket)

            elif msg_type == "clear_sentence":
                if manager.video_processor:
                    manager.video_processor.clear_sentence()
                    response = {
                        "type": "system",
                        "data": {"message": "Oración limpiada"},
                        "timestamp": ts
                    }
                    await manager.send_message(json.dumps(response), websocket)
            
            # 🔥 NUEVO: Limpiar grabación
            elif msg_type == "clear_recording":
                if client_id in manager.recording_buffers:
                    manager.recording_buffers[client_id] = []
                    response = {
                        "type": "system",
                        "data": {"message": "Grabación limpiada"},
                        "timestamp": ts
                    }
                    await manager.send_message(json.dumps(response), websocket)
                    print(f"🧹 Grabación limpiada para cliente {client_id}")

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"⚠️ Error en conexión WebSocket: {e}")
        manager.disconnect(client_id)

# 🌐 Endpoints REST
@app.get("/")
async def root():
    return {
        "message": "Servidor de reconocimiento de lenguaje de señas activo",
        "service": "sign-language-recognition",
        "version": "2.0 - Soporte para grabaciones"
    }

@app.get("/health")
async def health_check():
    processor_status = "ready" if manager.video_processor and manager.video_processor.initialized else "error"
    return {
        "status": "healthy",
        "service": "sign-language-recognition",
        "processor_status": processor_status,
        "active_connections": len(manager.active_connections),
        "available_classes": manager.video_processor.CLASSES if manager.video_processor else [],
        "recording_buffers_size": sum(len(buf) for buf in manager.recording_buffers.values())
    }

@app.post("/clear-sentence")
async def clear_sentence():
    """Endpoint para limpiar la oración acumulada"""
    if manager.video_processor:
        manager.video_processor.clear_sentence()
        return {"message": "Oración limpiada"}
    return {"error": "Processor no disponible"}

@app.post("/clear-recording/{client_id}")
async def clear_recording(client_id: str):
    """Endpoint para limpiar la grabación de un cliente"""
    if client_id in manager.recording_buffers:
        manager.recording_buffers[client_id] = []
        return {"message": f"Grabación limpiada para cliente {client_id}"}
    return {"error": "Cliente no encontrado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)