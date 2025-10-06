# video_processor.py
import cv2
import numpy as np
import base64
import json
import asyncio
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model

# 🔥 REMOVER la importación de WebSocket aquí - no es necesaria en este archivo

class VideoProcessor:
    def __init__(self):
        # Cargar modelo y metadata
        try:
            self.model = load_model('sign_language_model.h5')
            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            # Inicializar MediaPipe
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False, 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            )
            
            # Configuración
            self.NUM_FRAMES = self.metadata['num_frames']
            self.CLASSES = self.metadata['classes']
            self.FACE_KEYPOINTS_INDICES = self.metadata['face_keypoints_indices']
            
            # Buffer para frames
            self.frame_sequence = []
            self.hand_detection_flags = []
            self.is_gesture_active = False
            self.last_hand_detection = False
            self.best_predicted_class = "Ninguna"
            self.best_confidence = 0.0
            self.sentence = []
            
            self.initialized = True
            print(f"Modelo cargado exitosamente. Clases: {self.CLASSES}")
            
        except Exception as e:
            print(f"Error inicializando modelo: {e}")
            self.initialized = False
    
    def extract_keypoints(self, frame):
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
            # Decodificar imagen
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"error": "No se pudo decodificar la imagen"}
            
            # Extraer keypoints
            keypoints, has_hand_detection, results = self.extract_keypoints(frame)
            
            # Actualizar buffer
            self.frame_sequence.append(keypoints)
            self.hand_detection_flags.append(has_hand_detection)
            
            # Mantener solo los últimos NUM_FRAMES
            if len(self.frame_sequence) > self.NUM_FRAMES:
                self.frame_sequence.pop(0)
                self.hand_detection_flags.pop(0)
            
            # Inicializar respuesta
            response = {
                "status": "waiting",
                "current_prediction": "Ninguna",
                "current_confidence": 0.0,
                "best_prediction": self.best_predicted_class,
                "best_confidence": float(self.best_confidence),
                "sentence": " ".join(self.sentence) if self.sentence else "Ninguna",
                "frames_accumulated": len(self.frame_sequence),
                "hand_detected": has_hand_detection
            }
            
            # Hacer predicción si tenemos suficientes frames
            if len(self.frame_sequence) == self.NUM_FRAMES:
                hand_detection_count = sum(self.hand_detection_flags)
                keypoint_sum = np.sum(np.abs(self.frame_sequence))
                
                # Verificar si es un gesto válido
                if hand_detection_count >= self.NUM_FRAMES * 0.5 and keypoint_sum > 1.0:
                    keypoints_sequence = np.array([self.frame_sequence])
                    prediction = self.model.predict(keypoints_sequence, verbose=0)
                    predicted_class_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if confidence >= 0.6:  # Umbral de confianza
                        current_prediction = self.CLASSES[predicted_class_idx]
                        response.update({
                            "status": "gesture_detected",
                            "current_prediction": current_prediction,
                            "current_confidence": float(confidence),
                            "gesture_active": True
                        })
                        
                        # Actualizar mejor predicción del gesto actual
                        if confidence > self.best_confidence:
                            self.best_predicted_class = current_prediction
                            self.best_confidence = confidence
                        
                        self.is_gesture_active = True
                    else:
                        response.update({
                            "status": "low_confidence",
                            "current_prediction": "Ninguna",
                            "current_confidence": float(confidence)
                        })
                        self.is_gesture_active = False
                else:
                    response["status"] = "insufficient_hands"
                    self.is_gesture_active = False
            
            # Detectar fin de gesto
            if (self.last_hand_detection and not has_hand_detection and 
                self.is_gesture_active and self.best_confidence >= 0.6):
                self.sentence.append(self.best_predicted_class)
                self.best_predicted_class = "Ninguna"
                self.best_confidence = 0.0
                self.is_gesture_active = False
                response["sentence"] = " ".join(self.sentence)
                response["gesture_completed"] = True
            
            self.last_hand_detection = has_hand_detection
            
            # Actualizar respuesta con estado actual
            response.update({
                "best_prediction": self.best_predicted_class,
                "best_confidence": float(self.best_confidence)
            })
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    def set_processing_parameters(self, **kwargs):
        # Puedes agregar parámetros configurables aquí
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['confidence_threshold']
    
    def clear_sentence(self):
        """Limpiar la oración acumulada"""
        self.sentence = []
        self.best_predicted_class = "Ninguna"
        self.best_confidence = 0.0
    
    async def cleanup(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()

    # 🔥 NUEVO: Método para procesar grabaciones (SIN WebSocket en los parámetros)
    async def process_recording_frames(self, frames_data: list) -> dict:
        """Procesar una lista completa de frames de una grabación"""
        try:
            all_predictions = []
            
            print(f"🎬 Procesando {len(frames_data)} frames de grabación...")
            
            for i, frame_data in enumerate(frames_data):
                if i % 10 == 0:  # Log cada 10 frames
                    print(f"  📊 Procesando frame {i+1}/{len(frames_data)}")
                
                # Procesar frame individual
                result = await self.process_frame(frame_data)
                
                if ("current_prediction" in result and 
                    result["current_prediction"] != "Ninguna" and 
                    result.get("current_confidence", 0) > 0.1):
                    
                    all_predictions.append({
                        "frame": i,
                        "prediction": result["current_prediction"],
                        "confidence": result["current_confidence"]
                    })
            
            print(f"📈 Procesamiento completado. Predicciones encontradas: {len(all_predictions)}")
            
            # Analizar resultados agregados
            if not all_predictions:
                return {
                    "detected_word": "Ninguna",
                    "confidence": 0.0,
                    "message": "No se detectaron señas claras en la grabación",
                    "total_frames": len(frames_data)
                }
            
            # Encontrar la predicción más consistente
            from collections import Counter
            prediction_counts = Counter([p["prediction"] for p in all_predictions])
            most_common_prediction, count = prediction_counts.most_common(1)[0]
            
            # Calcular confianza promedio
            confidences = [p["confidence"] for p in all_predictions if p["prediction"] == most_common_prediction]
            avg_confidence = sum(confidences) / len(confidences)
            
            print(f"🎯 RESULTADO: '{most_common_prediction}' (confianza: {avg_confidence:.2f})")
            print(f"   Frames con detección: {count}/{len(frames_data)}")
            
            return {
                "detected_word": most_common_prediction,
                "confidence": float(avg_confidence),
                "total_frames": len(frames_data),
                "frames_with_detection": len(all_predictions),
                "prediction_consistency": f"{count}/{len(frames_data)} frames",
                "all_predictions_count": dict(prediction_counts)
            }
            
        except Exception as e:
            print(f"❌ Error procesando grabación: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error procesando grabación: {str(e)}"}