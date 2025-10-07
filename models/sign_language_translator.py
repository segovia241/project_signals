import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import json
from typing import Tuple, Dict, List
from config.settings import logger, MODEL_CONFIG

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
            self.model = load_model(MODEL_CONFIG["model_path"])
            with open(MODEL_CONFIG["metadata_path"], 'r') as f:
                self.metadata = json.load(f)
            
            # Configuración
            self.NUM_FRAMES = self.metadata['num_frames']
            self.CLASSES = self.metadata['classes']
            self.FACE_KEYPOINTS_INDICES = self.metadata['face_keypoints_indices']
            
            # Inicializar MediaPipe
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False, 
                min_detection_confidence=MODEL_CONFIG["min_detection_confidence"], 
                min_tracking_confidence=MODEL_CONFIG["min_tracking_confidence"]
            )
            
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise e
    
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, any]:
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
    
    def process_frame(self, frame: np.ndarray, frame_sequence: List[np.ndarray]) -> Tuple[Dict, List[np.ndarray]]:
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
