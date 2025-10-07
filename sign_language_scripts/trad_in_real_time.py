import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import json
from PIL import Image, ImageTk
import threading

class CameraSignLanguagePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Traductor de Lenguaje de Señas en Tiempo Real")
        self.root.geometry("800x600")
        
        # Inicializar MediaPipe y modelo
        self.initialize_model()
        
        # Variables
        self.is_playing = False
        self.cap = None
        self.frame_sequence = []
        
        # Crear interfaz
        self.create_widgets()
        
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
            
            self.model_loaded = True
            print("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model_loaded = False
    
    def create_widgets(self):
        """Crear la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="TRADUCTOR DE LENGUAJE DE SEÑAS EN TIEMPO REAL", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Botones de control
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.start_btn = ttk.Button(button_frame, text="▶ Iniciar Cámara", 
                                   command=self.toggle_camera)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Detener", 
                                 command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Información de estado
        info_frame = ttk.LabelFrame(main_frame, text="Estado", padding="5")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(info_frame, text="Presiona 'Iniciar Cámara' para comenzar")
        self.status_label.pack(anchor=tk.W)
        
        # Frame para video y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Panel de video
        video_frame = ttk.LabelFrame(content_frame, text="Cámara en Tiempo Real", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada", 
                                   background='black', foreground='white')
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(content_frame, text="Traducción", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Predicción actual
        ttk.Label(results_frame, text="SEÑA DETECTADA:", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.prediction_label = ttk.Label(results_frame, text="---", 
                                        font=('Arial', 24, 'bold'), foreground='blue')
        self.prediction_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Confianza
        ttk.Label(results_frame, text="CONFIANZA:", 
                 font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W)
        
        self.confidence_label = ttk.Label(results_frame, text="0%", 
                                        font=('Arial', 16))
        self.confidence_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 20))
        
        # Indicador de detección de manos
        ttk.Label(results_frame, text="ESTADO DE DETECCIÓN:", 
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W)
        
        self.detection_label = ttk.Label(results_frame, text="Sin manos detectadas", 
                                       font=('Arial', 12))
        self.detection_label.grid(row=5, column=0, sticky=tk.W, pady=(0, 20))
        
        # Todas las predicciones
        ttk.Label(results_frame, text="TODAS LAS PREDICCIONES:", 
                 font=('Arial', 10, 'bold')).grid(row=6, column=0, sticky=tk.W)
        
        self.predictions_text = tk.Text(results_frame, height=8, width=30, 
                                      font=('Arial', 9))
        self.predictions_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                command=self.predictions_text.yview)
        scrollbar.grid(row=7, column=1, sticky=(tk.N, tk.S))
        self.predictions_text.configure(yscrollcommand=scrollbar.set)
        
        # Configurar pesos para expansión
        main_frame.rowconfigure(3, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(7, weight=1)
        results_frame.columnconfigure(0, weight=1)
    
    def toggle_camera(self):
        """Iniciar o pausar la cámara"""
        if not self.is_playing:
            self.is_playing = True
            self.start_btn.config(text="⏸ Pausar")
            self.stop_btn.config(state=tk.NORMAL)
            self.start_camera()
        else:
            self.is_playing = False
            self.start_btn.config(text="▶ Continuar")
    
    def stop_camera(self):
        """Detener la cámara"""
        self.is_playing = False
        self.start_btn.config(text="▶ Iniciar Cámara")
        self.stop_btn.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Resetear interfaz
        self.video_label.config(image='')
        self.video_label.config(text="Cámara no iniciada")
        self.prediction_label.config(text="---")
        self.confidence_label.config(text="0%")
        self.detection_label.config(text="Sin manos detectadas")
        self.predictions_text.delete(1.0, tk.END)
        self.status_label.config(text="Cámara detenida")
    
    def start_camera(self):
        """Iniciar la cámara en un hilo separado"""
        if not self.model_loaded:
            self.status_label.config(text="Error: Modelo no cargado")
            return
        
        def camera_thread():
            # Inicializar cámara
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                self.root.after(0, lambda: self.status_label.config(text="Error: No se pudo acceder a la cámara"))
                return
            
            self.root.after(0, lambda: self.status_label.config(text="Cámara activa - Mostrando traducción en tiempo real"))
            
            self.frame_sequence = []
            
            while self.is_playing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Procesar frame
                keypoints, has_hand_detection, results = self.extract_keypoints(frame)
                self.frame_sequence.append(keypoints)
                
                # Mantener solo los últimos NUM_FRAMES
                if len(self.frame_sequence) > self.NUM_FRAMES:
                    self.frame_sequence.pop(0)
                
                # Dibujar landmarks en el frame
                annotated_frame = self.draw_landmarks(frame, results)
                
                # Hacer predicción si tenemos suficientes frames
                current_prediction = "---"
                current_confidence = 0
                all_predictions_array = np.zeros(len(self.CLASSES))
                
                if len(self.frame_sequence) == self.NUM_FRAMES:
                    hand_detection_count = sum([1 for _ in self.frame_sequence])  # Simplificado
                    keypoint_sum = np.sum(np.abs(self.frame_sequence))
                    
                    if hand_detection_count >= self.NUM_FRAMES * 0.5 and keypoint_sum > 1.0:
                        keypoints_sequence = np.array([self.frame_sequence])
                        prediction = self.model.predict(keypoints_sequence, verbose=0)
                        predicted_class_idx = np.argmax(prediction[0])
                        confidence = np.max(prediction[0])
                        
                        current_prediction = self.CLASSES[predicted_class_idx]
                        current_confidence = confidence
                        all_predictions_array = prediction[0]
                
                # Actualizar interfaz en el hilo principal
                self.root.after(0, self.update_prediction, 
                              current_prediction, current_confidence, all_predictions_array)
                
                # Actualizar estado de detección
                detection_text = "Manos detectadas" if has_hand_detection else "Sin manos detectadas"
                detection_color = "green" if has_hand_detection else "red"
                self.root.after(0, lambda: self.detection_label.config(
                    text=detection_text, foreground=detection_color))
                
                # Mostrar frame en la interfaz
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (400, 300))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.root.after(0, self.update_video_display, imgtk)
                
                # Pequeña pausa para controlar la velocidad
                cv2.waitKey(1)
            
            # Liberar cámara al finalizar
            if self.cap:
                self.cap.release()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=camera_thread)
        thread.daemon = True
        thread.start()
    
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
    
    def draw_landmarks(self, image, results):
        """Dibujar landmarks de MediaPipe en la imagen"""
        annotated_image = image.copy()
        
        # Dibujar landmarks de pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Dibujar landmarks de manos
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        
        # Dibujar landmarks de cara (solo contorno)
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        
        return annotated_image
    
    def update_video_display(self, image):
        """Actualizar la visualización del video"""
        self.video_label.configure(image=image)
        self.video_label.image = image
    
    def update_prediction(self, prediction, confidence, all_predictions):
        """Actualizar la predicción en la interfaz"""
        # Predicción principal
        self.prediction_label.config(text=prediction)
        
        # Confianza
        confidence_percent = int(confidence * 100)
        self.confidence_label.config(text=f"{confidence_percent}%")
        
        # Color basado en confianza
        if confidence_percent >= 80:
            color = "green"
        elif confidence_percent >= 60:
            color = "orange"
        else:
            color = "red"
        self.confidence_label.config(foreground=color)
        
        # Todas las predicciones
        self.predictions_text.delete(1.0, tk.END)
        for i, (class_name, pred) in enumerate(zip(self.CLASSES, all_predictions)):
            percent = int(pred * 100)
            bar = "█" * (percent // 5)  # Barra de progreso simple
            self.predictions_text.insert(tk.END, 
                                       f"{class_name:12} {percent:3d}% {bar}\n")

def main():
    root = tk.Tk()
    app = CameraSignLanguagePredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()