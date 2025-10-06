import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import json
import os
from PIL import Image, ImageTk
import threading

class VideoSignLanguagePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Lenguaje de Señas")
        self.root.geometry("800x600")
        
        # Inicializar MediaPipe y modelo
        self.initialize_model()
        
        # Variables
        self.video_path = None
        self.is_playing = False
        self.current_frame = 0
        self.cap = None
        
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
        title_label = ttk.Label(main_frame, text="PREDICTOR DE LENGUAJE DE SEÑAS", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Botones de control
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.select_btn = ttk.Button(button_frame, text="Seleccionar Video", 
                                   command=self.select_video)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.play_btn = ttk.Button(button_frame, text="▶ Analizar", 
                                 command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Detener", 
                                 command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # Información del video
        info_frame = ttk.LabelFrame(main_frame, text="Información del Video", padding="5")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.video_info_label = ttk.Label(info_frame, text="No hay video seleccionado")
        self.video_info_label.pack(anchor=tk.W)
        
        # Frame para video y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Panel de video
        video_frame = ttk.LabelFrame(content_frame, text="Video", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Selecciona un video para comenzar", 
                                   background='black', foreground='white')
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(content_frame, text="Resultados", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Predicción actual
        ttk.Label(results_frame, text="PREDICCIÓN ACTUAL:", 
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
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(results_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Todas las predicciones
        ttk.Label(results_frame, text="TODAS LAS PREDICCIONES:", 
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W)
        
        self.predictions_text = tk.Text(results_frame, height=8, width=30, 
                                      font=('Arial', 9))
        self.predictions_text.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                command=self.predictions_text.yview)
        scrollbar.grid(row=6, column=1, sticky=(tk.N, tk.S))
        self.predictions_text.configure(yscrollcommand=scrollbar.set)
        
        # Configurar pesos para expansión
        main_frame.rowconfigure(3, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(6, weight=1)
        results_frame.columnconfigure(0, weight=1)
    
    def select_video(self):
        """Seleccionar archivo de video"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.load_video_info()
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
    
    def load_video_info(self):
        """Cargar información del video seleccionado"""
        if self.video_path:
            file_name = os.path.basename(self.video_path)
            file_size = os.path.getsize(self.video_path) / (1024 * 1024)  # MB
            
            # Obtener información del video con OpenCV
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                info_text = (f"Archivo: {file_name}\n"
                           f"Tamaño: {file_size:.1f} MB\n"
                           f"Duración: {duration:.1f} segundos\n"
                           f"FPS: {fps:.1f}\n"
                           f"Frames: {frame_count}")
            else:
                info_text = f"Archivo: {file_name}\nError al leer el video"
            
            self.video_info_label.config(text=info_text)
    
    def toggle_play(self):
        """Iniciar o pausar el análisis"""
        if not self.is_playing:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pausar")
            self.analyze_video()
        else:
            self.is_playing = False
            self.play_btn.config(text="▶ Continuar")
    
    def stop_video(self):
        """Detener el análisis"""
        self.is_playing = False
        self.current_frame = 0
        self.play_btn.config(text="▶ Analizar")
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Resetear interfaz
        self.video_label.config(image='')
        self.video_label.config(text="Selecciona un video para comenzar")
        self.prediction_label.config(text="---")
        self.confidence_label.config(text="0%")
        self.progress_var.set(0)
        self.predictions_text.delete(1.0, tk.END)
    
    def analyze_video(self):
        """Analizar el video en un hilo separado"""
        if not self.video_path or not self.model_loaded:
            return
        
        def analysis_thread():
            self.cap = cv2.VideoCapture(self.video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            frame_sequence = []
            predictions_history = []
            
            while self.is_playing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.current_frame += 1
                
                # Actualizar progreso
                progress = (self.current_frame / total_frames) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                
                # Procesar frame
                keypoints, has_hand_detection, results = self.extract_keypoints(frame)
                frame_sequence.append(keypoints)
                
                # Mantener solo los últimos NUM_FRAMES
                if len(frame_sequence) > self.NUM_FRAMES:
                    frame_sequence.pop(0)
                
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
                        
                        # Actualizar interfaz en el hilo principal
                        self.root.after(0, self.update_prediction, 
                                      predicted_class, confidence, prediction[0])
                        
                        # Guardar en historial
                        predictions_history.append({
                            'frame': self.current_frame,
                            'prediction': predicted_class,
                            'confidence': confidence
                        })
                
                # Mostrar frame en la interfaz
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (400, 300))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.root.after(0, self.update_video_display, imgtk)
                
                # Control de velocidad (simular FPS real)
                delay = int(1000 / fps) if fps > 0 else 33
                cv2.waitKey(delay)
            
            # Al finalizar el video
            self.root.after(0, self.analysis_finished, predictions_history)
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=analysis_thread)
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
    
    def analysis_finished(self, predictions_history):
        """Manejar la finalización del análisis"""
        self.is_playing = False
        self.play_btn.config(text="▶ Analizar")
        
        # Mostrar resumen
        if predictions_history:
            # Encontrar la predicción más común
            from collections import Counter
            all_predictions = [p['prediction'] for p in predictions_history]
            most_common = Counter(all_predictions).most_common(1)[0]
            
            summary = f"\n--- RESUMEN ---\n"
            summary += f"Predicción más frecuente: {most_common[0]}\n"
            summary += f"Frames analizados: {len(predictions_history)}\n"
            
            self.predictions_text.insert(tk.END, summary)

def main():
    root = tk.Tk()
    app = VideoSignLanguagePredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()