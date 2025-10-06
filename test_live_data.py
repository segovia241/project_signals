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
import time

class RealTimeSignLanguagePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Lenguaje de Señas en Tiempo Real")
        self.root.geometry("1000x700")
        
        # Inicializar MediaPipe y modelo
        self.initialize_model()
        
        # Variables de control
        self.is_recording = False
        self.cap = None
        self.frame_sequence = []
        self.predictions_history = []
        
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
            self.mp_drawing = mp.solutions.drawing_utils
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
        main_frame.columnconfigure(0, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="PREDICTOR DE LENGUAJE DE SEÑAS EN TIEMPO REAL", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Botones de control
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.start_btn = ttk.Button(button_frame, text="🎥 Empezar Grabación", 
                                  command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Terminar", 
                                 command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(button_frame, text="🗑 Limpiar", 
                                  command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Estado de la cámara
        self.status_label = ttk.Label(button_frame, text="Cámara: Apagada", 
                                    foreground="red", font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.RIGHT)
        
        # Frame para cámara y resultados
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Panel de cámara
        camera_frame = ttk.LabelFrame(content_frame, text="Cámara en Vivo", padding="5")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        camera_frame.columnconfigure(0, weight=1)
        camera_frame.rowconfigure(0, weight=1)
        
        self.camera_label = ttk.Label(camera_frame, text="Presiona 'Empezar Grabación' para iniciar", 
                                    background='black', foreground='white', anchor=tk.CENTER)
        self.camera_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(content_frame, text="Resultados en Tiempo Real", padding="5")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        
        # Información de frames
        info_frame = ttk.Frame(results_frame)
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(info_frame, text="Frames capturados:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.frames_label = ttk.Label(info_frame, text="0", font=('Arial', 10, 'bold'))
        self.frames_label.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(info_frame, text="/", font=('Arial', 10)).pack(side=tk.LEFT)
        self.total_frames_label = ttk.Label(info_frame, text=str(self.NUM_FRAMES), 
                                          font=('Arial', 10, 'bold'))
        self.total_frames_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Predicción actual
        ttk.Label(results_frame, text="PREDICCIÓN ACTUAL:", 
                 font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        self.prediction_label = ttk.Label(results_frame, text="---", 
                                        font=('Arial', 28, 'bold'), foreground='blue')
        self.prediction_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 15))
        
        # Confianza
        ttk.Label(results_frame, text="CONFIANZA:", 
                 font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W)
        
        self.confidence_label = ttk.Label(results_frame, text="0%", 
                                        font=('Arial', 18))
        self.confidence_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 20))
        
        # Barra de progreso de confianza
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(results_frame, variable=self.confidence_var, 
                                            maximum=100, length=200)
        self.confidence_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Historial de predicciones
        ttk.Label(results_frame, text="HISTORIAL DE PREDICCIONES:", 
                 font=('Arial', 10, 'bold')).grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        self.history_text = tk.Text(results_frame, height=10, width=35, 
                                  font=('Arial', 9))
        self.history_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar para el historial
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                command=self.history_text.yview)
        scrollbar.grid(row=7, column=1, sticky=(tk.N, tk.S))
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        # Todas las probabilidades
        ttk.Label(results_frame, text="PROBABILIDADES:", 
                 font=('Arial', 10, 'bold')).grid(row=8, column=0, sticky=tk.W, pady=(10, 5))
        
        self.probabilities_text = tk.Text(results_frame, height=6, width=35, 
                                       font=('Arial', 8))
        self.probabilities_text.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configurar pesos para expansión
        main_frame.rowconfigure(2, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=2)
        content_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(7, weight=1)
        results_frame.rowconfigure(9, weight=1)
        results_frame.columnconfigure(0, weight=1)
    
    def start_recording(self):
        """Iniciar la grabación desde la cámara"""
        if not self.model_loaded:
            self.show_error("Modelo no cargado")
            return
        
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Cámara: Grabando", foreground="green")
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            self.show_error("No se pudo abrir la cámara")
            return
        
        # Reiniciar variables
        self.frame_sequence = []
        self.predictions_history = []
        self.history_text.delete(1.0, tk.END)
        self.probabilities_text.delete(1.0, tk.END)
        
        # Iniciar hilo de procesamiento
        self.processing_thread = threading.Thread(target=self.process_camera)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_recording(self):
        """Detener la grabación"""
        self.is_recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Cámara: Apagada", foreground="red")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Limpiar display de cámara
        self.camera_label.config(image='')
        self.camera_label.config(text="Presiona 'Empezar Grabación' para iniciar")
        
        # Mostrar resumen
        self.show_summary()
    
    def clear_results(self):
        """Limpiar todos los resultados"""
        self.predictions_history = []
        self.history_text.delete(1.0, tk.END)
        self.probabilities_text.delete(1.0, tk.END)
        self.prediction_label.config(text="---")
        self.confidence_label.config(text="0%")
        self.confidence_var.set(0)
        self.frames_label.config(text="0")
    
    def process_camera(self):
        """Procesar frames de la cámara en tiempo real"""
        while self.is_recording and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Extraer keypoints
            keypoints, has_hand_detection, results = self.extract_keypoints(frame)
            self.frame_sequence.append(keypoints)
            
            # Mantener solo los últimos NUM_FRAMES
            if len(self.frame_sequence) > self.NUM_FRAMES:
                self.frame_sequence.pop(0)
            
            # Actualizar contador de frames
            self.root.after(0, self.frames_label.config, 
                          {"text": str(len(self.frame_sequence))})
            
            # Hacer predicción si tenemos suficientes frames
            current_prediction = "---"
            current_confidence = 0
            all_probabilities = []
            
            if len(self.frame_sequence) == self.NUM_FRAMES:
                hand_detection_count = sum([1 for seq in self.frame_sequence])  # Simplificado
                keypoint_sum = np.sum(np.abs(self.frame_sequence))
                
                if hand_detection_count >= self.NUM_FRAMES * 0.3 and keypoint_sum > 0.5:
                    keypoints_sequence = np.array([self.frame_sequence])
                    prediction = self.model.predict(keypoints_sequence, verbose=0)
                    predicted_class_idx = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    
                    current_prediction = self.CLASSES[predicted_class_idx]
                    current_confidence = confidence
                    all_probabilities = prediction[0]
                    
                    # Guardar en historial
                    timestamp = time.strftime("%H:%M:%S")
                    self.predictions_history.append({
                        'time': timestamp,
                        'prediction': current_prediction,
                        'confidence': confidence
                    })
                    
                    # Actualizar interfaz
                    self.root.after(0, self.update_prediction_display, 
                                  current_prediction, confidence, all_probabilities)
            
            # Dibujar landmarks en el frame
            frame_with_landmarks = self.draw_landmarks(frame, results)
            
            # Convertir para display
            frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, self.update_camera_display, imgtk)
            
            # Pequeña pausa para no saturar la CPU
            time.sleep(0.03)
    
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
        image.flags.writeable = True
        
        # Dibujar landmarks de manos
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Dibujar landmarks de pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
        
        return image
    
    def update_camera_display(self, image):
        """Actualizar la visualización de la cámara"""
        self.camera_label.configure(image=image)
        self.camera_label.image = image
    
    def update_prediction_display(self, prediction, confidence, all_probabilities):
        """Actualizar la predicción en la interfaz"""
        # Predicción principal
        self.prediction_label.config(text=prediction)
        
        # Confianza
        confidence_percent = int(confidence * 100)
        self.confidence_label.config(text=f"{confidence_percent}%")
        self.confidence_var.set(confidence_percent)
        
        # Color basado en confianza
        if confidence_percent >= 80:
            color = "green"
        elif confidence_percent >= 60:
            color = "orange"
        else:
            color = "red"
        self.confidence_label.config(foreground=color)
        
        # Actualizar historial
        if self.predictions_history:
            latest = self.predictions_history[-1]
            history_entry = f"{latest['time']} - {latest['prediction']} ({confidence_percent}%)\n"
            self.history_text.insert(tk.END, history_entry)
            self.history_text.see(tk.END)
        
        # Actualizar probabilidades
        self.probabilities_text.delete(1.0, tk.END)
        for i, (class_name, prob) in enumerate(zip(self.CLASSES, all_probabilities)):
            percent = int(prob * 100)
            bar = "█" * (percent // 5)  # Barra de progreso simple
            color = "green" if class_name == prediction else "black"
            self.probabilities_text.insert(tk.END, 
                                         f"{class_name:12} {percent:3d}% {bar}\n")
            # Aplicar color a la predicción actual
            if class_name == prediction:
                self.probabilities_text.tag_add("highlight", 
                                              f"{i+1}.0", 
                                              f"{i+1}.end")
                self.probabilities_text.tag_config("highlight", 
                                                 foreground="blue", 
                                                 font=('Arial', 8, 'bold'))
    
    def show_summary(self):
        """Mostrar resumen al terminar la grabación"""
        if self.predictions_history:
            from collections import Counter
            all_predictions = [p['prediction'] for p in self.predictions_history if p['prediction'] != "---"]
            
            if all_predictions:
                most_common = Counter(all_predictions).most_common(1)[0]
                
                summary = f"\n--- RESUMEN FINAL ---\n"
                summary += f"Predicción más frecuente: {most_common[0]}\n"
                summary += f"Veces detectada: {most_common[1]}\n"
                summary += f"Total de predicciones: {len(all_predictions)}\n"
                
                self.history_text.insert(tk.END, summary)
    
    def show_error(self, message):
        """Mostrar mensaje de error"""
        self.status_label.config(text=f"Error: {message}", foreground="red")
    
    def __del__(self):
        """Destructor para liberar recursos"""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'holistic'):
            self.holistic.close()

def main():
    root = tk.Tk()
    app = RealTimeSignLanguagePredictor(root)
    
    # Manejar cierre de ventana
    def on_closing():
        app.is_recording = False
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()