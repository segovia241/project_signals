import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import stats

class SequenceGenerator:
    def __init__(self, model_path, metadata_path):
        """Inicializar el generador de secuencias"""
        self.model = load_model(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.CLASSES = self.metadata['classes']
        self.NUM_FRAMES = self.metadata['num_frames']
        self.FACE_KEYPOINTS_INDICES = self.metadata['face_keypoints_indices']
        
        # Estructura de keypoints (debes ajustar según tu modelo)
        self.TOTAL_KEYPOINTS = (21 * 3 * 2) + (33 * 3) + (20 * 3)  # 2 manos + pose + cara
        
        print(f"✅ Modelo cargado para {len(self.CLASSES)} clases: {self.CLASSES}")
    
    def generate_sequence_from_model(self, target_class, num_sequences=5, sequence_length=None):
        """Generar secuencia usando el modelo y gradientes"""
        print(f"🔍 Generando secuencia para: {target_class}")
        
        if sequence_length is None:
            sequence_length = self.NUM_FRAMES
        
        # Encontrar el índice de la clase objetivo
        target_idx = self.CLASSES.index(target_class)
        
        # Generar múltiples secuencias y promediar
        all_sequences = []
        
        for seq_idx in range(num_sequences):
            print(f"  Secuencia {seq_idx + 1}/{num_sequences}")
            # Inicializar secuencia aleatoria
            sequence = np.random.normal(0, 0.1, (sequence_length, self.TOTAL_KEYPOINTS))
            
            # Optimizar la secuencia para maximizar la predicción de la clase objetivo
            optimized_sequence = self.optimize_sequence_for_class(sequence, target_idx)
            all_sequences.append(optimized_sequence)
        
        # Promediar las secuencias
        final_sequence = np.mean(all_sequences, axis=0)
        
        # Normalizar y suavizar la secuencia
        final_sequence = self.normalize_sequence(final_sequence)
        final_sequence = self.smooth_sequence(final_sequence)
        
        return final_sequence.tolist()
    
    def optimize_sequence_for_class(self, sequence, target_idx, learning_rate=0.01, iterations=50):
        """Optimizar secuencia usando gradientes para maximizar la clase objetivo"""
        sequence_tensor = tf.Variable(sequence, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        best_sequence = sequence_tensor.numpy().copy()
        best_prob = 0.0
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Expandir dimensión para batch
                sequence_batch = tf.expand_dims(sequence_tensor, 0)
                
                # Predecir
                predictions = self.model(sequence_batch, training=False)
                target_prob = predictions[0, target_idx]
                
                # Loss: minimizar probabilidad negativa (maximizar probabilidad objetivo)
                loss = -target_prob
                
                # Regularización para suavidad
                smoothness_loss = tf.reduce_mean(tf.square(sequence_tensor[1:] - sequence_tensor[:-1]))
                loss += 0.1 * smoothness_loss
                
                # Regularización para evitar valores extremos
                magnitude_loss = tf.reduce_mean(tf.square(sequence_tensor))
                loss += 0.01 * magnitude_loss
            
            # Aplicar gradientes
            gradients = tape.gradient(loss, [sequence_tensor])
            optimizer.apply_gradients(zip(gradients, [sequence_tensor]))
            
            current_prob = target_prob.numpy()
            if current_prob > best_prob:
                best_prob = current_prob
                best_sequence = sequence_tensor.numpy().copy()
            
            if i % 10 == 0:
                print(f"    Iteración {i}: Probabilidad = {current_prob:.4f}")
        
        print(f"    Mejor probabilidad alcanzada: {best_prob:.4f}")
        return best_sequence
    
    def generate_sequence_from_pattern(self, target_class):
        """Generar secuencia basada en patrones genéricos"""
        sequence_length = self.NUM_FRAMES
        
        # Usar el índice de la clase para crear un patrón único
        class_idx = self.CLASSES.index(target_class)
        num_classes = len(self.CLASSES)
        
        sequence = []
        
        for frame in range(sequence_length):
            keypoints = []
            progress = frame / sequence_length
            
            # Crear un patrón basado en el índice de la clase
            # Esto asegura que cada clase tenga un movimiento único
            phase_shift = (class_idx / num_classes) * 2 * np.pi
            
            # Patrón para manos - diferente para cada clase
            for hand in range(2):
                hand_direction = -1 if hand == 0 else 1
                
                for i in range(21):
                    # Movimiento circular único para cada clase
                    angle = progress * 2 * np.pi + phase_shift + hand_direction * np.pi * 0.5
                    radius = 0.15 + 0.05 * (i / 21)
                    
                    base_x = 0.5 + radius * np.cos(angle)
                    base_y = 0.4 + radius * np.sin(angle) * 0.7
                    base_z = 0.1 + 0.05 * np.sin(progress * 4 * np.pi + phase_shift)
                    
                    keypoints.extend([base_x, base_y, base_z])
            
            # Pose - ligeramente diferente para cada clase
            for i in range(33):
                # Pequeñas variaciones basadas en la clase
                variation = 0.02 * np.sin(progress * np.pi + phase_shift + i * 0.1)
                base_x = 0.5 + variation
                base_y = 0.5 + variation * 0.5
                base_z = 0.0
                
                keypoints.extend([base_x, base_y, base_z])
            
            # Cara - variaciones sutiles
            for i in range(20):
                variation = 0.01 * np.sin(progress * 2 * np.pi + phase_shift + i * 0.2)
                base_x = 0.5 + variation
                base_y = 0.3 + variation * 0.3
                base_z = 0.0
                
                keypoints.extend([base_x, base_y, base_z])
            
            sequence.append(keypoints)
        
        return sequence
    
    def analyze_class_characteristics(self, target_class, num_samples=100):
        """Analizar qué características hacen única a cada clase"""
        print(f"📊 Analizando características para: {target_class}")
        
        target_idx = self.CLASSES.index(target_class)
        
        # Generar muestras aleatorias y ver cuáles tienen alta probabilidad para esta clase
        high_prob_samples = []
        
        for i in range(num_samples):
            sequence = np.random.normal(0, 0.3, (1, self.NUM_FRAMES, self.TOTAL_KEYPOINTS))
            prediction = self.model.predict(sequence, verbose=0)
            prob = prediction[0, target_idx]
            
            if prob > 0.3:  # Umbral para considerar "alta" probabilidad
                high_prob_samples.append(sequence[0])
        
        if high_prob_samples:
            # Analizar patrones comunes en las muestras de alta probabilidad
            avg_sequence = np.mean(high_prob_samples, axis=0)
            std_sequence = np.std(high_prob_samples, axis=0)
            
            print(f"  Encontradas {len(high_prob_samples)} muestras con alta probabilidad")
            return avg_sequence.tolist()
        else:
            print(f"  No se encontraron muestras con alta probabilidad, usando patrón genérico")
            return None
    
    def _ease_in_out(self, t):
        """Función de easing para movimientos suaves"""
        return t * t * (3 - 2 * t)
    
    def normalize_sequence(self, sequence):
        """Normalizar secuencia para valores válidos"""
        sequence = np.array(sequence)
        
        # Asegurar que los valores estén en rango razonable
        sequence = np.clip(sequence, -2.0, 2.0)
        
        # Escalar a rango típico de coordenadas normalizadas [0, 1]
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence) + 1e-8)
        
        return sequence
    
    def smooth_sequence(self, sequence, window_size=3):
        """Suavizar secuencia con filtro de media móvil"""
        if len(sequence) < window_size:
            return sequence
        
        smoothed = np.zeros_like(sequence)
        
        for i in range(len(sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            smoothed[i] = np.mean(sequence[start:end], axis=0)
        
        return smoothed

def create_sequences_from_model():
    """Crear secuencias usando el modelo entrenado"""
    
    # Crear directorio si no existe
    os.makedirs("sequences", exist_ok=True)
    
    try:
        # Inicializar generador
        generator = SequenceGenerator('sign_language_model.h5', 'model_metadata.json')
        
        print(f"🎯 Generando secuencias para {len(generator.CLASSES)} clases del modelo")
        print("=" * 50)
        
        # Generar secuencia para cada clase que el modelo conoce
        for class_name in generator.CLASSES:
            print(f"\n📝 Procesando: {class_name}")
            
            # Primero intentar análisis de características
            analyzed_sequence = generator.analyze_class_characteristics(class_name)
            
            if analyzed_sequence is not None:
                # Usar la secuencia analizada
                sequence = analyzed_sequence
                method = "análisis de características"
            else:
                # Intentar generación por gradientes
                try:
                    sequence = generator.generate_sequence_from_model(class_name)
                    method = "optimización por gradientes"
                except Exception as e:
                    print(f"   ⚠️  Error con gradientes: {e}")
                    print(f"   🔄 Usando patrón genérico para: {class_name}")
                    sequence = generator.generate_sequence_from_pattern(class_name)
                    method = "patrón genérico"
            
            # Validar la secuencia con el modelo
            validation_sequence = np.array([sequence])
            predictions = generator.model.predict(validation_sequence, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0, predicted_class_idx]
            predicted_class = generator.CLASSES[predicted_class_idx]
            
            print(f"   ✅ Secuencia generada ({method})")
            print(f"   📊 Validación: {predicted_class} ({confidence*100:.1f}% confianza)")
            
            # Guardar secuencia
            filename = f"sequences/{class_name}.json"
            with open(filename, 'w') as f:
                json.dump(sequence, f, indent=2)
            
            print(f"   💾 Guardado en: {filename}")
        
        print(f"\n🎉 Todas las secuencias generadas exitosamente!")
        print(f"📁 Ubicación: sequences/")
        print(f"📊 Resumen: {len(generator.CLASSES)} secuencias creadas")
        
        # Mostrar resumen final
        print("\n📋 Resumen de secuencias creadas:")
        for class_name in generator.CLASSES:
            filename = f"sequences/{class_name}.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    sequence = json.load(f)
                print(f"   • {class_name}: {len(sequence)} frames")
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        print("🔧 Intentando método alternativo...")
        create_fallback_sequences_from_metadata()

def create_fallback_sequences_from_metadata():
    """Crear secuencias básicas usando solo la metadata del modelo"""
    
    try:
        # Cargar metadata para obtener las clases
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        classes = metadata['classes']
        num_frames = metadata['num_frames']
        
        print(f"📝 Creando secuencias básicas para {len(classes)} clases")
        
        # Crear directorio si no existe
        os.makedirs("sequences", exist_ok=True)
        
        for class_idx, class_name in enumerate(classes):
            sequence = []
            
            print(f"  Generando: {class_name}")
            
            for frame in range(num_frames):
                keypoints = []
                progress = frame / num_frames
                
                # Crear un patrón único basado en el índice de la clase
                class_phase = (class_idx / len(classes)) * 2 * np.pi
                
                # Manos - diferentes para cada clase
                for hand in range(2):
                    hand_sign = -1 if hand == 0 else 1
                    
                    for i in range(21):
                        # Movimiento único por clase
                        angle = progress * 2 * np.pi + class_phase + hand_sign * np.pi
                        radius = 0.1 + 0.08 * (class_idx / len(classes))
                        
                        base_x = 0.5 + radius * np.cos(angle)
                        base_y = 0.4 + radius * np.sin(angle) * 0.8
                        base_z = 0.1
                        
                        keypoints.extend([base_x, base_y, base_z])
                
                # Pose
                for i in range(33):
                    keypoints.extend([0.5, 0.5, 0.0])
                
                # Cara
                for i in range(20):
                    keypoints.extend([0.5, 0.3, 0.0])
                
                sequence.append(keypoints)
            
            # Guardar secuencia
            filename = f"sequences/{class_name}.json"
            with open(filename, 'w') as f:
                json.dump(sequence, f)
            
            print(f"    ✅ {filename}")
        
        print(f"\n✅ Secuencias básicas creadas para {len(classes)} clases")
        
    except Exception as e:
        print(f"❌ Error en método alternativo: {e}")
        print("💡 Asegúrate de que el archivo model_metadata.json existe")

if __name__ == "__main__":
    create_sequences_from_model()