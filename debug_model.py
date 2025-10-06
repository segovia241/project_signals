# debug_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

def debug_model():
    print("🔍 Iniciando diagnóstico del modelo...")
    
    # Cargar modelo y metadata
    try:
        model = load_model('sign_language_model.h5')
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Modelo cargado exitosamente")
        print(f"📊 Clases: {metadata['classes']}")
        print(f"📏 Input shape: ({metadata['num_frames']}, {metadata['total_keypoints']})")
        
        # Crear datos de prueba
        num_frames = metadata['num_frames']
        total_keypoints = metadata['total_keypoints']
        
        # Prueba 1: Datos aleatorios
        print("\n🧪 Prueba 1: Datos aleatorios")
        test_data_random = np.random.random((1, num_frames, total_keypoints))
        prediction = model.predict(test_data_random, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"   Predicción: {metadata['classes'][predicted_class_idx]}")
        print(f"   Confianza: {confidence:.4f}")
        print(f"   Todas las confianzas: {[f'{c:.4f}' for c in prediction[0]]}")
        
        # Prueba 2: Datos ceros (como cuando no hay detección)
        print("\n🧪 Prueba 2: Datos con ceros")
        test_data_zeros = np.zeros((1, num_frames, total_keypoints))
        prediction_zeros = model.predict(test_data_zeros, verbose=0)
        predicted_class_idx_zeros = np.argmax(prediction_zeros)
        confidence_zeros = np.max(prediction_zeros)
        
        print(f"   Predicción: {metadata['classes'][predicted_class_idx_zeros]}")
        print(f"   Confianza: {confidence_zeros:.4f}")
        
        # Prueba 3: Verificar si el modelo está sesgado
        print("\n🧪 Prueba 3: Distribución de predicciones")
        test_samples = []
        for i in range(5):
            sample = np.random.normal(0, 0.1, (1, num_frames, total_keypoints))
            test_samples.append(sample)
        
        predictions_count = {}
        for i, sample in enumerate(test_samples):
            pred = model.predict(sample, verbose=0)
            class_idx = np.argmax(pred)
            class_name = metadata['classes'][class_idx]
            predictions_count[class_name] = predictions_count.get(class_name, 0) + 1
            print(f"   Muestra {i+1}: {class_name} ({np.max(pred):.4f})")
        
        print(f"\n📈 Conteo de predicciones: {predictions_count}")
        
        # Verificar si hay una clase dominante
        if len(predictions_count) == 1:
            print("❌ ALERTA: El modelo siempre predice la misma clase")
            print("   Posibles causas:")
            print("   - Overfitting durante el entrenamiento")
            print("   - Desbalance de clases en los datos")
            print("   - Modelo no entrenado correctamente")
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()