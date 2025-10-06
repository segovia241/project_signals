# debug_model_predictions.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

def debug_model_predictions():
    print("🔍 Diagnóstico detallado del modelo...")
    
    try:
        model = load_model('sign_language_model.h5')
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        CLASSES = metadata['classes']
        NUM_FRAMES = metadata['num_frames']
        TOTAL_KEYPOINTS = metadata['total_keypoints']
        
        print(f"✅ Modelo cargado - Clases: {CLASSES}")
        
        # Probar con diferentes tipos de datos de entrada
        test_cases = [
            ("Ceros (sin detección)", np.zeros((1, NUM_FRAMES, TOTAL_KEYPOINTS))),
            ("Aleatorio bajo", np.random.normal(0, 0.1, (1, NUM_FRAMES, TOTAL_KEYPOINTS))),
            ("Aleatorio medio", np.random.normal(0, 0.3, (1, NUM_FRAMES, TOTAL_KEYPOINTS))),
            ("Aleatorio alto", np.random.normal(0, 0.5, (1, NUM_FRAMES, TOTAL_KEYPOINTS))),
            ("Unos (máxima activación)", np.ones((1, NUM_FRAMES, TOTAL_KEYPOINTS))),
        ]
        
        for name, test_data in test_cases:
            prediction = model.predict(test_data, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            print(f"\n🧪 {name}:")
            print(f"   → Predicción: {CLASSES[predicted_idx]} ({confidence:.4f})")
            
            # Mostrar top 3 predicciones
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            for i, idx in enumerate(top_3_idx):
                print(f"   {i+1}. {CLASSES[idx]}: {prediction[0][idx]:.4f}")
        
        # Verificar si el modelo está sesgado
        print(f"\n📊 Análisis de sesgo:")
        all_predictions = []
        for _ in range(20):
            test_data = np.random.normal(0, 0.2, (1, NUM_FRAMES, TOTAL_KEYPOINTS))
            pred = model.predict(test_data, verbose=0)
            all_predictions.append(np.argmax(pred[0]))
        
        from collections import Counter
        prediction_counts = Counter(all_predictions)
        print(f"   Distribución: { {CLASSES[k]: v for k, v in prediction_counts.items()} }")
        
        if len(prediction_counts) == 1:
            print("   ❌ ALERTA: Modelo siempre predice la misma clase!")
            most_common_class = CLASSES[prediction_counts.most_common(1)[0][0]]
            print(f"   💡 Solución: El modelo necesita reentrenamiento")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_model_predictions()