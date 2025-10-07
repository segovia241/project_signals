# debug_live_data_colored.py
import cv2
import numpy as np
import mediapipe as mp

def debug_live_data():
    print("🎥 Probando datos en vivo de MediaPipe...")
    
    # Inicializar MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    # Colores
    COLOR_HAND = (0, 255, 0)   # Verde
    COLOR_FACE = (255, 0, 0)   # Azul
    COLOR_POSE = (0, 0, 255)   # Rojo
    
    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    
    print("📹 Mostrando información de keypoints (presiona 'q' para salir):")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        frame_count += 1
        if frame_count % 30 == 0:
            keypoints = []
            has_hand_detection = False
            
            # --- Manos ---
            hand_points = 0
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    has_hand_detection = True
                    hand_points += 21
                    for lm in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, COLOR_HAND, -1)
                        keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0.0] * (21 * 3))
            
            # --- Pose ---
            pose_points = 0
            if results.pose_landmarks:
                pose_points = 33
                for lm in results.pose_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, COLOR_POSE, -1)
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0.0] * (33 * 3))
            
            # --- Cara (solo algunos puntos ejemplo) ---
            face_points = 0
            if results.face_landmarks:
                # Elegimos solo 20 puntos de la cara para depuración
                selected_indices = list(range(0, 468, 23))[:20]  # 468 puntos en total
                face_points = len(selected_indices)
                for idx in selected_indices:
                    lm = results.face_landmarks.landmark[idx]
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, COLOR_FACE, -1)
                    keypoints.extend([lm.x, lm.y, lm.z])
            else:
                keypoints.extend([0.0] * (20 * 3))
            
            keypoint_sum = np.sum(np.abs(keypoints))
            
            print(f"Frame {frame_count}:")
            print(f"  Manos detectadas: {has_hand_detection}")
            print(f"  Puntos mano: {hand_points}, pose: {pose_points}, cara: {face_points}")
            print(f"  Suma keypoints: {keypoint_sum:.2f}")
            print(f"  ¿Gesto válido?: {has_hand_detection and keypoint_sum > 1.0}")
            print("-" * 50)
        
        # Mostrar video con puntos dibujados
        cv2.imshow('Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    debug_live_data()
