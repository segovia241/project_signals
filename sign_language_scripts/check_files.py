# check_files.py
import os

files = ['sign_language_model.h5', 'model_metadata.json', 'video_processor.py', 'main.py']

print("📁 Verificando archivos...")
for file in files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"{status} {file}: {'EXISTE' if exists else 'NO EXISTE'}")
    
    if exists and file.endswith('.h5'):
        size = os.path.getsize(file)
        print(f"   Tamaño: {size / (1024*1024):.2f} MB")