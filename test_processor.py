# test_processor.py
from video_processor import VideoProcessor
def test_processor():
    print("🧪 Probando inicialización de VideoProcessor...")
    processor = VideoProcessor()
    
    if processor.initialized:
        print("✅ TEST EXITOSO - Processor inicializado")
        print(f"Clases: {processor.CLASSES}")
    else:
        print("❌ TEST FALLIDO")
        if hasattr(processor, 'initialization_error'):
            print(f"Error: {processor.initialization_error}")

if __name__ == "__main__":
    test_processor()