import uvicorn
from api.endpoints import create_app
from config.settings import APP_CONFIG

# Crear aplicación FastAPI
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        reload=APP_CONFIG["reload"],
        log_level=APP_CONFIG["log_level"]
    )