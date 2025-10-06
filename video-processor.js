class VideoProcessor {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.websocket = null;
        this.isProcessing = false;
        this.fps = 10; // Frames por segundo
        
        this.initializeElements();
    }
    
    initializeElements() {
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('connectBtn').addEventListener('click', () => this.connectWebSocket());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopProcessing());
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480 
                } 
            });
            this.video.srcObject = this.stream;
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('connectBtn').disabled = false;
            
            this.updateStatus('Cámara activa', 'connected');
            
        } catch (error) {
            console.error('Error al acceder a la cámara:', error);
            this.updateStatus('Error de cámara: ' + error.message, 'disconnected');
        }
    }
    
    connectWebSocket() {
        this.websocket = new WebSocket('ws://localhost:8000/ws');
        
        this.websocket.onopen = () => {
            this.updateStatus('Conectado al servidor', 'connected');
            this.startProcessing();
            document.getElementById('connectBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.displayResults(data);
        };
        
        this.websocket.onclose = () => {
            this.updateStatus('Desconectado del servidor', 'disconnected');
            this.stopProcessing();
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Error de conexión', 'disconnected');
        };
    }
    
    startProcessing() {
        this.isProcessing = true;
        this.processFrame();
    }
    
    async processFrame() {
        if (!this.isProcessing || !this.websocket) return;
        
        // Configurar canvas con las dimensiones del video
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
        
        // Dibujar frame actual en canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convertir a JPEG con calidad reducida para optimización
        const imageData = this.canvas.toDataURL('image/jpeg', 0.7);
        
        // Enviar frame al servidor
        const message = {
            type: 'frame',
            data: imageData,
            timestamp: Date.now(),
            dimensions: {
                width: this.canvas.width,
                height: this.canvas.height
            }
        };
        
        this.websocket.send(JSON.stringify(message));
        
        // Programar próximo frame
        setTimeout(() => this.processFrame(), 1000 / this.fps);
    }
    
    displayResults(data) {
        const resultsDiv = document.getElementById('results');
        if (data.type === 'analysis') {
            resultsDiv.innerHTML = JSON.stringify(data.data, null, 2);
            
            // Ejemplo: Cambiar color basado en intensidad
            if (data.data.mean_intensity) {
                const intensity = data.data.mean_intensity;
                resultsDiv.style.borderLeft = `5px solid rgb(${intensity}, 100, 100)`;
            }
        }
    }
    
    stopProcessing() {
        this.isProcessing = false;
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('connectBtn').disabled = true;
        document.getElementById('stopBtn').disabled = true;
        
        this.updateStatus('Procesamiento detenido', 'disconnected');
        document.getElementById('results').innerHTML = 'Esperando datos...';
    }
    
    updateStatus(message, type) {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = `Estado: ${message}`;
        statusDiv.className = `status ${type}`;
    }
}

// Inicializar cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
    new VideoProcessor();
});