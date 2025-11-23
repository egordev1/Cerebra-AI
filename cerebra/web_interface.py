import asyncio
import websockets
import json
import logging
import threading
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import nest_asyncio
from pydantic import BaseModel
import time

from .core import CerebraAI
from .gpu_acceleration import GPUAccelerator
from .quantization import ModelQuantizer

logger = logging.getLogger(__name__)

# Применяем nest_asyncio для корректной работы с asyncio в Jupyter и других средах
nest_asyncio.apply()

class MessageRequest(BaseModel):
    message: str
    model_type: str = "L1"
    stream: bool = False

class CerebraWebInterface:
    """Класс для веб-интерфейса Cerebra AI с WebSocket поддержкой"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Cerebra AI Web Interface", version="1.0.0")
        self.cerebra_ai = CerebraAI()
        self.gpu_accelerator = GPUAccelerator()
        self.quantizer = ModelQuantizer()
        
        # Хранилище активных соединений
        self.active_connections: Dict[WebSocket, str] = {}
        
        # Настройка маршрутов
        self.setup_routes()
        
    def setup_routes(self):
        """Настройка маршрутов FastAPI"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_index():
            """Главная страница веб-интерфейса"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cerebra AI Web Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                    .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
                    .message { margin-bottom: 10px; padding: 5px; }
                    .user-message { background: #d4edda; border-radius: 5px; }
                    .ai-message { background: #cce5ff; border-radius: 5px; }
                    .input-area { display: flex; }
                    input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                    button { padding: 10px 20px; margin-left: 10px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #0056b3; }
                    select { padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-right: 10px; }
                    .status { padding: 10px; background: #e2e3e5; border-radius: 5px; margin-bottom: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Cerebra AI Web Interface</h1>
                    <div class="status" id="status">Состояние: Готов к работе</div>
                    <div class="chat-box" id="chat-box"></div>
                    <div class="input-area">
                        <select id="model-select">
                            <option value="L1">Synthesis-L1 (~29M параметров)</option>
                            <option value="L2">Synthesis-L2 (~220M параметров)</option>
                            <option value="L3">Synthesis-L3 (~1.3B параметров)</option>
                        </select>
                        <input type="text" id="message-input" placeholder="Введите сообщение..." onkeypress="if(event.key==='Enter') sendMessage()">
                        <button onclick="sendMessage()">Отправить</button>
                    </div>
                </div>
                
                <script>
                    const chatBox = document.getElementById('chat-box');
                    const messageInput = document.getElementById('message-input');
                    const modelSelect = document.getElementById('model-select');
                    const statusDiv = document.getElementById('status');
                    
                    let socket;
                    
                    function connectWebSocket() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws`;
                        
                        socket = new WebSocket(wsUrl);
                        
                        socket.onopen = function(event) {
                            statusDiv.innerHTML = 'Состояние: Подключен к Cerebra AI';
                            statusDiv.style.backgroundColor = '#d4edda';
                        };
                        
                        socket.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            addMessage(data.message, data.type);
                            
                            if (data.type === 'ai') {
                                statusDiv.innerHTML = 'Состояние: Готов к работе';
                                statusDiv.style.backgroundColor = '#e2e3e5';
                            }
                        };
                        
                        socket.onclose = function(event) {
                            statusDiv.innerHTML = 'Состояние: Соединение закрыто, переподключаюсь...';
                            statusDiv.style.backgroundColor = '#f8d7da';
                            setTimeout(connectWebSocket, 3000); // Пытаемся переподключиться через 3 секунды
                        };
                        
                        socket.onerror = function(error) {
                            console.error('WebSocket error:', error);
                            statusDiv.innerHTML = 'Состояние: Ошибка соединения';
                            statusDiv.style.backgroundColor = '#f8d7da';
                        };
                    }
                    
                    function addMessage(message, type) {
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message ${type}-message`;
                        messageDiv.textContent = message;
                        chatBox.appendChild(messageDiv);
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                    
                    function sendMessage() {
                        const message = messageInput.value.trim();
                        if (!message) return;
                        
                        const modelType = modelSelect.value;
                        
                        const data = {
                            message: message,
                            model_type: modelType
                        };
                        
                        socket.send(JSON.stringify(data));
                        
                        addMessage(message, 'user');
                        statusDiv.innerHTML = 'Состояние: Отправка сообщения...';
                        statusDiv.style.backgroundColor = '#fff3cd';
                        
                        messageInput.value = '';
                    }
                    
                    // Инициализация
                    connectWebSocket();
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections[websocket] = "active"
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    user_message = message_data.get("message", "")
                    model_type = message_data.get("model_type", "L1")
                    
                    # Отправляем сообщение пользователя
                    await websocket.send_text(json.dumps({
                        "type": "user",
                        "message": user_message
                    }))
                    
                    # Генерируем ответ от AI
                    ai_response = self.cerebra_ai.generate_response(user_message, model_type)
                    
                    # Отправляем ответ от AI
                    await websocket.send_text(json.dumps({
                        "type": "ai", 
                        "message": ai_response
                    }))
                    
            except WebSocketDisconnect:
                if websocket in self.active_connections:
                    del self.active_connections[websocket]
    
    def run(self, debug: bool = False):
        """Запуск веб-сервера"""
        logger.info(f"Запуск веб-интерфейса на {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            log_level="info"
        )
    
    def run_in_thread(self, debug: bool = False):
        """Запуск веб-сервера в отдельном потоке"""
        thread = threading.Thread(target=self.run, args=(debug,))
        thread.daemon = True
        thread.start()
        return thread

class CerebraAPIClient:
    """Клиент для взаимодействия с API Cerebra AI"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def send_message(self, message: str, model_type: str = "L1") -> str:
        """Отправка сообщения через WebSocket"""
        uri = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws"
        
        async with websockets.connect(uri) as websocket:
            # Отправляем сообщение
            await websocket.send(json.dumps({
                "message": message,
                "model_type": model_type
            }))
            
            # Получаем ответ
            response = await websocket.recv()
            response_data = json.loads(response)
            
            return response_data.get("message", "")

def create_web_interface(host: str = "0.0.0.0", port: int = 8000):
    """Создание и запуск веб-интерфейса"""
    web_interface = CerebraWebInterface(host, port)
    return web_interface