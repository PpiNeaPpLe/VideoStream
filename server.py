from flask import Flask, Response, render_template_string
import mss
import cv2
import numpy as np
import time
import threading
import socket
import struct
import json
import pyautogui
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Performance configuration - adjust these for better performance
TARGET_WIDTH = 1280   # Target width (will scale proportionally) - reduced for better performance
TARGET_HEIGHT = 720   # Target height (will scale proportionally) - reduced for better performance
JPEG_QUALITY = 70     # JPEG quality (higher = better quality but slower) - reduced for speed
TARGET_FPS = 30       # Target frame rate - reduced for better performance

class ScreenStreamer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def get_display_bounds(self):
        """Get the bounds of primary monitor"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # monitors[1] is the primary display
            return monitor

    def capture_loop(self):
        """Continuously capture screen frames"""
        monitor = self.get_display_bounds()
        print(f"Capturing screen: {monitor['width']}x{monitor['height']}")
        print(f"Streaming at: {TARGET_WIDTH}x{TARGET_HEIGHT} (scaled), {TARGET_FPS} FPS target, JPEG quality: {JPEG_QUALITY}")

        with mss.mss() as sct:
            while self.running:
                start_time = time.time()

                # Capture screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)

                # Convert BGRA to BGR (remove alpha channel)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Scale down for better performance
                scale_factor = min(TARGET_WIDTH / frame.shape[1], TARGET_HEIGHT / frame.shape[0])
                if scale_factor < 1.0:
                    new_width = int(frame.shape[1] * scale_factor)
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = encoded_frame.tobytes()

                # Update the latest frame
                with self.lock:
                    self.frame = frame_bytes

                # Control frame rate
                elapsed = time.time() - start_time
                target_time = 1.0 / TARGET_FPS
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.frame

    def stop(self):
        """Stop the capture loop"""
        self.running = False

class VideoStreamer:
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = True
        print(f"Video streamer initialized on port {port}")

    def video_server_loop(self):
        """Main loop for video streaming server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Video streaming server listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"Video client connected from {addr}")
                    threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
                except Exception as e:
                    if self.running:
                        print(f"Error accepting video client connection: {e}")

        except Exception as e:
            print(f"Video server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def handle_client(self, client_socket):
        """Handle individual video client connection"""
        try:
            while self.running:
                # Get latest frame from screen streamer
                frame_data = streamer.get_frame()
                if frame_data is None:
                    time.sleep(0.01)  # Wait a bit if no frame available
                    continue

                # Send frame size (4 bytes, big-endian)
                frame_size = len(frame_data)
                size_data = struct.pack('>L', frame_size)
                client_socket.sendall(size_data)

                # Send frame data
                client_socket.sendall(frame_data)

                # Control frame rate (target ~20 FPS for better performance)
                time.sleep(1/20)

        except Exception as e:
            print(f"Video client error: {e}")
        finally:
            client_socket.close()

    def stop(self):
        """Stop the video server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

class MouseController:
    def __init__(self, host='0.0.0.0', port=8081):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = True

        # Get screen dimensions for coordinate mapping
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Mouse controller initialized for screen: {self.screen_width}x{self.screen_height}")

    def handle_mouse_command(self, command):
        """Execute mouse command received from client"""
        try:
            cmd_type = command.get('type')

            if cmd_type == 'move':
                # Move mouse to absolute coordinates
                x = int(command.get('x', 0))
                y = int(command.get('y', 0))
                pyautogui.moveTo(x, y)

            elif cmd_type == 'click':
                # Mouse click (left by default)
                button = command.get('button', 'left')
                pyautogui.click(button=button)

            elif cmd_type == 'scroll':
                # Mouse scroll
                x = int(command.get('x', 0))
                y = int(command.get('y', 0))
                clicks = int(command.get('clicks', 1))
                pyautogui.scroll(clicks, x, y)

            elif cmd_type == 'drag':
                # Mouse drag
                x = int(command.get('x', 0))
                y = int(command.get('y', 0))
                pyautogui.dragTo(x, y, duration=0.1)

        except Exception as e:
            print(f"Error executing mouse command: {e}")

    def mouse_server_loop(self):
        """Main loop for mouse control server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Mouse control server listening on port {self.port}")

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"Mouse control client connected from {addr}")

                    # Handle client connection
                    data = b''
                    while self.running:
                        chunk = client_socket.recv(1024)
                        if not chunk:
                            break

                        data += chunk

                        # Process complete JSON messages
                        while b'\n' in data:
                            message, data = data.split(b'\n', 1)
                            if message:
                                try:
                                    command = json.loads(message.decode('utf-8'))
                                    self.handle_mouse_command(command)
                                except json.JSONDecodeError as e:
                                    print(f"Invalid JSON received: {e}")

                    client_socket.close()
                    print(f"Mouse control client disconnected from {addr}")

                except Exception as e:
                    print(f"Mouse server error: {e}")
                    break

        except Exception as e:
            print(f"Failed to start mouse server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def stop(self):
        """Stop the mouse control server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

# WebSocket event handlers for mouse control
@socketio.on('mouse_move')
def handle_mouse_move(data):
    """Handle mouse move events from web client"""
    try:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        pyautogui.moveTo(x, y)
        emit('mouse_ack', {'status': 'moved', 'x': x, 'y': y})
    except Exception as e:
        emit('mouse_error', {'error': str(e)})

@socketio.on('mouse_click')
def handle_mouse_click(data):
    """Handle mouse click events from web client"""
    try:
        button = data.get('button', 'left')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))

        # Move to position first, then click
        pyautogui.moveTo(x, y)
        pyautogui.click(button=button)
        emit('mouse_ack', {'status': 'clicked', 'button': button, 'x': x, 'y': y})
    except Exception as e:
        emit('mouse_error', {'error': str(e)})

@socketio.on('mouse_scroll')
def handle_mouse_scroll(data):
    """Handle mouse scroll events from web client"""
    try:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        delta_y = int(data.get('deltaY', 0))

        # Convert scroll delta to clicks (normalize)
        clicks = delta_y // 50  # Adjust divisor for sensitivity
        if clicks == 0:
            clicks = 1 if delta_y > 0 else -1

        pyautogui.scroll(clicks, x, y)
        emit('mouse_ack', {'status': 'scrolled', 'clicks': clicks, 'x': x, 'y': y})
    except Exception as e:
        emit('mouse_error', {'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Web client connected for mouse control")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Web client disconnected")

# Global instances
streamer = ScreenStreamer()
mouse_controller = MouseController()

def generate_frames():
    """Generator function for MJPEG stream"""
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.01)  # Small delay to prevent busy waiting

@app.route('/')
def index():
    """Serve the main page with embedded video stream and mouse control"""
    # Get screen dimensions for coordinate scaling
    screen_bounds = streamer.get_display_bounds()
    screen_width = screen_bounds['width']
    screen_height = screen_bounds['height']

    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Screen Stream with Mouse Control</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                font-family: Arial, sans-serif;
                overflow: hidden;
            }}
            .container {{
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .video-container {{
                position: relative;
                cursor: crosshair;
                border: 2px solid #333;
                box-shadow: 0 0 20px rgba(0,255,0,0.3);
            }}
            img {{
                max-width: 100%;
                max-height: 90vh;
                object-fit: contain;
                display: block;
            }}
            .status {{
                position: fixed;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.8);
                color: #00ff00;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;
            }}
            .instructions {{
                position: fixed;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.8);
                color: #fff;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                max-width: 300px;
            }}
            .cursor-indicator {{
                position: absolute;
                width: 20px;
                height: 20px;
                background: rgba(255,0,0,0.5);
                border: 2px solid red;
                border-radius: 50%;
                pointer-events: none;
                z-index: 999;
                transition: transform 0.1s ease;
            }}
        </style>
    </head>
    <body>
        <div class="status" id="status">Connecting...</div>
        <div class="instructions">
            <strong>Mouse Control:</strong><br>
            • Click anywhere on the stream to control mouse<br>
            • Right-click for right mouse button<br>
            • Use mouse wheel to scroll<br>
            • Green border indicates active control
        </div>
        <div class="container">
            <div class="video-container" id="videoContainer">
                <img src="/video_feed" id="videoStream" />
                <div class="cursor-indicator" id="cursorIndicator" style="display: none;"></div>
            </div>
        </div>

        <script>
            const videoStream = document.getElementById('videoStream');
            const videoContainer = document.getElementById('videoContainer');
            const cursorIndicator = document.getElementById('cursorIndicator');
            const statusDiv = document.getElementById('status');

            // Screen dimensions from server
            const SCREEN_WIDTH = {screen_width};
            const SCREEN_HEIGHT = {screen_height};

            // Initialize Socket.IO
            const socket = io();

            // Connection status
            socket.on('connect', function() {{
                statusDiv.textContent = 'Connected - Mouse Control Active';
                statusDiv.style.color = '#00ff00';
                videoContainer.style.borderColor = '#00ff00';
            }});

            socket.on('disconnect', function() {{
                statusDiv.textContent = 'Disconnected';
                statusDiv.style.color = '#ff0000';
                videoContainer.style.borderColor = '#ff0000';
            }});

            socket.on('mouse_ack', function(data) {{
                console.log('Mouse action acknowledged:', data);
                // Briefly flash the cursor indicator
                cursorIndicator.style.display = 'block';
                setTimeout(() => {{
                    cursorIndicator.style.display = 'none';
                }}, 200);
            }});

            socket.on('mouse_error', function(data) {{
                console.error('Mouse control error:', data.error);
                statusDiv.textContent = 'Error: ' + data.error;
                statusDiv.style.color = '#ffaa00';
                setTimeout(() => {{
                    statusDiv.textContent = 'Connected - Mouse Control Active';
                    statusDiv.style.color = '#00ff00';
                }}, 3000);
            }});

            // Mouse event handlers
            videoStream.addEventListener('mousemove', function(e) {{
                const rect = videoStream.getBoundingClientRect();
                const scaleX = SCREEN_WIDTH / rect.width;
                const scaleY = SCREEN_HEIGHT / rect.height;

                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);

                // Update cursor indicator position
                cursorIndicator.style.left = (e.clientX - rect.left - 10) + 'px';
                cursorIndicator.style.top = (e.clientY - rect.top - 10) + 'px';

                socket.emit('mouse_move', {{ x: x, y: y }});
            }});

            videoStream.addEventListener('click', function(e) {{
                e.preventDefault();
                const rect = videoStream.getBoundingClientRect();
                const scaleX = SCREEN_WIDTH / rect.width;
                const scaleY = SCREEN_HEIGHT / rect.height;

                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);

                const button = e.button === 2 ? 'right' : 'left';
                socket.emit('mouse_click', {{ button: button, x: x, y: y }});
            }});

            videoStream.addEventListener('contextmenu', function(e) {{
                e.preventDefault(); // Prevent default context menu
            }});

            videoStream.addEventListener('wheel', function(e) {{
                e.preventDefault();
                const rect = videoStream.getBoundingClientRect();
                const scaleX = SCREEN_WIDTH / rect.width;
                const scaleY = SCREEN_HEIGHT / rect.height;

                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);

                socket.emit('mouse_scroll', {{
                    x: x,
                    y: y,
                    deltaY: e.deltaY
                }});
            }});

            // Show cursor indicator on hover
            videoContainer.addEventListener('mouseenter', function() {{
                cursorIndicator.style.display = 'block';
            }});

            videoContainer.addEventListener('mouseleave', function() {{
                cursorIndicator.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    '''
    return html_template

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Initialize controllers (after class definitions)
video_streamer = VideoStreamer(host='0.0.0.0', port=8080)
mouse_controller = MouseController(host='0.0.0.0', port=8081)

if __name__ == '__main__':
    # Start the screen capture thread
    capture_thread = threading.Thread(target=streamer.capture_loop, daemon=True)
    capture_thread.start()

    # Start the video streaming server thread
    video_thread = threading.Thread(target=video_streamer.video_server_loop, daemon=True)
    video_thread.start()

    # Start the mouse control server thread
    mouse_thread = threading.Thread(target=mouse_controller.mouse_server_loop, daemon=True)
    mouse_thread.start()

    print("Screen streaming server starting...")
    print("Python client connects to: 192.168.x.x:8080 (video) and :8081 (mouse)")
    print("Web interface available at: http://192.168.x.x:8082")
    print("(Replace 192.168.x.x with your computer's IP address)")
    print("Or try: http://localhost:8082 for local web access")
    print("Press Ctrl+C to stop the server")

    try:
        socketio.run(app, host='0.0.0.0', port=8082)  # Web interface on port 8082
    except KeyboardInterrupt:
        print("\nStopping server...")
        streamer.stop()
        video_streamer.stop()
        mouse_controller.stop()
        print("Server stopped.")
