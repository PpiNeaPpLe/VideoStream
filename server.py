from flask import Flask, Response
import mss
import cv2
import numpy as np
import time
import threading
import socket
import json
import pyautogui

app = Flask(__name__)

# Performance configuration - adjust these for better performance
TARGET_WIDTH = 1920   # Target width (will scale proportionally)
TARGET_HEIGHT = 1080  # Target height (will scale proportionally)
JPEG_QUALITY = 90     # JPEG quality (higher = better quality but slower)
TARGET_FPS = 45       # Target frame rate

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
    """Serve the main page with embedded video stream"""
    return '''
    <html>
    <head>
        <title>Screen Stream</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            img {
                max-width: 100%;
                max-height: 100vh;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <img src="/video_feed" />
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the screen capture thread
    capture_thread = threading.Thread(target=streamer.capture_loop, daemon=True)
    capture_thread.start()

    # Start the mouse control server thread
    mouse_thread = threading.Thread(target=mouse_controller.mouse_server_loop, daemon=True)
    mouse_thread.start()

    print("Screen streaming server starting...")
    print("Open your browser and go to: http://192.168.x.x:8080")
    print("(Replace 192.168.x.x with your computer's IP address)")
    print("Or try: http://localhost:8080 for local access")
    print("Mouse control server running on port 8081")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host='0.0.0.0', port=8080, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping server...")
        streamer.stop()
        mouse_controller.stop()
        print("Server stopped.")
