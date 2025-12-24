import cv2
import socket
import struct
import numpy as np
import json
import threading
import argparse
import time
import os
import logging

# Set up logging (will be configured based on command line args)
logger = logging.getLogger(__name__)

def detect_faces(frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces in a frame using Haar cascades"""
    logger.debug("Starting face detection on frame")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    logger.debug(f"Converted frame to grayscale, shape: {gray.shape}")

    # Load the face cascade classifier
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    logger.debug(f"Face cascade path: {face_cascade_path}")

    # Check if cascade file exists
    if not os.path.exists(face_cascade_path):
        logger.error(f"Haar cascade file not found at {face_cascade_path}")
        return []

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        logger.error("Failed to load face cascade classifier")
        return []

    logger.debug("Face cascade classifier loaded successfully")

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    logger.info(f"Face detection completed - found {len(faces)} faces")
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            logger.info(f"Face {i+1}: position ({x}, {y}), size {w}x{h}")

    return faces

def draw_face_rectangles(frame, faces):
    """Draw rectangles around detected faces for visualization"""
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw center point
        center_x = x + w//2
        center_y = y + h//2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events from OpenCV window"""
    mouse_socket, window_info = param

    if mouse_socket is None:
        return

    try:
        # Scale coordinates back to original screen size
        scale_x = window_info['original_width'] / window_info['window_width']
        scale_y = window_info['original_height'] / window_info['window_height']

        screen_x = int(x * scale_x)
        screen_y = int(y * scale_y)

        if event == cv2.EVENT_MOUSEMOVE:
            # Send mouse move command
            command = {
                'type': 'move',
                'x': screen_x,
                'y': screen_y
            }

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Send left mouse click
            command = {
                'type': 'click',
                'button': 'left',
                'x': screen_x,
                'y': screen_y
            }

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Send right mouse click
            command = {
                'type': 'click',
                'button': 'right',
                'x': screen_x,
                'y': screen_y
            }

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Handle mouse wheel
            if flags > 0:
                clicks = 1  # Scroll up
            else:
                clicks = -1  # Scroll down

            command = {
                'type': 'scroll',
                'x': screen_x,
                'y': screen_y,
                'clicks': clicks
            }

        else:
            return  # Ignore other events

        # Send command to server
        try:
            message = json.dumps(command) + '\n'
            mouse_socket.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send mouse command: {e}")

    except Exception as e:
        print(f"Mouse callback error: {e}")

def receive_stream(host, port=8080, mouse_port=8081, enable_face_tracking=False):
    """Receive and display video stream from server with mouse control"""
    logger.info(f"Starting video stream client - Host: {host}, Video Port: {port}, Mouse Port: {mouse_port}, Face Tracking: {enable_face_tracking}")

    # Video stream socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Mouse control socket
    mouse_socket = None
    try:
        mouse_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connecting to mouse control server at {host}:{mouse_port}")
        mouse_socket.connect((host, mouse_port))
        logger.info(f"Mouse control connected successfully to {host}:{mouse_port}")
    except Exception as e:
        logger.error(f"Mouse control connection failed: {e}")
        mouse_socket = None

    logger.info(f"Connecting to video stream at {host}:{port}...")
    try:
        client_socket.connect((host, port))
        logger.info("Video stream connected! Receiving stream...")
    except Exception as e:
        logger.error(f"Failed to connect to video stream: {e}")
        if mouse_socket:
            mouse_socket.close()
        return

    payload_size = struct.calcsize('>L')
    data = b''
    window_name = 'Display 1 Stream'
    mouse_callback_set = False  # Track if mouse callback has been set

    try:
        while True:
            # Receive frame size
            while len(data) < payload_size:
                data += client_socket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('>L', packed_msg_size)[0]

            # Receive frame data
            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Decode frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                logger.debug(f"Decoded frame successfully, shape: {frame.shape}")
                # Get window info for coordinate scaling
                original_frame = frame.copy()  # Keep original for face detection

                # Perform face detection if enabled
                if enable_face_tracking and mouse_socket:
                    logger.debug("Face tracking enabled, processing frame for faces")
                    faces = detect_faces(original_frame)

                    if len(faces) > 0:
                        # Use the largest face (or first face if multiple)
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        face_x, face_y, face_w, face_h = largest_face

                        logger.info(f"Selected largest face: position ({face_x}, {face_y}), size {face_w}x{face_h}")

                        # Calculate center of face
                        face_center_x = face_x + face_w // 2
                        face_center_y = face_y + face_h // 2

                        logger.debug(f"Face center in frame coordinates: ({face_center_x}, {face_center_y})")

                        # Scale coordinates to original screen size
                        scale_x = 1920 / frame.shape[1]  # Assuming server streams at 1920x1080
                        scale_y = 1080 / frame.shape[0]

                        screen_x = int(face_center_x * scale_x)
                        screen_y = int(face_center_y * scale_y)

                        logger.info(f"Scaled to screen coordinates: ({screen_x}, {screen_y}) with scale factors ({scale_x:.2f}, {scale_y:.2f})")

                        # Send mouse move command to face position
                        command = {
                            'type': 'move',
                            'x': screen_x,
                            'y': screen_y
                        }

                        logger.debug(f"Sending face tracking mouse command: {command}")

                        try:
                            message = json.dumps(command) + '\n'
                            mouse_socket.sendall(message.encode('utf-8'))
                            logger.info(f"Face tracking: moved cursor to ({screen_x}, {screen_y})")
                        except Exception as e:
                            logger.error(f"Failed to send face tracking command: {e}")

                    # Draw face rectangles for visualization
                    frame = draw_face_rectangles(frame, faces)

                cv2.imshow(window_name, frame)

                # Set up mouse callback only once when window is first shown
                if not mouse_callback_set:
                    window_info = {
                        'original_width': 1920,  # Assuming server streams at 1920x1080
                        'original_height': 1080,
                        'window_width': frame.shape[1],
                        'window_height': frame.shape[0]
                    }

                    # Set mouse callback with mouse socket and window info
                    cv2.setMouseCallback(window_name, mouse_callback, (mouse_socket, window_info))
                    mouse_callback_set = True
                    logger.info("Mouse callback set up for window")
                    if enable_face_tracking:
                        logger.info("Face tracking enabled - cursor will follow detected faces")
                        logger.info("Face detection parameters: scale_factor=1.1, min_neighbors=5, min_size=(30,30)")

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                logger.warning("Failed to decode frame")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        if mouse_socket:
            mouse_socket.close()
        cv2.destroyAllWindows()
        print("Connection closed")

def test_mouse_control(host, port=8081):
    """Test mouse control by sending circular mouse movements"""
    print("Testing mouse control with circular movements...")

    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect((host, port))
        print(f"Connected to mouse control server at {host}:{port}")

        # Send circular mouse movements for testing
        center_x, center_y = 960, 540  # Center of a typical 1920x1080 screen
        radius = 200

        for angle in range(0, 360, 10):  # Move in a circle
            x = center_x + int(radius * np.cos(np.radians(angle)))
            y = center_y + int(radius * np.sin(np.radians(angle)))

            command = {
                'type': 'move',
                'x': x,
                'y': y
            }

            try:
                message = json.dumps(command) + '\n'
                test_socket.sendall(message.encode('utf-8'))
                print(f"Sent mouse move to ({x}, {y})")
                time.sleep(0.1)  # Small delay between movements
            except Exception as e:
                print(f"Failed to send test command: {e}")
                break

        print("Test completed")
        test_socket.close()

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Screen Stream Client with Mouse Control')
    parser.add_argument('--host', default='192.168.2.1',
                       help='Server IP address (default: 192.168.2.1)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Video stream port (default: 8080)')
    parser.add_argument('--mouse-port', type=int, default=8081,
                       help='Mouse control port (default: 8081)')
    parser.add_argument('--test', action='store_true',
                       help='Run mouse control test with circular movements')
    parser.add_argument('--face-tracking', action='store_true',
                       help='Enable automatic face tracking - cursor follows detected faces')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='DEBUG', help='Set logging level (default: DEBUG)')

    args = parser.parse_args()

    # Configure logging based on command line argument
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('client.log', mode='w')
        ]
    )

    if args.test:
        logger.info("Running mouse control test")
        test_mouse_control(args.host, args.mouse_port)
    else:
        logger.info("Starting video stream client")
        receive_stream(args.host, port=args.port, mouse_port=args.mouse_port,
                      enable_face_tracking=args.face_tracking)
