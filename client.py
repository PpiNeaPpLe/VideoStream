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

# Global cascade classifier cache for performance
_cascade_cache = {}

# Global face tracking state for ROI optimization
_last_face_roi = None
_frames_since_face = 0
_MAX_FRAMES_WITHOUT_FACE = 30  # Reset ROI after this many frames without face

def detect_faces_dnn(frame, confidence_threshold=0.5):
    """Detect faces using DNN model with actual confidence scores"""
    logger.debug(f"DNN face detection starting with confidence threshold {confidence_threshold}")
    try:
        # Load the DNN model
        model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "deploy.prototxt"

        # Check if model files exist, if not use Haar cascade fallback
        if not (os.path.exists(model_file) and os.path.exists(config_file)):
            logger.warning("DNN model files not found, falling back to Haar cascade")
            return detect_faces(frame)

        net = cv2.dnn.readNetFromCaffe(config_file, model_file)

        # Prepare input
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)

        # Detect faces
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype(int)

                # Convert to (x, y, w, h) format
                faces.append((x, y, x2-x, y2-y, confidence))

        logger.info(f"DNN face detection - found {len(faces)} faces above {confidence_threshold} confidence")
        for i, (x, y, w, h, conf) in enumerate(faces):
            logger.info(f"Face {i+1}: position ({x}, {y}), size {w}x{h}, confidence: {conf:.3f}")

        # Return just the bounding boxes for compatibility
        return [(x, y, w, h) for x, y, w, h, conf in faces]

    except Exception as e:
        logger.error(f"DNN face detection failed: {e}, falling back to Haar cascade")
        return detect_faces_haar(frame)

def detect_faces(frame, method='haar', confidence_threshold=0.5, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces using specified method"""
    logger.debug(f"detect_faces called with method='{method}', threshold={confidence_threshold}")
    if method == 'dnn':
        logger.debug("Using DNN face detection method")
        return detect_faces_dnn(frame, confidence_threshold)
    else:
        logger.debug("Using Haar cascade face detection method")
        return detect_faces_haar(frame, scale_factor, min_neighbors, min_size)

def detect_faces_haar(frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces in a frame using Haar cascades with caching for performance"""
    logger.debug("Starting face detection on frame")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    logger.debug(f"Converted frame to grayscale, shape: {gray.shape}")

    # Use cached cascade classifier for performance
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cache_key = cascade_path

    if cache_key not in _cascade_cache:
        logger.debug(f"Loading cascade classifier (first time): {cascade_path}")
        if not os.path.exists(cascade_path):
            logger.error(f"Haar cascade file not found at {cascade_path}")
            return []

        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            return []

        _cascade_cache[cache_key] = face_cascade
        logger.info("Cascade classifier loaded and cached")
    else:
        face_cascade = _cascade_cache[cache_key]

    # Single pass face detection (removed unnecessary confidence testing)
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
            # Calculate confidence proxy based on face size and detection strength
            face_area = w * h
            frame_area = gray.shape[0] * gray.shape[1]
            relative_size = face_area / frame_area * 100

            logger.info(f"Face {i+1}: position ({x}, {y}), size {w}x{h}, relative size: {relative_size:.1f}% of frame")

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

def receive_stream(host, port=8080, mouse_port=8081, enable_face_tracking=False,
                  face_detection_method='haar', face_confidence_threshold=0.5, stream_only=False,
                  face_processing_frequency=5, face_min_size=30):
    """Receive and display video stream from server with mouse control"""
    logger.info(f"Starting video stream client - Host: {host}, Video Port: {port}, Mouse Port: {mouse_port}, Face Tracking: {enable_face_tracking}, Stream Only: {stream_only}")

    if stream_only:
        logger.info("STREAM ONLY MODE - displaying video stream without processing")
    elif enable_face_tracking:
        logger.info("FACE TRACKING IS ENABLED - will detect faces and move cursor")
    else:
        logger.info("Face tracking is DISABLED - use --face-tracking to enable")

    # Video stream socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Mouse control socket (only needed if not stream-only mode)
    mouse_socket = None
    if not stream_only:
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
        client_socket.settimeout(5.0)  # 5 second timeout for receive operations
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
    frame_counter = 0  # For frame skipping in face detection

    try:
        frame_count = 0
        while True:
            frame_count += 1

            if not stream_only:
                logger.debug(f"Starting frame {frame_count} processing")

            # Receive frame size
            if not stream_only:
                logger.debug(f"Waiting for frame size data ({payload_size} bytes needed)")
            while len(data) < payload_size:
                if not stream_only:
                    logger.debug(f"Received {len(data)} bytes so far, waiting for {payload_size - len(data)} more")
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        logger.error("Server closed connection while waiting for frame size")
                        return
                    data += chunk
                    if not stream_only:
                        logger.debug(f"Received chunk of {len(chunk)} bytes, total now {len(data)}")
                except socket.timeout:
                    logger.error("Timeout waiting for frame size data")
                    return

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('>L', packed_msg_size)[0]

            # Receive frame data
            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Decode frame
            if not stream_only:
                logger.debug(f"Received frame data of length: {len(frame_data)} bytes")
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                if not stream_only:
                    logger.debug(f"Decoded frame successfully, shape: {frame.shape}")
                    logger.info(f"Processing frame {frame.shape} - face tracking enabled: {enable_face_tracking}")

                if not stream_only:
                    # Get window info for coordinate scaling
                    original_frame = frame.copy()  # Keep original for face detection
                    logger.debug("Frame copied for face detection")

                    # Perform face detection if enabled (skip frames for performance)
                    faces = []  # Initialize faces list
                    frame_counter += 1
                    should_detect_faces = enable_face_tracking and mouse_socket and (frame_counter % face_processing_frequency == 0)

                    logger.debug(f"Frame processing: enable_face_tracking={enable_face_tracking}, mouse_socket={mouse_socket is not None}, frame_counter={frame_counter}, should_detect={should_detect_faces}")
                    if should_detect_faces:
                        logger.info(f"Face tracking enabled, processing frame for faces using {face_detection_method} method (every {face_processing_frequency} frames)")

                        # Use ROI optimization for faster detection
                        detection_frame = original_frame
                        global _last_face_roi, _frames_since_face

                        if _last_face_roi is not None and _frames_since_face < _MAX_FRAMES_WITHOUT_FACE:
                            # Focus detection on last known face area with some padding
                            x, y, w, h = _last_face_roi
                            padding = max(w, h) // 2  # Add padding around the face
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(original_frame.shape[1], x + w + padding)
                            y2 = min(original_frame.shape[0], y + h + padding)

                            roi_frame = original_frame[y1:y2, x1:x2]
                            logger.debug(f"Using ROI for face detection: {roi_frame.shape} from original {original_frame.shape}")
                            detection_frame = roi_frame
                        else:
                            logger.debug("Using full frame for face detection (no recent face ROI)")

                        logger.debug(f"Frame shape for detection: {detection_frame.shape}")
                        faces = detect_faces(detection_frame, method=face_detection_method,
                                           confidence_threshold=face_confidence_threshold,
                                           min_size=(face_min_size, face_min_size))

                        # Adjust face coordinates back to full frame if using ROI
                        if _last_face_roi is not None and _frames_since_face < _MAX_FRAMES_WITHOUT_FACE and len(faces) > 0:
                            x_offset, y_offset = _last_face_roi[0] - max(0, _last_face_roi[0] - max(_last_face_roi[2], _last_face_roi[3]) // 2), \
                                               _last_face_roi[1] - max(0, _last_face_roi[1] - max(_last_face_roi[2], _last_face_roi[3]) // 2)
                            faces = [(x + x_offset, y + y_offset, w, h) for x, y, w, h in faces]
                            logger.debug(f"Adjusted face coordinates from ROI back to full frame")

                        # Update ROI tracking
                        if len(faces) > 0:
                            # Use the largest face for ROI tracking
                            largest_face = max(faces, key=lambda f: f[2] * f[3])
                            _last_face_roi = largest_face
                            _frames_since_face = 0
                            logger.debug(f"Updated face ROI: {largest_face}")
                        else:
                            _frames_since_face += face_processing_frequency
                            if _frames_since_face >= _MAX_FRAMES_WITHOUT_FACE:
                                _last_face_roi = None
                                logger.debug("Reset face ROI (no faces detected recently)")

                        logger.debug(f"Face detection returned {len(faces)} faces")
                    elif enable_face_tracking and mouse_socket is None:
                        logger.warning("Face tracking enabled but mouse socket is None - cannot send mouse commands")
                    elif not enable_face_tracking:
                        logger.debug("Face tracking disabled - skipping face detection")
                    else:
                        logger.debug("Face tracking conditions not met")

                    # Process detected faces and move mouse cursor
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

                    # Draw face rectangles for visualization (only if faces were detected)
                    if len(faces) > 0:
                        frame = draw_face_rectangles(frame, faces)

                cv2.imshow(window_name, frame)

                # Set up mouse callback only once when window is first shown (only if not stream-only)
                if not stream_only and not mouse_callback_set:
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
                if not stream_only:
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
    parser.add_argument('--face-detection-method', choices=['haar', 'dnn'],
                       default='haar', help='Face detection method: haar (fast, recommended) or dnn (accurate but slower)')
    parser.add_argument('--face-confidence-threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for DNN face detection (0.0-1.0)')
    parser.add_argument('--face-processing-frequency', type=int, default=5,
                       help='Process every Nth frame for face detection (higher = faster but less responsive, default: 5)')
    parser.add_argument('--face-min-size', type=int, default=30,
                       help='Minimum face size for detection in pixels (default: 30)')
    parser.add_argument('--stream', action='store_true',
                       help='Stream-only mode: display video without mouse control or face tracking for latency testing')
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
                      enable_face_tracking=args.face_tracking,
                      face_detection_method=args.face_detection_method,
                      face_confidence_threshold=args.face_confidence_threshold,
                      stream_only=args.stream,
                      face_processing_frequency=args.face_processing_frequency,
                      face_min_size=args.face_min_size)
