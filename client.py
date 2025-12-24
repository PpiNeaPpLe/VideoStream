import cv2
import socket
import struct
import numpy as np
import json
import threading
import argparse
import time

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

def receive_stream(host, port=8080, mouse_port=8081):
    """Receive and display video stream from server with mouse control"""
    # Video stream socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Mouse control socket
    mouse_socket = None
    try:
        mouse_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mouse_socket.connect((host, mouse_port))
        print(f"Mouse control connected to {host}:{mouse_port}")
    except Exception as e:
        print(f"Mouse control connection failed: {e}")
        mouse_socket = None

    print(f"Connecting to video stream at {host}:{port}...")
    try:
        client_socket.connect((host, port))
        print("Video stream connected! Receiving stream...")
    except Exception as e:
        print(f"Failed to connect to video stream: {e}")
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
                # Get window info for coordinate scaling
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
                    print("Mouse callback set up for window")

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to decode frame")

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
    parser.add_argument('--host', default='192.168.1.129',
                       help='Server IP address (default: 192.168.1.129)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Video stream port (default: 8080)')
    parser.add_argument('--mouse-port', type=int, default=8081,
                       help='Mouse control port (default: 8081)')
    parser.add_argument('--test', action='store_true',
                       help='Run mouse control test with circular movements')

    args = parser.parse_args()

    if args.test:
        test_mouse_control(args.host, args.mouse_port)
    else:
        receive_stream(args.host, port=args.port, mouse_port=args.mouse_port)
