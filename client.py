import cv2
import socket
import struct
import numpy as np
import json
import threading

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
                window_name = 'Display 1 Stream'
                cv2.imshow(window_name, frame)

                # Set up mouse callback if not already set
                window_info = {
                    'original_width': 1920,  # Assuming server streams at 1920x1080
                    'original_height': 1080,
                    'window_width': frame.shape[1],
                    'window_height': frame.shape[0]
                }

                # Set mouse callback with mouse socket and window info
                cv2.setMouseCallback(window_name, mouse_callback, (mouse_socket, window_info))

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

if __name__ == "__main__":
    # Replace with the IP address of the PC (Ethernet 2 interface)
    # You can find this by running: ipconfig in cmd and looking for "Ethernet 2"
    SERVER_IP = "169.254.210.134"  # Your PC's Ethernet 2 IP
    receive_stream(SERVER_IP, port=8080, mouse_port=8081)
