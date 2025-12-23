import cv2
import socket
import struct
import numpy as np

def receive_stream(host, port=5000):
    """Receive and display video stream from server"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print(f"Connecting to {host}:{port}...")
    try:
        client_socket.connect((host, port))
        print("Connected! Receiving stream...")
    except Exception as e:
        print(f"Failed to connect: {e}")
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
                cv2.imshow('Display 1 Stream', frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to decode frame")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print("Connection closed")

if __name__ == "__main__":
    # Replace with the IP address of the PC (Ethernet 2 interface)
    # You can find this by running: ipconfig in cmd and looking for "Ethernet 2"
    SERVER_IP = "169.254.210.134"  # Your PC's Ethernet 2 IP
    receive_stream(SERVER_IP)
