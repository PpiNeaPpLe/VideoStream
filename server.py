import mss
import cv2
import socket
import numpy as np
import struct
import time

# Performance configuration - adjust these for better performance
TARGET_WIDTH = 1920   # Target width (will scale proportionally)
TARGET_HEIGHT = 1080  # Target height (will scale proportionally)
JPEG_QUALITY = 60     # JPEG quality (lower = faster but worse quality)
TARGET_FPS = 60       # Target frame rate

def get_display_1_bounds():
    """Get the bounds of Display 1 (primary monitor)"""
    with mss.mss() as sct:
        # Monitor 1 is typically the primary display
        monitor = sct.monitors[1]  # monitors[0] is all monitors combined
        return monitor

def stream_screen(host='0.0.0.0', port=5000):
    """Capture Display 1 and stream it over TCP with performance monitoring"""
    # Setup socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")
    print("Waiting for client to connect...")

    # Get display bounds
    monitor = get_display_1_bounds()
    print(f"Capturing Display 1: {monitor['width']}x{monitor['height']}")
    print(f"Streaming at: {TARGET_WIDTH}x{TARGET_HEIGHT} (scaled), {TARGET_FPS} FPS target, JPEG quality: {JPEG_QUALITY}")

    with mss.mss() as sct:
        while True:
            conn, addr = server_socket.accept()
            print(f"Client connected from {addr}")

            # Enable TCP_NODELAY to reduce latency
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Performance monitoring
            frame_count = 0
            start_time = time.time()
            capture_times = []
            encode_times = []
            send_times = []

            try:
                while True:
                    frame_start = time.time()

                    # Capture screen
                    capture_start = time.time()
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

                    capture_end = time.time()
                    capture_time = capture_end - capture_start

                    # Encode frame as JPEG (configurable quality for performance)
                    encode_start = time.time()
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    encode_end = time.time()
                    encode_time = encode_end - encode_start

                    # Send frame size first, then frame data
                    send_start = time.time()
                    data = encoded_frame.tobytes()
                    size = len(data)
                    conn.sendall(struct.pack('>L', size))
                    conn.sendall(data)
                    send_end = time.time()
                    send_time = send_end - send_start

                    # Store timing data (keep last 100 frames)
                    capture_times.append(capture_time)
                    encode_times.append(encode_time)
                    send_times.append(send_time)
                    if len(capture_times) > 100:
                        capture_times.pop(0)
                        encode_times.pop(0)
                        send_times.pop(0)

                    frame_count += 1

                    # Print performance stats every 100 frames
                    if frame_count % 100 == 0:
                        avg_capture = sum(capture_times) / len(capture_times)
                        avg_encode = sum(encode_times) / len(encode_times)
                        avg_send = sum(send_times) / len(send_times)
                        fps = 100 / (time.time() - start_time)
                        start_time = time.time()

                        print("\n--- Server Performance Stats ---")
                        print(f"Avg Capture: {avg_capture*1000:.1f}ms")
                        print(f"Avg Encode:  {avg_encode*1000:.1f}ms")
                        print(f"Avg Send:    {avg_send*1000:.1f}ms")
                        print(f"Total/Frame: {(avg_capture + avg_encode + avg_send)*1000:.1f}ms")
                        print(f"FPS:         {fps:.1f}")
                        print(f"Frame Size:  {len(data)/1024:.1f}KB")

                    frame_end = time.time()
                    total_frame_time = frame_end - frame_start

                    # Dynamic frame rate control - aim for target FPS but don't sleep if we're already slow
                    target_frame_time = 1/TARGET_FPS
                    if total_frame_time < target_frame_time:
                        time.sleep(target_frame_time - total_frame_time)

            except (ConnectionResetError, BrokenPipeError):
                print(f"Client disconnected")
                conn.close()
            except Exception as e:
                print(f"Error: {e}")
                conn.close()

if __name__ == "__main__":
    stream_screen()
