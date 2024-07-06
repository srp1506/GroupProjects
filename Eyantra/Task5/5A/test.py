import socket
from time import sleep
import signal
import sys

def signal_handler(sig, frame):
    print('Clean-up!')
    cleanup()
    sys.exit(0)

def cleanup():
    s.close()
    print("Cleanup done")

ip = "192.168.201.180"  # Enter the IP address of the ESP32
port = 8002

# To understand the working of the code, visit https://docs.python.org/3/library/socket.html
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            # command = "right"
            command = input("Enter command (moveForward, sleep, right, sleep, left, sleep, halt): ")
            conn.sendall(str.encode(f"/command?cmd={command}"))
            sleep(1)
