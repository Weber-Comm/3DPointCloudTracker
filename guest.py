import socket
import threading


from CustomOut import custom_print, init_log_file


class DataReceiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.target_xyz = []

    def receiver(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            custom_print(f"Listening on {host}:{port}")

            while True:
                client_socket, addr = s.accept()
                custom_print(f"Connected by {addr}")
                with client_socket:
                    while True:
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        data_string = data.decode()
                        custom_print(
                            f"Received data: {data_string}, size = {len(data_string)} bytes"
                        )
                        xyz = [float(num) for num in data_string.split(",")]

                        with self.lock:
                            self.target_xyz = data


if __name__ == "__main__":
    init_log_file(log_filename="guest")

    data_receiver = DataReceiver()
    receiver_thread = threading.Thread(
        target=data_receiver.receiver, args=("localhost", 54321)
    )
    receiver_thread.start()
