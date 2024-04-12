import socket
import time
import threading

def transmitter_thread(host, port):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
        except socket.error as e:
            print("Socket error: " + str(e))
            return

        while True:
            

            data = f"{1},{2},{3}"


            try:
                s.sendall(data.encode())
                print(
                    f"Transmitted data: {data.encode()}, size = {len(data)} bytes"
                )
            except socket.error as e:
                print("Socket error during send: " + str(e))
                break

            time.sleep(0.1)

if __name__ == "__main__":
    transmitter = threading.Thread(
        target=transmitter_thread, args=("localhost", 54321)
    )

    transmitter.start()
    transmitter.join()