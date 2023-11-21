import socket
import struct
import pandas as pd
import numpy as np
import time
import os

from CustomOut import custom_print, init_log_file

def read_point_cloud_from_csv(file_path, retain_rate=0.15):
    """
    Read a 3D point cloud from a CSV file and randomly retain a subset of points.

    :param file_path: Path to the CSV file.
    :param retain_rate: Fraction of points to retain, between 0 and 1.
    :return: A numpy array of shape (M, 3) representing the 3D points, where M <= N.
    """
    # read CSV file
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'z', 'intensity'])

    # Delete rows containing NaN.
    df = df.dropna()

    # random sample point cloud with retain_rate probability 
    if 0 < retain_rate < 1:
        df = df.sample(frac=retain_rate)

    # convert DataFrame to NumPy array, and reserve first 3 cols（x, y, z）
    point_cloud = df[['x', 'y', 'z']].to_numpy()

    return point_cloud

def send_point_cloud_continuously(folder, host='localhost', port=12345):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    files.sort(key=lambda x: int(x.split('.')[0]))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        custom_print("Server started, waiting for a connection...")
        conn, addr = s.accept()
        with conn:
            custom_print(f'Connected by {addr}')
            count = 0
            for file in files:
                file_path = os.path.join(folder, file)
                pc = read_point_cloud_from_csv(file_path)
                data_string = pc.tobytes()
                data_size = len(data_string)

                count += 1
                header = struct.pack('>I', count) + struct.pack('>I', data_size)
                message = header + data_string
                conn.sendall(message)

                custom_print(f"Sent data #{count}: size = {data_size} bytes, shape = {pc.shape}")
                time.sleep(0.5)  



if __name__ == '__main__':
    init_log_file(log_filename='server')

    # Replace the folder path with the path to your point cloud folder.
    send_point_cloud_continuously('X:/pointclouds/')
