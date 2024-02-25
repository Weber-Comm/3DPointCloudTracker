import socket
import struct
import pandas as pd
import numpy as np
import time
import os

from CustomOut import custom_print, init_log_file


def read_point_cloud_from_csv(
    file_path,
    retain_rate=0.15,
    invert_x=False,
    bias_x=0,
    invert_y=False,
    bias_y=0,
    invert_z=True,
    bias_z=0,
):
    """
    Read a 3D point cloud from a CSV file, randomly retain a subset of points,
    and apply inversion and/or bias to x, y, z coordinates if specified.

    :param file_path: Path to the CSV file.
    :param retain_rate: Fraction of points to retain, between 0 and 1.
    :param invert_x: Boolean, whether to invert x-axis values.
    :param bias_x: Float, bias to add to x-axis values.
    :param invert_y: Boolean, whether to invert y-axis values.
    :param bias_y: Float, bias to add to y-axis values.
    :param invert_z: Boolean, whether to invert z-axis values.
    :param bias_z: Float, bias to add to z-axis values.
    :return: A numpy array of shape (M, 3) representing the 3D points, where M <= N.
    """
    # read CSV file
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "intensity"])

    # Delete rows containing NaN
    df = df.dropna()

    # Random sample point cloud with retain_rate probability
    if 0 < retain_rate < 1:
        df = df.sample(frac=retain_rate)

    # Apply inversion if specified
    if invert_x:
        df["x"] = -df["x"]
    if invert_y:
        df["y"] = -df["y"]
    if invert_z:
        df["z"] = -df["z"]

    # Apply bias
    df["x"] += bias_x
    df["y"] += bias_y
    df["z"] += bias_z

    # Convert DataFrame to NumPy array, and reserve first 3 cols（x, y, z）
    point_cloud = df[["x", "y", "z"]].to_numpy()

    return point_cloud


def send_point_cloud_continuously(folder, host="localhost", port=12345):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    files.sort(key=lambda x: int(x.split(".")[0]))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        custom_print(f"Listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            custom_print(f"Connected by {addr}")
            count = 0
            for file in files:
                file_path = os.path.join(folder, file)
                pc = read_point_cloud_from_csv(file_path)
                data_string = pc.tobytes()
                data_size = len(data_string)

                count += 1
                header = struct.pack(">I", count) + struct.pack(">I", data_size)
                message = header + data_string
                conn.sendall(message)

                custom_print(
                    f"Sent data #{count}: size = {data_size} bytes, shape = {pc.shape}"
                )
                time.sleep(0.2)


if __name__ == "__main__":
    init_log_file(log_filename="server")
    send_point_cloud_continuously(
        "D:/Backup/SH23/Beam/HardwarePlatform/code_v2/LiDAR_xyz_2"
    )
