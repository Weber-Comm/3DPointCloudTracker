import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import socket
import struct
import time

from CustomOut import custom_print, init_log_file


def point_cloud_callback(point_cloud_msg, conn):
    global count
    pc_data = pc2.read_points(
        point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True
    )
    pc_array = np.array([[x, y, z] for x, y, z in pc_data])

    data_string = pc_array.tobytes()
    data_size = len(data_string)

    count += 1
    header = struct.pack(">I", count) + struct.pack(">I", data_size)
    message = header + data_string
    conn.sendall(message)

    custom_print(
        f"Sent data #{count}: size = {data_size} bytes, shape = {pc_array.shape}"
    )


def send_point_cloud_continuously(host="localhost", port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        custom_print(f"Listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            custom_print(f"Connected by {addr}")

            rospy.init_node("point_cloud_listener", anonymous=True)
            rospy.Subscriber(
                "/rslidar_points", PointCloud2, point_cloud_callback, callback_args=conn
            )

            rospy.spin()


if __name__ == "__main__":
    count = 0
    init_log_file(log_filename="server")
    send_point_cloud_continuously("localhost", 12345)
