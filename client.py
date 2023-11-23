import socket
import struct
import numpy as np
import threading
import time
import json

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QDoubleSpinBox, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.cluster import DBSCAN

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pandas as pd

from CustomOut import custom_print, init_log_file

data_receiving_completed = threading.Event()

def receiver_thread(host, port, app):
    global count, data_receiving_completed

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        custom_print("Connected to the server.")
        while True:
            # Receive header：4 bytes for count, 4 bytes for data size
            header = s.recv(8)
            if not header:
                custom_print("No more data from server. Exiting.")
                data_receiving_completed.set()
                break

            count, data_size = struct.unpack('>II', header)
            custom_print(f"Receiving data #{count}: expecting size = {data_size} bytes")

            data = bytearray()
            while len(data) < data_size:
                packet = s.recv(min(1024, data_size - len(data)))
                if not packet:
                    break
                data.extend(packet)

            # Convert the received byte data to a point cloud numpy array
            # Ensure that the data type and array shape here match the sender
            point_cloud = np.frombuffer(data, dtype=np.float64).reshape(-1, 3)

            # Use a lock to ensure thread-safe updating of shared variables
            app.lockA.acquire()
            try:
                app.point_cloud_data = point_cloud
                app.update_count_label(count)
                app.data_processed = False
                
            finally:
                app.lockA.release()

            # Output the received data information
            custom_print(f"Received data #{count}: size = {len(data)} bytes")


def process_data_thread(app):
    global count, data_receiving_completed

    while not data_receiving_completed.is_set():
        
        if not app.clustering_active or app.data_processed:
            time.sleep(0.05)  # If processing is not activated, wait briefly
            continue

        app.lockA.acquire()
        try:
            point_cloud = app.point_cloud_data
            if point_cloud is not None:
                # app.point_cloud_data = None
                app.data_processed = True  # Set to True when data processing is complete
        finally:
            app.lockA.release()

        if point_cloud is not None:
            eps = app.eps_spinbox.value()
            min_samples = int(app.min_samples_spinbox.value())

            start_time = time.time()
            labels = cluster_points_dbscan(point_cloud, eps, min_samples)
            end_time = time.time()
            elapsed_time = end_time - start_time
            custom_print(f"DBSCAN processed #{count}: elapsed time = {elapsed_time} seconds")
            
            app.lockA.acquire()
            try:
                app.clustered_data = (point_cloud, labels)
            finally:
                app.lockA.release()

            if app.target_x is not None and app.target_y is not None and app.target_z is not None:
                centroids = calculate_centroids(point_cloud, labels)
                nearest_cluster = find_nearest_cluster(centroids, point_cloud, labels,
                                                       app.target_x, app.target_y, app.target_z,
                                                       app.min_volume_spinbox.value(),
                                                       app.max_volume_spinbox.value(),
                                                       app.min_points_spinbox.value())
                
                app.lockB.acquire()
                try:
                    app.is_target_newest_for_guest = False
                finally:
                    app.lockB.release()

                if nearest_cluster is not None:
                    class_member_mask = (labels == nearest_cluster)
                    xyz = point_cloud[class_member_mask]

                    # Calculate the boundary coordinates of the cluster
                    min_coords = np.min(xyz, axis=0)
                    max_coords = np.max(xyz, axis=0)
                    mean_coords = np.mean(xyz, axis=0)

                    # Create a 3D box for the target cluster
                    app.lockB.acquire()
                    try:
                        app.box = create_3d_box(min_coords, max_coords)
                        app.volume = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1]) * (max_coords[2] - min_coords[2])
                        app.target_x, app.target_y, app.target_z = mean_coords
                        app.target_xyz_seq = np.vstack((app.target_xyz_seq, mean_coords)) if app.target_xyz_seq is not None else mean_coords
                    finally:
                        app.lockB.release()

        else:
            time.sleep(0.01)

def transmitter_thread(host, port, app):
    '''
    Send app.target_x, app.target_y, app.target_z to another guest.
    '''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        try:
            s.connect((host, port))
        except socket.error as e:
            custom_print("Socket error: " + str(e))
            return
        

        while True:

            if app.is_target_newest_for_guest:
                time.sleep(0.01)
                continue

            data = f"{app.target_x},{app.target_y},{app.target_z}"

            app.lockB.acquire()
            try:
                app.is_target_newest_for_guest = True
            finally:
                app.lockB.release()

            try:
                s.sendall(data.encode())
                custom_print(f"Transmited data: {data.encode()}, size = {len(data)} bytes")
            except socket.error as e:
                custom_print("Socket error during send: " + str(e))
                break

            time.sleep(0.01)


def cluster_points_dbscan(point_cloud, eps=0.15, min_samples=5):
    """
    Cluster a 3D point cloud using DBSCAN algorithm.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
    labels = db.labels_
    return labels

def calculate_centroids(point_cloud, labels):
    unique_labels = set(labels)
    centroids = {}
    for k in unique_labels:
        if k != -1:  # exclude noise points
            class_member_mask = (labels == k)
            xyz = point_cloud[class_member_mask]
            centroid = np.mean(xyz, axis=0)
            centroids[k] = centroid
    return centroids

def find_nearest_cluster(centroids, point_cloud, labels, x0, y0, z0, min_volume, max_volume, min_points):
    nearest_cluster = None
    max_distance = float("inf")

    for k, centroid in centroids.items():

        # Calculate distance between the target point and the centroid
        distance = np.linalg.norm([x0 - centroid[0], y0 - centroid[1], z0 - centroid[2]])

        # Rule: distance should be less than max_distance
        if distance < max_distance:
            # Get all points in the cluster
            class_member_mask = (labels == k)
            xyz = point_cloud[class_member_mask]

            # Rule: number of points should be greater than min_points
            if len(xyz) < min_points:
                continue

            min_coords = np.min(xyz, axis=0)
            max_coords = np.max(xyz, axis=0)
            volume = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1]) * (max_coords[2] - min_coords[2])

            # Rule: volume should be between min_volume and max_volume
            if min_volume <= volume <= max_volume:
                # Update max_distance and nearest_cluster
                max_distance = distance
                nearest_cluster = k

    return nearest_cluster

def visualize_clusters(ax, 
                       point_cloud, 
                       labels, 
                       box=None, 
                       target_xyz_seq=None,
                       marker_size=5, 
                       alpha=0.8, 
                       xlim=None, 
                       ylim=None, 
                       zlim=None,
                       retain_rate=1):
    # Ensure that 0 <= retain_rate <= 1
    retain_rate = max(0, min(retain_rate, 1))

    # If retain_rate < 1, randomly sample a subset of the point cloud
    if retain_rate < 1:
        num_points = len(point_cloud)
        retain_indices = np.random.choice(num_points, int(num_points * retain_rate), replace=False)
        point_cloud = point_cloud[retain_indices]
        labels = labels[retain_indices]

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Use black for noise points

        class_member_mask = (labels == k)
        xyz = point_cloud[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=marker_size, c=[col], alpha=alpha)

    if box is not None:
        ax.add_collection3d(Poly3DCollection(box, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))

    # Check if target_xyz_seq is a numpy array of shape (N, 3), where N >= 1, rather than a single 3D point (3,)
    if target_xyz_seq is not None and target_xyz_seq.ndim > 1:
        ax.plot(target_xyz_seq[:, 0], target_xyz_seq[:, 1], target_xyz_seq[:, 2], color='black', linewidth=2)

    ax.set_title('Clustered 3D points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

def visualize_raw_point_cloud(ax, 
                              point_cloud, 
                              marker_size=5, 
                              alpha=0.8, 
                              target_xyz=None,
                              xlim=None, 
                              ylim=None, 
                              zlim=None, 
                              retain_rate=1):
    """
    Visualize the raw point cloud when there are no cluster labels.

    :param ax: The axes object to plot on.
    :param point_cloud: A numpy array of shape (N, 3) representing the 3D points.
    :param marker_size: Size of the markers in the plot.
    :param alpha: Transparency of the markers.
    :param xlim: A tuple (xmin, xmax) for the x-axis limit.
    :param ylim: A tuple (ymin, ymax) for the y-axis limit.
    :param zlim: A tuple (zmin, zmax) for the z-axis limit.
    :param retain_rate: Fraction of points to retain for visualization, between 0 and 1.
    """
    # Ensure retain_rate resonable
    retain_rate = max(0, min(retain_rate, 1))

    # If retain_rate < 1, randomly sample a subset of points
    if retain_rate < 1:
        num_points = len(point_cloud)
        retain_indices = np.random.choice(num_points, int(num_points * retain_rate), replace=False)
        point_cloud = point_cloud[retain_indices]

    # plot point cloud
        
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='blue', s=marker_size, alpha=alpha/2)

    # Target point
    ax.scatter(target_xyz[0], target_xyz[1], target_xyz[2], color='red', s=30, alpha=1)

    ax.set_title('Raw 3D Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

def create_3d_box(min_coords, max_coords):

    # Define the 8 vertices of the cube
    x = [min_coords[0], max_coords[0]]
    y = [min_coords[1], max_coords[1]]
    z = [min_coords[2], max_coords[2]]

    # Calculate the 6 faces of the cube
    verts = [
        [(x[0], y[0], z[0]), (x[0], y[1], z[0]), (x[1], y[1], z[0]), (x[1], y[0], z[0])],  # 底面
        [(x[0], y[0], z[1]), (x[0], y[1], z[1]), (x[1], y[1], z[1]), (x[1], y[0], z[1])],  # 顶面
        [(x[0], y[0], z[0]), (x[0], y[0], z[1]), (x[0], y[1], z[1]), (x[0], y[1], z[0])],  # 左面
        [(x[1], y[0], z[0]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[1], y[1], z[0])],  # 右面
        [(x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[0], z[1]), (x[0], y[0], z[1])],  # 前面
        [(x[0], y[1], z[0]), (x[1], y[1], z[0]), (x[1], y[1], z[1]), (x[0], y[1], z[1])]   # 后面
    ]


    return verts

def load_config(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_config(filename, config_data):
    with open(filename, "w") as file:
        json.dump(config_data, file, indent=4)

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Point Cloud Visualizer'
        self.left = 50
        self.top = 50
        self.width = 750
        self.height = 950

        self.config = load_config("config.json")
        self.initUI()

        self.lockA = threading.Lock()
        self.clustered_data = (None, None)
        self.point_cloud_data = None
        self.data_processed = True

        self.lockB = threading.Lock()
        self.initTarget()

        self.clustering_active = False 
        self.plotting_active = True 

        self.update_button_styles()

    def initTarget(self):
        self.lockB.acquire()
        try:
            self.target_x = self.initial_x_spinbox.value()
            self.target_y = self.initial_y_spinbox.value()
            self.target_z = self.initial_z_spinbox.value()
            self.target_xyz_seq = None
            self.box = None
            self.volume = None
            self.is_target_newest_for_guest = False
        finally:
            self.lockB.release()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        # Create central widget
        widget = QWidget(self)
        self.setCentralWidget(widget)

        # Main layout
        layout = QVBoxLayout(widget)

        # Horizontal layout with labels
        hlayout0_container = QWidget()  # Create container widget
        hlayout0 = QHBoxLayout(hlayout0_container)  # Add layout to the container
        self.count_label = QLabel("Count: 0", self)
        self.count_label.setFixedSize(100, 10)
        self.target_xyz_label = QLabel(None, self)  # Assuming you want to display some text
        self.volume_label = QLabel(None, self)  # Assuming you want to display some text
        hlayout0.addWidget(self.count_label)
        hlayout0.addWidget(self.target_xyz_label)
        hlayout0.addWidget(self.volume_label)
        hlayout0_container.setStyleSheet("background-color: lightblue;")  # Set background color for container
        layout.addWidget(hlayout0_container)
            

        # Create and add controls
        # EPS and Min Samples
        self.eps_layout, self.eps_spinbox = self.create_double_spinbox("EPS", 0.01, 2, 0.15, 0.05)
        self.min_samples_layout, self.min_samples_spinbox = self.create_double_spinbox("Min Samples", 1, 50, 5, 1)
        # Marker Size and Alpha
        self.marker_size_layout, self.marker_size_spinbox = self.create_double_spinbox("Marker Size", 0.2, 10, 5, 0.05)
        self.alpha_layout, self.alpha_spinbox = self.create_double_spinbox("Alpha", 0.1, 1.0, 0.8, 0.05)
        hlayout1 = QHBoxLayout()
        hlayout1.addLayout(self.eps_layout)
        hlayout1.addLayout(self.min_samples_layout)
        hlayout1.addLayout(self.marker_size_layout)
        hlayout1.addLayout(self.alpha_layout)
        layout.addLayout(hlayout1)

        # X-Min and X-Max
        self.xlim_min_layout, self.xlim_min_spinbox = self.create_double_spinbox("X-Min", -15, 15, -5, 0.5)
        self.xlim_max_layout, self.xlim_max_spinbox = self.create_double_spinbox("X-Max", -15, 15, 5, 0.5)
        # Y-Min and Y-Max
        self.ylim_min_layout, self.ylim_min_spinbox = self.create_double_spinbox("Y-Min", -15, 15, -5, 0.5)
        self.ylim_max_layout, self.ylim_max_spinbox = self.create_double_spinbox("Y-Max", -15, 15, 5, 0.5)

        hlayout3 = QHBoxLayout()
        hlayout3.addLayout(self.xlim_min_layout)
        hlayout3.addLayout(self.xlim_max_layout)
        hlayout3.addLayout(self.ylim_min_layout)
        hlayout3.addLayout(self.ylim_max_layout)
        layout.addLayout(hlayout3)

        self.initial_x_layout, self.initial_x_spinbox = self.create_double_spinbox("Initial-x", -1000, 1000, 0, 0.1)
        self.initial_y_layout, self.initial_y_spinbox = self.create_double_spinbox("Initial-y", -1000, 1000, 0, 0.1)
        self.initial_z_layout, self.initial_z_spinbox = self.create_double_spinbox("Initial-z", -1000, 1000, 0, 0.05)
        hlayout5 = QHBoxLayout()
        hlayout5.addLayout(self.initial_x_layout)
        hlayout5.addLayout(self.initial_y_layout)
        hlayout5.addLayout(self.initial_z_layout)
        layout.addLayout(hlayout5)

        self.min_volume_layout, self.min_volume_spinbox = self.create_double_spinbox("Min-Volume", 0, 10, 0.01, 0.02)
        self.max_volume_layout, self.max_volume_spinbox = self.create_double_spinbox("Max-Volume", 0, 10, 1, 0.02)
        self.min_points_layout, self.min_points_spinbox = self.create_double_spinbox("Min_Points", 0, 1000, 5, 2)
        hlayout6 = QHBoxLayout()
        hlayout6.addLayout(self.min_volume_layout)
        hlayout6.addLayout(self.max_volume_layout)
        hlayout6.addLayout(self.min_points_layout)
        layout.addLayout(hlayout6)

        # Refresh Rate and Refresh Button
        self.refresh_rate_layout, self.refresh_rate_spinbox = self.create_double_spinbox("Refresh Rate (s)", 0.2, 100, 1.0, 0.2)
        # self.refresh_button = QPushButton('Refresh', self)
        # self.refresh_button.clicked.connect(self.update_plot)
        hlayout7 = QHBoxLayout()
        hlayout7.addLayout(self.refresh_rate_layout)
        # hlayout5.addWidget(self.refresh_button)
        # layout.addLayout(hlayout5)

        # Clustering control button
        self.toggle_clustering_button = QPushButton('Activate Clustering and Tracking', self)
        self.toggle_clustering_button.clicked.connect(self.toggle_clustering)

        # Plotting control button
        self.toggle_plotting_button = QPushButton('Pause Plotting', self)
        self.toggle_plotting_button.clicked.connect(self.toggle_plotting)

        # hlayout6 = QHBoxLayout()
        hlayout7.addWidget(self.toggle_clustering_button)
        hlayout7.addWidget(self.toggle_plotting_button)
        layout.addLayout(hlayout7)

        self.canvas = PlotCanvas(self, width=5, height=8)
        layout.addWidget(self.canvas)

        self.refresh_rate_spinbox.valueChanged.connect(self.update_refresh_rate)

        widget.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # 1000 ms
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.eps_spinbox.setValue(self.config.get("eps", 0.15))
        self.min_samples_spinbox.setValue(self.config.get("min_samples", 5))
        self.marker_size_spinbox.setValue(self.config.get("marker_size", 5))
        self.alpha_spinbox.setValue(self.config.get("alpha", 0.8))
        self.xlim_min_spinbox.setValue(self.config.get("xlim_min", -7.5))
        self.xlim_max_spinbox.setValue(self.config.get("xlim_max", 7.5))
        self.ylim_min_spinbox.setValue(self.config.get("ylim_min", -7.5))
        self.ylim_max_spinbox.setValue(self.config.get("ylim_max", 7.5))
        self.initial_x_spinbox.setValue(self.config.get("initial_x", 0))
        self.initial_y_spinbox.setValue(self.config.get("initial_y", 0))
        self.initial_z_spinbox.setValue(self.config.get("initial_z", 0))
        self.min_volume_spinbox.setValue(self.config.get("min_volume", 0))
        self.max_volume_spinbox.setValue(self.config.get("max_volume", 0))
        self.min_points_spinbox.setValue(self.config.get("min_points", 0))
        self.refresh_rate_spinbox.setValue(self.config.get("refresh_rate", 1))

        self.eps_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.min_samples_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.marker_size_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.alpha_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.xlim_min_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.xlim_max_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.ylim_min_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.ylim_max_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.initial_x_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.initial_y_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.initial_z_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.min_volume_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.max_volume_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.min_points_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.refresh_rate_spinbox.valueChanged.connect(self.on_parameter_changed)

    def on_parameter_changed(self):
        # Call this method whenever any parameter changes
        self.config["eps"] = self.eps_spinbox.value()
        self.config["min_samples"] = self.min_samples_spinbox.value()
        self.config["marker_size"] = self.marker_size_spinbox.value()
        self.config["alpha"] = self.alpha_spinbox.value()
        self.config["xlim_min"] = self.xlim_min_spinbox.value()
        self.config["xlim_max"] = self.xlim_max_spinbox.value()
        self.config["ylim_min"] = self.ylim_min_spinbox.value()
        self.config["ylim_max"] = self.ylim_max_spinbox.value()
        self.config["initial_x"] = self.initial_x_spinbox.value()
        self.config["initial_y"] = self.initial_y_spinbox.value()
        self.config["initial_z"] = self.initial_z_spinbox.value()
        self.config["min_volume"] = self.min_volume_spinbox.value()
        self.config["max_volume"] = self.max_volume_spinbox.value()
        self.config["min_points"] = self.min_points_spinbox.value()
        self.config["refresh_rate"] = self.refresh_rate_spinbox.value()

        save_config("config.json", self.config)

    def toggle_clustering(self):
        self.config = load_config("config.json")
        self.clustering_active = not self.clustering_active
        self.update_button_styles()
        self.initTarget()
        custom_print(f"Clustering and tracking {'started' if self.clustering_active else 'stopped'}")

    def toggle_plotting(self):
        self.config = load_config("config.json")
        self.plotting_active = not self.plotting_active
        self.update_button_styles()
        custom_print(f"Plotting {'started' if self.plotting_active else 'stopped'}")

    def update_button_styles(self):
        if self.clustering_active:
            # self.toggle_clustering_button.setStyleSheet("background-color: green;")
            self.toggle_clustering_button.setText('Pause Clustering and Tracking')
        else:
            # self.toggle_clustering_button.setStyleSheet("background-color: red;")
            self.toggle_clustering_button.setText('Activate Clustering and Tracking')

        if self.plotting_active:
            # self.toggle_plotting_button.setStyleSheet("background-color: green;")
            self.toggle_plotting_button.setText('Pause Plotting')
        else:
            # self.toggle_plotting_button.setStyleSheet("background-color: red;")
            self.toggle_plotting_button.setText('Activate Plotting')

    def update_count_label(self, count):
        self.count_label.setText(f"Count: {count}")

    def update_target_xyz_label(self):
        target_x_value_string = f"{self.target_x}"
        target_x_value_string = target_x_value_string[:target_x_value_string.find('.') + 3]
        target_y_value_string = f"{self.target_y}"
        target_y_value_string = target_y_value_string[:target_y_value_string.find('.') + 3]
        target_z_value_string = f"{self.target_z}"
        target_z_value_string = target_z_value_string[:target_z_value_string.find('.') + 3]
        self.target_xyz_label.setText(f"Target x: {target_x_value_string}, Target y: {target_y_value_string}, Target z: {target_z_value_string}")
        

    def update_volume_label(self):
        volume_value_string = f"{self.volume}"
        volume_value_string = volume_value_string[:volume_value_string.find('.') + 4]
        self.volume_label.setText("Volume: " + volume_value_string)

    def update_refresh_rate(self):
        refresh_rate = self.refresh_rate_spinbox.value() * 1000  # Convert to milliseconds
        self.timer.setInterval(int(refresh_rate))

    def create_double_spinbox(self, label_text, min_val, max_val, init_val, step):
        # Create a horizontal layout
        layout = QHBoxLayout()

        # Create and add label
        label = QLabel(label_text)
        layout.addWidget(label)

        # Create and add double precision spin box
        spinbox = QDoubleSpinBox(self)
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(init_val)     # Set initial value
        spinbox.setSingleStep(step)    # Set adjustment step
        layout.addWidget(spinbox)

        return layout, spinbox
    

    def update_plot(self):

        if not self.plotting_active:
            return

        self.lockA.acquire()
        try:
            if self.clustering_active:
                point_cloud, labels = self.clustered_data
            else:
                point_cloud = self.point_cloud_data
                labels = None
            if point_cloud is None:
                return
        finally:
            self.lockA.release()
        
        self.canvas.axes.clear()
        
        # If there are cluster labels, plot by cluster and plot the target cluster box
        if labels is not None and self.clustering_active:
            visualize_clusters(self.canvas.axes, point_cloud, labels,
                               self.box,
                               self.target_xyz_seq,
                               self.marker_size_spinbox.value(),
                               self.alpha_spinbox.value(),
                               (self.xlim_min_spinbox.value(), self.xlim_max_spinbox.value()),
                               (self.ylim_min_spinbox.value(), self.ylim_max_spinbox.value()))
        else:
            # If there are no cluster labels, plot all points with a single color
            visualize_raw_point_cloud(self.canvas.axes, point_cloud,
                                      self.marker_size_spinbox.value(),
                                      self.alpha_spinbox.value(),
                                      (self.initial_x_spinbox.value(), self.initial_y_spinbox.value(), self.initial_z_spinbox.value()), 
                                      (self.xlim_min_spinbox.value(), self.xlim_max_spinbox.value()),
                                      (self.ylim_min_spinbox.value(), self.ylim_max_spinbox.value()))

        # Re-draw the canvas
        self.canvas.draw()
        self.update_target_xyz_label()
        self.update_volume_label()
        

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')  # Add 3D projection
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)



if __name__ == '__main__':

    count = 0

    init_log_file(log_filename='client')

    app = QApplication(sys.argv)
    ex = App()

    receiver = threading.Thread(target=receiver_thread, args=('localhost', 12345, ex))
    processor = threading.Thread(target=process_data_thread, args=(ex,))
    transmitter = threading.Thread(target=transmitter_thread, args=('localhost', 54321, ex))

    receiver.start()
    processor.start()
    transmitter.start()

    ex.show()
    sys.exit(app.exec_())