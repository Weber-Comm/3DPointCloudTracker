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
import matplotlib.pyplot as plt
import pandas as pd

from CustomOut import custom_print, init_log_file

data_receiving_completed = threading.Event()

import datetime

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
            app.lock.acquire()
            try:
                app.point_cloud_data = point_cloud
                app.update_count(count)
                app.data_processed = False
            finally:
                app.lock.release()

            # Output the received data information
            custom_print(f"Received data #{count}: size = {len(data)} bytes")


def process_data_thread(app):
    global count, data_receiving_completed

    while not data_receiving_completed.is_set():
        if not app.process_active:
            time.sleep(0.1)  # If processing is not activated, wait briefly
            continue

        app.lock.acquire()
        try:
            point_cloud = app.point_cloud_data
            if point_cloud is not None:
                app.point_cloud_data = None
                app.data_processed = True  # Set to True when data processing is complete
        finally:
            app.lock.release()

        if point_cloud is not None:
            eps = app.eps_spinbox.value()
            min_samples = int(app.min_samples_spinbox.value())

            start_time = time.time()
            labels = cluster_points_dbscan(point_cloud, eps, min_samples)
            end_time = time.time()
            elapsed_time = end_time - start_time
            custom_print(f"DBSCAN processed #{count}: elapsed time = {elapsed_time} seconds")
            
            app.lock.acquire()
            try:
                app.clustered_data = (point_cloud, labels)
            finally:
                app.lock.release()

        else:
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
            xy = point_cloud[class_member_mask]
            centroid = np.mean(xy, axis=0)
            centroids[k] = centroid
    return centroids

def find_nearest_cluster(centroids, 
                         point_cloud, 
                         labels, 
                         x0, 
                         y0, 
                         min_area, 
                         max_area, 
                         min_points):
    nearest_cluster = None
    min_distance = float("inf")

    for k, centroid in centroids.items():
        distance = np.linalg.norm([x0 - centroid[0], y0 - centroid[1]])
        if distance < min_distance:
            class_member_mask = (labels == k)
            xy = point_cloud[class_member_mask]

            if len(xy) < min_points:
                continue

            min_coords = np.min(xy, axis=0)
            max_coords = np.max(xy, axis=0)
            area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])

            # Ensure area between min_area and max_area
            if min_area <= area <= max_area:  
                min_distance = distance
                nearest_cluster = k

    return nearest_cluster

def visualize_clusters(ax, 
                       point_cloud, 
                       labels, 
                       marker_size=5, 
                       alpha=0.8, 
                       xlim=None, 
                       ylim=None, 
                       target_x=None, 
                       target_y=None, 
                       retain_rate=1, 
                       min_area=0.02, 
                       max_area=1,
                       min_points=10):
    """
    Visualize the 2D projection of a clustered 3D point cloud on the given axes.

    :param ax: The axes object to plot on.
    :param point_cloud: A numpy array of shape (N, 3) representing the 3D points.
    :param labels: An array of shape (N,) containing cluster labels for each point.
    :param marker_size: Size of the markers in the plot.
    :param alpha: Transparency of the markers.
    :param xlim: A tuple (xmin, xmax) for the x-axis limit.
    :param ylim: A tuple (ymin, ymax) for the y-axis limit.
    :param retain_rate: Fraction of points to retain for visualization, between 0 and 1.
    """
    # Ensure retain_rate resonable
    retain_rate = max(0, min(retain_rate, 1))

    # If retain_rate < 1, randomly sample a subset of points
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
        xy = point_cloud[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, c=[col], alpha=alpha, edgecolors='none')

    if target_x is not None and target_y is not None:
        centroids = calculate_centroids(point_cloud, labels)
        nearest_cluster = find_nearest_cluster(centroids, point_cloud, labels, target_x, target_y, min_area, max_area, min_points)  # 传入 max_area

        if nearest_cluster is not None:
            class_member_mask = (labels == nearest_cluster)
            xy = point_cloud[class_member_mask]

            min_coords = np.min(xy, axis=0) 
            max_coords = np.max(xy, axis=0)  

            min_x, min_y = min_coords[0], min_coords[1]
            max_x, max_y = max_coords[0], max_coords[1]

            ax.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', linewidth=2))

    ax.set_title('Clustered points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


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
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 600

        self.config = load_config("config.json")
        self.initUI()
        self.clustered_data = (None, None)
        self.lock = threading.Lock()
        self.point_cloud_data = None
        self.data_processed = True


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create central widget
        widget = QWidget(self)
        self.setCentralWidget(widget)
        
        # Main layout
        layout = QVBoxLayout(widget)

        self.count_label = QLabel("Count: 0", self)
        layout.addWidget(self.count_label)

        # Create and add controls
        # EPS and Min Samples
        self.eps_layout, self.eps_spinbox = self.create_double_spinbox("EPS", 0.01, 2, 0.15, 0.05)
        self.min_samples_layout, self.min_samples_spinbox = self.create_double_spinbox("Min Samples", 1, 50, 5, 1)
        hlayout1 = QHBoxLayout()
        hlayout1.addLayout(self.eps_layout)
        hlayout1.addLayout(self.min_samples_layout)
        layout.addLayout(hlayout1)

        # Marker Size and Alpha
        self.marker_size_layout, self.marker_size_spinbox = self.create_double_spinbox("Marker Size", 1, 10, 5, 1)
        self.alpha_layout, self.alpha_spinbox = self.create_double_spinbox("Alpha", 0.1, 1.0, 0.8, 0.05)
        hlayout2 = QHBoxLayout()
        hlayout2.addLayout(self.marker_size_layout)
        hlayout2.addLayout(self.alpha_layout)
        layout.addLayout(hlayout2)

        # X-Min and X-Max
        self.xlim_min_layout, self.xlim_min_spinbox = self.create_double_spinbox("X-Min", -15, 15, -5, 0.5)
        self.xlim_max_layout, self.xlim_max_spinbox = self.create_double_spinbox("X-Max", -15, 15, 5, 0.5)
        hlayout3 = QHBoxLayout()
        hlayout3.addLayout(self.xlim_min_layout)
        hlayout3.addLayout(self.xlim_max_layout)
        layout.addLayout(hlayout3)

        # Y-Min and Y-Max
        self.ylim_min_layout, self.ylim_min_spinbox = self.create_double_spinbox("Y-Min", -15, 15, -5, 0.5)
        self.ylim_max_layout, self.ylim_max_spinbox = self.create_double_spinbox("Y-Max", -15, 15, 5, 0.5)
        hlayout4 = QHBoxLayout()
        hlayout4.addLayout(self.ylim_min_layout)
        hlayout4.addLayout(self.ylim_max_layout)
        layout.addLayout(hlayout4)

        # Refresh Rate and Refresh Button
        self.refresh_rate_layout, self.refresh_rate_spinbox = self.create_double_spinbox("Refresh Rate (s)", 0.2, 100, 1.0, 0.2)
        # self.refresh_button = QPushButton('Refresh', self)
        # self.refresh_button.clicked.connect(self.update_plot)
        hlayout5 = QHBoxLayout()
        hlayout5.addLayout(self.refresh_rate_layout)
        # hlayout5.addWidget(self.refresh_button)
        # layout.addLayout(hlayout5)

        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        # hlayout6 = QHBoxLayout()
        hlayout5.addWidget(self.start_button)
        hlayout5.addWidget(self.stop_button)
        layout.addLayout(hlayout5)

        self.canvas = PlotCanvas(self, width=5, height=4)
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
        self.refresh_rate_spinbox.setValue(self.config.get("refresh_rate", 1))

        self.eps_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.min_samples_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.marker_size_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.alpha_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.xlim_min_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.xlim_max_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.ylim_min_spinbox.valueChanged.connect(self.on_parameter_changed)
        self.ylim_max_spinbox.valueChanged.connect(self.on_parameter_changed)
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
        self.config["refresh_rate"] = self.refresh_rate_spinbox.value()

        save_config("config.json", self.config)

    def update_count(self, count):
        self.count_label.setText(f"Count: {count}")

    def update_refresh_rate(self):
        refresh_rate = self.refresh_rate_spinbox.value() * 1000  # Convert to milliseconds
        self.timer.setInterval(int(refresh_rate))

    def start_processing(self):
        self.process_active = True
        self.config = load_config("config.json")
        self.update_plot()  # Update the plot immediately

    def stop_processing(self):
        self.process_active = False

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

        if not self.process_active:
            return
        self.lock.acquire()
        try:
            point_cloud, labels = self.clustered_data
            if point_cloud is None or labels is None:
                return  # If there is no new data, return directly
        finally:
            self.lock.release()

        # Clear the axes for the updated plot
        self.canvas.axes.clear()

        # Plot the clusters
        visualize_clusters(self.canvas.axes, point_cloud, labels,
                           self.marker_size_spinbox.value(),
                           self.alpha_spinbox.value(),
                           (self.xlim_min_spinbox.value(), self.xlim_max_spinbox.value()),
                           (self.ylim_min_spinbox.value(), self.ylim_max_spinbox.value()),
                           target_x=2.3,
                           target_y=-0.95)

        # Redraw the canvas
        self.canvas.draw()

        

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot(self, point_cloud, labels, marker_size, alpha, xlim, ylim):
        self.axes.clear()
        visualize_clusters(self.axes, point_cloud, labels, marker_size, alpha, xlim, ylim)
        self.draw()



if __name__ == '__main__':

    count = 0

    init_log_file(log_filename='client')

    app = QApplication(sys.argv)
    ex = App()
    ex.process_active = False

    receiver = threading.Thread(target=receiver_thread, args=('localhost', 12345, ex))
    processor = threading.Thread(target=process_data_thread, args=(ex,))

    receiver.start()
    processor.start()

    ex.show()
    sys.exit(app.exec_())