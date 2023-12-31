## 3DPointCloudTracker

### **Script Function**

The `server.py` script reads 3D LiDAR point clouds from a specified path (which can be replaced with any desired method of data acquisition) and periodically transmits this data to **Network Port 1**.

The `client.py` script's **receiver thread** continuously receives point clouds from **Network Port 1**. The **processor thread** asynchronously processes the point clouds, using DBSCAN clustering, and persistently tracks a particular cluster (the **target**) by setting an initial anchor point. The **transmitter thread** sends the tracking results, `ex.target_x`, `ex.target_y`, and `ex.target_z`, in real-time to **Network Port 2**. Key parameters and visualizations are managed through a PyQt GUI interface.

The `guest.py` script creates a subprocess, the **receiver thread**, which is used to receive tracking results,`ex.target_x`, `ex.target_y`, and `ex.target_z` from **Network Port 2** for downstream tasks.

### **Operation Steps**

1. Run `main.py`. Or run `python server.py` , `python guest.py` and `python client.py` in separate terminals, sequentially.

![1](https://github.com/Webersan/3DPointCloudTracker/assets/75790375/48630d52-be5a-4b5e-9cfa-b6259680ecda)

2. Set the **red initial anchor point** (`Initial-x`, `Initial-y`, `Initial-z`) for tracking, and the anchor point should be as close as possible to the object being tracked. Click `Activate Clustering and Tracking` to start the tracking process.

![2](https://github.com/Webersan/3DPointCloudTracker/assets/75790375/82dd31d0-dbd1-44b5-b396-5f265e65f05a)

3. If the tracking process is not satisfied, click `Pause Clustering and Tracking`, reset anchor point, and restart.

### **Client.py Configuration** 

**Clustering**

1. `DBSCAN-EPS`: a parameter for the DBSCAN clustering algorithm. It specifies the maximum distance between two samples for them to be considered as in the same neighborhood. A smaller EPS value means that points need to be closer together to be considered part of the same cluster, leading to a greater number of smaller clusters.
2. `DBSCAN-Min-Samples`: a parameter for the DBSCAN clustering algorithm. It represents the minimum number of points required to form a cluster. A higher value will result in fewer clusters, as it requires more points to be close together (within the EPS distance) to form a cluster.

**Canvas and plotting**

1. `Plot-Maker-Size`: the size of the markers (points) in the 3D plot. 
2. `Plot-Alpha`: transparency of the markers from 0 to 1 (no transparency).
3. `Plot-X-Min` and `Plot-X-Max`: the minimum and maximum limits of the x-axis on the plot.
4. `Plot-Y-Min` and `Plot-Y-Max`: the minimum and maximum limits of the y-axis on the plot.

**Tracking**

1. `Initial-x`, `Initial-y` and `Initial-z`: the anchor point to begin tracking, which should be as close as possible to the object being tracked.
2. `Max D to Last Centroid`: the maximum allowed distance between the centroid of the currently found cluster (target) and the centroid of the previously found cluster (last target), ensuring spatial proximity in the tracking process.
3. `H Thres.`: height threshold, the minimum height value for points in the 3D point cloud data, filtering out points that fall below this threshold and focusing analysis on points above it.
4. `Min-Volume`, `Max-Volume`, `Min-Area` and `Max-Area`: the minimum volume (xyz), maximum volume (xyz), minimum area (xy) and maximum area (xy) of a cluster to be considered in the tracking process to filter out smaller or larger clusters that are not of interest.
5. `Min_Points`: the minimum number of points that a cluster must contain to be considered in the tracking process. Clusters with fewer points than this threshold are considered as noise.