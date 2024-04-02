import cv2
import time
from ultralytics import YOLO
import socket
import threading
import queue
import json

########################################## x-axis
#                                        #
#      (x1,y1)                           #
#                                        #
#                                        #
#                                        #
#                                        #
#                                        #
#                      (x2,y2)           #
#                                        #
##########################################
# y-axis

MAX_FPS = 10

# Bounding box mode, choose from "xyxy", "xywh", "xyxyn", "xywhn"
BBOX_MODE = "xyxyn"

# Server address
HOST = 'localhost'

# Server port
PORT = 12345  

def transmit_data(host, port, data_queue):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Try to connect to the server
        sock.connect((host, port))
        while True:
            # Get data from the queue
            data = data_queue.get()
            if data is None:
                break   # Use None as an exit signal
            message = str(data).encode('utf-8')
            sock.sendall(message)
            print("message sent: ", message)
            time.sleep(0.005)

if __name__ == "__main__":

    # Initialize a queue for data transmission

    data_queue = queue.Queue()

    # Start the data transmission thread
    transmit_thread = threading.Thread(target=transmit_data, args=(HOST, PORT, data_queue), daemon=True)
    transmit_thread.start()
    print("Connecting to host {}, port {}...".format(HOST, PORT))

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the default camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # 
    target_interval = 1 / MAX_FPS

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Initialize variables for calculating the frame rate
    fps = 0
    frame_count = 0
    time_flag_A = time.time() # Time flag for limiting the frame rate
    time_flag_B = time.time() # Time flag for updating the FPS

    # Loop through the frames from the camera
    while True:
        # Read a frame from the camera
        success, frame = cap.read()

        if success:
            
            # Limit the frame rate
            elapsed = time.time() - time_flag_A
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            time_flag_A = time.time()

            # Update the FPS counter and time
            frame_count += 1
            if time.time() - time_flag_B >= 1.0:  # Update FPS per second
                fps = frame_count / (time.time() - time_flag_B)
                frame_count = 0
                time_flag_B = time.time()

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            # Assuming `results.plot()` returns a frame that can be displayed directly
            annotated_frame = results[0].plot()

            # Print the bounding boxes of the detected objects
            for r in results:
                # print(r.boxes.xyxy)

                if BBOX_MODE == "xyxy":
                    data_str = json.dumps(r.boxes.xyxy.cpu().numpy().tolist())
                elif BBOX_MODE == "xywh":
                    data_str = json.dumps(r.boxes.xywh.cpu().numpy().tolist())
                elif BBOX_MODE == "xyxyn":
                    data_str = json.dumps(r.boxes.xyxyn.cpu().numpy().tolist())
                elif BBOX_MODE == "xywhn":
                    data_str = json.dumps(r.boxes.xywhn.cpu().numpy().tolist())
                else:
                    raise ValueError("Invalid BBOX_MODE: " + BBOX_MODE)

                print(data_str)
                data_queue.put(data_str)


            # If the `plot` method does not work as expected, you might need to modify how the results are visualized.
            # In such a case, refer to the YOLO documentation or the Results object's available methods for proper usage.
            
            # Display the frame rate on the annotated frame
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # If there's an error reading a frame, print an error message
            print("Error: Could not read frame.")
            break
    
    
    data_queue.put(None)
    transmit_thread.join()  # Wait for the data transmission thread to finish

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
