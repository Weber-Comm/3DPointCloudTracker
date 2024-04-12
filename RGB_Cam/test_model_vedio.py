from ultralytics import YOLO
import cv2
import os
import glob

if __name__ == "__main__":
    
    # Load model with your weights
    weights_path = 'last.pt'
    model = YOLO(weights_path)

    # Path to the input video
    video_path = './YOLO test vedios/VID_20240408_143544.mp4'

    # Base directory to save the test results
    base_save_dir = './tests/'

    # Find the next available test folder
    test_folders = glob.glob(os.path.join(base_save_dir, 'test*'))
    if test_folders:
        # Extract folder numbers and find the maximum
        folder_nums = [int(folder.split('test')[-1]) for folder in test_folders]
        next_folder_num = max(folder_nums) + 1
    else:
        next_folder_num = 1

    # Create the new test folder
    new_test_folder = os.path.join(base_save_dir, f'test{next_folder_num}')
    os.makedirs(new_test_folder, exist_ok=True)

    # Prepare to capture video and get properties
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare video writer
    output_video_path = os.path.join(new_test_folder, 'result_video.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model([frame])
        
        # Get the frame with drawn detections
        frame_with_detections = results[0].plot()

        # Write frame to output video
        out.write(frame_with_detections)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")