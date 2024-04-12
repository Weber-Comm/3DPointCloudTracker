from ultralytics import YOLO
import glob
import os

if __name__ == "__main__":

    # Load model with your weights
    weights_path = 'last.pt'
    model = YOLO(weights_path)

    # Directory where images are located
    images_dir = './datasets/agv/images/train/*.jpg'

    # Get list of all image paths
    image_paths = glob.glob(images_dir)

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

    # Run inference on all images and save the results
    results = model(image_paths)

    # Process and save results
    for i, result in enumerate(results):
        # result.show()  # Optionally display the result on screen
        
        # Save each result image to the new test folder
        save_path = os.path.join(new_test_folder, f'result_{i}.jpg')
        result.save(filename=save_path)

