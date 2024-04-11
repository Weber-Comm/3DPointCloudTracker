from ultralytics import YOLO
import torch

if __name__ == "__main__":

    print(torch.__version__)
    print(torch.cuda.is_available())

    # Load a model
    model = YOLO('my_model.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='my_data.yaml', epochs=250, batch=8,imgsz=640, device=0)#