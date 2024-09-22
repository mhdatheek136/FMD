import torch
from ultralytics import YOLO

def main():
    # Check for GPU availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU not available, using CPU.")
        # Optionally set device to 'cpu' and continue, or exit as needed
        raise SystemExit(1)

    # Define the path to your dataset.yaml file
    dataset_yaml = 'dataset.yaml'

    # Define the path to the pretrained YOLOv8 model
    pretrained_model = 'yolov8m.pt'  # You can replace this with 'yolov8s.pt', 'yolov8m.pt', or 'yolov8l.pt' based on the model size you prefer

    # Define the directory to save the trained model
    output_dir = 'D:\Face_Mask_Detector'

    # Create a YOLO model instance with the pretrained weights
    model = YOLO(pretrained_model)

    # Train the YOLOv10 model with specified parameters
    model.train(
        data=dataset_yaml,     # Path to the dataset configuration YAML file (which contains train/val paths and class info)
        epochs=100,            # Number of epochs to train for; set this based on how long you want training to run
        imgsz=640,             # Image size (resolution) for training; 640x640 is a good trade-off between speed and accuracy
        batch=8,               # Batch size; determines how many images are processed per iteration (optimized for your GPU memory)
        device=device,         # Specify the device for training; '0' for GPU or '-1' for CPU (you'll use GPU here)
        project=output_dir,    # Directory where all training outputs (weights, logs, etc.) will be saved
        workers=8,             # Number of CPU workers for data loading (optimized for your 8-core, 16-thread CPU)
        name='yolov10_training_01',  # Name of the experiment, used for saving the outputs and logs
        exist_ok=True          # If True, allows overwriting an existing project directory (useful if you're running multiple sessions)
    )

    print("Training completed.")

if __name__ == '__main__':
    main()
