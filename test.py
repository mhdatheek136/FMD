from ultralytics import YOLO

if __name__ == '__main__':

    # Define the directory to save the trained model
    output_dir = 'D:\Face_Mask_Detector'

    # Load the best model weights from your training
    model = YOLO('yolov8_training_01/weights/best.pt')

    # Validate on the test set
    results = model.val(
        data='dataset_test.yaml',  # Path to your dataset YAML file
        batch=16,  # Adjust batch size based on your hardware
        imgsz=640,  # Image size, typically 640x640 or whatever was used during training
        project = output_dir,  # Specify the output directory for results
        name='yolov8_test_01'
    )

    # Print the validation results
    print(results)
