# train_yolo.py

from ultralytics import YOLO
import os

def main():
    # --- 1. Load a Pre-trained YOLOv8 Model ---
    # We use 'yolov8n.pt', which is the smallest and fastest version.
    # The '.pt' file will be downloaded automatically the first time you run this.
    print("Loading pre-trained YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded.")

    # --- 2. Define the Path to Your Dataset Configuration File ---
    # This is the .yaml file that Roboflow created for you.
    # Make sure the path is correct relative to where you run the script.
    data_yaml_path = os.path.join('dataset', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: The data configuration file was not found at '{data_yaml_path}'")
        print("Please make sure you have extracted the Roboflow dataset into a 'dataset' folder.")
        return

    # --- 3. Train the Model ---
    print("Starting model training... This may take a while depending on your hardware.")
    # The 'train' method handles everything: loading data, training, and validation.
    # 'epochs': The number of times the model will see the entire dataset. 50 is a good start.
    # 'imgsz': The image size the model will be trained on. 640 is standard for YOLO.
    results = model.train(data=data_yaml_path, epochs=50, imgsz=640)
    
    print("\n--- Training Complete ---")
    print("The training results and the best model have been saved in the 'runs' folder.")
    print("The best model is usually found at: runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()

