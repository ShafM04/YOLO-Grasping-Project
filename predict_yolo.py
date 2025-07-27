# predict_yolo.py

from ultralytics import YOLO
import cv2
import numpy as np
import os

def main():
    # --- 1. Load Your Custom-Trained Model ---
    # The model is saved in the 'runs' folder after training.
    # Make sure this path is correct.
    model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"Error: The model file was not found at '{model_path}'")
        print("Please make sure you have trained the model successfully.")
        return

    print(f"Loading custom model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully.")

    # --- 2. Initialize Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting real-time object detection...")
    print("Show your object to the camera.")
    print("Press 'q' to quit.")

    # --- 3. Real-Time Detection Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 4. Make Predictions ---
        # The model.predict() method can take an image source directly.
        # 'stream=True' is efficient for video feeds.
        results = model(frame, stream=True)

        # --- 5. Process and Visualize Results ---
        for result in results:
            # Get the bounding boxes from the result
            boxes = result.boxes
            
            for box in boxes:
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to integers

                # Get the confidence score
                confidence = box.conf[0]
                
                # Get the class ID and name
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create the label text
                label = f"{class_name} {confidence:.2f}"
                
                # Put the label text above the bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Custom YOLOv8 Object Detection', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("System shut down.")


if __name__ == '__main__':
    main()

