import cv2
import torch  # If you're using a PyTorch model
from ultralytics import YOLO

# Define a function to assign a color to each class
def get_class_color(class_id):
    # Define a list of colors (RGB tuples)
    colors = [
        (0, 255, 0),  # Class 0: Green
        (255, 0, 0),  # Class 1: Blue
        (0, 0, 255),  # Class 2: Red
        (0, 255, 255),  # Class 3: Yellow
        (255, 255, 0),  # Class 4: Cyan
        (255, 0, 255)   # Class 5: Magenta
    ]
    return colors[class_id % len(colors)]  # Assign color based on class_id, cycle if more than 6 classes

def annotate_video_with_model(input_video_path, model):
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    
    out = cv2.VideoWriter(f'F:/coding_projects/hardhat_detection_on_construction_site/annotated_videos_results/annotated_{input_video_path.split("/")[-1].replace('mp4', 'avi')}', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get predictions from the model (results is a list of detections)
        results = model(frame)  # Pass the frame to the model
        
        # Handle the case where results is a list
        if isinstance(results, list):
            # Extract the detections from the list (first element is usually the result)
            detections = results[0]  # First item in the list is the detections

            # Get the boxes, confidence scores, and labels
            boxes = detections.boxes.xyxy  # The bounding boxes are stored in .xyxy
            confidences = detections.boxes.conf  # The confidence scores are stored in .conf
            class_ids = detections.boxes.cls  # Class IDs for detections

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i]  # Unpack box coordinates
                confidence = confidences[i]  # Confidence score for this detection
                class_id = int(class_ids[i])  # Class ID for this detection (convert to int)

                # Apply confidence threshold
                if confidence > 0.5:  # Only annotate detections with confidence > 0.5
                    # Get color based on class ID
                    color = get_class_color(class_id)

                    # Get the label name from the model's names dictionary
                    label = model.names[class_id]  # Get the actual label name

                    # Draw the bounding box with the chosen color
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                    # Create label text (Class name with confidence)
                    label_text = f'{label} ({confidence:.2f})'

                    # Put the label text on the frame
                    cv2.putText(frame, label_text, (int(x_min), int(y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    # print(label)

        # Display the annotated frame
        cv2.imshow('Annotated Video', frame)

        # Write the frame to the output video
        out.write(frame)
        

        # Wait for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video resources and close window
    cap.release()
    print(frame_height, frame_width, fps)
    out.release()
    cv2.destroyAllWindows()

# Example usage with your custom model
model = YOLO('F:/coding_projects/hardhat_detection_on_construction_site/runs/detect/train35/weights/best.pt').to('cuda')
input_video = 'F:/coding_projects/hardhat_detection_on_construction_site/hardhat_videos/Construction_vid_4.mp4'
annotate_video_with_model(input_video, model)
