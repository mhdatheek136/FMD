import cv2
from ultralytics import YOLO
import math
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def draw_line_between_points(image, point1, point2):
    pt1 = (int(point1[0]), int(point1[1]))
    pt2 = (int(point2[0]), int(point2[1]))
    return cv2.line(image, pt1, pt2, (0, 0, 255), thickness=2)


# Load the face mask detection model
mask_model = YOLO("yolov8_training_01/weights/best.pt")

threshold_mask = 0.1

# Open video stream
video_capture = cv2.VideoCapture('predict_data/people.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('predict_data/people_output.mp4', fourcc, 30, (int(video_capture.get(3)), int(video_capture.get(4))))

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect masks in the frame
    mask_results = mask_model(frame)[0]

    # Create a list to store people without masks
    people_without_masks = []

    for mask_result in mask_results.boxes.data.tolist():
        x1_face, y1_face, x2_face, y2_face, score_mask, class_id_mask = mask_result

        if score_mask > threshold_mask:
            # Check mask classification
            if class_id_mask == 2:  # Mask worn incorrectly
                color = (0, 255, 255)  # Yellow
                people_without_masks.append((x1_face, y1_face, x2_face, y2_face))
            elif class_id_mask == 1:  # With mask
                color = (0, 255, 0)  # Green
            else:  # Without mask
                color = (0, 0, 255)  # Red
                people_without_masks.append((x1_face, y1_face, x2_face, y2_face))

            # Draw bounding box around the face
            cv2.rectangle(frame, (int(x1_face), int(y1_face)), (int(x2_face), int(y2_face)), color, 2)

    # Draw lines between people without masks if they are too close
    for i, person1 in enumerate(people_without_masks):
        for person2 in people_without_masks[i + 1:]:
            # Calculate centroids of the bounding boxes
            centroid1 = ((person1[0] + person1[2]) // 2, (person1[1] + person1[3]) // 2)
            centroid2 = ((person2[0] + person2[2]) // 2, (person2[1] + person2[3]) // 2)

            # Calculate the height of each face bounding box
            height1 = person1[3] - person1[1]
            height2 = person2[3] - person2[1]

            # Check if the heights are significantly different
            if height1 > 2 * height2 or height2 > 2 * height1:
                continue

            # Average height as a threshold
            avg_face_height = (height1 + height2) // 2
            avg_person_height = avg_face_height * 8

            # Calculate the distance between centroids
            distance = calculate_distance(centroid1, centroid2)

            # Check if the distance is less than the threshold
            if distance < avg_person_height:
                draw_line_between_points(frame, centroid1, centroid2)

    # Write the frame to the output video file
    output_video.write(frame)

    # Resize the frame for display
    frame = ResizeWithAspectRatio(frame, width=640)

    # Display the frame
    cv2.imshow('Mask Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
