import cv2
from ultralytics import YOLO
import math
import torch
import os

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

# Load the image
input_image_path = 'predict_data/img7.jpg'  # Replace with your image path
image = cv2.imread(input_image_path)

# Detect masks in the image
mask_results = mask_model(image)[0]

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
        cv2.rectangle(image, (int(x1_face), int(y1_face)), (int(x2_face), int(y2_face)), color, 3)

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
            draw_line_between_points(image, centroid1, centroid2)

# Resize the image for display
image = ResizeWithAspectRatio(image, width=640)

# Generate output image name based on input image name
input_name = os.path.splitext(os.path.basename(input_image_path))[0]  # Get the input image name without extension
image_name_output = f"{input_name}_output.jpg"  # Append "_output" to the name

# Use the same directory as the input image
output_path = os.path.join(os.path.dirname(input_image_path), image_name_output)

# Save the processed image
cv2.imwrite(output_path, image)

# Display the image
cv2.imshow('Mask Detection', image)

# Wait for key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

