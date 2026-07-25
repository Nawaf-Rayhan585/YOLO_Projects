import os
import cv2
from ultralytics import YOLO
import pandas as pd

# Load your YOLO model
model = YOLO("yolo11s-pose.pt")

# Video path
cap = cv2.VideoCapture('susp.mp4')

# Get video properties
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = round(frames / fps)

frame_total = 2000
i = 0
a = 295 # Start from 848 forcefully

all_data = []

# Define output path for cropped images
output_path_dir = r'C:\Users\freed\Downloads\yolopose\images1'

while cap.isOpened():
    # Set the position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))
    flag, frame = cap.read()

    if not flag:
        break

    # Save full frame image
    pa = r'C:\Users\freed\Downloads\yolopose\images'
    image_path = f'{pa}\img_{i}.jpg'
    cv2.imwrite(image_path, frame)

    # Run YOLO detection
    results = model(frame, verbose=False)

    for r in results:
        bound_box = r.boxes.xyxy  # Get bounding boxes
        conf = r.boxes.conf.tolist()  # Confidence score
        keypoints = r.keypoints.xyn.tolist()  # Human keypoints

        for index, box in enumerate(bound_box):
            if conf[index] > 0.75:
                x1, y1, x2, y2 = box.tolist()
                cropped_person = frame[int(y1):int(y2), int(x1):int(x2)]
                output_path = os.path.join(output_path_dir, f'person_nn_{a}.jpg')

                data = {'image_name': f'person_nn_{a}.jpg'}

                # Save keypoint data
                for j in range(len(keypoints[index])):
                    data[f'x{j}'] = keypoints[index][j][0]
                    data[f'y{j}'] = keypoints[index][j][1]

                all_data.append(data)
                cv2.imwrite(output_path, cropped_person)
                a += 1  # Increment image number for the next person

    i += 1

print(f"Total frames processed: {i-1}, Total cropped images saved: {a-1}")
cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Path to your CSV file
csv_file_path = r'C:\Users\freed\Downloads\yolopose\nkeypoint.csv'

# Check if the file exists to determine whether to append or create new
if not os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path, index=False)  # Create new file if it doesn't exist
else:
    df.to_csv(csv_file_path, mode='a', header=False, index=False)  # Append if it exists

print(f"Keypoint data saved to {csv_file_path}")
