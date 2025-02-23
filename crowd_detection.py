import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial import distance

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8m.pt")  

# Parameters
CROWD_THRESHOLD = 3  
DISTANCE_THRESHOLD = 60
INDIVIDUAL_DISTANCE = 20
FRAME_THRESHOLD = 10  

# Tracking memory
tracked_people = {}  # {ID: (x, y, age (how many frames has the person appered) )}
next_id = 0  
crowd_data = []

# Open video
video_path = "dataset_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_centroids = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                detected_centroids.append((center_x, center_y))

    # **Update tracked people**
    updated_people = {}
    matched_ids = set()

    for new_center in detected_centroids:
        assigned_id = None
        min_dist = INDIVIDUAL_DISTANCE

        for person_id, (old_center, age) in tracked_people.items():
            if person_id in matched_ids:
                continue

            if distance.euclidean(new_center, old_center) < min_dist:
                assigned_id = person_id
                min_dist = distance.euclidean(new_center, old_center)

        if assigned_id is not None:
            updated_people[assigned_id] = (new_center, 0)  
            matched_ids.add(assigned_id)
        else:
            updated_people[next_id] = (new_center, 0)  
            next_id += 1  

    tracked_people = {pid: (center, age + 1) for pid, (center, age) in updated_people.items() if age < FRAME_THRESHOLD}

    # **Detect crowds**
    active_people = list(tracked_people.values())
    persons = [p[0] for p in active_people]  
    crowds = []

    if len(persons) >= CROWD_THRESHOLD:
        for i, p1 in enumerate(persons):
            group = [p1]
            for j, p2 in enumerate(persons):
                if i != j and distance.euclidean(p1, p2) <= DISTANCE_THRESHOLD:
                    group.append(p2)
            if len(group) >= CROWD_THRESHOLD:
                crowds.append([group,len(group)])

    unique_crowds = []
    for group, n in crowds:
        group_set = set(group)
        if group_set not in unique_crowds:
            unique_crowds.append([group_set,n])

    # Store data if a crowd is persistent
    if unique_crowds:
        crowd_data.append([cap.get(cv2.CAP_PROP_POS_FRAMES), unique_crowds[0][1]])

    # **Draw results**
    # for person in persons:
    #     cv2.circle(frame, person, 5, (0, 255, 0), -1)  

    for group, n in unique_crowds:
        for person in group:
            cv2.circle(frame, person, 5, (0, 0, 255), -1)  

    cv2.imshow("Crowd Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# **Save Results**
df = pd.DataFrame(crowd_data, columns=["Frame Number", "Crowd Count"])
df.to_csv("crowd_detection_results.csv", index=False)

cap.release()
cv2.destroyAllWindows()
