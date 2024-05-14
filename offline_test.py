import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import Tracker

import torch

torch.cuda.set_device(0) # Set to your desired GPU number

# coordinates 1 cars
area1=[(311,297),(711,297),(729,329),(299,329)]
area2=[(281,344),(734,335),(760,365),(266,373)]


downcar={}
downcarcounter=[]

upcar={}
upcarcounter=[]

# Load YOLO model
model = YOLO('yolov8m.pt')

# Open the video file for reading
video_path = "road.mp4"
cap = cv2.VideoCapture(video_path)

# Define a mouse event callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a named window and set mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Read class labels from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
count = 0
tracker = Tracker()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1020, 500))  # Output video filename, codec, FPS, and frame size

# Main loop to read frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Filter out car objects
    car_list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        if 'car' in class_list[int(d)]:
            car_list.append([int(x1), int(y1), int(x2), int(y2)])
            print("car shown")

    # Update tracker with car bounding boxes
    bbox_idx = tracker.update(car_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        #detect the car in the polygon by printing the result as 1 else -1
        result=cv2.pointPolygonTest(np.array(area1,np.int32),(cx,cy),False)
        print(result)

        # only show the car's bounding box, when it will be in the middle of the polygon

        # Draw bounding box and ID
        if(result>0):
            downcar[id1]=(cx,cy)
        # now when the car entering inside, then only we will show the bounding box
        if id1 in downcar:
            result1=cv2.pointPolygonTest(np.array(area2,np.int32),(cx,cy),False)
            if result1>0:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                # downcarcounter.append(id1)

                # only increment the counter 1 time only, so that we can count the car only 1 time
                if downcarcounter.count(id1)==0:
                    downcarcounter.append(id1)

############################################################################################################################################################################
                                    #  NOW FOR INSIDE GOING CARS
############################################################################################################################################################################               
        
        #detect the car in the polygon by printing the result as 1 else -1
        result2=cv2.pointPolygonTest(np.array(area2,np.int32),(cx,cy),False)
        print(result2)

        # only show the car's bounding box, when it will be in the middle of the polygon

        # Draw bounding box and ID
        if(result2>0):
            upcar[id1]=(cx,cy)
        # now when the car entering inside, then only we will show the bounding box
        if id1 in upcar:
            result3=cv2.pointPolygonTest(np.array(area1,np.int32),(cx,cy),False)
            if result3>0:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                # upcarcounter.append(id1)

                # only increment the counter 1 time only, so that we can count the car only 1 time
                if upcarcounter.count(id1)==0:
                    upcarcounter.append(id1)
                    # Stop the screen and visualize the car
                    # cv2.imshow("RGB", frame)
                    # key = cv2.waitKey(0) & 0xFF
                    # if key == 27:
                    #     break  # Break out of the loop when escape key is pressed


    # Draw the cv2 polylines of area1
    cv2.polylines(frame, [np.array(area1,np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)   
    cv2.polylines(frame, [np.array(area2,np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)   
    # cv2.polylines(frame, [np.array(area1,np.int32)], True, (0, 0, 255), 2)  

    print("downcarcounter")
    print(downcarcounter)
    print(len(downcarcounter))

    print("upcarcounter")
    print(upcarcounter)
    print(len(upcarcounter))

    cvzone.putTextRect(frame, f'Down: {len(downcarcounter)}', (50, 50), 1, 1)
    cvzone.putTextRect(frame, f'Up: {len(upcarcounter)}', (50, 100), 1, 1)

    
    # Write the frame to the output video
    out.write(frame)

    # Display frame
    cv2.imshow("RGB", frame)

    # Check for exit key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close OpenCV windows
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
