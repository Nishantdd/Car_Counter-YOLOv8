from ultralytics import YOLO
import cv2
import cvzone
from math import ceil
# https://github.com/abewley/sort/blob/master/sort.py  -   Script used to track the vehicles from one frame to another
from sort import *


cap=cv2.VideoCapture("Images/video.mp4")
model=YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", #Giving the class numbers a name to identify objects in custom bounding boxes
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  


mask=cv2.imread("Yolo-CarCounter/mask_video.png")
# mask=cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)


#Tracking a car from one frame to another
tracker=Sort(max_age=20, min_hits=3, iou_threshold=0.3)


limits = [216,562,350,641]
total_count=[]

while True:
    success,img=cap.read()
    img_region=cv2.bitwise_and(img,mask)
    graphics=cv2.imread("Yolo-CarCounter/Graphics.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,graphics,(0,0))
    results=model(img_region, stream = True)

    detections=np.empty((0,5))


    for r in results:
        boxes=r.boxes
        for box in boxes:
            cls=int(box.cls[0])
            conf=(ceil(box.conf[0]*100))/100

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            b,h = x2-x1,y2-y1

            cvzone.cornerRect(img, (x1,y1,b,h), l=5, t=2, rt=2)
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1+5,y1-7), offset=4, scale=0.8, thickness=1)


            currentArray=np.array([x1,y1,x2,y2,conf])
            detections=np.vstack((detections, currentArray))

    resultsTracker=tracker.update(detections)

    cv2.line(img, (40,457),(1250,457), (0,0,255), 4)


    for results in resultsTracker: 
        x1, y1, x2, y2, Id = results   #ID is not used to track number of vehicles in real-time, we can count vehicles as long as the ID remains same
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(results)
        b,h = x2-x1, y2-y1
        # cvzone.cornerRect(img, (x1,y1,b,h), l=5, t=2, rt=1, colorR=(255,0,0))
        # cvzone.putTextRect(img, f'{classNames[cls]} {int(Id)}', (x1+5,y1-7), offset=4, scale=0.8, thickness=1) #Tracker rectangle
        # cv2.circle(img,(cx,cy),4,(255,255,0),thickness=2) #To display a circle around the center of TrackerBox

        cx,cy = (x1+x2)//2, (y1+y2)//2
        if 40<cx<1250 and 445<cy<470:
            if total_count.count(Id)==0:
                total_count.append(Id)
                cv2.line(img, (40,457),(1250,457), (0,255,0), 4)
        cv2.putText(img,f'{len(total_count)}',(200,70),fontFace=cv2.FONT_HERSHEY_COMPLEX,thickness=2,fontScale=1,color=(255,255,255))
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if (key==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()