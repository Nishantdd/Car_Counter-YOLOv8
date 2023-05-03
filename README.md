# Car_Counter-YOLOv8
This code uses the YOLO (You Only Look Once) object detection model to track vehicles in a video. It first loads the YOLO model and defines a list of class names for the detected objects. Then it reads the input video, applies a mask to isolate the region of interest, and overlays graphics on top of the video.

The script uses the SORT (Simple Online Real-time Tracking) algorithm to track vehicles from frame to frame. The detected objects are drawn as bounding boxes with their corresponding class names and confidence scores. The script counts the number of vehicles that cross a designated line and displays the count on the video.

The code uses the OpenCV library for reading and displaying videos and images, and the cvzone library for drawing bounding boxes and text.

**Requirements**
The following libraries are required to run the program:
1. ultralytics
2. cv2
3. cvzone
4. math

You also need the YOLOv8 weights and configuration file. You can download the weights file from [https://github.com/ultralytics/yolov8/releases](https://github.com/ultralytics/ultralytics).

**Usage**
Install the required libraries:
 > pip install ultralytics cv2 cvzone
 
**Download the YOLOv8 weights** and configuration file and **place them in a folder called "Yolo-Weights"** in the same directory as the program.

Download the "mask_video.png" and "Graphics.png" files from the "Yolo-CarCounter" folder in this repository and place them in the same directory as the program.
Note : You will have to create the mask_video.png file yourself for suitable results in case using a different video file than provided in repository.

Modify the "limits" variable in the program to define the region where cars will be counted. The format is [y1, y2, x1, x2], where y1 is the top limit, y2 is the bottom limit, x1 is the left limit, and x2 is the right limit.

**Run the program:**
python car_counter.py
Press "q" to quit the program.

**Acknowledgments**
The SORT algorithm used in this program was created by Alex Bewley. You can find more information about it at [https://github.com/abewley/sort](https://github.com/abewley/sort).
