## Documentation 


# Research 

- video 1 (speed distance ball aquisition)
- video 2 (passes logic)
- video 3 (intercepting ball logic and passes)

- used to develop project. 
- focus on different videos 

- locating different objects in frame and identify them


# Image Classification/Recognition 
(what is inside of an image) 
- usually have predefined list of classes 
- model has ability to produce one of those four classes
- prduces a number/index like 0 -4 to identify object

# Image Localization
- produces 4 numbers and they indicate the rectangle around object
- model doesnt draw the rectangle it produces 4 numbers (position of the rectangle and the size)
- middle of rectangle position is (x,y) (width x , height y)
- preproccesing function after recieving output from the model we can draw bounding box
- functions in openCV take two main points (top left and bottom right) to identify rectangle location and size
- (x1,y1) (x2,y2)

# Object Detection 
- identifies what is an object in image and title of object
- different classes for different objects (player) (ball)
- each would have bounding box

- object locations tracked over frames using object detection framework/model (YOLO)

# YOLO Model Architecture (V1 - V11) (Computer Vision Model)
- process image once and produce multiple boxes

# Ultralytics 
- computer vision model library 
- includes YOLO Models
- This is a python Library 


![alt text](image.png)





## Workflow Overview
The detection script operates through a sequence of automated steps, from model acquisition to video output.

Model Initialization and Caching
Upon its initial execution, the script automatically fetches the pre-trained YOLOv8 model from the Ultralytics hub. This model is downloaded and saved locally within the project directory, ensuring it is readily available for all subsequent runs without needing to be downloaded again.

Frame-by-Frame Video Processing
The model processes the input video sequentially, one frame at a time, rather than loading the entire video into memory. For each frame, it performs the following steps:

Isolates the single frame as an image.

Runs the object detection inference to identify objects and their bounding boxes.

Proceeds to the next frame to repeat the process until the video is complete.

Output and Results

Console Logs: The detection results for each frame, including bounding box coordinates and class predictions, are stored in a results object and printed to the console for real-time monitoring.

Saved Video: When the save=True parameter is set, the script automatically generates a runs/detect/predict/ directory. The final processed video, with the detection results visually rendered onto each frame, is saved within this folder.

Related
On first run, the script fetches the YOLoV8 model from Ultralytics and downloads it into your project folder
The model processes each video frame individually, not the entire video at once
For each frame, it detects objects and outputs bounding boxes, which are stored in "results"
Results are printed out for review
With save=true, a "runs" folder is automatically created containing the output detection video








## First Test

- number next to the object label is the confidence number (how confident is the model that this is actually a "person" object )
- problems:
    picking up crowd, 
    misclassification of objects,
    basketball isn't always detected


![alt text](image-1.png)




## Next steps

- Accuracy needs to improved through fine tuning already trained model (give it examples with labels (labeled dataset) and allow model to execute similar output by learning)
- this is transferred learning 

# Labeled Datasets w/ RoboFlow
- single image with bounding boxes on players and other scene objects and labeling it
- also labeling scoreboard overlay
- dataset is a list of images labeled according to our specific needs
- roboflow allows us to use existing datasets and retrieve them, more images means more accuracy and stable performance
- the dataset has multiple classes (Player, Referee, Basketball)



# Added Notebooks folder for training
- optimal to run this on a capable machine. we will upload code via collab