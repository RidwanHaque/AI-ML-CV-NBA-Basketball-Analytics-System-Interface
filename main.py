from ultralytics import YOLO
# Load a pre-trained YOLOv8 model   

model = YOLO("models/player_detecter.pt")  # Load a pre-trained YOLOv8 model

results = model.predict("input_videos/video_1.mp4", save=True) # Predict objects in the video and save the output
print(results) # Print the results of the prediction
print("======================") # Separator for clarity
for box in results[0].boxes: # Iterate through detected boxes
    print(box)  # Print each detected box

#video consists of multiple frames, each frame is an image