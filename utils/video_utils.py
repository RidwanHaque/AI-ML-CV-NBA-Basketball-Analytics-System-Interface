# Library for reading images/videos is open CV

import cv2
import os   # helps us to interact with the operating system and get directory paths and file names

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)  # Open the video file
    frames = []  # List to store frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames  # Return the list of frames

def save_video(output_video_frames, output_video_path):
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))  # Create VideoWriter object

    for frame in output_video_frames:
        out.write(frame)
    out.release()

