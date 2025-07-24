from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append("../")  # Adjust the path to import from the parent directory
from utils import read_stub, save_stub  # Import stub utilities

# we can ignore the tracking for the ball and just use the detection


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0, len(frames), batch_size):
            batch_frames=frames[i:i+batch_size]
            batch_detections=self.model.predict(batch_frames, conf=0.5)
            detections+=batch_detections
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        tracks = read_stub(read_from_stub, stub_path) 
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
            

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)   
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv["Ball"]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        
        save_stub(stub_path, tracks)  # Save the tracks to a stub file
        # the whole stub thing was to provide checkpoints in the code so that if it crashes, we can resume from the last checkpoint
        # this is useful for long videos where the tracker takes a lot of time to run   
        return tracks
