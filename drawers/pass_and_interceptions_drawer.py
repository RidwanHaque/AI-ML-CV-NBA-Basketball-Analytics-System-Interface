import cv2
import numpy as np

class PassInterceptionDrawer:
    """
    A class responsible for calculating and drawing pass and interception statistics
    on a sequence of video frames.
    """
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for Team 1 and Team 2.

        Args:
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            tuple: A tuple of four integers (team1_pass_total, team2_pass_total,
                team1_interception_total, team2_interception_total) indicating the total
                number of passes and interceptions for both teams.
        """
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)
                
            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)
                
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

    def draw(self, video_frames, passes, interceptions):
        """
        Draw pass and interception statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            list: A list of frames with pass and interception statistics drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, passes, interceptions):
        """
        Draw a semi-transparent overlay of pass and interception counts on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame on which the overlay will be drawn.
            frame_num (int): The index of the current frame.
            passes (list): A list of pass events up to this frame.
            interceptions (list): A list of interception events up to this frame.

        Returns:
            numpy.ndarray: The frame with the semi-transparent overlay and statistics.
        """
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness=2

        # Overlay Position
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.16) 
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)  
        rect_y2 = int(frame_height * 0.90)
        # Text positions
        text_x = int(frame_width * 0.19)  
        text_y1 = int(frame_height * 0.80)  
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Get stats until current frame
        passes_till_frame = passes[:frame_num+1]
        interceptions_till_frame = interceptions[:frame_num+1]
        
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, 
            interceptions_till_frame
        )

        cv2.putText(
            frame, 
            f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}",
            (text_x, text_y1), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )
        
        cv2.putText(
            frame, 
            f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}",
            (text_x, text_y2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )


        return frame