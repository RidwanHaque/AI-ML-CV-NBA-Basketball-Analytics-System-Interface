import cv2 
import numpy as np

class TeamBallControlDrawer:
    """
    A class responsible for calculating and drawing team ball control statistics on video frames.
    """
    def __init__(self):
        pass

    def get_team_ball_control(self,player_assignment,ball_aquisition):
        """
        Calculate which team has ball control for each frame.
        """
        team_ball_control = []
        for player_assignment_frame,ball_aquisition_frame in zip(player_assignment,ball_aquisition):
            if ball_aquisition_frame == -1:
                team_ball_control.append(-1)
                continue
            if ball_aquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue
            if player_assignment_frame[ball_aquisition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        team_ball_control= np.array(team_ball_control) 
        return team_ball_control

    def draw(self,video_frames,player_assignment,ball_aquisition):
        """
        Draw team ball control statistics on a list of video frames.
        """
        team_ball_control = self.get_team_ball_control(player_assignment,ball_aquisition)

        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame_drawn = self.draw_frame(frame,frame_num,team_ball_control)
            output_video_frames.append(frame_drawn)
        return output_video_frames
    
    def draw_frame(self,frame,frame_num,team_ball_control):
        """
        Draw a professional semi-transparent overlay of team ball control percentages on a single frame.
        """
        overlay = frame.copy()
        
        # Professional styling parameters
        font_scale = 0.8
        font_thickness = 2
        header_font_scale = 0.9
        border_thickness = 3
        
        # Overlay Position - moved to lower part of screen
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.02)   # Left margin
        rect_y1 = int(frame_height * 0.65)  # Lower on screen (was 0.02)
        rect_x2 = int(frame_width * 0.25)   # Compact width
        rect_y2 = int(frame_height * 0.78)  # Lower end (was 0.15)
        
        # Text positions with better spacing
        text_x = rect_x1 + 15
        header_y = rect_y1 + 25
        text_y1 = rect_y1 + 50
        text_y2 = rect_y1 + 75

        # Draw main background with rounded corners effect
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)  # Dark background
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (100, 200, 255), border_thickness)  # Blue border
        
        # Semi-transparent overlay
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate statistics
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_ball_control_till_frame.shape[0]
        
        team_1_percentage = (team_1_num_frames / total_frames) * 100 if total_frames > 0 else 0
        team_2_percentage = (team_2_num_frames / total_frames) * 100 if total_frames > 0 else 0

        # Draw header
        cv2.putText(frame, "BALL CONTROL", (text_x, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, header_font_scale, (255, 255, 255), font_thickness + 1)
        
        # Draw team statistics with color coding
        cv2.putText(frame, f"Team 1: {team_1_percentage:.1f}%", (text_x, text_y1), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), font_thickness)  # Green
        cv2.putText(frame, f"Team 2: {team_2_percentage:.1f}%", (text_x, text_y2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 255), font_thickness)  # Red

        return frame