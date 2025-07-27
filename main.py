from utils import read_video, save_video
import argparse
import os
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer, CourtKeypointDrawer, TacticalViewDrawer, SpeedAndDistanceDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator




from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH, 
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():

    args = parse_args()
    # Read Video
    video_frames = read_video(args.input_video)

    # initialize Player Tracker
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    # initialize Ball Tracker
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    # initialize Court Key Point Detector
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)





    # run Tracker (this is going to produce the player tracks but wont visualize them)
    player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path=os.path.join(args.stub_path, "player_track_stubs.pkl"))

    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=os.path.join(args.stub_path, "ball_track_stubs.pkl"))





    # get court keypoints
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames, read_from_stub=True, stub_path = os.path.join(args.stub_path, "court_key_points_stubs.pkl"))


    # Interpolate Ball Tracks (remove wrong ball detections)
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)


    # assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=True, stub_path=os.path.join(args.stub_path,"stubs/player_assignment_stub.pkl"))


    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)


    # detect passes and interceptions 
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # tactical view
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints,player_tracks)


    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )

    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # draw output
    # initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()


    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_assignment, ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)


    # draw team ball control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_aquisition)




    # draw passes an interceptions 
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions)


    # draw court key points
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints)

    # draw tactical view
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    )


    # Speed and Distance Drawer
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    # Save Video
    save_video(output_video_frames, args.output_video)

if __name__ == "__main__":
    main()
