from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector



def main():

    # Read Video
    video_frames = read_video("input_videos/video_3.mp4")

    # initialize Player Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")
    # initialize Ball Tracker
    ball_tracker = BallTracker("models/ball_detector_model.pt")






    # run Tracker (this is going to produce the player tracks but wont visualize them)
    player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")

    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl")



    # Interpolate Ball Tracks (remove wrong ball detections)
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)


    # assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=True, stub_path="stubs/player_assignment_stub.pkl")


    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)


    # detect passes and interceptions 
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)







    # draw output
    # initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()

    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_assignment, ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)


    # draw team ball control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_aquisition)




    # draw passes an interceptions 
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions)


    # Save Video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
