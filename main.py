from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer



def main():

    # Read Video
    video_frames = read_video("input_videos/video_1.mp4")

    # initialize Player Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")
    # initialize Ball Tracker
    ball_tracker = BallTracker("models/ball_detector_model.pt")


    # run Tracker (this is going to produce the player tracks but wont visualize them)
    player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")

    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl")

    

    # draw output
    # initialize drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw object tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Save Video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
