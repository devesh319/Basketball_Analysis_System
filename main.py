from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from drawers.player_track_drawer import PlayerTrackDrawer


def main():

    # Read Video
    video_frames = read_video("./input/video_1.mp4")

    # Intialize Player Tracker
    player_tracker = PlayerTracker("./models/player_detector.pt")

    # Track Players
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="checkpoints/player_tracker_stub.pkl",
    )

    # Initialize Player Drawer
    player_drawer = PlayerTrackDrawer()

    # Draw Player Tracks
    output_video_frames = player_drawer.draw(video_frames, player_tracks)

    # Write Video
    save_video(output_video_frames, "./output/video_1.avi")
    print("Run Successful!")


if __name__ == "__main__":
    main()
