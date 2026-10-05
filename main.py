from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from drawers.player_tracks_drawer import PlayerTrackDrawer
from drawers.ball_tracks_drawer import BallTrackDrawer
from team_assigner.team_assigner import TeamAssigner


def main():

    # Read Video
    video_frames = read_video("./input/video_1.mp4")

    # Intialize Trackers
    player_tracker = PlayerTracker("./models/player_detector.pt")
    ball_tracker = BallTracker("./models/ball_detector.pt")

    # Track Players
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="checkpoints/player_tracker_stub.pkl",
    )
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="checkpoints/ball_tracker_stub.pkl",
    )

    # Remove Wrong Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Teams to Players
    team_assigner = TeamAssigner()
    players_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stubs=True,
        stub_path="checkpoints/team_assigner_stub.pkl",
    )

    # Initialize Drawers
    player_drawer = PlayerTrackDrawer()
    ball_drawer = BallTrackDrawer()

    # Draw Tracks
    output_video_frames = player_drawer.draw(
        video_frames, player_tracks, players_assignment
    )
    output_video_frames = ball_drawer.draw(output_video_frames, ball_tracks)

    # Write Video
    save_video(output_video_frames, "./output/video_1.avi")
    print("Run Successful!")


if __name__ == "__main__":
    main()
