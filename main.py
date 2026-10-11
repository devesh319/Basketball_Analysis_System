from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_keypoint_detector.court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from speed_distance_calculator.speed_distance_calculator import (
    SpeedAndDistanceCalculator,
)
from drawers.player_tracks_drawer import PlayerTracksDrawer
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.team_ball_control_drawer import TeamBallControlDrawer
from drawers.pass_interception_drawer import PassInterceptionDrawer
from drawers.court_keypoints_drawer import CourtKeypointDrawer
from drawers.tactical_view_drawer import TacticalViewDrawer
from drawers.speed_distance_drawer import SpeedAndDistanceDrawer
from team_assigner.team_assigner import TeamAssigner
from ball_acquisition.ball_acquisition import BallAquisitionDetector
from pass_interception_detector.pass_interception_detector import (
    PassInterceptionDetector,
)


def main():

    # Read Video
    video_frames = read_video("./input/video_1.mp4")

    # Intialize Trackers
    player_tracker = PlayerTracker("./models/player_detector_v2.pt")
    ball_tracker = BallTracker("./models/ball_detector.pt")

    # Initialize Court Keypoints Detector
    court_keypoint_detector = CourtKeypointDetector(
        "./models/court_keypoint_detector.pt"
    )

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

    # Get Court Keypoints
    court_keypoints = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_stub=True,
        stub_path="checkpoints/court_keypoints_stub.pkl",
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
        read_from_stub=True,
        stub_path="checkpoints/team_assigner_stub.pkl",
    )

    # Find Ball Acquisition
    ball_acquisition = BallAquisitionDetector()
    ball_acquisition = ball_acquisition.detect_ball_possession(
        player_tracks, ball_tracks
    )

    # Passes and Interceptions
    pass_interceptions_detector = PassInterceptionDetector()
    passes = pass_interceptions_detector.detect_passes(
        ball_acquisition, players_assignment
    )
    interceptions = pass_interceptions_detector.detect_interceptions(
        ball_acquisition, players_assignment
    )

    # Get Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(
        court_keypoints
    )
    tactical_player_positions = (
        tactical_view_converter.transform_players_to_tactical_view(
            court_keypoints_per_frame, player_tracks
        )
    )

    # Speed and Distance Calculator
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(
        tactical_player_positions
    )
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(
        player_distances_per_frame
    )

    # Initialize Drawers
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    ball_acqusition_drawer = TeamBallControlDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_distance_drawer = SpeedAndDistanceDrawer()

    # Draw Tracks
    output_video_frames = player_drawer.draw(
        video_frames, player_tracks, players_assignment, ball_acquisition
    )
    output_video_frames = ball_drawer.draw(output_video_frames, ball_tracks)

    # Draw Court Keypoints
    output_video_frames = court_keypoint_drawer.draw(
        output_video_frames, court_keypoints
    )

    # Write Ball Acquisition
    output_video_frames = ball_acqusition_drawer.draw(
        output_video_frames, players_assignment, ball_acquisition
    )

    # Write Passes & Interceptions
    output_video_frames = pass_and_interceptions_drawer.draw(
        output_video_frames, passes, interceptions
    )

    # Draw Speed and Distance
    output_video_frames = speed_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame,
    )

    # Draw Tactical View
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        players_assignment,
        ball_acquisition,
    )

    # Write Video
    save_video(output_video_frames, "./output/video_1.avi")
    print("Run Successful!")


if __name__ == "__main__":
    main()
