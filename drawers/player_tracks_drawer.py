from drawers.utils import draw_ellipse


class PlayerTrackDrawer:
    def __init__(self, team1_color=[255, 245, 238], team2_color=[128, 0, 0]):
        self.default_team_id = 1
        self.team1_color = team1_color
        self.team2_color = team2_color

    def draw(self, video_frames, tracks, players_assignment):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # As this frame is a copy by reference, we duplicate it
            frame = frame.copy()

            player_dict = tracks[frame_num]

            player_assignment_for_frame = players_assignment[frame_num]

            for track_id, player in player_dict.items():

                team_id = player_assignment_for_frame.get(
                    track_id, self.default_team_id
                )

                color = self.team1_color if team_id == 1 else self.team2_color

                frame = draw_ellipse(frame, player["bbox"], color, track_id)

            output_video_frames.append(frame)

        return output_video_frames
