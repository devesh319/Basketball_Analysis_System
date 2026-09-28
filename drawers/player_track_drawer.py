from drawers.utils import draw_ellipse


class PlayerTrackDrawer:
    def __init__(self):
        pass

    def draw(self, video_frames, tracks):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # As this frame is a copy by reference, we duplicate it
            frame = frame.copy()

            player_dict = tracks[frame_num]

            for track_id, player in player_dict.items():

                frame = draw_ellipse(frame, player["box"], (0, 0, 255), track_id)

            output_video_frames.append(frame)

        return output_video_frames
