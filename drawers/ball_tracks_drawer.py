from drawers.utils import draw_pointer


class BallTrackDrawer:
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # As this frame is a copy by reference, we duplicate it
            frame = frame.copy()

            ball_dict = tracks[frame_num]

            for _, ball in ball_dict.items():
                bbox = ball["bbox"]

                if bbox is None:
                    continue

                frame = draw_pointer(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(frame)

        return output_video_frames
