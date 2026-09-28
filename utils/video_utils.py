import cv2
import os
from typing import List
import numpy as np


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Reads a video from the path
    """
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def save_video(output_video_frames: List[np.ndarray], output_video_path: str) -> None:
    """
    Writes a video to the given path. Needs video frames for it.
    """
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        filename=output_video_path,
        fourcc=fourcc,
        fps=24.0,
        frameSize=(output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )

    for frame in output_video_frames:
        out.write(frame)

    out.release()
