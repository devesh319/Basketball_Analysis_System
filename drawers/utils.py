import cv2
import numpy as np
from utils.bbox_utils import get_bbox_center, get_bbox_width


def draw_ellipse(frame, bbox, color, track_id=None):

    y2 = int(bbox[3])
    xc, _ = get_bbox_center(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(xc, y2),
        axes=(int(width), int(width * 0.35)),
        angle=0,
        startAngle=-45,
        endAngle=235,
        color=color,
        lineType=cv2.LINE_4,
        thickness=2,
    )

    rect_width = 40
    rect_height = 20

    x1_rect = int(xc - rect_width / 2)
    x2_rect = int(xc + rect_width / 2)
    y1_rect = int(y2 - rect_height / 2 + 15)
    y2_rect = int(y2 + rect_height / 2 + 15)

    if track_id is not None:
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

        x1_text = x1_rect + 10 if track_id < 100 else x1_rect

        cv2.putText(
            frame,
            str(track_id),
            (x1_text, y1_rect + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


def draw_pointer(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_bbox_center(bbox)

    triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame
