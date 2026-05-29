import math


def distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    Points may be tuples/lists like (x, y) or (id, x, y).
    """
    if len(p1) == 3:
        _, x1, y1 = p1
    else:
        x1, y1 = p1

    if len(p2) == 3:
        _, x2, y2 = p2
    else:
        x2, y2 = p2

    return math.hypot(x2 - x1, y2 - y1)


def smooth_point(current, previous, alpha=0.5):
    """
    Smooth between current and previous points.
    alpha is the smoothing factor for the previous point.
    """
    if previous is None:
        return current

    cur_x, cur_y = current
    prev_x, prev_y = previous
    smooth_x = int(prev_x * alpha + cur_x * (1 - alpha))
    smooth_y = int(prev_y * alpha + cur_y * (1 - alpha))
    return smooth_x, smooth_y


def is_finger_extended(tip, pip, threshold=0.04):
    """
    Return True if a finger tip is extended relative to its PIP joint.
    Works with normalized landmark coordinates.
    """
    return tip[2] < pip[2] - threshold


def get_finger_states(landmarks, threshold=0.04):
    """
    Determine which fingers are extended using landmark tip/PIP positions.
    Returns a dictionary with thumb, index, middle, ring, and pinky states.
    """
    if not landmarks or len(landmarks) < 21:
        return {}

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    thumb_y_extended = thumb_tip[2] < thumb_ip[2] - threshold
    thumb_x_delta = abs(thumb_tip[1] - thumb_ip[1]) > (threshold * 2)
    thumb_extended = thumb_y_extended or thumb_x_delta

    return {
        'thumb': thumb_extended,
        'index': is_finger_extended(index_tip, index_pip, threshold),
        'middle': is_finger_extended(middle_tip, middle_pip, threshold),
        'ring': is_finger_extended(ring_tip, ring_pip, threshold),
        'pinky': is_finger_extended(pinky_tip, pinky_pip, threshold)
    }
