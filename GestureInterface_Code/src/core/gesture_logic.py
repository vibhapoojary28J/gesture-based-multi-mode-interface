from src.core.utils import get_finger_states

# Gesture stability buffer for smoother control
gesture_history = []
STABLE_FRAMES = 3


def detect_gesture(landmarks):
    """
    Detect hand gestures from landmark finger states.
    Returns one of the supported gesture strings or 'UNKNOWN'.
    """
    states = get_finger_states(landmarks)
    if not states:
        return 'UNKNOWN'

    thumb = states['thumb']
    index = states['index']
    middle = states['middle']
    ring = states['ring']
    pinky = states['pinky']

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_vertical = thumb_tip[2] - thumb_ip[2]
    thumb_horizontal = abs(thumb_tip[1] - thumb_ip[1])
    thumb_up = thumb_vertical < -0.02 and abs(thumb_vertical) > thumb_horizontal * 0.45
    thumb_down = thumb_vertical > 0.02 and abs(thumb_vertical) > thumb_horizontal * 0.45

    extended_count = sum([thumb, index, middle, ring, pinky])

    # Stable rules for common gestures
    if extended_count == 5:
        return 'OPEN_PALM'
    if thumb_down and not any([index, middle, ring, pinky]):
        return 'THUMBS_DOWN'
    if thumb_up and not any([index, middle, ring, pinky]):
        return 'THUMBS_UP'
    if extended_count == 0:
        return 'FIST'
    if index and middle and not any([ring, pinky]):
        return 'TWO_FINGERS'
    if index and not any([thumb, middle, ring, pinky]):
        return 'POINT'

    return 'UNKNOWN'


def detect_stable_gesture(landmarks):
    """
    Return a gesture only once it has been stable for several frames.
    This reduces false triggers for media and click actions.
    """
    global gesture_history
    raw_gesture = detect_gesture(landmarks)

    if raw_gesture == 'UNKNOWN':
        gesture_history.clear()
        return 'UNKNOWN'

    gesture_history.append(raw_gesture)
    if len(gesture_history) > STABLE_FRAMES:
        gesture_history.pop(0)

    if len(gesture_history) == STABLE_FRAMES and all(g == raw_gesture for g in gesture_history):
        return raw_gesture

    return 'UNKNOWN'
