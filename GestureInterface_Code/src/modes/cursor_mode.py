import cv2
import time
import pyautogui
from src.core.utils import smooth_point, get_finger_states

# Global variables for smoothing and click control
prev_cursor_x = None
prev_cursor_y = None
last_click_time = 0
prev_two_fingers = False
CLICK_DELAY = 0.35


def run_cursor_mode(frame, landmarks):
    """
    Control mouse cursor using index finger tip with smoothing.
    Click with peace symbol, pause on fist.
    Returns the frame with visual feedback overlay.
    """
    global prev_cursor_x, prev_cursor_y, last_click_time, prev_two_fingers

    output = frame.copy()
    cursor_color = (0, 200, 255)
    status_text = 'Move index finger to control cursor'

    if not landmarks or len(landmarks) < 21:
        prev_gesture = None
        cv2.putText(output, 'No hand detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return output

    states = get_finger_states(landmarks)
    if not states:
        prev_gesture = None
        cv2.putText(output, 'No hand detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return output

    index_tip = landmarks[8]
    index_point = (int(index_tip[1] * frame.shape[1]), int(index_tip[2] * frame.shape[0]))
    fist_gesture = not any([states['thumb'], states['index'], states['middle'], states['ring'], states['pinky']])

    if fist_gesture:
        prev_cursor_x = None
        prev_cursor_y = None
        prev_gesture = 'FIST'
        cursor_color = (0, 0, 255)
        status_text = 'Paused: make a fist to freeze cursor'
        cv2.circle(output, index_point, 22, cursor_color, 3)
        cv2.circle(output, index_point, 10, cursor_color, -1)
        cv2.putText(output, status_text, (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cursor_color, 2)
        return output

    # Map index tip position to screen coordinates for cursor movement
    screen_width, screen_height = pyautogui.size()
    raw_x = int(index_tip[1] * screen_width)
    raw_y = int(index_tip[2] * screen_height)

    if prev_cursor_x is None or prev_cursor_y is None:
        prev_cursor_x, prev_cursor_y = raw_x, raw_y

    cursor_x, cursor_y = smooth_point((raw_x, raw_y), (prev_cursor_x, prev_cursor_y), alpha=0.75)
    cursor_x = max(0, min(screen_width - 1, cursor_x))
    cursor_y = max(0, min(screen_height - 1, cursor_y))
    pyautogui.moveTo(cursor_x, cursor_y)
    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

    two_fingers = states['index'] and states['middle'] and not states['ring'] and not states['pinky']
    now = time.time()

    if two_fingers:
        cursor_color = (0, 255, 0)
        status_text = 'Peace sign detected: click ready'
        if two_fingers and not prev_two_fingers and (now - last_click_time) > CLICK_DELAY:
            pyautogui.click()
            last_click_time = now
            status_text = 'Click performed!'
    else:
        cursor_color = (0, 200, 255)
        status_text = 'Move index finger to control cursor'

    prev_two_fingers = two_fingers
    cv2.circle(output, index_point, 22, cursor_color, 3)
    cv2.circle(output, index_point, 10, cursor_color, -1)
    cv2.putText(output, status_text, (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cursor_color, 2)

    return output
