import cv2
import time
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import CAMERA_INDEX, MODE_LIST, DEFAULT_MODE, SWITCH_DELAY, WINDOW_NAME, DETECTION_MODE
from src.core.hand_tracking import HandTracker
from src.core.gesture_logic import detect_gesture, detect_stable_gesture
from src.modes.cursor_mode import run_cursor_mode
from src.modes.drawing_mode import run_drawing_mode
from src.modes.media_mode import run_media_mode
from src.ml.model import load_model
from src.ml.predict import predict_gesture


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        return

    cv2.namedWindow('Gesture Interface', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Interface', 800, 600)

    tracker = HandTracker()
    current_mode = DEFAULT_MODE
    last_switch_time = 0
    current_detection = DETECTION_MODE
    ml_model = None

    if current_detection == 'ml':
        ml_model = load_model()
        if ml_model is None:
            print('Warning: ML model not found. Falling back to rule-based detection.')
            current_detection = 'rule'

    print('Starting Gesture Interface. Press q to quit.')
    print(f'Gesture detection mode: {current_detection}')

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print('Error: Could not read frame.')
                break

            frame = cv2.flip(frame, 1)
            landmarks = tracker.detect_hands(frame)

            if current_detection == 'ml' and ml_model is not None:
                gesture = predict_gesture(landmarks, ml_model)
            else:
                gesture = detect_gesture(landmarks)

            stable_media_gesture = detect_stable_gesture(landmarks)
            current_time = time.time()
            if gesture == 'PEACE' and (current_time - last_switch_time) > SWITCH_DELAY:
                current_index = MODE_LIST.index(current_mode)
                current_mode = MODE_LIST[(current_index + 1) % len(MODE_LIST)]
                last_switch_time = current_time
                print(f'Switched to mode: {current_mode}')

            media_action = ''
            if current_mode == 'cursor':
                frame = run_cursor_mode(frame, landmarks)
            elif current_mode == 'drawing':
                frame = run_drawing_mode(frame, landmarks)
            elif current_mode == 'media':
                media_action = run_media_mode(stable_media_gesture)

            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Detection: {current_detection}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(frame, f'Mode: {current_mode}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if media_action:
                cv2.putText(frame, media_action, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Gesture Interface', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                current_detection = 'ml' if current_detection == 'rule' else 'rule'
                if current_detection == 'ml':
                    ml_model = load_model()
                    if ml_model is None:
                        current_detection = 'rule'
                        print('ML model unavailable. Staying in rule-based mode.')
                print(f'Switched detection mode to: {current_detection}')

    except Exception as e:
        print(f'Error: {e}')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            tracker.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
