import os
import sys
import threading
import time

import cv2
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.hand_tracking import HandTracker
from src.core.gesture_logic import detect_gesture, detect_stable_gesture
from src.modes.cursor_mode import run_cursor_mode
from src.modes.drawing_mode import run_drawing_mode
from src.modes.media_mode import run_media_mode
from src.modes.presentation_mode import PresentationMode

presentation_mode = PresentationMode()

# Live camera thread state
if 'camera_thread' not in globals():
    camera_thread = None
if 'camera_stop_event' not in globals():
    camera_stop_event = threading.Event()
if 'camera_running' not in globals():
    camera_running = False
if 'current_mode' not in globals():
    current_mode = 'Cursor'
if 'current_gesture' not in globals():
    current_gesture = 'UNKNOWN'
if 'status_message' not in globals():
    status_message = 'Stopped'
if 'hand_detected' not in globals():
    hand_detected = False
if 'landmarks_count' not in globals():
    landmarks_count = 0
if 'fullscreen_enabled' not in globals():
    fullscreen_enabled = False
if 'window_name' not in globals():
    window_name = 'Gesture Control'


def init_state():
    st.session_state.setdefault('selected_mode', 'Cursor')
    st.session_state.setdefault('status_message', status_message)
    st.session_state.setdefault('current_gesture', current_gesture)
    st.session_state.setdefault('camera_running', camera_running)
    st.session_state.setdefault('hand_detected', hand_detected)
    st.session_state.setdefault('landmarks_count', landmarks_count)
    st.session_state.setdefault('fullscreen_mode', fullscreen_enabled)


def update_streamlit_state():
    st.session_state.status_message = status_message
    st.session_state.current_gesture = current_gesture
    st.session_state.camera_running = camera_running
    st.session_state.hand_detected = hand_detected
    st.session_state.landmarks_count = landmarks_count


def camera_loop():
    global camera_running, current_gesture, status_message, hand_detected, landmarks_count
    tracker = None
    stop_event = camera_stop_event

    print('Camera loop: starting')
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_message = 'Unable to open webcam.'
            camera_running = False
            print('Camera loop: unable to open webcam')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            cv2.startWindowThread()
        except Exception:
            pass
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen_enabled:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1)
        else:
            cv2.resizeWindow(window_name, 1280, 720)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except Exception:
            pass

        status_message = 'Webcam opened. Initializing hand tracker...'
        print('Camera loop: webcam opened')

        try:
            tracker = HandTracker()
            status_message = 'Hand tracker loaded. Running live stream.'
        except Exception as exc:
            status_message = f'Hand tracker failed: {exc}'
            cap.release()
            cv2.destroyWindow(window_name)
            camera_running = False
            print('Camera loop: hand tracker failed', exc)
            return

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                status_message = 'Failed to read webcam frame.'
                print('Camera loop: failed to read frame')
                break

            frame = cv2.flip(frame, 1)
            landmarks = tracker.detect_hands(frame)
            hand_detected = bool(landmarks)
            landmarks_count = len(landmarks) if landmarks else 0

            if hand_detected:
                current_gesture = detect_gesture(landmarks)
                stable_gesture = detect_stable_gesture(landmarks)
                status_message = f'Gesture: {current_gesture}'
            else:
                current_gesture = 'No hand detected'
                stable_gesture = 'UNKNOWN'
                status_message = 'No hand detected'

            mode = current_mode
            if mode == 'Cursor':
                frame = run_cursor_mode(frame, landmarks)
            elif mode == 'Drawing':
                frame = run_drawing_mode(frame, landmarks)
            elif mode == 'Presentation':
                frame, presentation_action, _ = presentation_mode.run(frame, landmarks)
                if presentation_action:
                    status_message = presentation_action
            elif mode == 'Media':
                media_action = run_media_mode(stable_gesture)
                if media_action:
                    status_message = media_action

            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1 if fullscreen_enabled else 0)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        camera_running = False
        status_message = 'Stopped'
        print('Camera loop: stopped')
    except Exception as e:
        print('Camera loop error:', e)
        camera_running = False
        status_message = f'Error: {e}'
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def start_camera():
    global camera_thread, camera_stop_event, camera_running, status_message
    print('Start camera called')
    if camera_running and camera_thread is not None and camera_thread.is_alive():
        print('Camera already running')
        return

    camera_stop_event.clear()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    camera_running = True
    status_message = 'Starting camera...'
    print('Camera thread started')


def stop_camera():
    global status_message, camera_running
    if not camera_running:
        return
    camera_stop_event.set()
    camera_running = False
    status_message = 'Stopping camera...'
    print('Camera stop requested')


def main():
    global current_mode, fullscreen_enabled, camera_running, current_gesture, status_message
    st.set_page_config(page_title='Gesture-Based Multi-Mode Interface', layout='wide')
    init_state()

    current_mode = st.session_state.selected_mode
    fullscreen_enabled = st.session_state.fullscreen_mode
    camera_running = st.session_state.camera_running
    current_gesture = st.session_state.current_gesture
    status_message = st.session_state.status_message

    st.title('Gesture-Based Multi-Mode Interface')
    st.markdown('Use the sidebar to start and stop the live gesture camera. The webcam feed opens in a separate OpenCV window.')

    with st.sidebar:
        st.header('Control Panel')
        selected_mode = st.selectbox('Select Mode', ['Cursor', 'Drawing', 'Presentation', 'Media'], key='selected_mode')
        st.checkbox('Fullscreen Mode', key='fullscreen_mode')
        if st.button('Start Camera'):
            current_mode = selected_mode
            fullscreen_enabled = st.session_state.fullscreen_mode
            start_camera()
        if st.button('Stop Camera'):
            stop_camera()
        st.divider()
        st.subheader('Instructions')
        st.write('- Start the camera to open the live OpenCV video window.')
        st.write('- Use mode selection for cursor, drawing, or media control.')
        st.write('- Press q in the OpenCV window to stop.')
        st.write('- Drag the window corners to resize it.')

    current_mode = selected_mode
    update_streamlit_state()

    st.subheader('Status')
    st.metric('Camera running', 'Yes' if camera_running else 'No')
    st.metric('Current mode', current_mode)
    st.metric('Detected gesture', current_gesture)
    st.write('Status message:', status_message)

    if camera_running:
        st.info('Live webcam stream is running. Check the OpenCV window for the feed.')
    else:
        st.warning('Camera is stopped. Click Start Camera to begin.')


if __name__ == '__main__':
    main()
