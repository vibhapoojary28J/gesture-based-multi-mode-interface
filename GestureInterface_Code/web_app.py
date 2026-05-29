import atexit
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

from src.config import CAMERA_INDEX, DEFAULT_MODE, MODE_LIST
from src.core.gesture_logic import detect_gesture, detect_stable_gesture
from src.core.hand_tracking import HandTracker
from src.modes.cursor_mode import run_cursor_mode
from src.modes.drawing_mode import run_drawing_mode
from src.modes.media_mode import run_media_mode
from src.modes.presentation_mode import PresentationMode

app = Flask(__name__, template_folder='templates')

camera_lock = threading.Lock()
camera = None
tracker = None
current_mode = DEFAULT_MODE.title()
current_gesture = 'UNKNOWN'
app_status = 'Stopped'
camera_running = False
presentation_mode = PresentationMode()


def create_capture(index):
    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_MSMF'):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return None


def init_camera():
    global camera, tracker, app_status
    with camera_lock:
        if camera is not None and camera.isOpened():
            return True

        camera = create_capture(CAMERA_INDEX)
        if camera is None:
            app_status = 'Camera unavailable'
            camera = None
            return False

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        tracker = HandTracker()
        app_status = 'Running'
        return True


def release_resources():
    global camera, tracker, app_status, camera_running
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
            tracker = None
        camera_running = False
        app_status = 'Stopped'


def generate_frames():
    global current_gesture, app_status
    while camera_running:
        with camera_lock:
            if camera is None:
                break
            success, frame = camera.read()

        if not success or frame is None:
            app_status = 'Camera read failed'
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        landmarks = tracker.detect_hands(frame)
        gesture = detect_gesture(landmarks)
        stable_gesture = detect_stable_gesture(landmarks)
        current_gesture = gesture

        selected_mode = current_mode.lower()
        media_action = ''
        presentation_action = ''
        if selected_mode == 'cursor':
            frame = run_cursor_mode(frame, landmarks)
        elif selected_mode == 'drawing':
            frame = run_drawing_mode(frame, landmarks)
        elif selected_mode == 'presentation':
            frame, presentation_action, _ = presentation_mode.run(frame, landmarks)
        elif selected_mode == 'media':
            media_action = run_media_mode(stable_gesture)

        cv2.putText(frame, f'Mode: {current_mode}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Gesture: {gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if media_action:
            cv2.putText(frame, media_action, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if presentation_action:
            cv2.putText(frame, presentation_action, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            continue

        frame_bytes = encoded_image.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html', current_mode=current_mode, current_gesture=current_gesture, camera_running=camera_running)


@app.route('/video')
def video_feed():
    if not camera_running:
        return Response('Camera is off', status=503, mimetype='text/plain')
    if not init_camera():
        return Response('Camera unavailable', status=503, mimetype='text/plain')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/on', methods=['POST'])
def camera_on():
    global camera_running, app_status
    if camera_running:
        return jsonify(success=True, status='Already running')

    if not init_camera():
        return jsonify(success=False, message=app_status), 503

    camera_running = True
    app_status = 'Running'
    return jsonify(success=True, status=app_status)


@app.route('/camera/off', methods=['POST'])
def camera_off():
    release_resources()
    return jsonify(success=True, status=app_status)


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.get_json(silent=True) or {}
    selected = data.get('mode', '').strip().lower()
    if selected in [mode.lower() for mode in MODE_LIST]:
        current_mode = selected.title()
        return jsonify(success=True, mode=current_mode)
    return jsonify(success=False, message='Invalid mode'), 400


@app.route('/status')
def status():
    return jsonify(mode=current_mode, gesture=current_gesture, status=app_status, camera_running=camera_running)


@atexit.register
def cleanup():
    release_resources()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, threaded=True)
