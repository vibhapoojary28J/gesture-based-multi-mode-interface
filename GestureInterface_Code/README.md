# Gesture-Based Multi-Mode Interface

## Project Title
Gesture-Based Multi-Mode Interface

## Description
This project implements a real-time gesture-controlled interface using webcam input, MediaPipe hand tracking, and PyAutoGUI. It supports multiple interaction modes for cursor control, drawing, and media control.

## Features
- Real-time hand detection with MediaPipe
- Rule-based gesture recognition for common hand poses
- Multi-mode interaction:
  - Cursor control
  - Drawing mode with virtual canvas overlay
  - Media control using gestures
- Optional ML-based gesture detection support
- Smooth cursor movement and gesture stability enhancements

## Folder Structure
AML/
  models/                    # Pretrained gesture model and assets
  src/
    config.py              # Project configuration constants
    main.py                # Main application entry point
    core/
      hand_tracking.py    # MediaPipe hand detection
      gesture_logic.py    # Rule-based gesture recognition
      utils.py           # Shared helper utilities
    modes/
      cursor_mode.py     # Cursor control mode
      drawing_mode.py    # Drawing mode with canvas overlay
      media_mode.py      # Media control mode
    ml/
      model.py           # ML model loader
      predict.py         # ML gesture prediction
  run.py                    # Project run entrypoint
  requirements.txt          # Python dependency list
  README.md                 # Project documentation

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the application:
   ```bash
   python run.py
   ```

### Streamlit UI
If you want to use the Streamlit interface, run:
```bash
streamlit run app.py
```

3. Use the webcam window to perform gestures.
4. Press `q` to quit.

## Libraries Used
- `opencv-python` - webcam capture and image display
- `mediapipe` - hand landmark detection
- `numpy` - array operations and image processing
- `pyautogui` - mouse control and media key actions
- `streamlit` - web-based user interface for controlling the camera and mode selection

## Future Scope
- Add additional gesture commands and UI feedback
- Improve ML model training for gesture accuracy
- Add calibration for different camera resolutions
- Implement handwriting recognition in drawing mode
- Support multi-hand interactions and custom mode mappings
