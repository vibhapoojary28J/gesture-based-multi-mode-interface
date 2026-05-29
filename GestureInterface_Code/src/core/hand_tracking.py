import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialize the HandTracker with MediaPipe Tasks API.
        """
        model_path = "models/hand_landmarker.task"
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def detect_hands(self, frame):
        """
        Detect hand landmarks from a webcam frame.
        Returns a list of landmarks as (id, x, y) tuples, or None if no hand detected.
        Also draws landmarks on the frame.
        """
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect
        timestamp = int(time.time() * 1000)  # milliseconds
        result = self.landmarker.detect_for_video(mp_image, timestamp)
        
        landmarks = None
        if result.hand_landmarks:
            # Take the first hand
            hand_landmarks = result.hand_landmarks[0]
            
            # Extract landmarks as list of (id, x, y) normalized
            landmarks = []
            for id, lm in enumerate(hand_landmarks):
                landmarks.append((id, lm.x, lm.y))
        
        return landmarks

    def close(self):
        """
        Close the landmarker.
        """
        self.landmarker.close()
