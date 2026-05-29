import cv2
import time
import pyautogui
from collections import deque
from src.core.utils import get_finger_states


class PresentationMode:
    def __init__(self, cooldown=1.2, buffer_size=12, wave_threshold=0.10, noise_threshold=0.02):
        self.cooldown = cooldown
        self.last_action_time = 0
        self.x_buffer = deque(maxlen=buffer_size)
        self.wave_threshold = wave_threshold
        self.noise_threshold = noise_threshold
        self.current_gesture = 'UNKNOWN'
        self.action_text = ''

    def _detect_wave(self):
        if len(self.x_buffer) < self.x_buffer.maxlen:
            return None

        diffs = [self.x_buffer[i + 1] - self.x_buffer[i] for i in range(len(self.x_buffer) - 1)]
        clean_diffs = [d for d in diffs if abs(d) > self.noise_threshold]
        if len(clean_diffs) < 3:
            return None

        signs = [1 if d > 0 else -1 for d in clean_diffs]
        direction_changes = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
        if direction_changes < 1:
            return None

        first_sign = signs[0]
        last_sign = signs[-1]
        span = abs(self.x_buffer[-1] - self.x_buffer[0])
        if span < self.wave_threshold:
            return None

        if first_sign > 0 and last_sign < 0:
            return 'WAVE_RIGHT'
        if first_sign < 0 and last_sign > 0:
            return 'WAVE_LEFT'
        return None

    def _detect_swipe(self):
        if len(self.x_buffer) < self.x_buffer.maxlen:
            return None
        span = self.x_buffer[-1] - self.x_buffer[0]
        if abs(span) < self.wave_threshold:
            return None
        return 'WAVE_RIGHT' if span > 0 else 'WAVE_LEFT'

    def _perform_action(self, wave_type):
        current_time = time.time()
        if (current_time - self.last_action_time) < self.cooldown:
            return ''

        time.sleep(0.1)
        if wave_type == 'WAVE_RIGHT':
            pyautogui.press('right')
            self.last_action_time = current_time
            return 'Wave Right Detected - Next Slide'
        if wave_type == 'WAVE_LEFT':
            pyautogui.press('left')
            self.last_action_time = current_time
            return 'Wave Left Detected - Previous Slide'

        return ''

    def run(self, frame, landmarks):
        output = frame.copy()
        self.action_text = ''
        self.current_gesture = 'UNKNOWN'
        exit_mode = False

        if not landmarks or len(landmarks) < 21:
            self.x_buffer.clear()
            cv2.putText(output, 'No hand detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return output, self.action_text, exit_mode

        states = get_finger_states(landmarks)
        if not states:
            self.x_buffer.clear()
            cv2.putText(output, 'No hand detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return output, self.action_text, exit_mode

        index_tip = landmarks[8]
        index_point = (int(index_tip[1] * frame.shape[1]), int(index_tip[2] * frame.shape[0]))
        self.x_buffer.append(index_tip[1])

        all_extended = sum(states.values()) == 5
        all_closed = sum(states.values()) == 0

        if all_extended:
            self.current_gesture = 'OPEN_PALM'
            self.action_text = 'Exit presentation mode'
            exit_mode = True
        elif all_closed:
            self.current_gesture = 'FIST'
            self.action_text = 'Paused (fist detected)'
        else:
            wave = self._detect_wave() or self._detect_swipe()
            if wave:
                action_text = self._perform_action(wave)
                if action_text:
                    self.current_gesture = wave
                    self.action_text = action_text
                    self.x_buffer.clear()
                else:
                    self.current_gesture = wave
                    self.action_text = 'Wave detected, waiting...'
            else:
                self.current_gesture = 'TRACKING'
                self.action_text = 'Wave left/right for slides'

        pointer_color = (0, 255, 0) if self.current_gesture not in ['FIST', 'UNKNOWN'] else (0, 0, 255)
        cv2.circle(output, index_point, 16, pointer_color, 2)
        cv2.circle(output, index_point, 8, pointer_color, -1)

        cv2.putText(output, 'Mode: Presentation', (20, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(output, f'Gesture: {self.current_gesture}', (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2)
        cv2.putText(output, f'Action: {self.action_text}', (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 255, 200), 2)

        return output, self.action_text, exit_mode
