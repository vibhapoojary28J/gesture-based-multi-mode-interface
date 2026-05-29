import pyautogui
import time

# Delay state to avoid repeated triggers
last_action_time = 0
ACTION_COOLDOWN = 0.5  # seconds


def run_media_mode(gesture):
    """
    Execute media control actions using gestures and return a display-friendly action string.
    """
    global last_action_time
    current_time = time.time()
    action_text = ''

    if (current_time - last_action_time) < ACTION_COOLDOWN:
        return action_text

    if gesture == 'THUMBS_UP':
        pyautogui.press('volumeup')
        action_text = 'Volume Up'
    elif gesture == 'THUMBS_DOWN':
        pyautogui.press('volumedown')
        action_text = 'Volume Down'
    elif gesture == 'FIST':
        pyautogui.press('playpause')
        action_text = 'Play/Pause'
    elif gesture == 'TWO_FINGERS':
        pyautogui.press('nexttrack')
        action_text = 'Next Track'
    elif gesture == 'POINT':
        pyautogui.press('prevtrack')
        action_text = 'Previous Track'

    if action_text:
        last_action_time = current_time

    return action_text

