import cv2
import numpy as np
import time

from src.core.utils import distance, smooth_point, get_finger_states

# Drawing state and canvas
canvas = None
prev_point = None
clear_hold_start = None

draw_color = (0, 0, 255)
selected_color_name = 'RED'
brush_thickness = 12
current_tool = 'PEN'
current_gesture = 'UNKNOWN'

tool_presets = [
    ('PEN', 12),
    ('BRUSH', 22),
    ('MARKER', 34)
]

color_palette = [
    ('RED', (0, 0, 255)),
    ('BLUE', (255, 0, 0)),
    ('GREEN', (0, 255, 0)),
    ('YELLOW', (0, 255, 255)),
    ('PURPLE', (128, 0, 128)),
    ('BLACK', (0, 0, 0)),
    ('WHITE', (255, 255, 255))
]

tool_buttons = []
color_buttons = []


def build_palettes(frame_width, color_radius=20, tool_radius=22, top_margin=90, spacing=24):
    global color_buttons, tool_buttons
    color_buttons = []
    tool_buttons = []

    tool_diameter = tool_radius * 2
    tool_total_width = len(tool_presets) * tool_diameter + (len(tool_presets) - 1) * spacing
    tool_start_x = max(40, (frame_width - tool_total_width) // 2)
    tool_y = top_margin

    for idx, (name, size) in enumerate(tool_presets):
        cx = tool_start_x + idx * (tool_diameter + spacing) + tool_radius
        tool_buttons.append({
            'name': name,
            'size': size,
            'center': (cx, tool_y),
            'radius': tool_radius
        })

    color_diameter = color_radius * 2
    color_total_width = len(color_palette) * color_diameter + (len(color_palette) - 1) * spacing
    color_start_x = max(frame_width - color_total_width - 40, tool_start_x + tool_total_width + 30)
    color_y = tool_y

    for idx, (name, color) in enumerate(color_palette):
        cx = color_start_x + idx * (color_diameter + spacing) + color_radius
        color_buttons.append({
            'name': name,
            'color': color,
            'center': (cx, color_y),
            'radius': color_radius
        })


def draw_tool_icon(frame, center, tool_name, radius):
    x, y = center
    if tool_name == 'PEN':
        cv2.line(frame, (x, y - 8), (x, y + 8), (40, 40, 40), 3)
        cv2.line(frame, (x, y - 8), (x - 10, y - 2), (40, 40, 40), 3)
        cv2.circle(frame, (x, y + 8), 4, (40, 40, 40), -1)
    elif tool_name == 'BRUSH':
        cv2.rectangle(frame, (x - 8, y - 6), (x + 8, y + 6), (40, 40, 40), -1)
        cv2.line(frame, (x - 8, y - 6), (x - 8, y - radius + 2), (40, 40, 40), 4)
        cv2.line(frame, (x + 8, y - 6), (x + 8, y - radius + 4), (40, 40, 40), 3)
    elif tool_name == 'MARKER':
        cv2.rectangle(frame, (x - 8, y - 10), (x + 8, y + 10), (40, 40, 40), -1)
        cv2.line(frame, (x - 8, y - 10), (x + 8, y - 10), (255, 255, 255), 2)
        cv2.circle(frame, (x, y + 10), 4, (255, 255, 255), -1)


def draw_toolbar(frame, gesture, hover_color=None, hover_tool=None):
    h, w = frame.shape[:2]
    toolbar_height = 230
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, toolbar_height), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, 'Gesture Drawing Studio', (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2)
    cv2.putText(frame, f'Gesture: {gesture}', (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
    cv2.putText(frame, f'Tool: {current_tool}', (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (205, 205, 205), 2)
    cv2.putText(frame, f'Color: {selected_color_name}', (30, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (205, 205, 205), 2)

    for button in tool_buttons:
        cx, cy = button['center']
        selected = button['name'] == current_tool
        is_hover = hover_tool is not None and hover_tool['name'] == button['name']
        base_color = (215, 225, 255) if selected else (235, 235, 235)
        outline_color = (0, 255, 255) if is_hover else (255, 255, 255)
        cv2.circle(frame, (cx, cy), button['radius'] + (10 if selected else 8), base_color, -1)
        cv2.circle(frame, (cx, cy), button['radius'] + 6, outline_color, 3)
        draw_tool_icon(frame, (cx, cy), button['name'], button['radius'])

    for button in color_buttons:
        cx, cy = button['center']
        color = button['color']
        selected = button['name'] == selected_color_name
        is_hover = hover_color is not None and hover_color['name'] == button['name']
        if selected:
            cv2.circle(frame, (cx, cy), button['radius'] + 12, (255, 255, 255), 4)
        if is_hover:
            cv2.circle(frame, (cx, cy), button['radius'] + 16, (0, 255, 255), 3)
        cv2.circle(frame, (cx, cy), button['radius'] + 2, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), button['radius'], color, -1)

    cv2.putText(frame, 'Pinch on tool or color to select. Open palm to clear.', (30, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 210, 220), 2)
    draw_brush_preview(frame)


def draw_status_panel(frame, action_text=''):
    x1, y1 = 20, frame.shape[0] - 180
    padding = 18
    line_height = 34
    text_lines = [
        f'Mode: {current_tool}',
        f'Color: {selected_color_name}',
        f'Brush size: {brush_thickness}',
        action_text or 'Hover a tool/color and pinch to select.'
    ]
    panel_width = 520
    x2 = x1 + panel_width
    y2 = y1 + padding * 2 + len(text_lines) * line_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 26, 38), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for idx, text in enumerate(text_lines):
        y = y1 + padding + (idx * line_height)
        cv2.putText(frame, text, (x1 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)


def draw_brush_preview(frame):
    margin = 30
    preview_radius = 22
    preview_x = frame.shape[1] - margin - preview_radius
    preview_y = frame.shape[0] - margin - preview_radius

    cv2.circle(frame, (preview_x, preview_y), preview_radius + 2, (255, 255, 255), 2)
    cv2.circle(frame, (preview_x, preview_y), preview_radius, draw_color, -1)
    inner_radius = max(8, min(int(brush_thickness / 2), preview_radius - 6))
    cv2.circle(frame, (preview_x, preview_y), inner_radius, (255, 255, 255), 2)
    cv2.putText(frame, f'{current_tool} {brush_thickness}px', (preview_x - 120, preview_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)


def point_in_circle(point, center, radius):
    return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 <= radius ** 2


def detect_gesture_mode(landmarks):
    if not landmarks or len(landmarks) < 21:
        return 'UNKNOWN', None

    states = get_finger_states(landmarks)
    if not states:
        return 'UNKNOWN', None

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    all_extended = sum(states.values()) == 5
    index_only = states['index'] and not states['middle'] and not states['ring'] and not states['pinky']
    index_middle = states['index'] and states['middle'] and not states['ring'] and not states['pinky']
    pinch_dist = distance(index_tip, thumb_tip)
    pinch = pinch_dist < 0.05

    if all_extended:
        return 'PAUSE', pinch_dist
    if pinch:
        return 'PINCH', pinch_dist
    if index_only or index_middle:
        return 'DRAW', pinch_dist

    return 'UNKNOWN', pinch_dist


def map_brush_thickness(distance_value, min_size=8, max_size=48):
    mapped = int(np.interp(distance_value, [0.02, 0.14], [min_size, max_size]))
    return max(min(mapped, max_size), min_size)


def run_drawing_mode(frame, landmarks):
    global canvas, prev_point, draw_color, selected_color_name, brush_thickness, current_tool, current_gesture, clear_hold_start

    frame_height, frame_width = frame.shape[:2]
    if canvas is None or canvas.shape[:2] != (frame_height, frame_width):
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    if not color_buttons or len(color_buttons) != len(color_palette) or len(tool_buttons) != len(tool_presets):
        build_palettes(frame_width)

    output = frame.copy()

    gesture, pinch_distance = 'UNKNOWN', None
    index_point = None

    palette_hover = False
    selected_color = None
    tool_hover = False
    selected_tool = None
    action_text = ''

    if landmarks and len(landmarks) >= 9:
        index_tip = landmarks[8]
        hand_x = int(index_tip[1] * frame_width)
        hand_y = int(index_tip[2] * frame_height)
        index_point = (hand_x, hand_y)

        gesture, pinch_distance = detect_gesture_mode(landmarks)
        current_gesture = gesture

        for button in tool_buttons:
            if point_in_circle(index_point, button['center'], button['radius'] + 20):
                tool_hover = True
                selected_tool = button
                break

        for button in color_buttons:
            if point_in_circle(index_point, button['center'], button['radius'] + 20):
                palette_hover = True
                selected_color = button
                break

        if gesture == 'PINCH' and selected_tool is not None:
            current_tool = selected_tool['name']
            brush_thickness = selected_tool['size']
            clear_hold_start = None
            prev_point = None
            action_text = f'Tool selected: {current_tool}'

        elif gesture == 'PINCH' and selected_color is not None:
            selected_color_name = selected_color['name']
            draw_color = selected_color['color']
            clear_hold_start = None
            prev_point = None
            action_text = f'Color selected: {selected_color_name}'

        elif gesture == 'DRAW':
            clear_hold_start = None
            if prev_point is None:
                prev_point = index_point
            smooth_point_xy = smooth_point(index_point, prev_point, alpha=0.75)
            cv2.line(canvas, prev_point, smooth_point_xy, draw_color, brush_thickness)
            prev_point = smooth_point_xy
            action_text = 'Drawing...'

        elif gesture == 'PINCH' and pinch_distance is not None:
            if selected_color is None and selected_tool is None:
                brush_thickness = map_brush_thickness(pinch_distance)
                action_text = f'Brush size: {brush_thickness}'
            prev_point = None
            clear_hold_start = None
            if selected_color is None and selected_tool is None:
                brush_thickness = map_brush_thickness(pinch_distance)
                action_text = f'Brush size: {brush_thickness}'
            prev_point = None
            clear_hold_start = None

        elif gesture == 'PAUSE':
            if clear_hold_start is None:
                clear_hold_start = time.time()
                action_text = 'Hold open palm to clear canvas'
            elif time.time() - clear_hold_start > 1.2:
                canvas = np.zeros_like(canvas)
                prev_point = None
                current_tool = 'PEN'
                action_text = 'Canvas cleared'
        else:
            prev_point = None
            if palette_hover and selected_color is not None:
                action_text = f'Hovering over {selected_color["name"]} - pinch to select'
            elif tool_hover and selected_tool is not None:
                action_text = f'Hovering over {selected_tool["name"]} - pinch to select'
            else:
                action_text = 'Ready to draw'

        if palette_hover and selected_color is not None:
            cv2.circle(output, selected_color['center'], selected_color['radius'] + 14, (0, 255, 255), 2)
        if tool_hover and selected_tool is not None:
            cv2.circle(output, selected_tool['center'], selected_tool['radius'] + 14, (0, 255, 255), 2)

    else:
        prev_point = None

    overlay = cv2.addWeighted(output, 0.7, canvas, 0.3, 0)
    draw_toolbar(overlay, current_gesture, hover_color=selected_color, hover_tool=selected_tool)
    draw_status_panel(overlay, action_text)

    if index_point is not None:
        cursor_color = draw_color
        cursor_radius = min(max(int(brush_thickness * 0.8), 16), 32)
        cv2.circle(overlay, index_point, cursor_radius + 10, (255, 255, 255), 2)
        cv2.circle(overlay, index_point, cursor_radius, cursor_color, 3)
        cv2.circle(overlay, index_point, 8, (255, 255, 255), -1)

    return overlay
