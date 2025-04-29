import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
import os
import platform
import numpy as np

try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

is_paused = False
screen_width, screen_height = pyautogui.size()
camera_width, camera_height = screen_width, screen_height

cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

control_mode = "mouse"
prev_x, prev_y = 0, 0  # Previous position for swipe detection
prev_distance = None  # Previous pinch distance for zoom



def set_control_mode():
    global control_mode
    selected = mode_var.get()
    modes = ["mouse", "volume", "brightness", "zoom", "swipe"]
    control_mode = modes[selected - 1]
    print(f"Control mode set to: {control_mode}")

def adjust_brightness(level):
    system_os = platform.system()
    if system_os == "Windows" and sbc:
        try:
            sbc.set_brightness(level)
        except Exception as e:
            print(f"Brightness adjustment failed: {e}")
    elif system_os == "Darwin":  # macOS
        os.system(f"brightness {level / 100.0}")
    else:
        print("Brightness adjustment is not supported on this OS.")

def perform_zoom(distance, prev_distance, zoom_in_threshold=15, zoom_out_threshold=15):
    if prev_distance is not None:
        if distance > prev_distance + zoom_in_threshold:
            print("Zooming In")
            pyautogui.hotkey("command", "+")
        elif distance < prev_distance - zoom_out_threshold:
            print("Zooming Out")
            pyautogui.hotkey("command", "-")
    return distance

def perform_swipe(index_x, prev_x, swipe_threshold=50):
    if prev_x is not None:
        dx = index_x - prev_x
        if dx > swipe_threshold:
            print("Swiped Right")
            pyautogui.hotkey("command", "right")
        elif dx < -swipe_threshold:
            print("Swiped Left")
            pyautogui.hotkey("command", "left")
    return index_x

def start_camera_feed():
    global prev_x, prev_y, prev_distance
    system_os = platform.system()
    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
                    index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                    distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

                    if control_mode == "mouse":
                        pyautogui.moveTo(index_x, index_y)

                    elif control_mode == "volume":
                        volume_level = min(int(distance), 100)
                        print(f"Volume set to: {volume_level}")
                        os.system(f"osascript -e 'set volume output volume {volume_level}'")

                    elif control_mode == "brightness":
                        brightness_level = min(int(distance), 100)
                        print(f"Brightness set to: {brightness_level}")
                        adjust_brightness(brightness_level)

                    elif control_mode == "zoom":
                        prev_distance = perform_zoom(distance, prev_distance)

                    elif control_mode == "swipe":
                        prev_x = perform_swipe(index_x, prev_x)

                    cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            if show_camera.get() == 1:
                cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Hand Gesture Control")
root.geometry("300x350")

show_camera = tk.IntVar(value=1)
camera_checkbox = tk.Checkbutton(root, text="Show Camera Feed", variable=show_camera)
camera_checkbox.pack()

mode_var = tk.IntVar(value=1)
modes = ["Mouse", "Volume", "Brightness", "Zoom", "Swipe"]
for i, mode in enumerate(modes, 1):
    tk.Radiobutton(root, text=mode, variable=mode_var, value=i, command=set_control_mode).pack()

start_button = tk.Button(root, text="Start", command=start_camera_feed)
start_button.pack()

root.mainloop()