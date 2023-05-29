import cv2
import mediapipe as mp
import time
import pyautogui
import tkinter as tk
from pynput import keyboard

pTime=0
pyautogui.FAILSAFE = False


# Global variable to track the input pause state
is_paused = False

# Function to handle keyboard events
def on_key_press(key):
    global is_paused
    if key == keyboard.Key.space:
        is_paused = not is_paused
        if is_paused:
            print("Input paused")

        else:
            print("Input resumed")
    elif key == keyboard.Key.enter:

        pyautogui.click()

# Set up the keyboard listener

listener = keyboard.Listener(on_press=on_key_press)
listener.start()

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Set desired camera vision size
camera_width, camera_height = screen_width, screen_height

# Initialize Mediapipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Variables for cursor movement
cursor_speed = 100  # Adjust the cursor speed as needed (lower value for faster movement)
cursor_x = 0
cursor_y = 0

# Function to toggle camera feed display
def toggle_camera_feed():
    if show_camera.get() == 1:
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        print("hi")


# Create the Tkinter GUI
root = tk.Tk()
root.title("Hand Detection")
root.geometry("300x100")

# Create a checkbox for showing camera feed
show_camera = tk.IntVar(value=1)
camera_checkbox = tk.Checkbutton(root, text="Show Camera Feed", variable=show_camera, command=toggle_camera_feed)
camera_checkbox.pack()

# Function to start the camera feed
def start_camera_feed():
    # Main loop for processing the camera feed
    while True:
        # Read frame from the camera
        _, frame = cap.read()
        frame = cv2.flip(frame, 1) #uncomment if youre facing towards the camera, keep it as a comment if it is a body cam

        # Convert the frame to RGB for Mediapipe
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks with Mediapipe
        result = hands.process(framergb)

        if result.multi_hand_landmarks and not is_paused:
            # Get the landmarks for the first detected hand
            hand_landmarks = result.multi_hand_landmarks[0]

            # Example: Use pyautogui to move the mouse based on hand coordinates
            # You can customize this according to your game's input requirements
            target_x = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
            target_y = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

            # Smoothly move the cursor to the target position
            # cursor_x += int((target_x - cursor_x) / cursor_speed)
            # cursor_y += int((target_y - cursor_y) / cursor_speed)
            pyautogui.moveTo(target_x, target_y)

        # Display the frame in the camera feed window if enabled
        if show_camera.get() == 1:
            cv2.imshow("Camera Feed", frame)
            cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # ...

        # Check for 'q' key press to exit the program
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Button to start the camera feed
start_button = tk.Button(root, text="Start Camera Feed", command=start_camera_feed)
start_button.pack()

# Start the Tkinter event loop
root.mainloop()
