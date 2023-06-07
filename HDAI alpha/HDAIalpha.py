import cv2
import mediapipe as mp
import time
import pyautogui
import tkinter as tk
from pynput import keyboard

# I have taken help from ChatGPT, but the code isn't in its entirety
# Bugs may prevail, don't squash em, they're upcoming features
# For now try with the camera on only

# Global settings, comment to disable them
is_paused = False  # sets the code state to running (for mouse input)
pTime = 0  # for calculating fps, no change needed
#pyautogui.FAILSAFE = False  # if enabled, terminates the program when your hand/palm goes to the corner of the screen
screen_width, screen_height = pyautogui.size()  # screen and camera dimensions
camera_width, camera_height = screen_width, screen_height
sign_image = cv2.imread("sign.png")  # Replace "sign.png" with the path to your sign image

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

# Mediapipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Variables for cursor movement - linear smoothening - disabled by default
# cursor_speed = 100  # Adjust the cursor speed as needed (lower value for faster movement)
# cursor_x = 0
# cursor_y = 0

# Hand State
def is_hand_open(hand_landmarks):
    # Define the landmark points for an open hand gesture
    open_landmark_ids = [mpHands.HandLandmark.THUMB_TIP,
                         mpHands.HandLandmark.INDEX_FINGER_TIP,
                         mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                         mpHands.HandLandmark.RING_FINGER_TIP,
                         mpHands.HandLandmark.PINKY_TIP]

    for landmark_id in open_landmark_ids:
        if hand_landmarks.landmark[landmark_id].y > hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y:
            return False
    return True


# Keyboard events
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


# Keyboard listener
listener = keyboard.Listener(on_press=on_key_press)
listener.start()


# Camera feed display
def toggle_camera_feed():
    if show_camera.get() == 1:
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        print("Camera feed off")


# Tkinter Stuff
root = tk.Tk()
root.title("Hand Detection")
root.geometry("300x100")
show_camera = tk.IntVar(value=1)
camera_checkbox = tk.Checkbutton(root, text="Show Camera Feed", variable=show_camera, command=toggle_camera_feed)
camera_checkbox.pack()

# Cooldown settings
cooldown_duration = 2  # Cooldown duration in seconds
cooldown_start_time = time.time() - cooldown_duration


def start_camera_feed():
    global cooldown_start_time
    hand_open_state = False

    while True:
        if not is_paused:
            success, frame = cap.read()
            if not success:
                break

            # Frame Operations
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Landmarks and Gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the landmarks for the first detected hand - keep it disabled
                    # mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    # hand_landmarks = results.multi_hand_landmarks[0] #disable, use only for testing

                    # Calculate Box Coordinates
                    landmark_x = [landmark.x for landmark in hand_landmarks.landmark]
                    landmark_y = [landmark.y for landmark in hand_landmarks.landmark]
                    min_x = min(landmark_x)
                    max_x = max(landmark_x)
                    min_y = min(landmark_y)
                    max_y = max(landmark_y)
                    x = int(min_x * frame.shape[1])
                    y = int(min_y * frame.shape[0])
                    w = int((max_x - min_x) * frame.shape[1])
                    h = int((max_y - min_y) * frame.shape[0])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # a box around the hand

                    # Check if hand is open
                    if is_hand_open(hand_landmarks):
                        current_time = time.time()
                        if not hand_open_state and current_time - cooldown_start_time >= cooldown_duration:
                            hand_open_state = True
                            cooldown_start_time = current_time
                            print("Hand Opened")
                            cv2.putText(frame, "Hand Open", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Sign Image Code
                            if h > 0 and w > 0:
                                sign_height, sign_width, _ = sign_image.shape
                                resized_sign = cv2.resize(sign_image, (w, h))
                                frame[y:y + h, x:x + w] = resized_sign
                            else:
                                print("resize issues")
                    else:
                        hand_open_state = False

                    # Get the index fingertip position
                    target_x = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                    target_y = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
                    #print(target_y, target_x)      # prints index finger's tip (pointer)'s location wrt your screen

                    # Cursor Movement 
                    pyautogui.moveTo(target_x, target_y)
                    # cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1) #cursor location, disable for better fps

            # Camera enabled
            if show_camera.get() == 1:
                cv2.imshow("Camera Feed", frame)
                cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Check for 'q' key press to exit the program
            if cv2.waitKey(1) == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Start the camera feed through tkinter
start_button = tk.Button(root, text="Execute", command=start_camera_feed)

# tkinter stuff again
start_button.pack()
root.mainloop()
