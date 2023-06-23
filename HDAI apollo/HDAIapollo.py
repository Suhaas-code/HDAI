import cv2
import mediapipe as mp
import time
import pyautogui
from pynput import keyboard

# Global settings, comment to disable them
is_paused = False  # sets the code state to running (for mouse input)
pTime = 0  # for calculating fps, no change needed
# pyautogui.FAILSAFE = False  # if enabled, terminates the program when your hand/palm goes to the corner of the screen
screen_width, screen_height = pyautogui.size()  # screen and camera dimensions
camera_width, camera_height = screen_width, screen_height
show_camera = 1


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

# Cooldown settings
cooldown_duration = 1  # Cooldown duration in seconds
cooldown_start_time = time.time() - cooldown_duration


def start_camera_feed():
    global cooldown_start_time

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
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # a box around the hand
                    t = 637
                    u = 957

                    cv2.putText(frame, "Press q to quit and space to pause the operation", (t, u + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    print("press space bar to pause")
                    # Check if thumb and index finger meet
                    thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip_x = int(thumb_tip.x * frame.shape[1])
                    thumb_tip_y = int(thumb_tip.y * frame.shape[0])
                    index_finger_tip_x = int(index_finger_tip.x * frame.shape[1])
                    index_finger_tip_y = int(index_finger_tip.y * frame.shape[0])

                    # Distance between thumb and index fingertips
                    distance = ((thumb_tip_x - index_finger_tip_x) ** 2 + (
                            thumb_tip_y - index_finger_tip_y) ** 2) ** 0.5

                    if distance < 50:  # Adjust the distance threshold as needed
                        current_time = time.time()
                        cv2.putText(frame, "Thumb and Index Finger Meet", (t, u - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        if current_time - cooldown_start_time >= cooldown_duration:
                            cooldown_start_time = current_time

                            print("Thumb and Index Finger Meet")
                            # pyautogui.click() # experimental feature, mouse click

                            # Sign Code : If thumb and index finger meet
                            if h > 0 and w > 0:
                                # Center point of the thumb and index finger landmarks
                                thumb_center = (int(thumb_tip_x), int(thumb_tip_y))
                                index_center = (int(index_finger_tip_x), int(index_finger_tip_y))

                                # Shows if the thumb and index finger meet
                                radius = int(distance / 2)
                                cv2.circle(frame, thumb_center, radius, (0, 0, 255), cv2.FILLED)
                                cv2.circle(frame, index_center, radius, (0, 0, 255), cv2.FILLED)
                            else:
                                print("Resize issues")

                    # Scroll Condition : Pinky finger should be opened but to avoid errors,
                    # additional condition index finger should be lower than the pinky finger,
                    # invert the hand upside down to reproduce the desired scrolling
                    pinky_tip_y = int(hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y * screen_height)
                    is_pinky_lifted = pinky_tip_y < hand_landmarks.landmark[
                        mpHands.HandLandmark.INDEX_FINGER_TIP].y * screen_height

                    # Scroll feature
                    if is_pinky_lifted:
                        scroll_amount = int(distance * -0.2)  # Adjust the scroll sensitivity as needed
                        pyautogui.scroll(scroll_amount)
                        #print(scroll_amount)
                        cv2.putText(frame, "Scrolling", (t, u - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Index Finger Location
                    target_x = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                    target_y = int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
                    # print(target_y, target_x)      # prints index finger's tip's location wrt your screen

                    # Cursor Movement
                    pyautogui.moveTo(target_x, target_y)
                    # cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1)       # cursor location

            # Camera enabled
            if show_camera == 1:
                cv2.imshow("Camera Feed", frame)
                cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Check for 'q' key press to exit the program
            if cv2.waitKey(1) == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Start the camera feed
start_camera_feed()
