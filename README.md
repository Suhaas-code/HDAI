# HDAI
Hand Detection As Input - a python project to identify hand movement as an input through computer vision that utilizes the Mediapipe library for hand tracking and provides feature gestures as input.

### Libraries used:
opencv-python
mediapipe
pythonautogui
pynput

## Usage
1. Install the required libraries mentioned in the code, such as `cv2`, `mediapipe`, `time`, `pyautogui`, `tkinter`, and `pynput`.
2. Replace the value of the `sign_image` variable with the path to your own sign image file.
3. Run the script, and the application will start capturing video from the default camera.
4. The hand detection and gesture recognition will be performed in real-time on the camera feed.
5. When the thumb and index finger meet closely, the corresponding actions will be triggered, such as mouse click simulation and sign image overlay.
6. The checkbox option in the GUI can be used to toggle the display of the camera feed window.
7. The button in the GUI initiates the execution of the camera feed and the hand detection functionality.
8. Use the spacebar key to pause/resume the input processing.
9. Use the enter key to trigger a mouse click action.
10. Press 'q' in the camera feed window or close the window to exit the application.
