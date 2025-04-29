import cv2
import mediapipe as mp
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Initialize MediaPipe Hands before processing images
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Function to process hand image and predict hand presence and handedness
def process_hand_image(image_path):
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image at {image_path}")
        return None, None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(image_rgb)

    # Determine if hand is detected and handedness
    hand_detected = 1 if results.multi_hand_landmarks else 0
    handedness = None
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    return hand_detected, handedness

# Function to evaluate the model on the CSV file
def test_model(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Lists to store actual and predicted values
    actual_hand = []
    predicted_hand = []
    actual_handedness = []
    predicted_handedness = []

    # Loop through the rows of the CSV file
    for _, row in df.iterrows():
        print(f"Processing image {row['sno']} at {row['path']}")
        image_path = row['path']
        actual_hand_label = 1 if row['primary_label'] == 'TRUE' else 0
        actual_handedness_label = row['secondary_label']

        # Process the image
        predicted_hand_label, predicted_handedness_label = process_hand_image(image_path)

        if predicted_hand_label is None:
            continue  # Skip this iteration if there is an issue with the image processing

        # Store the actual and predicted values
        actual_hand.append(actual_hand_label)
        predicted_hand.append(predicted_hand_label)
        actual_handedness.append(actual_handedness_label)
        predicted_handedness.append(predicted_handedness_label if predicted_hand_label == 1 else None)

    # Confusion matrix and performance metrics for hand detection
    cm_hand = confusion_matrix(actual_hand, predicted_hand)
    accuracy_hand = accuracy_score(actual_hand, predicted_hand)
    precision_hand = precision_score(actual_hand, predicted_hand)
    recall_hand = recall_score(actual_hand, predicted_hand)
    f1_hand = f1_score(actual_hand, predicted_hand)

    # Confusion matrix and performance metrics for handedness (only for images where hand is detected)
    cm_handedness = confusion_matrix([h for h in actual_handedness if h != 'None'], 
                                     [h for h in predicted_handedness if h is not None])
    accuracy_handedness = accuracy_score([h for h in actual_handedness if h != 'None'], 
                                        [h for h in predicted_handedness if h is not None])
    precision_handedness = precision_score([h for h in actual_handedness if h != 'None'], 
                                          [h for h in predicted_handedness if h is not None], average='binary', pos_label='LEFT')
    recall_handedness = recall_score([h for h in actual_handedness if h != 'None'], 
                                    [h for h in predicted_handedness if h is not None], average='binary', pos_label='LEFT')
    f1_handedness = f1_score([h for h in actual_handedness if h != 'None'], 
                              [h for h in predicted_handedness if h is not None], average='binary', pos_label='LEFT')

    # Display the results
    print("Hand Detection Results:")
    print("Confusion Matrix:\n", cm_hand)
    print(f"Accuracy: {accuracy_hand:.4f}")
    print(f"Precision: {precision_hand:.4f}")
    print(f"Recall: {recall_hand:.4f}")
    print(f"F1 Score: {f1_hand:.4f}")

    print("\nHandedness Detection Results:")
    print("Confusion Matrix:\n", cm_handedness)
    print(f"Accuracy: {accuracy_handedness:.4f}")
    print(f"Precision: {precision_handedness:.4f}")
    print(f"Recall: {recall_handedness:.4f}")
    print(f"F1 Score: {f1_handedness:.4f}")

# Example usage
csv_path = "../HDAI/HDAI apollo/images/input/test_data.csv"  # Replace with your CSV path
test_model(csv_path)
