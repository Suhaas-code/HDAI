import cv2
import mediapipe as mp
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.9
)
mp_draw = mp.solutions.drawing_utils

# Function to process hand image, predict hand presence and save annotated image
def process_hand_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image at {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_detected = 1 if results.multi_hand_landmarks else 0

    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imwrite(output_path, image)
    return hand_detected

# Function to save and plot evaluation metrics in the images folder
def save_and_plot_metrics(metric_values, metric_name, output_dir):
    plt.figure()
    plt.plot(metric_values)
    plt.title(f'{metric_name} over iterations')
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'graph_{metric_name}_plot.png'))
    plt.close()

# Evaluate model performance using CSV input
def test_model(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    actual_hand = []
    predicted_hand = []

    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    mcc_values = []
    balanced_acc_values = []
    kappa_values = []

    for _, row in df.iterrows():
        image_path = row['path']
        image_sno = str(row['sno'])
        actual_label_str = str(row['primary_label']).strip().lower()
        actual_hand_label = 1 if actual_label_str == 'true' else 0

        output_path = os.path.join(output_dir, f"annotated_{image_sno}.jpg")
        predicted_hand_label = process_hand_image(image_path, output_path)

        actual_hand.append(actual_hand_label)
        predicted_hand.append(predicted_hand_label)

        # Metrics
        cm_hand = confusion_matrix(actual_hand, predicted_hand)
        accuracy_hand = accuracy_score(actual_hand, predicted_hand)
        precision_hand = precision_score(actual_hand, predicted_hand, zero_division=0)
        recall_hand = recall_score(actual_hand, predicted_hand, zero_division=0)
        f1_hand = f1_score(actual_hand, predicted_hand, zero_division=0)
        mcc_hand = matthews_corrcoef(actual_hand, predicted_hand)
        balanced_acc = balanced_accuracy_score(actual_hand, predicted_hand)
        kappa = cohen_kappa_score(actual_hand, predicted_hand)

        accuracy_values.append(accuracy_hand)
        precision_values.append(precision_hand)
        recall_values.append(recall_hand)
        f1_values.append(f1_hand)
        mcc_values.append(mcc_hand)
        balanced_acc_values.append(balanced_acc)
        kappa_values.append(kappa)

    # Save and plot metrics
    save_and_plot_metrics(accuracy_values, 'Accuracy', output_dir)
    save_and_plot_metrics(precision_values, 'Precision', output_dir)
    save_and_plot_metrics(recall_values, 'Recall', output_dir)
    save_and_plot_metrics(f1_values, 'F1 Score', output_dir)
    save_and_plot_metrics(mcc_values, 'MCC', output_dir)
    save_and_plot_metrics(balanced_acc_values, 'Balanced Accuracy', output_dir)
    save_and_plot_metrics(kappa_values, 'Cohen\'s Kappa', output_dir)

    # Print final results
    print("Hand Detection Results:")
    print(f"Accuracy: {accuracy_values[-1]:.4f}")
    print(f"Precision: {precision_values[-1]:.4f}")
    print(f"Recall: {recall_values[-1]:.4f}")
    print(f"F1 Score: {f1_values[-1]:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc_values[-1]:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_values[-1]:.4f}")
    print(f"Cohen’s Kappa: {kappa_values[-1]:.4f}")

# Run
output_dir = "HDAI apollo/images/output/"
os.makedirs(output_dir, exist_ok=True)

# Clear existing images in output_dir
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

csv_path = "HDAI apollo/images/test.csv"
test_model(csv_path, output_dir)
