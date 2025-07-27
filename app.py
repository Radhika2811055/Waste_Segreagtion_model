import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import matplotlib.pyplot as plt
import json
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Unable to access the webcam.")

# Load the trained model and labels
try:
    classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
except Exception as e:
    raise RuntimeError(f"Error loading model or labels: {e}")

# Initialize bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
pathList.sort()  # Ensure consistent ordering of bins
for path in pathList:
    imgBin = cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED)
    if imgBin is None:
        raise RuntimeError(f"Error: Unable to load bin image at {path}")
    imgBinsList.append(imgBin)

# Map class IDs to corresponding bin indices
classDic = {
    0: 0,  # Default bin (e.g., unknown or no waste detected)
    1: 0,  # Recyclable waste
    2: 0,  # Recyclable waste
    3: 0,  # Recyclable waste
    4: 0,  # Recyclable waste
    5: 0,  # Recyclable waste
    6: 1,  # Non-Recyclable waste
    7: 1,  # Non-Recyclable waste
    8: 1,  # Non-Recyclable waste
    9: 1,  # Non-Recyclable waste
    10: 1,  # Non-Recyclable waste
}

# Initialize dictionary to store waste count
waste_count = {i: 0 for i in range(len(classDic))}

# File to save waste data
output_file = "waste_data.json"

# Load existing data if the file exists and is valid
try:
    with open(output_file, "r") as file:
        waste_count = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    # Initialize default waste_count if file is missing or corrupted
    waste_count = {i: 0 for i in range(len(classDic))}
    print("Initialized default waste_count due to missing or corrupted file.")

# Function to save waste count to a file safely
def save_waste_count():
    temp_file = "temp_waste_data.json"
    try:
        with open(temp_file, "w") as file:
            json.dump(waste_count, file)
        os.rename(temp_file, output_file)
    except Exception as e:
        print(f"Error saving waste count: {e}")

# Main loop
while True:
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Error: Unable to read frame from webcam.")
        break
    imgResize = cv2.resize(img, (454, 340))

    # Predict the waste type
    try:
        prediction = classifier.getPrediction(img)
        classID = prediction[1]
        print(f"Predicted Class: {classID}")
    except Exception as e:
        print(f"Prediction error: {e}")
        continue

    # Update waste count
    waste_count[classID] = waste_count.get(classID, 0) + 1

    # Save updated waste count to file every 10 iterations
    if sum(waste_count.values()) % 10 == 0:
        save_waste_count()

    # Determine the corresponding bin
    classIDBin = classDic.get(classID, 0)

    # Load the background and overlay the bin
    imgBackground = cv2.imread('Resources/background.jpg')
    if imgBackground is None:
        print("Error: Background image could not be loaded.")
        break

    # Ensure bin image exists before overlay
    if classIDBin < len(imgBinsList):
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
    else:
        print(f"Error: Invalid bin index {classIDBin}.")

    # Overlay the resized webcam feed on the background
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Display output with background and bins in another window
    cv2.imshow("Output", imgBackground)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Break loop on pressing 'q'
    if key == ord('q'):
        break

    # Display graph on pressing 'g'
    if key == ord('g'):
        labels = [f"Class {k}" for k in waste_count.keys()]
        counts = list(waste_count.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='green')
        plt.xlabel("Waste Classes")
        plt.ylabel("Count")
        plt.title("Waste Segregation Count")
        plt.show()

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save data before exiting
save_waste_count()
