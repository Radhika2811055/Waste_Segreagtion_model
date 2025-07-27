import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the trained model and labels
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

# Initialize bin images
imgBinsList = []
pathFolderBins = "Resources/Bins"
pathList = os.listdir(pathFolderBins)
pathList.sort()  # Ensure consistent ordering of bins
for path in pathList:
    img = cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Unable to load bin image at {path}")
        exit(1)
    # Ensure all images have an alpha channel (convert RGB to RGBA if necessary)
    if img.shape[2] == 3:  # If image has only 3 channels (RGB)
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque alpha channel
        img = cv2.merge((b, g, r, alpha))
    imgBinsList.append(img)

# Map class IDs to corresponding bin indices
classDic = {
    0: 0,  # Default bin (e.g., unknown or no waste detected)
    1: 0,  # Biodegradable waste
    2: 0,  # Biodegradable waste
    3: 1,  # Hazardous waste
    4: 1,  # Recyclable waste
    5: 1,  # Recyclable waste
    6: 0,  # Biodegradable waste
    7: 1,  # Hazardous waste
    8: 0   # Biodegradable waste
}

while True:
    # Capture frame from webcam
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))  # Resize for overlay

    # Predict the waste type
    prediction = classifier.getPrediction(img)
    classID = prediction[1]
    print(f"Predicted Class: {classID}")

    # Determine the corresponding bin
    classIDBin = classDic.get(classID, 0)

    # Load the background and overlay the bin
    imgBackground = cv2.imread('Resources/background.jpg')
    if imgBackground is None:
        print("Error: Unable to load background image.")
        break

    try:
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))
    except Exception as e:
        print(f"Error during overlay: {e}")
        break

    # Overlay the resized webcam feed on the background
    imgBackground[148:148 + 340, 159:159 + 454] = imgResize

    # Combine the webcam feed and the output into one frame
    imgCombined = np.hstack((cv2.resize(img, (640, 480)), cv2.resize(imgBackground, (640, 480))))

    # Display the combined output
    cv2.imshow("Combined Output", imgCombined)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
