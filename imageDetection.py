import cv2 as cv
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog 
from skimage import exposure
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

"""
cam_port = 1
cam = cv.VideoCapture(cam_port)

result, image = cam.read()
print(result)
if result:
    cv.imshow("Capture Test", image)
    cv.imwrite("Capture Test.png", image)
    template matching
else:
    print("Error: No image detected")
"""

# SIFT w/ FLANN based matcher
templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results/siftWithFLANNResults"

# Mapping of template filenames to hexadecimal values
template_to_hex = {
    "24_timesignature.PNG": 0x24,
    "34_timesignature.PNG": 0x34,
    "44_timesignature.PNG": 0x44,
    "40_tempo.PNG": 0x28,
    "50_tempo.PNG": 0x32,
    "60_tempo.PNG": 0x3C,
    "70_tempo.PNG": 0x46,
    "80_tempo.PNG": 0x50,
    "90_tempo.PNG": 0x5A,
    "100_tempo.PNG": 0x64,
    "110_tempo.PNG": 0x6E,
    "120_tempo.PNG": 0x78,
    "130_tempo.PNG": 0x82,
    "140_tempo.PNG": 0x8C,
    "150_tempo.PNG": 0x96,
    "160_tempo.PNG": 0xA0,
    "170_tempo.PNG": 0xAA,
    "180_tempo.PNG": 0xB4,
    "190_tempo.PNG": 0xBE,
    "200_tempo.PNG": 0xC8
}

hex_values = []  # Array to store hexadecimal values

# Store the good matches here
good = []
# Loop through the music sheets
for sheetFilename in os.listdir(sheetDirectory):
    f = os.path.join(sheetDirectory, sheetFilename)
    musicSheet = cv.imread(f ,cv.IMREAD_GRAYSCALE)  # trainImage

    # Loop through the templates
    for templateFilename in os.listdir(templateDirectory):
        file = os.path.join(templateDirectory, templateFilename)
        template = cv.imread(file ,cv.IMREAD_GRAYSCALE)  # queryImage

        # Cross-correlation between templates and music sheet
        res = cv.matchTemplate(musicSheet, template, cv.TM_CCOEFF_NORMED)
        # Get the min and max correlation value as well as locations of the matched points
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # threshold to filter valid matches
        threshold = 0.8
        if max_val > threshold:
            # Define the region of interest (ROI) around the max correlation location
            h, w = template.shape
            top_left = max_loc  # This is the top left point of the match
            bottom_right = (top_left[0] + w, top_left[1] + h)
            roi = musicSheet[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

             # SIFT with FLANN matcher on ROI
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(roi, None)

            # Check the number of keypoints and adjust k accordingly
            min_keypoints = min(len(kp1), len(kp2))
            k_value = min(2, min_keypoints)  # Ensure k is at most 2, but not more than the number of keypoints

            if k_value < 2:
                print(f"Warning: Only {min_keypoints} keypoints found. Adjusting k to {k_value}")

            if k_value > 0:
                # KD-Tree (nearest neighbor searches)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
                search_params = dict(checks=75)  # checks = # of times to check consistency
                # match the key points
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=k_value)  # Use adjusted k_value

                matchesMask = [[0, 0] for i in range(len(matches))]
                if k_value == 2:
                    for i, (m, n) in enumerate(matches):
                        if m.distance < 0.7 * n.distance:
                            matchesMask[i] = [1, 0]
                else:
                    for i, m in enumerate(matches):
                        matchesMask[i] = [1, 0]  # If k_value is 1, keep all matches

                draw_params = dict(matchColor=(0, 255, 0),
                                singlePointColor=(255, 0, 0),
                                matchesMask=matchesMask,
                                flags=cv.DrawMatchesFlags_DEFAULT)
                
                matchingResult = cv.drawMatchesKnn(template, kp1, roi, kp2, matches, None, **draw_params)

                # Save the plot as an image
                outputFilename = f"{sheetFilename}_{templateFilename}"
                outputPath = os.path.join(outputDirectory, outputFilename)
                cv.imwrite(outputPath, matchingResult)

                # Store the hexadecimal value corresponding to the matched template
                hex_value = template_to_hex.get(templateFilename)
                if hex_value is not None:
                    hex_values.append(hex_value)
    
print(hex_values)
