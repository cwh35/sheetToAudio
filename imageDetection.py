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

# directory = "templates"
# template_features = []

# # Harris Corner Detection on templates
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f):
#         # Read in image
#         img = cv.imread(f)
#         # Convert to gray scale
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#         gray = np.float32(gray)
#         dst = cv.cornerHarris(gray, 2, 3, 0.04)

#         # Dilated for marking corners
#         dst = cv.dilate(dst, None)

#         # Threshold for optimum value
#         img[dst > 0.01*dst.max()] = [0, 0, 255]

#         # Store result in a folder for visual representation
#         cv.imwrite(f"templateCorners/{filename}.png", img)

#         # Extract corner features (coordinates) and store them in a list
#         template_corners = np.argwhere(dst > 0.01 *dst.max())
#         template_features.append(template_corners)

# sheetDirectory = "sheets"
# music_sheet_features = []

# # Harris Corner Detection on the unlabeled music sheets
# for sheetFilename in os.listdir(sheetDirectory):
#     f = os.path.join(sheetDirectory, sheetFilename)
#     if os.path.isfile(f):
#          # Read in image
#         img = cv.imread(f)
#         # Convert to gray scale
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#         gray = np.float32(gray)
#         dst = cv.cornerHarris(gray, 2, 3, 0.04)

#         # Dilated for marking corners
#         dst = cv.dilate(dst, None)

#         # Threshold for optimum value
#         img[dst > 0.01*dst.max()] = [0, 0, 255]

#         cv.imwrite(f"sheetCorners/{sheetFilename}.png", img)

#         # Extract corner features (coordinates) and store them in a list
#         music_sheet_corners = np.argwhere(dst > 0.01 *dst.max())
#         music_sheet_features.append(music_sheet_corners)


# SIFT w/ FLANN based matcher
templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results/siftWithFLANNResults"

good = []
# Loop through the music sheets
for sheetFilename in os.listdir(sheetDirectory):
    f = os.path.join(sheetDirectory, sheetFilename)
    musicSheet = cv.imread(f ,cv.IMREAD_GRAYSCALE) # trainImage

    # Loop through the templates
    for templateFilename in os.listdir(templateDirectory):
        file = os.path.join(templateDirectory, templateFilename) 
        template = cv.imread(file ,cv.IMREAD_GRAYSCALE) # queryImage

         # Cross-correlation between templates and music sheet
        res = cv.matchTemplate(musicSheet, template, cv.TM_CCOEFF_NORMED)
        # Get the min and max correlation value as well as locations of the matched points
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # threshold to filter valid matches
        threshold = 0.93
        if max_val > threshold:
            # SIFT with FLANN matcher
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(musicSheet, None)

            # KD-Tree (nearest neighbor searches)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
            search_params = dict(checks=75) # checks = # of times to check consistency
            # match the key points
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            matchesMask = [[0, 0] for i in range(len(matches))]
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                                singlePointColor=(255, 0, 0),
                                matchesMask=matchesMask,
                                flags=cv.DrawMatchesFlags_DEFAULT)

            matchingResult = cv.drawMatchesKnn(template, kp1, musicSheet, kp2, matches, None, **draw_params)

            # Save the plot as an image
            outputFilename = f"{sheetFilename}_{templateFilename}"
            outputPath = os.path.join(outputDirectory, outputFilename)
            cv.imwrite(outputPath, matchingResult)
   
