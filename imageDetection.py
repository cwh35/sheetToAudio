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

# feature mtaching w/ ORB descriptor
# img1 = cv.imread('templates/halfnote_lowf.PNG',cv.IMREAD_GRAYSCALE) # queryImage
# img2 = cv.imread('sheets/samplesheet.PNG',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate ORB detector
# orb = cv.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# # create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# feature matching w/ SIFT descriptor
img2 = cv.imread('templates/halfnote_lowf.png', cv.IMREAD_GRAYSCALE).astype('uint8')
img2 = cv.resize(img2, (400,400))
img1 = cv.imread('sheets/samplesheet.png', cv.IMREAD_GRAYSCALE).astype('uint8')
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


