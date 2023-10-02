import cv2 as cv
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog 
from skimage import exposure
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

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
directory = "templates"
count = 0
'''
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        # Read in image
        img = imread(f)
        # Resize image
        new_img = resize(img, (128*4, 64*4))
        # For color images
        if len(new_img.shape) == 3:
            fd, hog_img = hog(new_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, channel_axis=-1)
            plt.axis("off")
            plt.imshow(hog_img, cmap="gray")
            plt.savefig("template_features_output/feature {}.png".format(count))
        # For gray images
        else:
            fd, hog_img = hog(new_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)
            plt.axis("off")
            plt.imshow(hog_img, cmap="gray")
            plt.savefig("template_features_output/feature {}.png".format(count))
        count += 1
'''
# feature matching w/ SIFT descriptor
img2 = cv.imread('templates/quarterrest.png', cv.IMREAD_GRAYSCALE).astype('uint8')
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

"""    
plt.axis("off")
plt.imshow(img)z
#plt.show()
print(img.shape)
"""
