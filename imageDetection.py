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
else:
    print("Error: No image detected")
"""
directory = "templates"
count = 0

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        img = imread(f)
        new_img = resize(img, (128*4, 64*4))
        if len(new_img.shape) == 3:
            fd, hog_img = hog(new_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, channel_axis=-1)
            plt.axis("off")
            plt.imshow(hog_img, cmap="gray")
            plt.savefig("template_features_output/feature {}.png".format(count))
        else:
            fd, hog_img = hog(new_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)
            plt.axis("off")
            plt.imshow(hog_img, cmap="gray")
            plt.savefig("template_features_output/feature {}.png".format(count))
        count += 1
# testing HOG using skimage library

"""    
plt.axis("off")
plt.imshow(img)
#plt.show()
print(img.shape)
"""
# resize the image
# create/visualize HOG features
