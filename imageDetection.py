import cv2 as cv
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog 
from skimage import exposure
import matplotlib.pyplot as plt

cam_port = 1
cam = cv.VideoCapture(cam_port)

result, image = cam.read()
print(result)
if result:
    cv.imshow("Capture Test", image)
    cv.imwrite("Capture Test.png", image)
else:
    print("Error: No image detected")

# testing HOG using skimage library

img = imread('templates/flat.png')
plt.axis("off")
plt.imshow(img)
print(img.shape)

# resize the image
new_img = resize(img, (128*4), (64*4))
plt.axis("off")
plt.imshow(new_img)
print(new_img.shape)

# create/visualize HOG features
fd, hog_img = hog(new_img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True,multichannel=True)
plt.axis("off")
plt.imshow(hog_img, cmap="gray")
plt.show()