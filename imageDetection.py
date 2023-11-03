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
outputDirectory = "results/siftResults"

# Mapping of template filenames to hexadecimal values
timeSignatureDict = {
    "24_timesignature.PNG": 0x24,
    "34_timesignature.PNG": 0x34,
    "44_timesignature.PNG": 0x44,
}
tempoDict = {
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
    "200_tempo.PNG": 0xC8,
}
keySignatureDict = {
    "cmajor": 0x0,
    "flat_f.PNG": 0x1,
    "flat_bflt.PNG": 0x2,
    "flat_eflt.PNG": 0x3,
    "flat_aflt.PNG": 0x4,
    "flat_dflt.PNG": 0x5,
    "flat_gflt.PNG": 0x6,
    "flat_cflt.PNG": 0x7,
    "sharp_g.PNG": 0x8,
    "sharp_d.PNG": 0x9,
    "sharp_a.PNG": 0xA,
    "sharp_e.PNG": 0xB,
    "sharp_b.PNG": 0xC,
    "sharp_fshrp.PNG": 0xD,
    "sharp_cshrp.PNG": 0xE,
}
dynamicsDict = {
    "forte.PNG": 0x0,
    "fortissimo.PNG": 0x1,
    "piano.PNG": 0x2,
    "pianissimo.PNG": 0x3,
}
notesAndRestsDict = {
    "quarternote_lowc.PNG": 0x00,
    "quarternote_lowd.PNG": 0x10,
    "quarternote_lowe.PNG": 0x20,
    "quarternote_lowf.PNG": 0x30,
    "quarternote_lowg.PNG": 0x40,
    "quarternote_lowa.PNG": 0x50,
    "quarternote_lowb.PNG": 0x60,
    "quarternote_middlec.PNG": 0x70,
    "quarternote_highd.PNG": 0x80,
    "quarternote_highe.PNG": 0x90,
    "quarternote_highf.PNG": 0xA0,
    "quarternote_highg.PNG": 0xB0,
    "quarternote_higha.PNG": 0xC0,
    "quarternote_highb.PNG": 0xD0,
    "quarternote_highc.PNG": 0xE0,
    "halfnote_lowc.PNG": 0x01,
    "halfnote_lowd.PNG": 0x11,
    "halfnote_lowe.PNG": 0x21,
    "halfnote_lowf.PNG": 0x31,
    "halfnote_lowg.PNG": 0x41,
    "halfnote_lowa.PNG": 0x51,
    "halfnote_lowb.PNG": 0x61,
    "halfnote_middlec.PNG": 0x71,
    "halfnote_highd.PNG": 0x81,
    "halfnote_highe.PNG": 0x91,
    "halfnote_highf.PNG": 0xA1,
    "halfnote_highg.PNG": 0xB1,
    "halfnote_higha.PNG": 0xC1,
    "halfnote_highb.PNG": 0xD1,
    "halfnote_highc.PNG": 0xE1,
    "wholenote_lowc.PNG": 0x02,
    "wholenote_lowd.PNG": 0x12,
    "wholenote_lowe.PNG": 0x22,
    "wholenote_lowf.PNG": 0x32,
    "wholenote_lowg.PNG": 0x42,
    "wholenote_lowa.PNG": 0x52,
    "wholenote_lowb.PNG": 0x62,
    "wholenote_middlec.PNG": 0x72,
    "wholenote_highd.PNG": 0x82,
    "wholenote_highe.PNG": 0x92,
    "wholenote_highf.PNG": 0xA2,
    "wholenote_highg.PNG": 0xB2,
    "wholenote_higha.PNG": 0xC2,
    "wholenote_highb.PNG": 0xD2,
    "wholenote_highc.PNG": 0xE2,
    "eighthnote_lowc.PNG": 0x03,
    "eighthnote_lowd.PNG": 0x13,
    "eighthnote_lowe.PNG": 0x23,
    "eighthnote_lowf.PNG": 0x33,
    "eighthnote_lowg.PNG": 0x43,
    "eighthnote_lowa.PNG": 0x53,
    "eighthnote_lowb.PNG": 0x63,
    "eighthnote_middlec.PNG": 0x73,
    "eighthnote_highd.PNG": 0x83,
    "eighthnote_highe.PNG": 0x93,
    "eighthnote_highf.PNG": 0xA3,
    "eighthnote_highg.PNG": 0xB3,
    "eighthnote_higha.PNG": 0xC3,
    "eighthnote_highb.PNG": 0xD3,
    "eighthnote_highc.PNG": 0xE3,
    "sixteenthnote_lowc.PNG": 0x04,
    "sixteenthnote_lowd.PNG": 0x14,
    "sixteenthnote_lowe.PNG": 0x24,
    "sixteenthnote_lowf.PNG": 0x34,
    "sixteenthnote_lowg.PNG": 0x44,
    "sixteenthnote_lowa.PNG": 0x54,
    "sixteenthnote_lowb.PNG": 0x64,
    "sixteenthnote_middlec.PNG": 0x74,
    "sixteenthnote_highd.PNG": 0x84,
    "sixteenthnote_highe.PNG": 0x94,
    "sixteenthnote_highf.PNG": 0xA4,
    "sixteenthnote_highg.PNG": 0xB4,
    "sixteenthnote_higha.PNG": 0xC4,
    "sixteenthnote_highb.PNG": 0xD4,
    "sixteenthnote_highc.PNG": 0xE4,
    "quarterrest.PNG": 0xF5,
    "halfrest.PNG": 0xF6,
    "wholerest.PNG": 0xF7,
    "eighthrest.PNG": 0xF8,
    "sixteenthrest.PNG": 0xF9,
}

dict_list = [timeSignatureDict, tempoDict, keySignatureDict, dynamicsDict, notesAndRestsDict]

matchedFiles = [] # To store templates that were recognized in the music sheet

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
        threshold = 0.9
        if max_val > threshold:
            #print(max_val, " ", templateFilename)
            # Define the region of interest (ROI) around the max correlation location
            h, w = template.shape
            top_left = max_loc  # This is the top left point of the match
            bottom_right = (top_left[0] + w, top_left[1] + h)
            roi = musicSheet[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

             # SIFT with FLANN matcher on ROI
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(roi, None)

            # # SURF with FLANN matcher on ROI
            # surf = cv.xfeatures2d.SURF_create()
            # kp1, des1 = surf.detectAndCompute(template, None)
            # kp2, des2 = surf.detectAndCompute(roi, None)


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
                
                # Concatenate template and roi horizontally
                concatenated_image = cv.hconcat([template, roi])

                # Get the dimensions of the template
                h_template, w_template = template.shape

                # Draw a vertical line between the images
                cv.line(concatenated_image, (w_template, 0), (w_template, h_template), (0, 0, 255), 5)

                # Save the concatenated image
                outputFilename = f"{sheetFilename}_{templateFilename}"
                outputPath = os.path.join(outputDirectory, outputFilename)
                cv.imwrite(outputPath, concatenated_image)

                
                #matchingResult = cv.drawMatchesKnn(template, kp1, roi, kp2, matches, None, **draw_params)
                
                # # Save the plot as an image
                # outputFilename = f"{sheetFilename}_{templateFilename}"
                # outputPath = os.path.join(outputDirectory, outputFilename)
                # cv.imwrite(outputPath, matchingResult)


                matchedFiles.append(templateFilename)

                # Store the hexadecimal value corresponding to the matched template
                hex_value = None
                for dictionary in dict_list:
                    hex_value = dictionary.get(templateFilename)
                    if hex_value is not None:
                        break
                if hex_value is not None:
                    hex_values.append(hex_value)
    
print(hex_values)
#print(matchedFiles)