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

templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results"

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
    "quarterrest.PNG": 0xF5,
    "halfrest.PNG": 0xF6,
    "wholerest.PNG": 0xF7,
    "eighthrest.PNG": 0xF8,
    "sixteenthrest.PNG": 0xF9,
}

dict_list = [timeSignatureDict, tempoDict, keySignatureDict, dynamicsDict, notesAndRestsDict]

matchedFiles = [] # To store templates that were recognized in the music sheet

hex_values = []  # Array to store hexadecimal values

# Create a dictionary to store the highest correlation value for each ROI
roi_dict = {}

def are_nearby(box1, box2, tolerance):
    return abs(box1[0][0] - box2[0][0]) <= tolerance and abs(box1[0][1] - box2[0][1]) <= tolerance

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
            h, w = template.shape
            top_left = max_loc  # This is the top left point of the match
            bottom_right = (top_left[0] + w, top_left[1] + h)
            roi = musicSheet[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Check each existing ROI to see if this one is nearby
            for existing_roi, data in roi_dict.items():
                if are_nearby(existing_roi, (top_left, bottom_right), tolerance=25):  # assuming a tolerance of 15 pixels
                    # If the new match has a higher correlation value, update the existing ROI data
                    if max_val > data['max_val']:
                        data['max_val'] = max_val
                        data['templateFilename'] = templateFilename
                        break
            else:
                # If this ROI doesn't match any existing ROI, add it as a new entry
                roi_dict[(top_left, bottom_right)] = {'max_val': max_val, 'templateFilename': templateFilename, 'roi': roi}
   
            # Concatenate template and roi horizontally
            concatenated_image = cv.hconcat([template, roi])

            # Get the dimensions of the template
            h_template, w_template = template.shape

            # Draw a vertical line between the images
            cv.line(concatenated_image, (w_template, 0), (w_template, h_template), (0, 0, 255), 4)

            # Save the concatenated image
            outputFilename = f"{sheetFilename}_{templateFilename}"
            outputPath = os.path.join(outputDirectory, outputFilename)
            cv.imwrite(outputPath, concatenated_image)

            matchedFiles.append(templateFilename)

            # Store the hexadecimal value corresponding to the matched template
            hex_value = None
            for dictionary in dict_list:
                hex_value = dictionary.get(templateFilename)
                if hex_value is not None:
                    break
            if hex_value is not None:
                hex_values.append(hex_value)
    
count = 1
# Now, roi_dict contains the highest correlation value and corresponding template for each ROI
for roi_id, data in roi_dict.items():
    print(f"ROI: {roi_id}, Max Val: {data['max_val']}, Template: {data['templateFilename']}")
    print(count)
    count +=1
