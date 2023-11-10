import cv2 as cv
import os
import numpy as np
from pprint import pprint
from MTM import matchTemplates

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
    "cmajor": 0x00,
    "flat_f.PNG": 0x01,
    "flat_bflt.PNG": 0x02,
    "flat_eflt.PNG": 0x03,
    "flat_aflt.PNG": 0x04,
    "flat_dflt.PNG": 0x05,
    "flat_gflt.PNG": 0x06,
    "flat_cflt.PNG": 0x07,
    "sharp_g.PNG": 0x08,
    "sharp_d.PNG": 0x09,
    "sharp_a.PNG": 0x0A,
    "sharp_e.PNG": 0x0B,
    "sharp_b.PNG": 0x0C,
    "sharp_fshrp.PNG": 0x0D,
    "sharp_cshrp.PNG": 0x0E,
}
dynamicsDict = {
    "forte.PNG": 0x00,
    "fortissimo.PNG": 0x01,
    "piano.PNG": 0x02,
    "pianissimo.PNG": 0x03,
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

templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results"

dict_list = [timeSignatureDict, tempoDict, keySignatureDict, dynamicsDict, notesAndRestsDict]

# Array to store hexadecimal values
hex_values = []

def get_center(box):
    return ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)

def are_nearby(box1, box2, tolerance):
    center1 = get_center(box1)
    center2 = get_center(box2)
    return abs(center1[0] - center2[0]) <= tolerance and abs(center1[1] - center2[1]) <= tolerance

def split_image(image, shift_pixels=0, pixelDivision = 170):
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Check if the height is divisible by pixelDivision (height of one line of music)
    if height % pixelDivision != 0:
        raise ValueError("The height of the image is not divisible by 160 pixels.")

    # Calculate the number of lines in the image
    num_lines = (height - shift_pixels) // pixelDivision

    # Initialize an empty list to store the line images
    lines_of_music = []

    # Split the image into individual lines of music
    for i in range(num_lines):
        # Calculate the starting and ending y-coordinates of the current line
        start_y = i * pixelDivision + shift_pixels
        end_y = start_y + pixelDivision

        # Extract the current line from the image
        line_image = image[start_y:end_y + 1, :]

        # Save the current line image to a file
        cv.imwrite(f"line{i}.PNG", line_image)

        # Append the line image to the list
        lines_of_music.append(line_image)

    return lines_of_music

def get_hex_value(template_filename):
    # Iterate over all the dictionaries in dict_list
    for dictionary in dict_list:
        # Check if the template filename is in the current dictionary
        if template_filename in dictionary:
            # Return the corresponding hex value
            return dictionary[template_filename]
    # If the template filename is not found in any dictionary, handle the case (e.g., return None or raise an error)
    return None  # or raise ValueError(f"No hex value found for template: {template_filename}")

all_matches = []
listTemplate = []

# Load templates into listTemplate
for filename in os.listdir(templateDirectory):
    template_img = cv.imread(os.path.join(templateDirectory, filename), cv.IMREAD_GRAYSCALE)
    listTemplate.append((filename.split('.')[0], template_img))

# Loop through the music sheets
for sheetFilename in os.listdir(sheetDirectory):
    f = os.path.join(sheetDirectory, sheetFilename)
    musicSheet = cv.imread(f, cv.IMREAD_GRAYSCALE)

    # Split the music sheet into lines
    lines_of_music = split_image(musicSheet)

    # Loop through each line of music
    for idx, line in enumerate(lines_of_music):
        # Get matches using matchTemplates
        hits = matchTemplates(listTemplate,
                              line,
                              score_threshold=0.93,
                              searchBox=(0, 0, 3000, 750),
                              method=cv.TM_CCOEFF_NORMED,
                              maxOverlap=0.2)
        
        pprint(hits)

        # Process each hit
        for hit in hits:
            # Assuming each hit is a tuple in the form (template_name, score, top_left, bottom_right)
            template_name, score, top_left, bottom_right = hit

            # Construct match dictionary
            match_dict = {
                'ROI': (top_left, bottom_right),
                'max_val': score,
                'templateFilename': template_name,
                'sheetFilename': sheetFilename,
                'lineIndex': idx
            }
            # Add to all_matches
            all_matches.append(match_dict)

# Grouping matches
groups = []
for match in all_matches:
    placed = False
    for group in groups:
        if any(are_nearby(match['ROI'], existing_match['ROI'], tolerance=20) for existing_match in group):
            group.append(match)
            placed = True
            break
    if not placed:
        groups.append([match])

# Processing groups to select the best match
best_matches = []
for group in groups:
    max_match = max(group, key=lambda x: x['max_val'])
    best_matches.append(max_match)

# Sorting best matches
sorted_best_matches = sorted(best_matches, key=lambda match: (match['lineIndex'], match['ROI'][0][0]))

# Extracting hex values and printing template filenames
hex_array = []
for match in sorted_best_matches:
    hex_value = get_hex_value(match['templateFilename'])
    if hex_value is not None:
        hex_string = hex(hex_value) if isinstance(hex_value, int) else hex_value
        hex_array.append(hex_string)
        print(f"{match['templateFilename']} - {hex_string}")
    else:
        print(f"{match['templateFilename']} - No hex value found")
