import cv2 as cv
import os
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

def split_image(image, shift_pixels=0):
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Check if the height is divisible by 160 (height of one line of music)
    if height % 160 != 0:
        raise ValueError("The height of the image is not divisible by 160 pixels.")

    # Calculate the number of lines in the image
    num_lines = (height - shift_pixels) // 160

    # Initialize an empty list to store the line images
    lines_of_music = []

    # Split the image into individual lines of music
    for i in range(num_lines):
        # Calculate the starting and ending y-coordinates of the current line
        start_y = i * 160 + shift_pixels
        end_y = start_y + 160

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

# Loop through the music sheets
for sheetFilename in os.listdir(sheetDirectory):
    f = os.path.join(sheetDirectory, sheetFilename)
    musicSheet = cv.imread(f ,cv.IMREAD_GRAYSCALE)  # trainImage

    # Split the music sheet into lines
    lines_of_music = split_image(musicSheet)

    # Loop through each line of music
    for idx, line in enumerate(lines_of_music):
        # Loop through the templates
        for templateFilename in os.listdir(templateDirectory):
            file = os.path.join(templateDirectory, templateFilename)
            template = cv.imread(file ,cv.IMREAD_GRAYSCALE)  # queryImage

            # Cross-correlation between templates and music sheet
            res = cv.matchTemplate(line, template, cv.TM_CCOEFF_NORMED)
            # Get the min and max correlation value as well as locations of the matched points
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            # threshold to filter valid matches
            threshold = 0.9
            if max_val > threshold:
                h, w = template.shape
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)

                all_matches.append({'ROI': (top_left, bottom_right),
                                    'max_val': max_val,
                                    'templateFilename': templateFilename,
                                    'sheetFilename': sheetFilename,
                                    'lineIndex': idx})
groups = []
for match in all_matches:
    placed = False
    for group in groups:
        for existing_match in group:
            if are_nearby(match['ROI'], existing_match['ROI'], tolerance=20):
                group.append(match)
                placed = True
                break
        if placed:
            break
    if not placed:
        groups.append([match])

for group in groups:
    max_match = max(group, key=lambda x: x['max_val'])

    # Extracting information from max_match
    top_left, bottom_right = max_match['ROI']
    max_val = max_match['max_val']
    templateFilename = max_match['templateFilename']
    sheetFilename = max_match['sheetFilename']
    
    # Load the template image
    template = cv.imread(os.path.join(templateDirectory, templateFilename), cv.IMREAD_GRAYSCALE)
    
    # Get region of interest from musicSheet
    roi = musicSheet[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # Make sure the images have the same height before concatenating
    h = max(template.shape[0], roi.shape[0])
    template = cv.copyMakeBorder(template, 0, h - template.shape[0], 0, 0, cv.BORDER_CONSTANT, value=[0,0,0])
    roi = cv.copyMakeBorder(roi, 0, h - roi.shape[0], 0, 0, cv.BORDER_CONSTANT, value=[0,0,0])
    
    # Concatenate the template and roi images horizontally
    concatenated_image = cv.hconcat([template, roi])
    
    # Vertical line to separate template and matched object
    line_position = template.shape[1]
    cv.line(concatenated_image, (line_position, 0), (line_position, concatenated_image.shape[0]), (0, 0, 0), 4)
    
    # Save image
    outputFilename = f"{templateFilename}_{sheetFilename}_match.png"
    outputPath = os.path.join(outputDirectory, outputFilename)
    cv.imwrite(outputPath, concatenated_image)

best_matches = []

for group in groups:
    max_match = max(group, key=lambda x: x['max_val'])
    best_matches.append(max_match)

# Now sort these best matches by line number and from left to right
sorted_best_matches = sorted(best_matches, key=lambda match: (match['lineIndex'], match['ROI'][0][0]))

# Extract hex values and print template filenames for sorted best matches
hex_array = [get_hex_value(match['templateFilename']) for match in sorted_best_matches if get_hex_value(match['templateFilename']) is not None]

# Extract hex values for sorted best matches
hex_array = []
for match in sorted_best_matches:
    # Get the hex value for the current template filename
    hex_value = get_hex_value(match['templateFilename'])
    if hex_value is not None:
        # Ensure hex_value is a string in hexadecimal format
        hex_string = hex(hex_value) if isinstance(hex_value, int) else hex_value
        hex_array.append(hex_string)
        # Print the template filename and its corresponding hex value
        print(f"{match['templateFilename']} - {hex_string}")
    else:
        # Handle the case where there is no corresponding hex value
        print(f"{match['templateFilename']} - No hex value found")