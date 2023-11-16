import cv2 as cv
import os
import numpy as np
from pprint import pprint
import pandas as pd
from MTM import matchTemplates
from sklearn.cluster import KMeans

# Set the LOKY_MAX_CPU_COUNT environment variable
# Replace '4' with the number of cores you wish to use
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


# Mapping of template filenames to hexadecimal values
timeSignatureDict = {
    "24_timesignature": 0x24,
    "34_timesignature": 0x34,
    "44_timesignature": 0x44,
}
tempoDict = {
    "40_tempo": 0x28,
    "50_tempo": 0x32,
    "60_tempo": 0x3C,
    "70_tempo": 0x46,
    "80_tempo": 0x50,
    "90_tempo": 0x5A,
    "100_tempo": 0x64,
    "110_tempo": 0x6E,
    "120_tempo": 0x78,
    "130_tempo": 0x82,
    "140_tempo": 0x8C,
    "150_tempo": 0x96,
    "160_tempo": 0xA0,
    "170_tempo": 0xAA,
    "180_tempo": 0xB4,
    "190_tempo": 0xBE,
    "200_tempo": 0xC8,
}
keySignatureDict = {
    "cmajor": 0x00,
    "flat_f": 0x01,
    "flat_bflt": 0x02,
    "flat_eflt": 0x03,
    "flat_aflt": 0x04,
    "flat_dflt": 0x05,
    "flat_gflt": 0x06,
    "flat_cflt": 0x07,
    "sharp_g": 0x08,
    "sharp_d": 0x09,
    "sharp_a": 0x0A,
    "sharp_e": 0x0B,
    "sharp_b": 0x0C,
    "sharp_fshrp": 0x0D,
    "sharp_cshrp": 0x0E,
}
dynamicsDict = {
    "fortissimo": 0x00,
    "forte": 0x01,
    "piano": 0x02,
    "pianissimo": 0x03,
}
notesAndRestsDict = {
    "quarternote_lowc": 0x00,
    "quarternote_lowd": 0x10,
    "quarternote_lowe": 0x20,
    "quarternote_lowf": 0x30,
    "quarternote_lowg": 0x40,
    "quarternote_lowa": 0x50,
    "quarternote_lowb": 0x60,
    "quarternote_middlec": 0x70,
    "quarternote_highd": 0x80,
    "quarternote_highe": 0x90,
    "quarternote_highf": 0xA0,
    "quarternote_highg": 0xB0,
    "quarternote_higha": 0xC0,
    "quarternote_highb": 0xD0,
    "quarternote_highc": 0xE0,
    "halfnote_lowc": 0x01,
    "halfnote_lowd": 0x11,
    "halfnote_lowe": 0x21,
    "halfnote_lowf": 0x31,
    "halfnote_lowg": 0x41,
    "halfnote_lowa": 0x51,
    "halfnote_lowb": 0x61,
    "halfnote_middlec": 0x71,
    "halfnote_highd": 0x81,
    "halfnote_highe": 0x91,
    "halfnote_highf": 0xA1,
    "halfnote_highg": 0xB1,
    "halfnote_higha": 0xC1,
    "halfnote_highb": 0xD1,
    "halfnote_highc": 0xE1,
    "wholenote_lowc": 0x02,
    "wholenote_lowd": 0x12,
    "wholenote_lowe": 0x22,
    "wholenote_lowf": 0x32,
    "wholenote_lowg": 0x42,
    "wholenote_lowa": 0x52,
    "wholenote_lowb": 0x62,
    "wholenote_middlec": 0x72,
    "wholenote_highd": 0x82,
    "wholenote_highe": 0x92,
    "wholenote_highf": 0xA2,
    "wholenote_highg": 0xB2,
    "wholenote_higha": 0xC2,
    "wholenote_highb": 0xD2,
    "wholenote_highc": 0xE2,
    "quarterrest": 0xF5,
    "halfrest": 0xF6,
    "wholerest": 0xF7,
    "eighthrest": 0xF8,
    "sixteenthrest": 0xF9,
    "trebleclef": 0xFF,
}
default_values = {
    'keySignature': 'cmajor',
    'tempo': '120_tempo',
    'timeSignature': '44_timesignature'
}

def get_hex_value(template_name):
    # Returns the hex value corresponding to the template name
    return combined_dict.get(template_name, None)

def get_beats_per_measure(time_signature_hex):
    if time_signature_hex == 0x24:  # 2/4 Time
        return 2
    elif time_signature_hex == 0x34:  # 3/4 Time
        return 3
    elif time_signature_hex == 0x44:  # 4/4 Time
        return 4
    else:
        return 4  # default time signature is 4/4 Time
    
def get_beat_value(template_name, current_time_signature):
    if 'quarternote' in template_name or 'quarterrest' in template_name:
        return 1  # Quarter notes/rests are 1 beat
    elif 'halfnote' in template_name or 'halfrest' in template_name:
        return 2  # Half notes/rests are 2 beats
    elif 'wholenote' in template_name:
        return 4  # Whole notes/rests are 4 beats
    elif 'wholerest' in template_name:
        # Whole rests represent a full measure, regardless of the time signature
        return current_time_signature
    elif 'eighthrest' in template_name:
        return 1/8  # Eighth rests are 1/8 beat
    elif 'sixteenthrest' in template_name:
        return 1/16  # Sixteenth rests are 1/16 beat
    else:
        return 0  # No beat value or not a note/rest

def count_treble_clefs(hits):
    # Counting the number of treble clefs based on the 'TemplateName' column
    return len(hits[hits['TemplateName'] == 'trebleclef'])

def cluster_and_sort_hits(hits, cluster_range=180):  # Increased range
    # Count treble clefs to determine the number of clusters
    n_clusters = count_treble_clefs(hits)

    # Handle the case when no treble clefs are detected
    if n_clusters == 0:
        print("No treble clefs detected. Clustering cannot be performed.")
        return hits

    # Extract x and y coordinates for clustering and sorting
    hits['x'] = hits['BBox'].apply(lambda bbox: bbox[0])
    hits['y'] = hits['BBox'].apply(lambda bbox: bbox[1])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init = 15, random_state=0).fit(hits[['y']])
    hits['Cluster'] = kmeans.labels_

    # Sort clusters by their mean y-coordinate to maintain top-down order
    cluster_order = hits.groupby('Cluster')['y'].mean().sort_values().index

    # Sort hits within each cluster primarily by x-coordinate
    sorted_hits = pd.DataFrame()
    for cluster_id in cluster_order:
        cluster = hits[hits['Cluster'] == cluster_id]

        # Warning if a cluster exceeds the y-coordinate range
        if cluster['y'].max() - cluster['y'].min() > cluster_range:
            print(f"Warning: Cluster {cluster_id} exceeds y-coordinate range of {cluster_range} pixels.")

        cluster = cluster.sort_values(by=['x', 'y'])
        sorted_hits = pd.concat([sorted_hits, cluster])

    # Drop the added columns if not needed in the final output
    sorted_hits = sorted_hits.drop(columns=['x', 'y'])

    return sorted_hits


combined_dict = {**timeSignatureDict, **tempoDict, **keySignatureDict, 
                 **dynamicsDict, **notesAndRestsDict}

templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results"

listTemplate = []
# USE CLUSTERING ALGORITHM FOR SPLITTING OF LINES
for filename in os.listdir(templateDirectory):
    template_img = cv.imread(os.path.join(templateDirectory, filename))
    template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    listTemplate.append((filename.split('.')[0], template_img))
sheet = "sheets/Sample Sheet 4.png"
sheet_img = cv.imread(sheet)
sheet_img = cv.cvtColor(sheet_img, cv.COLOR_BGR2GRAY)

hits = matchTemplates(listTemplate,
                      sheet_img,
                      score_threshold=0.93,
                      searchBox=(0, 0, 3000, 750),
                      method=cv.TM_CCOEFF_NORMED,
                      maxOverlap=0.3)
# Process the hits
sorted_hits = cluster_and_sort_hits(hits)

# Get mean y-coordinate for each cluster and sort them
cluster_order = hits.groupby('Cluster')['y'].mean().sort_values().index

# Mapping from old cluster IDs to new sequential IDs
cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}

# Apply mapping
sorted_hits['Cluster'] = sorted_hits['Cluster'].map(cluster_mapping)

# Reset the indexing for adding in the default values
sorted_hits.reset_index(drop=True, inplace=True)

# Get each cluster
clusters = sorted_hits['Cluster'].unique()

for cluster in clusters:

    # Filter the DataFrame for the current cluster
    cluster_data = sorted_hits[sorted_hits['Cluster'] == cluster]

    # Flag to track if any template name belongs to keySignatureDict
    found_in_dict = False

    # Store the index of the 'trebleclef' row
    trebleclef_index = None

    for index, row in cluster_data.iterrows():

        template_name = row['TemplateName']
        
        # Check if the template name matches any key in keySignatureDict
        if any(key in template_name for key in keySignatureDict):
            found_in_dict = True
        elif template_name == "trebleclef":
            trebleclef_index = index

    # If no template names from keySignatureDict are found and 'trebleclef' exists
    if not found_in_dict and trebleclef_index is not None:

        # Define your new row here with the 'cmajor' template name
        new_row = {'TemplateName': 'cmajor', 'BBox': 'default', 'Score': '1.000000', 'Cluster': cluster, 'HexValue': keySignatureDict.get('cmajor')}
        
        # Insert the new row under the 'trebleclef' row in the original DataFrame
        sorted_hits = pd.concat([sorted_hits.iloc[:trebleclef_index + 1], pd.DataFrame([new_row]), sorted_hits.iloc[trebleclef_index + 1:]]).reset_index(drop=True)

# Convert the 'TemplateName' column to the 'HexValue' column DataFrame
sorted_hits['HexValue'] = sorted_hits['TemplateName'].apply(get_hex_value)



# Convert hex values to integers and store in a NumPy array
int_values = sorted_hits['HexValue'].dropna().values
int_array = np.array(int_values, dtype=int)

print(sorted_hits)
print("The number of matches found in the music sheet:", len(sorted_hits))
print("Corresponding hex values in decimal form: ", int_array)

measures_per_line = []
current_measures = 0
current_beats = 0
beats_per_measure = 4  # Default to 4/4 time

for index, row in sorted_hits.iterrows():
    if row['TemplateName'] in timeSignatureDict:
        beats_per_measure = get_beats_per_measure(row['HexValue'])

    elif row['TemplateName'] == 'trebleclef':
        if current_measures > 0 or current_beats > 0:
            measures_per_line.append(current_measures + int(current_beats > 0))
            current_measures = 0
        current_beats = 0  # Reset for the new line

    elif row['TemplateName'] in notesAndRestsDict:
        beat_value = get_beat_value(row['TemplateName'], beats_per_measure)
        current_beats += beat_value
        while current_beats >= beats_per_measure:
            current_measures += 1
            current_beats -= beats_per_measure

# Handle the last line if it doesn't end with a treble clef
if current_measures > 0 or current_beats > 0:
    measures_per_line.append(current_measures + int(current_beats > 0))

print("Measures per line: ", measures_per_line)

# Make a copy of sorted_hits for modifications
modified_hits = sorted_hits.copy()

# Group by 'Cluster' to process each cluster separately
clusters = modified_hits.groupby('Cluster')

# Container for storing modified data from each cluster
modified_output = []

# for i, (cluster_id, cluster) in enumerate(clusters):
#     # Remove extra treble clefs except for the first cluster
#     if i >= 0:
#         cluster = cluster[cluster['TemplateName'] != 'trebleclef']

#     modified_output.append(cluster)

# # Combine all clusters back into a single DataFrame for the modified output
# modified_hits = pd.concat(modified_output)

# # Map TemplateName to HexValue
# modified_hits['HexValue'] = modified_hits['TemplateName'].apply(get_hex_value)

# # Convert hex values to integers and store in a NumPy array
# int_values2 = modified_hits['HexValue'].dropna().values
# int_array2 = np.array(int_values2, dtype=int)

# # Extract y-coordinates
# hits['y'] = hits['BBox'].apply(lambda bbox: bbox[1])

# # Plotting the y-coordinates
# plt.scatter(hits['y'], [0] * len(hits), alpha=0.5)
# plt.title("Distribution of Y-Coordinates")
# plt.xlabel("Y-Coordinate")
# plt.ylabel("Frequency (Dummy)")
# plt.show()
