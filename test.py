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

def remove_signatures_at_line_end(hits):
    indices_to_drop = []

    # Ensure clusters are sorted
    clusters = sorted(hits['Cluster'].unique())

    for i in range(len(clusters) - 1):
        # Get the data for the current cluster
        current_cluster_data = hits[hits['Cluster'] == clusters[i]]

        # Get the first entry of the next cluster
        next_cluster_first_entry = hits[hits['Cluster'] == clusters[i + 1]].iloc[0]

        # Check the last two entries of the current cluster
        if len(current_cluster_data) >= 2:
            # Get the last two entries
            last_entries = current_cluster_data.iloc[-2:]

            # Check if any of these last entries are key or time signatures
            for _, last_entry in last_entries.iterrows():
                if last_entry['TemplateName'] in keySignatureDict or \
                   last_entry['TemplateName'] in timeSignatureDict:
                    # Check if the next cluster starts with a treble clef
                    if next_cluster_first_entry['TemplateName'] == 'trebleclef':
                        # Mark the index of the last entry for removal
                        indices_to_drop.append(last_entry.name)

    # Drop the identified indices from the hits DataFrame
    hits_dropped = hits.drop(indices_to_drop).reset_index(drop=True)
    return hits_dropped


def cluster_and_sort_hits(hits, cluster_range=180):
    # Count treble clefs to determine the number of clusters
    n_clusters = count_treble_clefs(hits)

    # If no treble clefs are detected
    if n_clusters == 0:
        print("No treble clefs detected. Clustering cannot be performed.")
        return hits

    # Extract x and y coordinates for clustering and sorting
    hits['x'] = hits['BBox'].apply(lambda bbox: bbox[0] + bbox[2] // 2)
    hits['y'] = hits['BBox'].apply(lambda bbox: bbox[1] + bbox[3] // 2)

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

        cluster = cluster.sort_values(by='x')

        # This is for updating the x and y part of the BBox Column
        # with the new "centered" coordinates
        for index, row in cluster.iterrows():
            # Extract the center coordinates
            x_center, y_center = row['x'], row['y']

            # Extract the original width and height from the BBox
            _, _, width, height = row['BBox']

            # Update the BBox value in the DataFrame
            cluster.at[index, 'BBox'] = (x_center, y_center, width, height)

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
                      searchBox=(0, 0, 3000, 1500),
                      method=cv.TM_CCOEFF_NORMED,
                      maxOverlap=0.3)
# Process the hits
sorted_hits = cluster_and_sort_hits(hits)

print("Number of initial matches before post-processing:", len(sorted_hits))

# Get each cluster (the numbers of them --> ex: 3 lines should have clusters 0, 1, 2)
clusters = sorted(sorted_hits['Cluster'].unique())

# Get mean y-coordinate for each cluster and sort them
cluster_order = hits.groupby('Cluster')['y'].mean().sort_values().index

# Mapping from old cluster IDs to new sequential IDs
cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}

# Apply mapping
sorted_hits['Cluster'] = sorted_hits['Cluster'].map(cluster_mapping)
# Process the hits to remove signatures at the end of a line
sorted_hits = remove_signatures_at_line_end(sorted_hits)

# Variables to store the first cluster's tempo and time signature
first_cluster_tempo = None
first_cluster_time_signature = None

for i, cluster in enumerate(clusters):
    # Filter the DataFrame for the current cluster
    cluster_data = sorted_hits[sorted_hits['Cluster'] == cluster]

    # Flag to track if template name belongs to keySignatureDict
    found_in_keySigDict = False

    # Store the index of the 'trebleclef' row
    trebleclef_index = None

    # Iterate through each cluster
    for index, row in cluster_data.iterrows():
        # Get the current template name
        template_name = row['TemplateName']
        
        # Check if the template name matches any key in keySignatureDict
        if any(key in template_name for key in keySignatureDict):
            found_in_keySigDict = True
        elif template_name == "trebleclef":
            # Set the index of the treble clef
            trebleclef_index = index

    # Insert key signature if not found
    if not found_in_keySigDict and trebleclef_index is not None:
        new_row = {'TemplateName': 'cmajor', 
                    'BBox': 'Default Key Signature', 
                    'Score': '1.000000', 
                    'Cluster': cluster, 
                    'HexValue': keySignatureDict.get('cmajor')}
        sorted_hits = pd.concat([sorted_hits.iloc[:trebleclef_index + 1], pd.DataFrame([new_row]), sorted_hits.iloc[trebleclef_index + 1:]]).reset_index(drop=True)
        keySig_index = trebleclef_index + 1
    else:
        keySig_index = trebleclef_index

    # Process Tempo
    tempo_row = cluster_data[cluster_data['TemplateName'].str.contains("_tempo")]
    # If it's the first cluster
    if i == 0:
        if not tempo_row.empty:
            first_cluster_tempo = tempo_row.iloc[0]['TemplateName']
        else:
            # Set to default tempo if not found and insert a row for it
            first_cluster_tempo = '120_tempo'
            new_tempo_row = {'TemplateName': first_cluster_tempo, 
                            'BBox': 'Default Tempo', 
                            'Score': '1.000000', 
                            'Cluster': cluster, 
                            'HexValue': tempoDict.get(first_cluster_tempo.split('_')[0])}
            sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 3], pd.DataFrame([new_tempo_row]), sorted_hits.iloc[keySig_index + 2:]]).reset_index(drop=True)
    # For subsequent clusters
    elif tempo_row.empty:
        # Insert a row with the carried-over tempo
        new_tempo_row = {'TemplateName': first_cluster_tempo, 
                        'BBox': 'Carried-over Tempo', 
                        'Score': '1.000000', 
                        'Cluster': cluster, 
                        'HexValue': tempoDict.get(first_cluster_tempo.split('_')[0])}
        sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 3], pd.DataFrame([new_tempo_row]), sorted_hits.iloc[keySig_index + 2:]]).reset_index(drop=True)

    # Insert or carry forward time signature
    timeSig_row = cluster_data[cluster_data['TemplateName'].str.contains("_timesignature")]
    if timeSig_row.empty and (i == 0 or first_cluster_time_signature is None):
        first_cluster_time_signature = '44_timesignature'
        new_timeSig_row = {'TemplateName': first_cluster_time_signature, 
                        'BBox': 'Default Time Signature', 
                        'Score': '1.000000', 
                        'Cluster': cluster, 
                        'HexValue': timeSignatureDict.get('44_timesignature')}
        sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 2], pd.DataFrame([new_timeSig_row]), sorted_hits.iloc[keySig_index + 3:]]).reset_index(drop=True)
    elif timeSig_row.empty:
        new_timeSig_row = {'TemplateName': first_cluster_time_signature, 
                        'BBox': 'Original Time Signature', 
                        'Score': '1.000000', 
                        'Cluster': cluster, 
                        'HexValue': timeSignatureDict.get('44_timesignature')}
        sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 2], pd.DataFrame([new_timeSig_row]), sorted_hits.iloc[keySig_index + 3:]]).reset_index(drop=True)
    elif i == 0:
        first_cluster_time_signature = timeSig_row.iloc[0]['TemplateName']


# Convert the 'TemplateName' column to the 'HexValue' column DataFrame
sorted_hits['HexValue'] = sorted_hits['TemplateName'].apply(get_hex_value)

# Convert hex values to integers and store in a NumPy array
int_values = sorted_hits['HexValue'].dropna().values
int_array = np.array(int_values, dtype=int)

print(sorted_hits)
print("The number of matches post-processing:", len(sorted_hits))
print("Corresponding hex values in decimal form: ", int_array)

def draw_matches_on_sheet(sheet_img, hits):
    for index, row in hits.iterrows():
        bbox = row['BBox']
        # Extract the top-left corner's x and y coordinates
        x, y = bbox[0], bbox[1]

        # Draw a small circle at the top-left corner of each match
        cv.circle(sheet_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # Save the image with the matches highlighted
    cv.imwrite(os.path.join(outputDirectory, 'Annotated_Sheet.png'), sheet_img)

draw_matches_on_sheet(sheet_img, hits)

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
