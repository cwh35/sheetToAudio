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
    "trebleclef": 0xFF,
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
}

def get_hex_value(template_name):
    # Returns the hex value corresponding to the template name
    return combined_dict.get(template_name, None)

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
sheet = "sheets/Sample Sheet 14.png"
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

# Map TemplateName to HexValue
sorted_hits['HexValue'] = sorted_hits['TemplateName'].apply(get_hex_value)

# Convert hex values to integers and store in a NumPy array
int_values = sorted_hits['HexValue'].dropna().values
int_array = np.array(int_values, dtype=int)

print(sorted_hits)
print("The number of matches found in the music sheet:", len(sorted_hits))
print("Corresponding hex values in decimal form: ", int_array)

# Make a copy of sorted_hits for modifications
modified_hits = sorted_hits.copy()

# Group by 'Cluster' to process each cluster separately
clusters = modified_hits.groupby('Cluster')

# Container for storing modified data from each cluster
modified_output = []

for i, (cluster_id, cluster) in enumerate(clusters):
    # Remove extra treble clefs except for the first cluster
    if i >= 0:
        cluster = cluster[cluster['TemplateName'] != 'trebleclef']

    modified_output.append(cluster)

# Combine all clusters back into a single DataFrame for the modified output
modified_hits = pd.concat(modified_output)

# Map TemplateName to HexValue
modified_hits['HexValue'] = modified_hits['TemplateName'].apply(get_hex_value)

# Convert hex values to integers and store in a NumPy array
int_values2 = modified_hits['HexValue'].dropna().values
int_array2 = np.array(int_values2, dtype=int)

#print(modified_hits)
#print(int_array2)

# # Extract y-coordinates
# hits['y'] = hits['BBox'].apply(lambda bbox: bbox[1])

# # Plotting the y-coordinates
# plt.scatter(hits['y'], [0] * len(hits), alpha=0.5)
# plt.title("Distribution of Y-Coordinates")
# plt.xlabel("Y-Coordinate")
# plt.ylabel("Frequency (Dummy)")
# plt.show()


# #specify path for export
# path = r'outputs/output_data.txt'

# #export DataFrame to text file
# with open(path, 'a') as f:
#     df_string = hits.to_string(header=True, index=True)
#     f.write(df_string)

# Group files by sheet number
# sheet_groups = defaultdict(list)
# for filename in os.listdir(sheetDirectory):
#     match = re.match(r"Sample Sheet (\d+)(-\d+)?", filename)
#     if match:
#         sheet_number = match.group(1)
#         sheet_groups[sheet_number].append(os.path.join(sheetDirectory, filename))
# # Sort each group to maintain order (e.g., -1, -2, etc.)
# for sheet in sheet_groups:
#     sheet_groups[sheet].sort()
# for sheet_number, files in sheet_groups.items():
#     # Read and concatenate images vertically
#     images = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in files]
#     concatenated_image = cv.vconcat(images) if len(images) > 1 else images[0]

img1 = cv.imread('sheets/Sample Sheet 13-1.png')
img2 = cv.imread('sheets/Sample Sheet 13-2.png')

img = cv.vconcat([img1, img2])
sheet_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)