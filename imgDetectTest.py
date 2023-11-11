import cv2 as cv
import os
import numpy as np
from pprint import pprint
import pandas as pd
from MTM import matchTemplates
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the LOKY_MAX_CPU_COUNT environment variable
# Replace '4' with the number of cores you wish to use
os.environ['LOKY_MAX_CPU_COUNT'] = '4'



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

templateDirectory = "templates"
sheetDirectory = "sheets"
outputDirectory = "results"

listTemplate = []
# USE CLUSTERING ALGORITHM FOR SPLITTING OF LINES
for filename in os.listdir(templateDirectory):
    template_img = cv.imread(os.path.join(templateDirectory, filename))
    template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    listTemplate.append((filename.split('.')[0], template_img))
sheet = "sheets/Test Sheet-1.png"
sheet_img = cv.imread(sheet)
sheet_img = cv.cvtColor(sheet_img, cv.COLOR_BGR2GRAY)

hits = matchTemplates(listTemplate,
                      sheet_img,
                      score_threshold=0.94,
                      searchBox=(0, 0, 3000, 750),
                      method=cv.TM_CCOEFF_NORMED,
                      maxOverlap=0.2)
# Process the hits
sorted_hits = cluster_and_sort_hits(hits)

# Print the sorted hits
print(sorted_hits)
print(len(hits))

# Extract y-coordinates
hits['y'] = hits['BBox'].apply(lambda bbox: bbox[1])

# Plotting the y-coordinates
plt.scatter(hits['y'], [0] * len(hits), alpha=0.5)
plt.title("Distribution of Y-Coordinates")
plt.xlabel("Y-Coordinate")
plt.ylabel("Frequency (Dummy)")
plt.show()