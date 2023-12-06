#!/usr/bin/env python
# Music Notation to Audio - Sheet Music Trainer
# Jack Carnovale, Cameron Henning, Chloe Hale
#
# The following program using computer vision to analyze PNG images
# that are loaded on a flashdrive, that are images of sheet music from 
# NoteFlight. Using template matching, the notes and properties of the 
# piece are determined and put into a uint16 array
#
# This array is then sent to a function to allow for audio and visual 
# playback. When a notes are played, two LEDs blink to follow the note 
# position, an LCD screen shows the note that is played, and a serial
# command is sent to an arduino controller to play the audio
#
# This structure is wrapped in a looping menu
#



# Library Imports
import time
import board
import busio
import serial
import numpy as np
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from rpi_lcd import LCD
import neopixel 
import cv2 as cv
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from MTM import matchTemplates


#set up buttons
playBtn=25
processBtn=24
GPIO.setmode(GPIO.BCM)
GPIO.setup(processBtn,GPIO.IN)
GPIO.setup(playBtn,GPIO.IN)

#create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

#create ADC object
ads= ADS.ADS1115(i2c)

#create single ended input for tempo and instrument and song selection
selector=AnalogIn(ads,ADS.P0)

horiz=30
vert=40

#create LCD screen
lcd=LCD()

strip=neopixel.NeoPixel(board.D12,horiz+2,brightness=1.0/20.0)
color=(255,192,0)
coff=(0,0,0)

#configure serial comms
srl = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 38400,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)
    
def OpenMenu():     
	#LCD Opening function
	
	#Song selection
	process = 0
	sel_max = 26800
	lcd.text("Scanning USB",1)
	
	#get usb directory
	basepath='/media/pi'
	foundDrive=0
	while (foundDrive == 0):
		mediaDevs=os.listdir(basepath)
		foundDrive = len(mediaDevs)
	basepath = basepath + '/' + mediaDevs[0]
	
	#get pngs
	lcd.clear()
	lcd.text("Select Song",1)
	
	#get pngs
	allSongs=os.listdir(basepath)
	allSongs=[f for f in allSongs if f.endswith(".png")]
	
	
	#read in song names
	if len(allSongs)>0:
		
		while process == 0:
			options=len(allSongs)+1.0
			sel=int(selector.value/sel_max*options)        
			
			if sel < 0:
				sel=0
			
			if sel == 0:
				lcd.text("Go To Menu",2)
			else:
				lcd.text(allSongs[sel-1],2)
            
			process = GPIO.input(processBtn)
        
			if process == 1:
				break
		if sel == 0:
			return 0,0,0,0
		else:
			song_path=basepath+ '/'+allSongs[sel-1]
	else:
		return 0,0,0,0
	
	#Instrument selection
	process=0
    
	lcd.clear()
	lcd.text("Select Instrument",1)
	time.sleep(1)
	
	"""
	Set instrument ranges with the voltage values
	Determine range with the number of instruments we're incoorperating
	"""
	options=9.0
	while process == 0:  
		sel=int(selector.value/sel_max*options)
		if sel < 0:
				sel=0
				
		if sel == 2:
			lcd.text("Alto Sax",2)
    
		elif sel == 3:
			lcd.text('Clarinet',2)
			
		elif sel == 4:
			lcd.text("Piano",2)
            
		elif sel == 5:
			lcd.text("Tenor Sax",2)
			
		elif sel == 6:
			lcd.text("Trumpet",2)
			
		elif sel == 7:
			lcd.text("Violin",2)
			
		elif sel == 8:
			lcd.text("Voice",2)
			
		elif sel==1:
			lcd.text("Tones",2)
		else:
			lcd.text("Go To Menu",2)
		
		
		process = GPIO.input(processBtn)
		
		if process == 1:
			break
	if sel==0:
		return 0,0,0,0
	
	instr=sel-1
	
	#Playback selection
	process=0
	
	lcd.clear()
	lcd.text("Select Playback",1)
	time.sleep(1)
	options=5.0
	while process == 0:  
		sel=int(selector.value/sel_max*options)
		
		if sel < 0:
				sel=0
		
		if sel == 2:
			lcd.text("3/4 speed",2)
			tempo_factor=3.0/4.0
		
		elif sel == 3:
			lcd.text("1/2 speed",2)
			tempo_factor=2.0/4.0
		
		elif sel == 4:
			lcd.text("1/4 speed",2)
			tempo_factor=1.0/4.0
			
		elif sel == 1:
			lcd.text("Original Tempo",2)
			tempo_factor=1.0
		else:
			lcd.text("Go To Menu",2)
		
		process = GPIO.input(24)
		
		if process == 1:
			break
	
	if sel==0:
		return 0,0,0,0
	
	return song_path,instr,tempo_factor,1

#Computer Vision
def processSong(song_name):
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
		"fortissimo": 0x100,
		"forte": 0x200,
		"piano": 0x300,
		"pianissimo": 0x400,
	}
	notesAndRestsDict = {
		"sixteenthnote_lowc": 0x00,
		"sixteenthnote_lowd": 0x10,
		"sixteenthnote_lowe": 0x20,
		"sixteenthnote_lowf": 0x30,
		"sixteenthnote_lowg": 0x40,
		"sixteenthnote_lowa": 0x50,
		"sixteenthnote_lowb": 0x60,
		"sixteenthnote_middlec": 0x70,
		"sixteenthnote_highd": 0x80,
		"sixteenthnote_highe": 0x90,
		"sixteenthnote_highf": 0xA0,
		"sixteenthnote_highg": 0xB0,
		"sixteenthnote_higha": 0xC0,
		"sixteenthnote_highb": 0xD0,
		"sixteenthnote_highc": 0xE0,
		"eighthnote_lowc": 0x01,
		"eighthnote_lowd": 0x11,
		"eighthnote_lowe": 0x21,
		"eighthnote_lowf": 0x31,
		"eighthnote_lowg": 0x41,
		"eighthnote_lowa": 0x51,
		"eighthnote_lowb": 0x61,
		"eighthnote_middlec": 0x71,
		"eighthnote_highd": 0x81,
		"eighthnote_highe": 0x91,
		"eighthnote_highf": 0xA1,
		"eighthnote_highg": 0xB1,
		"eighthnote_higha": 0xC1,
		"eighthnote_highb": 0xD1,
		"eighthnote_highc": 0xE1,
		"quarternote_lowc": 0x02,
		"quarternote_lowd": 0x12,
		"quarternote_lowe": 0x22,
		"quarternote_lowf": 0x32,
		"quarternote_lowg": 0x42,
		"quarternote_lowa": 0x52,
		"quarternote_lowb": 0x62,
		"quarternote_middlec": 0x72,
		"quarternote_highd": 0x82,
		"quarternote_highe": 0x92,
		"quarternote_highf": 0xA2,
		"quarternote_highg": 0xB2,
		"quarternote_higha": 0xC2,
		"quarternote_highb": 0xD2,
		"quarternote_highc": 0xE2,
		"halfnote_lowc": 0x03,
		"halfnote_lowd": 0x13,
		"halfnote_lowe": 0x23,
		"halfnote_lowf": 0x33,
		"halfnote_lowg": 0x43,
		"halfnote_lowa": 0x53,
		"halfnote_lowb": 0x63,
		"halfnote_middlec": 0x73,
		"halfnote_highd": 0x83,
		"halfnote_highe": 0x93,
		"halfnote_highf": 0xA3,
		"halfnote_highg": 0xB3,
		"halfnote_higha": 0xC3,
		"halfnote_highb": 0xD3,
		"halfnote_highc": 0xE3,
		"wholenote_lowc": 0x04,
		"wholenote_lowd": 0x14,
		"wholenote_lowe": 0x24,
		"wholenote_lowf": 0x34,
		"wholenote_lowg": 0x44,
		"wholenote_lowa": 0x54,
		"wholenote_lowb": 0x64,
		"wholenote_middlec": 0x74,
		"wholenote_highd": 0x84,
		"wholenote_highe": 0x94,
		"wholenote_highf": 0xA4,
		"wholenote_highg": 0xB4,
		"wholenote_higha": 0xC4,
		"wholenote_highb": 0xD4,
		"wholenote_highc": 0xE4,
		"sixteenthrest": 0x05,
		"eighthrest": 0x06,
		"quarterrest": 0x07,
		"halfrest": 0x08,
		"wholerest": 0x09,
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
			# Whole notes represent a full measure, regardless of time signature
			return current_time_signature 
		elif 'wholerest' in template_name:
			# Whole rests represent a full measure, regardless of the time signature
			return current_time_signature
		elif 'eighthrest' in template_name or 'eighthnote' in template_name:
			return .5  # Eighth notes and rests are half a beat
		elif 'sixteenthrest' in template_name or 'sixteenthnote' in template_name:
			return .25  # Sixteenth notes and rests are a quarter of a beat
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
		# sorted_hits = sorted_hits.drop(columns=['x', 'y'])
		return sorted_hits


	combined_dict = {**timeSignatureDict, **tempoDict, **keySignatureDict, 
					 **dynamicsDict, **notesAndRestsDict}

	templateDirectory = "/home/pi/Documents/sheetToAudio-main/templates"
	sheetDirectory = "/home/pi/Documents/sheetToAudio-main/sheets"


	listTemplate = []
	# USE CLUSTERING ALGORITHM FOR SPLITTING OF LINES
	for filename in os.listdir(templateDirectory):
		template_img = cv.imread(os.path.join(templateDirectory, filename))
		template_img = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
		listTemplate.append((filename.split('.')[0], template_img))
		
	# Set directory for image to be read in
	sheet = song_name
	sheet_img = cv.imread(sheet)
	sheet_img = cv.cvtColor(sheet_img, cv.COLOR_BGR2GRAY)

	hits = matchTemplates(listTemplate,
						  sheet_img,
						  score_threshold=0.925,
						  searchBox=(0, 0, sheet_img.shape[1], sheet_img.shape[0]),
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

	# Convert the 'TemplateName' column to the 'HexValue' column DataFrame
	sorted_hits['HexValue'] = sorted_hits['TemplateName'].apply(get_hex_value)

	# Variables to store the first cluster's tempo and time signature
	first_cluster_tempo = None
	first_cluster_time_signature = None

	# Variable for the carry over time signature
	carry_over_time_signature = None

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
						'x': '0',
						'y': '0', 
						'Cluster': cluster, 
						'HexValue': keySignatureDict.get('cmajor')}
			sorted_hits = pd.concat([sorted_hits.iloc[:trebleclef_index + 1], pd.DataFrame([new_row]), sorted_hits.iloc[trebleclef_index + 1:]]).reset_index(drop=True)
			keySig_index = trebleclef_index + 1
		else:
			keySig_index = trebleclef_index + 1

		timeSig_row = cluster_data[cluster_data['TemplateName'].str.contains("_timesignature")]

		if i == 0:
			if timeSig_row.empty:
				# Setting the default for the first cluster
				carry_over_time_signature = '44_timesignature'
				new_timeSig_row = {'TemplateName': carry_over_time_signature,
								   'BBox': 'Default Time Signature',
								   'Score': '1.000000',
								   'x': '0',
								   'y': '0',
								   'Cluster': cluster,
								   'HexValue': timeSignatureDict.get(carry_over_time_signature)}
				sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 1], pd.DataFrame([new_timeSig_row]), sorted_hits.iloc[keySig_index + 1:]]).reset_index(drop=True)
			else:
				# Set to identified time signature for the first cluster
				carry_over_time_signature = timeSig_row.iloc[0]['TemplateName']
		else:
			if timeSig_row.empty:
				# Carry over the last identified time signature
				new_timeSig_row = {'TemplateName': carry_over_time_signature,
								   'BBox': 'Carried-Over Time Signature',
								   'Score': '1.000000',
								   'x': '0',
								   'y': '0',
								   'Cluster': cluster,
								   'HexValue': timeSignatureDict.get(carry_over_time_signature)}
				sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 1], pd.DataFrame([new_timeSig_row]), sorted_hits.iloc[keySig_index + 1:]]).reset_index(drop=True)
			else:
				# Update the time signature when a new one is found
				carry_over_time_signature = timeSig_row.iloc[0]['TemplateName']
			
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
								'x': '0',
								'y': '0',  
								'Cluster': cluster, 
								'HexValue': tempoDict.get(first_cluster_tempo)}
				sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 2], pd.DataFrame([new_tempo_row]), sorted_hits.iloc[keySig_index + 2:]]).reset_index(drop=True)
		# For subsequent clusters
		elif tempo_row.empty:
			# Insert a row with the carried-over tempo
			new_tempo_row = {'TemplateName': first_cluster_tempo, 
							'BBox': 'Carried-over Tempo', 
							'Score': '1.000000',
							'x': '0',
							'y': '0',  
							'Cluster': cluster, 
							'HexValue': tempoDict.get(first_cluster_tempo)}
			sorted_hits = pd.concat([sorted_hits.iloc[:keySig_index + 2], pd.DataFrame([new_tempo_row]), sorted_hits.iloc[keySig_index + 2:]]).reset_index(drop=True)

	# Convert hex values to integers and store in a NumPy array
	int_values = sorted_hits['HexValue'].dropna().values
	int_array = np.array(int_values, dtype=int)

	print(sorted_hits)
	print("The number of matches post-processing:", len(sorted_hits))
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
	
	return int_array, measures_per_line

#get frequencies and note names based on key signature
#frequencies are characters that are sent to the Arduino to be played as notes
#noteNames are to be shown on the LCD screen
def handleKeySignature(keySig):   
    #F 
    if keySig==1:
        freqs=[chr(60),chr(62),chr(64),chr(65),chr(67),chr(69),chr(70),chr(72),chr(74),chr(76),chr(77),chr(79),chr(81),chr(82),chr(84),chr(0)]
        noteNames=["C4","D4","E4","F4","G4","A4","Bflat4","C5","D5","E5","F5","G5","A5","Bflat5","C6","REST"]

    #B flat
    elif keySig==2:
        freqs=[chr(60),chr(62),chr(63),chr(65),chr(67),chr(69),chr(70),chr(72),chr(74),chr(75),chr(77),chr(79),chr(81),chr(82),chr(84),chr(0)]
        noteNames=["C4","D4","Eflat4","F4","G4","A4","Bflat4","C5","D5","Eflat5","F5","G5","A5","Bflat5","C6","REST"]

    #E Flat
    elif keySig==3:
        freqs=[chr(60),chr(62),chr(63),chr(65),chr(67),chr(68),chr(70),chr(72),chr(74),chr(75),chr(77),chr(79),chr(80),chr(82),chr(84),chr(0)]
        noteNames=["C4","D4","Eflat4","F4","G4","Aflat4","Bflat4","C5","D5","Eflat5","F5","G5","Aflat5","Bflat5","C6","REST"]

    #A flat
    elif keySig==4:
        freqs=[chr(60),chr(61),chr(63),chr(65),chr(67),chr(68),chr(70),chr(72),chr(73),chr(75),chr(77),chr(79),chr(80),chr(82),chr(84),chr(0)]
        noteNames=["C4","Dflat4","Eflat4","F4","G4","Aflat4","Bflat4","C5","Dflat5","Eflat5","F5","G5","Aflat5","Bflat5","C6","REST"]

    #D flat
    elif keySig==5:
        freqs=[chr(60),chr(61),chr(63),chr(65),chr(66),chr(68),chr(70),chr(72),chr(73),chr(75),chr(77),chr(78),chr(80),chr(82),chr(84),chr(0)]
        noteNames=["C4","Dflat4","Eflat4","F4","Gflat4","Aflat4","Bflat4","C5","Dflat5","Eflat5","F5","Gflat5","Aflat5","Bflat5","C6","REST"]

    #G flat
    elif keySig==6:
        freqs=[chr(59),chr(61),chr(63),chr(65),chr(66),chr(68),chr(70),chr(71),chr(73),chr(75),chr(77),chr(78),chr(80),chr(82),chr(83),chr(0)]
        noteNames=["Cflat4","Dflat4","Eflat4","F4","Gflat4","Aflat4","Bflat4","Cflat5","Dflat5","Eflat5","F5","Gflat5","Aflat5","Bflat5","Cflat6","REST"]

    #C flat
    elif keySig==7:
        freqs=[chr(59),chr(61),chr(63),chr(64),chr(66),chr(68),chr(70),chr(71),chr(73),chr(75),chr(76),chr(78),chr(80),chr(82),chr(83),chr(0)]
        noteNames=["Cflat4","Dflat4","Eflat4","Fflat4","Gflat4","Aflat4","Bflat4","Cflat5","Dflat5","Eflat5","Fflat5","Gflat5","Aflat5","Bflat5","Cflat6","REST"]

    #G sharp
    elif keySig==8:
        freqs=[chr(60),chr(62),chr(64),chr(66),chr(67),chr(69),chr(71),chr(72),chr(74),chr(76),chr(78),chr(79),chr(81),chr(83),chr(84),chr(0)]
        noteNames=["C4","D4","E4","Fsharp4","G4","A4","B4","C5","D5","E5","Fsharp5","G5","A5","B5","C6","REST"]

    #D sharp
    elif keySig==9:
        freqs=[chr(61),chr(62),chr(64),chr(66),chr(67),chr(69),chr(71),chr(73),chr(74),chr(76),chr(78),chr(79),chr(81),chr(83),chr(85),chr(0)]
        noteNames=["Csharp4","D4","E4","Fsharp4","G4","A4","B4","Csharp5","D5","E5","Fsharp5","G5","A5","B5","Csharp6","REST"]

    #A sharp
    elif keySig==10:
        freqs=[chr(61),chr(62),chr(64),chr(66),chr(68),chr(69),chr(71),chr(73),chr(74),chr(76),chr(78),chr(80),chr(81),chr(83),chr(85),chr(0)]
        noteNames=["Csharp4","D4","E4","Fsharp4","Gsharp4","A4","B4","Csharp5","D5","E5","Fsharp5","Gsharp5","A5","B5","Csharp6","REST"]

    #E sharp
    elif keySig==11:
        freqs=[chr(61),chr(63),chr(64),chr(66),chr(68),chr(69),chr(71),chr(73),chr(75),chr(76),chr(78),chr(80),chr(81),chr(83),chr(85),chr(0)]
        noteNames=["Csharp4","Dsharp4","E4","Fsharp4","Gsharp4","A4","B4","Csharp5","Dsharp5","E5","Fsharp5","Gsharp5","A5","B5","Csharp6","REST"]

    #B sharp
    elif keySig==12:
        freqs=[chr(61),chr(63),chr(64),chr(66),chr(68),chr(70),chr(71),chr(73),chr(75),chr(76),chr(78),chr(80),chr(82),chr(83),chr(85),chr(0)]
        noteNames=["Csharp4","Dsharp4","E4","Fsharp4","Gsharp4","Asharp4","B4","Csharp5","Dsharp5","E5","Fsharp5","Gsharp5","Asharp5","B5","Csharp6","REST"]

    #F sharp
    elif keySig==13:
        freqs=[chr(61),chr(63),chr(65),chr(66),chr(68),chr(70),chr(71),chr(73),chr(75),chr(77),chr(78),chr(80),chr(82),chr(83),chr(85),chr(0)]
        noteNames=["Csharp4","Dsharp4","Esharp4","Fsharp4","Gsharp4","Asharp4","B4","Csharp5","Dsharp5","Esharp5","Fsharp5","Gsharp5","Asharp5","B5","Csharp6","REST"]

    #C sharp
    elif keySig==14:
        freqs=[chr(61),chr(63),chr(65),chr(66),chr(68),chr(70),chr(72),chr(73),chr(75),chr(77),chr(78),chr(80),chr(82),chr(84),chr(85),chr(0)]
        noteNames=["Csharp4","Dsharp4","Esharp4","Fsharp4","Gsharp4","Asharp4","Bsharp4","Csharp5","Dsharp5","Esharp5","Fsharp5","Gsharp5","Asharp5","Bsharp5","Csharp6","REST"]

    #C Major
    else:
        freqs=[chr(60),chr(62),chr(64),chr(65),chr(67),chr(69),chr(71),chr(72),chr(74),chr(76),chr(77),chr(79),chr(81),chr(83),chr(84),chr(0)]
        noteNames=["C4","D4","E4","F4","G4","A4","B4","C5","D5","E5","F5","G5","A5","B5","C6","REST"]
    return freqs,noteNames
	
#audio output
def playSong(song,measures,tempo_factor,instr):
    #track measure and line with lights
    meas=0
    line=0
    numLines=len(measures)
    
    #configure tempo
    tempo=(song[3]*tempo_factor).astype(np.uint8)
    
    #constant instrument
    statusByte=chr(ord('9')+instr)
    
    #volume parameter
    vol='0'
    
    #configure lookup tables
    notes=["Sixteenth","Eighth","Quarter","Half","Whole"]
    mStep=[1,2,4,8,16]
    durations=[1.0/tempo/4.0*60.0,1.0/tempo/2.0*60.0,1.0/tempo*60.0,2.0/tempo*60.0,4.0/tempo*60.0]
    waittime=1.0/tempo/32.0
    
    #countdown
    lcd.clear()
    lcd.text("Begin In:",1)
    lcd.text("3",2)
    time.sleep(1)
    lcd.text("2",2)
    time.sleep(1)
    lcd.text("1",2)
    time.sleep(1)
    lcd.clear()

    #loop through notes
    #go line by line, keeping track of measure
    k=0
    lenSong=len(song)
    while (line<numLines and k<lenSong):
        #lnlght=int(line/9.0*vert+1)+horiz+1
        #strip[lnlght]=(255,192,0)
        #determine number of spaces based on time sig and num measures
        tSig=(song[k+2]&0xF0)>>4
        m=measures[line]*4*(tSig)
        meas=0
        
        #get lookup tables for frequencies and notes based on key sig
        freqs,noteNames=handleKeySignature(song[k+1])
        k=k+4
        
        #loop through line while not at end
        while (meas<m and k<lenSong):
            lght=int(meas/m*horiz+2)
            sB=statusByte
            #if dynamic, adjust volume
            if song[k]>0xFF:
                vol=chr(48+((song[k]-0x0100)>>8))
                
                k=k+1
            else:
                note=song[k]&0x0F   #index value representing duration (quarter, whole, half,etc)
                if note<=4:
                    f=song[k]>>4    #spot on the line (C4,C5, etc)
                else:
                    f=15
                    note=note%5
                    sB='8'
                d=durations[note]
                mS=mStep[note]
                if (note==4):
                    d=d*tSig/4.0
                    mS=mS*tSig/4.0
                    
                strip[lght]=(255,192,0)    
                srl.write((sB+freqs[f]+vol).encode())
                lcd.text(noteNames[f],1)
                lcd.text(notes[note],2)
                time.sleep(d-waittime)
                srl.write(b'800')
                time.sleep(waittime)
                meas=meas+mS
                k = k + 1
                strip[lght]=(0,0,0)
        line = line + 1
        #strip[lnlght]=(0,0,0)
            
def main(args):
	while 1:
		strip.fill(coff)
		lcd.clear()
		lcd.text("Push Process",1)
		lcd.text("To Start",2)
		startP = 0
		while startP == 0:
			startP = GPIO.input(processBtn)
		time.sleep(1)
		lcd.clear()
		#get selections from menu
		song_name,instr,tempo_factor,keepGoing=OpenMenu()
		
		#process song
		lcd.clear()
		
		if keepGoing:
			lcd.text("Processing. . .",1)
			song,measures=processSong(song_name)
		
			#When processing is done, wait for play
			playing=1
			while playing==1:
				lcd.clear()
				lcd.text("Ready",1)
				lcd.text("Push Play",2)
				playNow = 0
			
				#push play to start
				while playNow == 0:
					playNow = GPIO.input(playBtn)
				time.sleep(0.5)
			
				playSong(song,measures,tempo_factor,instr)
				strip.fill(coff)
				#ask to replay
				lcd.clear()
				lcd.text("Replay Song?",1)
				lcd.text("Push Play",2)
				userChoice = 0
				while userChoice == 0:
					replay = GPIO.input(playBtn)
					goMenu = GPIO.input(processBtn)
				
					if goMenu == 1:
						userChoice=1
					elif replay == 1:
						userChoice=2
				lcd.clear()
			
				#go to menu
				if userChoice!=2:
					playing=0
			
		
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
