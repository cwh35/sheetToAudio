# sheetToAudio
This project is for ECE 1896 - Senior Design Fall 2023 at the University of Pittsburgh

Research paper written for this project is listed as "Final Paper.pdf"

The purpose of this project is to help people learn sheet music by teaching them the notes of a song, tempo, time signature, and key signature to familiarize them with musical notation. It also gives the user an idea of what the piece sounds like at different playback speeds as well as with different instruments.

The computer vision aspect of the project uses MTM (multi-template matching) to match the different notes, rests, key signatures, tempos, and time signatures. The resulting matches are then put through a clustering algorithm to put the notes in the proper order. The amount of clusters is determined by the number of treble clefs recognized. Each cluster is sorted by the y-coordinate --> music is read top to bottom. Each individual cluster is then sorted by the x-coordinate --> music is read left to right. Once everything is sorted, post-processing logic takes place. This logic accounts for missing key signatures, time signatures, and tempos. Once the post-processing logic is complete, the templates have corresponding hex values that are converted to integers and placed into an array. This array is then passed to the next module for audio processing. Extra logic within the code calculates the number of measures per line and sends that information in an array as well to help the next module know when the line has ended.

The main code integrated with the raspberri pi is 'note2audio.py'
