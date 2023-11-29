#!/usr/bin/env python
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

#Computer Vision
def processSong(song_sel):
    song=np.array([0x44,0x78,0x00,0x00,0x00,0x71,0x00,0x00,0x00, 0x32, 0x01, 0x10, 0x00]).astype(np.uint8)
    return song

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
def playSong(song,tempo):
    #configure lookup tables
    measure=0
    line=0
    statusByte=chr(ord('9')+instr)
    freqs,noteNames=handleKeySignature(song[2])
    notes=["Quarter","Half","Whole","Eighth","Sixteenth"]
    durations=[1.0/tempo*60,2.0/tempo*60,4.0/tempo*60,1.0/tempo/2.0*60,1.0/tempo/4.0*60]
    waittime=1.0/tempo/16.0*60

    #loop through notes
    for k in range(3,len(song),2):
        note=song[k]&0x0F
        if note<=4:
            f=song[k]>>4
        else:
            f=15
            note=note-5
        d=durations[(song[k]&0x0F)%5]

        srl.write((statusByte+freqs[f]+chr(song[k+1]+48)).encode())
        #strip[measure]=(255,192,0)
        strip[0]=color
        lcd.text(noteNames[f],1)
        lcd.text(notes[note],2)
        time.sleep(d-waittime)
        srl.write(b'800')
        #strip[measure]=coff
        strip[0]=coff
        measure=measure+(2**note)
        if measure>=16:
            line=line+1
            measure=0
        time.sleep(waittime)

#create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

#create ADC object
ads= ADS.ADS1115(i2c)

#create LCD screen
lcd=LCD()

#configure serial comms
srl = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 38400,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)

#create light strip, golden yellow
strip=neopixel.NeoPixel(board.D18,30,brightness=1.0/20)
color=(255,192,0)
coff=(0,0,0)

#create two single ended inputs for tempo and instrument selection
instr_sel=AnalogIn(ads,ADS.P0)
tempo_sel=AnalogIn(ads,ADS.P1)

#max selection based on experimental values
sel_max=26450

#get song selection
song_sel=0

#process song into bytes
song=processSong(song_sel)
lcd.clear()
lcd.text("Processing...",1)
time.sleep(4)

freqs=[chr(60),chr(62),chr(64),chr(65),chr(67),chr(69),chr(71),chr(72),chr(74),chr(76),chr(77),chr(79),chr(81),chr(83),chr(84),chr(0)]
noteNames=["C4","D4","E4","F4","G4","A4","B4","C5","D5","E5","F5","G5","A5","B5","C6","REST"]

#get selections
#instr=int(sel_max/instr_sel.value*8+0.75)
#tempo_factor=int(sel_max/tempo_sel.value*4+1)/4.0

#configure for output
tempo=(120)
insts=["Tones","Alto Sax","Clarinet","Piano","Tenor Sax","Trumpet","Violin","Voice"]
lcd.text("Playing",1)
time.sleep(1)
d=1.0/tempo*60
waittime=1.0/tempo/16.0*60

#loop through notes
for k in range(0,8):
    statusByte=chr(ord('9')+k)
    lcd.clear()
    lcd.text(insts[k],1)
    srl.write(((statusByte+freqs[0]+'0')).encode())
    lcd.text(noteNames[0],2)
    time.sleep(d-waittime)
    srl.write(b'800')
    time.sleep(waittime)
        

lcd.clear()

