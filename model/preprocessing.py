#http://essentia.upf.edu/documentation/
import essentia
import essentia.standard
from essentia.standard import *
import pickle
import os
import numpy as np

sample_rate = 44100
frame_size = 32768
hop_size = 16384
frame_rate = (frame_size - hop_size) / sample_rate
zero_padding = 0


window  = Windowing(size  = frame_size, zeroPadding = zero_padding, type = "blackmanharris62")
spectrum = Spectrum(size = frame_size + zero_padding)
mfcc = MFCC(sampleRate = sample_rate,numberCoefficients = 40)
train = []


def parse(file_path, file_name):
    try:
          
        mfccs = []
         
        audio   = MonoLoader(filename = file_path) # load audio and desampel to mono
        
        samples = audio()
        frames  = FrameGenerator(audio = samples, frameSize = frame_size, hopSize = hop_size)
        
        total_frames = frames.num_frames()
        n_frames = 0
        for frame in frames:

            frame_windowed = window(frame)
            frame_spectrum = spectrum(frame_windowed)
            mfcc_bands, mfcc_coeffs = mfcc(frame_spectrum)
            mfccs.append(mfcc_coeffs)
            n_frames += 1
            
        train.append(
            {"name":file_name,"mfccs":mfccs})
        print ("%d/%d: -%s" %(len(train),len(files),file_path))
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)


print ("[*] Processing data")
files = os.listdir('/data')
for f in files:
    parse(os.path.join('/data',f),f)
    
if os.path.exists('/model/tmp') is False:
    os.mkdir('/model/tmp')
output = open('/model/tmp/data.pkl', 'wb')
pickle.dump(train, output)
output.close()
print ("[*] End processing data")