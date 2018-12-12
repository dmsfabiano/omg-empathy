#import file_operations as fp
#import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.optimizers import RMSprop, adam,adamax, Nadam
#from sklearn.preprocessing import StandardScaler
#from keras import backend as K
#from keras import callbacks
#import tensorflow as tf

import os
from pydub import AudioSegment
from tqdm import tqdm

def splitAudio(path, savePath):
    audios = os.listdir(path)
    #stepSize = 40
    stepSize = 40 * 25
    for originalAudio in tqdm(audios):
        if originalAudio.endswith('.wav'):
		    audioPath = path + "/" + originalAudio
		    audioNoExtension = os.path.splitext(originalAudio)[0]

		    originalWavAudioSegment = AudioSegment.from_wav(audioPath)
		    
		    for i in range(0, len(originalWavAudioSegment) - stepSize, stepSize):
		        newSplitAudio = originalWavAudioSegment[i:i + stepSize]
		        #newSplitAudio.export('{0}{1}_frame_{2}.wav'.format(savePath, audioNoExtension, str(i / stepSize)), format="wav")
		        newSplitAudio.export('{0}{1}_sec_{2}.wav'.format(savePath, audioNoExtension, str(i / stepSize)), format="wav")

if __name__ == "__main__":
    print('Starting splitting Training audio')
    # Path where the audios are
    #path = "/data/Validation/Videos/"
    #path = "D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\data\\Training\\Videos\\"
    path = '/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio/Training/'
    #path = "/media/neil/30CA58BFCA588350/data/Training/Videos/"
    #path = "../data/Validation/Videos"

    # Path where the audios will be saved
    #savePath ="/data/audio/Validation/"
    #savePath = "D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\data\\audio-reduced\\Training\\"
    savePath = '/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split-1-sec/Training/'
    #savePath = "/media/neil/30CA58BFCA588350/data/audio-reduced/Training/"
    #savePath = "../data/audio/Validation"

    splitAudio(path,savePath)

    print('Starting splitting Validation audio')
    path = '/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio/Validation/'
    savePath = '/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-split-1-sec/Validation/'
    splitAudio(path,savePath)