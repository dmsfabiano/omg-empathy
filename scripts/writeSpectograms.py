from PIL import Image
import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
from multiprocessing import Pool,Lock
from cv2 import imwrite, resize
from scipy import signal

def fusion(data_container):
   
    scaler = MinMaxScaler(feature_range=(0.5,1))
    variance_container = np.array([np.var(signal) for signal in data_container])
    variance_container = scaler.fit_transform(np.reshape(variance_container, newshape=(-1,1)))
    
    weighted_container = np.array([0.0 for i in range(0,len(data_container[0]))])
    
    for index,signal in enumerate(data_container):
        signal = np.multiply(np.asarray(signal),variance_container[index])
        weighted_container = np.add(weighted_container,signal)
    return weighted_container

def getData(subject_list,story_list, directory):
    x_separated,y_separated = fp.read_all_data(subject_list,story_list, directory)

    x = np.asarray([story for subject in x_separated for story in subject])
    y = np.asarray([story for subject in y_separated for story in subject])
    
    return x,y

# Get data
train_sbj_list = [1,2,3,4,5,6,7,8,9,10]
train_story_list = [2,4,5,8]
validation_story_list =  [1]

x_train,y_train = getData(train_sbj_list,train_story_list,'../data/results/Training/')
x_validation,y_validation = getData(train_sbj_list,validation_story_list,'../data/results/Validation/')

train_audio = []
validation_audio = []

for video in x_train:
	for frame in video:
		train_audio.append(np.asarray(frame[272:5272]))

for video in x_validation:
	for frame in video:
		validation_audio.append(np.asarray(frame[272:5272]))

def writeImages(audio,j,flag):
	f,t,_ = signal.spectrogram(audio)
	height = max(f)
	width = max(t)
	matrix = np.zeros((int(height)+1, int(width)+1, 1), dtype = "uint8")
	for value in range(len(t)):
		matrix[int(f[value])][int(t[value])] = 255
	imwrite('../data/Spectograms/Training/frame_'+str(j)+'_.png' if flag == False else '../data/Spectograms/Validation/frame_'+str(j)+'_.png', resize(matrix,(256,256)))

for j,face in enumerate(train_audio):
	writeImages(face,j,False)
for j,face in enumerate(validation_audio):
	writeImages(face,j,True)
