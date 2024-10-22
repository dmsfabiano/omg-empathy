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

train_landmarks_subject, train_landmarks_actor, train_audio = [],[],[]
validation_landmarks_subject, validation_landmarks_actor, validation_audio = [],[],[]

for video in x_train:
	for frame in video:
		train_landmarks_subject.append(np.asarray(frame[0:136]))
		train_landmarks_actor.append(np.asarray(frame[136:272]))

for video in x_validation:
	for frame in video:
		validation_landmarks_subject.append(np.asarray(frame[0:136]))
		validation_landmarks_actor.append(np.asarray(frame[136:272]))

# 1 Fuse both faces (landmarks)
train_fused_faces, validation_fused_faces = [], []

for i in range(0,len(train_landmarks_subject)):
    train_fused_faces.append(fusion([train_landmarks_subject[i],train_landmarks_actor[i]]))
train_fused_faces = np.asarray(train_fused_faces)

for i in range(0,len(validation_landmarks_subject)):
    validation_fused_faces.append(fusion([validation_landmarks_subject[i],validation_landmarks_actor[i]]))
validation_fused_faces = np.asarray(validation_fused_faces)

def writeImages(landmarks,j,flag):
	xlm,ylm = [],[]
	for i in range(0,len(landmarks),2):
		xlm.append(int(landmarks[i]))
		ylm.append(int(landmarks[i+1]))
	height = max(ylm)
	width = max(xlm)
	matrix = np.zeros((height+1, width+1, 1), dtype = "uint8")
	for value in range(len(xlm)):
		matrix[ylm[value]][xlm[value]] = 255
	indicies = np.where(matrix==255)
	bbox = np.min(indicies[0]), np.max(indicies[0]), np.min(indicies[1]), np.max(indicies[1])
	matrix = matrix[bbox[0]:bbox[1], bbox[2]:bbox[3]]
	imwrite('../data/Images/Training/frame_'+str(j)+'_.png' if flag == False else '../data/Images/Validation/frame_'+str(j)+'_.png', resize(matrix,(128,128)))

for j,face in enumerate(train_fused_faces):
	writeImages(face,j,False)
for j,face in enumerate(validation_fused_faces):
	writeImages(face,j,True)
