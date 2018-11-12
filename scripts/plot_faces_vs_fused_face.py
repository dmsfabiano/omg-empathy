import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt

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

train_landmarks_subject, train_landmarks_actor, train_audio = [],[],[]
validation_landmarks_subject, validation_landmarks_actor, validation_audio = [],[],[]

for video in x_train:
	for frame in video:
		train_landmarks_subject.append(np.asarray(frame[0:136]))
		train_landmarks_actor.append(np.asarray(frame[136:272]))

x_lm, x2_lm, y_lm, y2_lm = [], [], [], []
for i in range(0,len(train_landmarks_subject[0]),2):
	x_lm.append(train_landmarks_subject[0][i])
	y_lm.append(train_landmarks_subject[0][i+1])
	x2_lm.append(train_landmarks_actor[0][i])
	y2_lm.append(train_landmarks_actor[0][i+1])

plt.scatter(x_lm,y_lm)
plt.savefig('../data/Visualization/face_subject.png')
plt.clf()

plt.scatter(x2_lm, y2_lm)
plt.savefig('../data/Visualization/face_actor.png')
plt.clf()

# 1 Fuse both faces (landmarks)
train_fused_faces = []

for i in range(0,len(train_landmarks_subject)):
    train_fused_faces.append(fusion([train_landmarks_subject[i],train_landmarks_actor[i]]))
    break
train_fused_faces = np.asarray(train_fused_faces)

x3_lm, y3_lm = [], []
for i in range(0,len(train_fused_faces[0]),2):
	x3_lm.append(train_fused_faces[0][i])
	y3_lm.append(train_fused_faces[0][i+1])

plt.scatter(x3_lm,y3_lm)
plt.savefig('../data/Visualization/face_fused.png')
