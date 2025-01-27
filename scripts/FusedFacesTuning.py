import file_operations as fp
import numpy as np
import networks as net
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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


temp = []
for video in y_train:
	temp.extend(video)
y_train = temp.copy()

temp = []
for video in y_validation:
	temp.extend(video)
y_validation = temp.copy()

# DDNet Structure


output_neurons = [10,100,1000,2500,5000,10000]


# 1 Fuse both faces (landmarks)
train_fused_faces, validation_fused_faces = [], []

for i in range(0,len(train_landmarks_subject)):
    train_fused_faces.append(fusion([train_landmarks_subject[i],train_landmarks_actor[i]]))
train_fused_faces = np.asarray(train_fused_faces)

for i in range(0,len(validation_landmarks_subject)):
    validation_fused_faces.append(fusion([validation_landmarks_subject[i],validation_landmarks_actor[i]]))
validation_fused_faces = np.asarray(validation_fused_faces)

scaler = StandardScaler()
train_fused_faces = scaler.fit_transform(train_fused_faces)
validation_fused_faces = scaler.fit_transform(validation_fused_faces)

# 2 Create and Train Face Feature Extractor (NEEDS TUNING ~ HARDCODED HYPERPARAMETERS)

for outNeurons in output_neurons:
    faceFeatureExtractor = net.CreateRegressor(input_neurons = len(train_fused_faces[0]), output_neurons=outNeurons,hidden_layers=5,learning_rate=0.1,
                                           optimizer='adam',hidden_neurons=len(train_fused_faces[0]))
    
    reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10,verbose=1,mode='min',min_delta=0.001,min_lr=0.0000001) 
    faceFeatureExtractor,history = net.trainRegressor(faceFeatureExtractor,train_fused_faces,y_train,epochs=100,batches= 250,calls=[reduceLR], verb=2)
    
    loss_values = history.history['loss']
    ccc_values = history.history['ccc']
    
    print('Average loss: {0} +/- {1}, ccc: {2} +/- {3}'.format(np.mean(loss_values),np.std(loss_values),np.mean(ccc_values),np.std(ccc_values)))
    
    metric = faceFeatureExtractor.evaluate(validation_fused_faces,y_validation, verbose=0)
    mse = metric[1]
    CCC = metric[2]
    mae = metric[3]
    r = metric[4]
    
    print('Fused Faces statistics on testing: {0} mse, {1} ccc, {2} mae, {3} r \n with parameters: {4} dim neurons '.format(
            mse,CCC,mae,r,outNeurons))

