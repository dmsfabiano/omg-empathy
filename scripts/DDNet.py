import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from sklearn.preprocessing import StandardScaler
from keras import backend as K

def fusion(data_container):
   
    scaler = MinMaxScaler(feature_range=(0.5,1))
    variance_container = np.array([np.var(signal) for signal in data_container])
    variance_container = scaler.fit_transform(np.reshape(variance_container, newshape=(-1,1)))
    
    weighted_container = np.array([0.0 for i in range(0,len(data_container[0]))])
    
    for index,signal in enumerate(data_container):
        signal = np.multiply(np.asarray(signal),variance_container[index])
        weighted_container = np.add(weighted_container,signal)
    return weighted_container

def getOptimizer(name,rate):
    if name is 'adamax':
        return adamax(lr=rate)
    elif name is 'adam':
        return adam(lr=rate)
    elif name is 'nadam':
        return Nadam(lr=rate)
    else:
        return RMSprop(lr=rate)

def CreateRegressor(input_neurons, output_neurons,hidden_layers,learning_rate, optimizer,hidden_neurons):

    
    model = Sequential()
    model.add(Dense(units = hidden_neurons, kernel_initializer = 'uniform',activation = 'relu', input_dim = input_neurons))
    
    for i in range(0,hidden_layers):
        model.add(Dense(units = hidden_neurons, kernel_initializer = 'uniform',activation = 'relu'))
        if i % 2 == 1:
            model.add(Dropout(0.3))
    model.add(Dense(units = output_neurons, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dense(1,activation='linear'))

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mean_squared_error', metrics = ['accuracy'])
            
    return model

def trainRegressor(model,x,y,epochs,batches,verb=1):
    model.fit(x, y, batch_size = batches, epochs = epochs, verbose=verb)
    return model

def getDeepFeatures(featureDetector,x):
    getLastLayer = K.function([featureDetector.layers[0].input], [featureDetector.layers[-2].output])
    return getLastLayer([x])[0]

def RegressorPrediction(model,x):
    return model.predict(x)
            
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
		train_audio.append(np.asarray([f[0] for f in frame[272:5272]]))

for video in x_validation:
	for frame in video:
		validation_landmarks_subject.append(np.asarray(frame[0:136]))
		validation_landmarks_actor.append(np.asarray(frame[136:272]))
		validation_audio.append(np.asarray([f[0] for f in frame[272:5272]]))

temp = []
for video in y_train:
	temp.extend(video)
y_train = temp.copy()

temp = []
for video in y_validation:
	temp.extend(video)
y_validation = temp.copy()

# DDNet Structure

# 1 Fuse both faces (landmarks)
train_fused_faces, validation_fused_faces = [], []

for i in range(0,len(train_landmarks_subject)):
    train_fused_faces.append(fusion([train_landmarks_subject[i],train_landmarks_actor[i]]))
train_fused_faces = np.asarray(train_fused_faces)

for i in range(0,len(validation_landmarks_subject)):
    validation_fused_faces.append(fusion([validation_landmarks_subject[i],validation_landmarks_actor[i]]))
validation_fused_faces = np.asarray(validation_fused_faces)

# 2 Create and Train Face Feature Extractor (NEEDS TUNING ~ HARDCODED HYPERPARAMETERS)
faceFeatureExtractor = CreateRegressor(input_neurons = len(train_fused_faces[0]), output_neurons=100,hidden_layers=10,learning_rate=0.1, optimizer='adam',hidden_neurons=int((136+1)/2))
faceFeatureExtractor = trainRegressor(faceFeatureExtractor,train_fused_faces,y_train,epochs=250,batches= 500)
print('Fused Faces Accuracy: {}%'.format(faceFeatureExtractor.evaluate(validation_fused_faces,y_validation, verbose=1)[1]*100))


# 3 Create and Train audio Feature Extractor
audioFeatureExtractor = CreateRegressor(input_neurons = len(train_audio[0]), output_neurons=100,hidden_layers=2,learning_rate=0.001, optimizer='adam',hidden_neurons=int((len(train_audio[0])+1)/2))
audioFeatureExtractor = trainRegressor(audioFeatureExtractor,train_audio,y_train,epochs=100,batches= 500)
print('Audio Accuracy: {}%'.format(audioFeatureExtractor.evaluate(validation_audio,y_validation, verbose=1)[1]*100))

# 4 Crete Deep feature Container
train_faceFeatures,train_audioFeatures = [],[]
validation_faceFeatures,validation_audioFeatures = [],[]

# 4.1 training
for instance in train_fused_faces:
    train_faceFeatures.append(np.array(getDeepFeatures(faceFeatureExtractor,instance)))
train_faceFeatures = np.asarray(train_faceFeatures)

for instance in train_audio:
    train_audioFeatures.append(np.array(getDeepFeatures(audioFeatureExtractor,instance)))
train_audioFeatures = np.asarray(train_audioFeatures)

# 4.2 validation
for instance in validation_fused_faces:
    validation_faceFeatures.append(np.array(getDeepFeatures(faceFeatureExtractor,instance)))
validation_faceFeatures = np.asarray(validation_faceFeatures)

for instance in validation_audio:
    validation_audioFeatures.append(np.array(getDeepFeatures(audioFeatureExtractor,instance)))
validation_audioFeatures = np.asarray(validation_audioFeatures)

# 5 Fuse Deep features
train_fused_deep_features, validation_fused_deep_features = [], []

# 5.1 training
for i in range(0,len(train_faceFeatures)):
    train_fused_deep_features.append(fusion([train_faceFeatures[i],train_audioFeatures[i]]))
train_fused_deep_features = np.asarray(train_fused_deep_features)

# 5.2 validation
for i in range(0,len(validation_faceFeatures)):
    validation_fused_deep_features.append(fusion([validation_faceFeatures[i],validation_audioFeatures[i]]))
validation_fused_deep_features = np.asarray(validation_fused_deep_features)

# 6 Create Final Regressor
regressor = CreateRegressor(input_neurons = 100, output_neurons=100,hidden_layers=2,learning_rate=0.001, optimizer='adam',hidden_neurons=int((100+1)/2))
regressor = trainRegressor(regressor,train_fused_deep_features,y_train,epochs=100,batches= 500,verb=1)

# 7 Test on validation
scores = regressor.evaluate(validation_fused_deep_features,y_validation, verbose=1)[1]*100
print('Final Network Results: {}%'.format(scores))


