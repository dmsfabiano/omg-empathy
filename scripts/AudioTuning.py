import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras import metrics
from scipy.stats import pearsonr


def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc


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

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mean_squared_error', metrics =  ['mse','accuracy',ccc])
            
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
		train_audio.append(np.asarray(frame[272:5272]))

for video in x_validation:
	for frame in video:
		validation_audio.append(np.asarray(frame[272:5272]))

temp = []
for video in y_train:
	temp.extend(video)
y_train = temp.copy()

temp = []
for video in y_validation:
	temp.extend(video)
y_validation = temp.copy()

# DDNet Structure

optimizers = ['adam', 'rms','adamax']
rates = [0.1,0.01,0.001,0.0001,0.00001]
hidden_layers = [0,1,2,5,10]
hidden_neurons = [int((136+1)/2),136*2]
output_neurons = [10,100,1000]


for opt in optimizers:
    for rate in rates:
        for hLayers in hidden_layers:
            for hNeurons in hidden_neurons:
                for outNeurons in output_neurons:
                    audioFeatureExtractor = CreateRegressor(input_neurons = len(train_audio[0]), output_neurons=outNeurons,hidden_layers=hLayers,learning_rate=rate,
                                                            optimizer=opt,hidden_neurons=hNeurons)
                    audioFeatureExtractor = trainRegressor(audioFeatureExtractor,np.asarray(train_audio),y_train,epochs=100,batches= 250)
                    
                    metric = audioFeatureExtractor.evaluate(np.asarray(validation_audio),y_validation, verbose=1)
                    mse = metric[1]
                    accuracy = metric[2] * 100
                    CCC = metric[3]
                    
                    print('Audio statistics: {0} mse, {1}% accuracy, and {3} ccc \n with parameters: {4} optimizer, rate: {5}, {6} hlayers, {7} hneurons, {8} dim neurons'.format(
                            mse,accuracy,CCC,opt,rate,hLayers,hNeurons,outNeurons))
