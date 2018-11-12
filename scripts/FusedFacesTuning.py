import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from keras import metrics
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import tensorflow as tf
from scipy.stats import pearsonr

def ccc(y_true, y_pred):
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)

    #rho,_ = pearsonr(y_pred,y_true)
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
                                                                             
    std_predictions = K.std(y_pred)

    std_gt = K.std(y_true)
    

    ccc = tf.divide(2 * tf.multiply(tf.multiply(pearson_r,std_gt),std_predictions),tf.add(tf.add(tf.multiply(std_predictions,std_predictions),tf.multiply(std_gt,std_gt)), 
        tf.multiply(tf.subtract(pred_mean,true_mean),tf.subtract(pred_mean,true_mean)))) 

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

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mean_squared_error', metrics = ['mse','accuracy',ccc])
            
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

for video in x_validation:
	for frame in video:
		validation_landmarks_subject.append(np.asarray(frame[0:136]))
		validation_landmarks_actor.append(np.asarray(frame[136:272]))

scaler = StandardScaler()
train_landmarks_subject = scaler.fit_transform(train_landmarks_subject)
train_landmarks_actor = scaler.fit_transform(train_landmarks_actor)

validation_landmarks_subject = scaler.fit_transform(validation_landmarks_subject)
validation_landmarks_actor = scaler.fit_transform(validation_landmarks_actor)


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


# 1 Fuse both faces (landmarks)
train_fused_faces, validation_fused_faces = [], []

for i in range(0,len(train_landmarks_subject)):
    train_fused_faces.append(fusion([train_landmarks_subject[i],train_landmarks_actor[i]]))
train_fused_faces = np.asarray(train_fused_faces)

for i in range(0,len(validation_landmarks_subject)):
    validation_fused_faces.append(fusion([validation_landmarks_subject[i],validation_landmarks_actor[i]]))
validation_fused_faces = np.asarray(validation_fused_faces)

# 2 Create and Train Face Feature Extractor (NEEDS TUNING ~ HARDCODED HYPERPARAMETERS)
for opt in optimizers:
    for rate in rates:
        for hLayers in hidden_layers:
            for hNeurons in hidden_neurons:
                for outNeurons in output_neurons:
                    faceFeatureExtractor = CreateRegressor(input_neurons = len(train_fused_faces[0]), output_neurons=outNeurons,hidden_layers=hLayers,learning_rate=rate,
                                                           optimizer=opt,hidden_neurons=hNeurons)
                    faceFeatureExtractor = trainRegressor(faceFeatureExtractor,train_fused_faces,y_train,epochs=100,batches=250)
                    
                    metric = faceFeatureExtractor.evaluate(validation_fused_faces,y_validation, verbose=1)
                    mse = metric[1]
                    accuracy = metric[2] * 100
                    CCC = metric[3]
                    
                    print('Fused Faces statistics: {0} mse, {1}% accuracy, and {3} ccc \n with parameters: {4} optimizer, rate: {5}, {6} hlayers, {7} hneurons, {8} dim neurons'.format(
                            mse,accuracy,CCC,opt,rate,hLayers,hNeurons,outNeurons))

