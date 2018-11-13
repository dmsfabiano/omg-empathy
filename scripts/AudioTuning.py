import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam,adamax, Nadam
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras import callbacks
import tensorflow as tf

def ccc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    rho = K.maximum(K.minimum(r, 1.0), -1.0)

    numerator = tf.multiply(tf.multiply(tf.scalar_mul(2,rho),K.std(x)),K.std(y))
    
    mean_differences = K.square(my - mx)
    std_predictions_squared = K.square(K.std(y))
    std_true_squared = K.square(K.std(x))
    
    denominator = tf.add(tf.add(std_predictions_squared,std_true_squared),mean_differences)
    
    return numerator/denominator


def ccc_loss(y_true, y_pred):
    c_value = ccc(y_true,y_pred)  
    return (1 - c_value)/2

def pearsonr(y_true,ypred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

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

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = ccc_loss , metrics =  ['mse','accuracy',ccc, 'mae', pearsonr])
            
    return model

def trainRegressor(model,x,y,epochs,batches,verb=1,calls):
    history = model.fit(x, y, batch_size = batches, epochs = epochs, verbose=verb,callbacks=calls)
    return model, history

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

scaler = StandardScaler()
train_audio = scaler.fit_transform(train_audio)

validation_audio = scaler.fit_transform(validation_audio)

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

for outNeurons in output_neurons:
    audioFeatureExtractor = CreateRegressor(input_neurons = len(train_audio[0]), output_neurons=outNeurons,hidden_layers=5,learning_rate=0.1,
                                            optimizer='adam',hidden_neurons=int((len(train_audio) + 1)/2))
                                            
    reduceLR = callbacks.ReduceLROnPlateau(monitor=ccc_loss,factor=0.2,patience=10,verbose=1,mode='min')
    
    audioFeatureExtractor,history = trainRegressor(audioFeatureExtractor,np.asarray(train_audio),y_train,epochs=250,batches= 250,calls=[reduceLR], verb=2)
    
    loss_values = history.history['loss']
    mse_values = history.history['mse']
    acc_values = history.history['accuracy']
    
    print('Average ccc: {0} +/- {1}, mse: {2} +/- {3}, accuracy: {4} +/- {5}'.format(np.mean(loss_values),np.std(loss_values),np.mean(mse_values),np.std(mse_values)),
          np.mean(acc_values),np.std(acc_values))
    
    metric = audioFeatureExtractor.evaluate(np.asarray(validation_audio),y_validation, verbose=0)
    mse = metric[1]
    accuracy = metric[2] * 100
    CCC = metric[3]
    mae = metric[4]
    r = metric[5]
    
    print('Audio statistics on testing: {0} ccc, {1} mse, {2} mae, {3} accuracy\n with parameters: {4} output dimension neurons '.format(
            CCC,mse,mae,accuracy,outNeurons))
