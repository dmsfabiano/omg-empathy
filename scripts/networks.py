from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ConvLSTM2D, MaxPooling3D, Input, Conv3D, concatenate
import operator
from keras.optimizers import RMSprop, adam,adamax, Nadam, SGD
from keras import backend as K
from keras.applications import Xception
from keras.utils import multi_gpu_model
import tensorflow as tf
import cv2
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def imageLoader(files,batch_size=256,time=None):
    
    def loadImages(files):
        x = []    
        for file in files:
            x.append(cv2.imread(file[0][0]))
        if time is not None:
            y = [[] for _ in x]
            for i in range(1,len(x)+1):
                count = 1
                while count <= time:
                    if i - count >= 0:
                        y[i-1].append(x[i-count])
                    else:
                        y[i-1].append(x[0])
                    count += 1
        if time is None:
            return np.reshape(np.asarray(x), (len(files),256,256,3))
        else:
            return np.reshape(np.asarray(y), (len(files),time,256,256,3))
    
    def loadValues(values):
        y = []    
        for value in values:
            y.append(value[1])
        return np.asarray(y)
    
    L = len(files)
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end,L)
            x = loadImages(files[batch_start:limit])
            y = loadValues(files[batch_start:limit])
            yield (x,y)
            
            batch_start += batch_size
            batch_end += batch_size

def ccc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x) + K.epsilon()
    my = K.mean(y) + K.epsilon()
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

def focal_loss(y_true,y_pred,gamma=2, alpha=0.25):
	eps = K.epsilon()
    
	y_pred=K.clip(y_pred,eps,1.- eps)
    
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def ccc_loss(y_true, y_pred):
    c_value = ccc(y_true,y_pred)  
    return (1 - c_value)/2

def pearsonr(y_true,y_pred):
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

def unison_shuffle(data_container):
    p = np.random.permutation(len(data_container[0]))
    
    placeholder = [data[p] for data in data_container]
    
    return placeholder


def getOptimizer(name,rate):
    if name == 'adamax':
        return adamax(lr=rate)
    elif name == 'adam':
        return adam(lr=rate)
    elif name == 'nadam':
        return Nadam(lr=rate)
    elif name == 'sgd':
        return SGD(lr=rate, momentum=0.9, nesterov=True)
    else:
        return RMSprop(lr=rate)

def finalNetwork(batch_size, initial_filter_value, time):

	# INITIAL FACE CHANNEL

	input_x = Input(shape=(time,256,256,3))
	x = Conv3D(filters=initial_filter_value, kernel_size=3, padding='same', activation='relu')(input_x)
	x = BatchNormalization()(x)

	x = MaxPooling3D(pool_size=(2,2,2))(x)
	x = Dropout(0.2)(x)

	x = Conv3D(filters=initial_filter_value, kernel_size=3, padding='same', activation='relu')(x)
	x = BatchNormalization()(x)

	x = MaxPooling3D(pool_size=(4,4,4))(x)
	x = Dropout(0.2)(x)

	x = Conv3D(filters=initial_filter_value*2, kernel_size=3, padding='same', activation='relu')(x)
	x = BatchNormalization()(x)

	# END INITIAL FACE CHANNEL

	# INITIAL AUDIO CHANNEL

	input_y = Input(shape=(256,256,3))
	y = Conv2D(filters=initial_filter_value, kernel_size=3, padding='same', activation='relu')(input_y)
	
	# END INITIAL AUDIO CHANNEL

	# CREATE CHANNEL FOR IMAGE

#	cross_channel_x = Flatten()(x)
#	cross_channel_x = Dense((25*256*256*8)//128)(cross_channel_x)
	
	# END CHANNEL

	# CREATE CHANNEL FOR AUDIO

#	cross_channel_y = Flatten()(y)
#	cross_channel_y = Dense((25*256*256*8)//128)(cross_channel_y)

	# END CHANNEL FOR AUDIO

	#CHANNEL MERGE

#	merged_channel = concatenate([cross_channel_x, cross_channel_y])
#	merged_channel = Reshape((2,25,2048,2), input_shape=(1,204800))(merged_channel)
#	merged_channel = Conv3D(filters=8, kernel_size=5, padding='same')(merged_channel)

#	merge_input_x = Flatten()(merged_channel)
#	merge_input_x = Dense(2048)(merge_input_x)
#	merged_channel_output_x = Reshape((1,16,16,8), input_shape=(1,2048))(merge_input_x)
#	merge_input_x = None
	
#	merge_input_y = Flatten()(merged_channel)
#	merge_input_y = Dense(256*256*3)(merge_input_y)
#	merged_channel_output_y = Reshape((256,256,3), input_shape=(1,256*256*3))(merge_input_y)
#	merge_input_y = None
#	merged_channel = None

	# END CHANNEL MERGE

	# START END OF VISUAL

#	x = concatenate([x, merged_channel_output_x])
#	merged_channel_output_x = None

	x = Flatten()(x)
	x = Dense(100, activation='relu')(x)
#	x = Dropout(0.5)(x)
	#x = Dense(150, activation='relu')(x)
	#x = Dropout(0.5)(x)
	
	visual_channel = Dense(time, activation='linear')(x)
	# FINISH VISUAL

	# START END OF AUDIO

#	y = concatenate([y, merged_channel_output_y])
#	merged_channel_output_y = None

	y = Conv2D(filters=initial_filter_value*2, kernel_size=3, padding='same', activation='relu')(y)
	
	y = Flatten()(y)
	y = Dense(100, activation='relu')(y)
#	y = Dropout(0.5)(y)
	
	audio_channel = Dense(time, activation='linear')(y)

	# FINISH AUDIO
	model = Model(inputs=[input_x,input_y], outputs=[visual_channel,audio_channel])
	model.summary()
	return multi_gpu_model(model)


def CreateLSTMConv2DRegressor(shape, output_neurons, learning_rate, optimizer, kernel, initial_dimension, decreasing, return_sequence = True):
    func = None
    if decreasing:
        func = operator.floordiv
    else:
        func = operator.mul

    model = Sequential()

    model.add(ConvLSTM2D(filters=initial_dimension, kernel_size=kernel, input_shape=shape,activation='relu', padding='valid', return_sequences = return_sequence))
    model.add(BatchNormalization())
    
    model.add(Dropout(0.2))
    
    model.add(ConvLSTM2D(filters=initial_dimension, kernel_size=kernel, activation='relu', padding='valid', return_sequences = return_sequence))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2,2,2), padding='same'))
    next_dimension = func(initial_dimension, 2)
    
    model.add(ConvLSTM2D(filters=next_dimension, kernel_size=kernel, activation='relu', padding='same', return_sequences = return_sequence))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    model.add(ConvLSTM2D(filters=next_dimension, kernel_size=kernel, activation='relu', padding='same', return_sequences = return_sequence))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2,2,2), padding='same'))

    model.add(Flatten())
    model.add(Dense(output_neurons,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))
    
    model = multi_gpu_model(model,gpus=8)
    model.compile(optimizer = getOptimizer(optimizer, learning_rate), loss = 'mse', metrics = ['mse', 'accuracy', ccc, 'mae', ccc_loss])
    return model  
    
def CreateRegressor(input_neurons, output_neurons,hidden_layers,learning_rate, optimizer,hidden_neurons):

    
    model = Sequential()
    model.add(Dense(units = hidden_neurons, kernel_initializer = 'uniform',activation = 'relu', input_dim = input_neurons))
    
    for i in range(0,hidden_layers):
        model.add(Dense(units = hidden_neurons, kernel_initializer = 'uniform',activation = 'relu'))

    model.add(Dense(units = output_neurons, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dense(1,activation='linear'))

    model = multi_gpu_model(model,gpus=8)
    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mse', metrics = ['mse','accuracy',ccc, 'mae'])
            
    return model


def CreateConv2DRegressor(shape, output_neurons,learning_rate, optimizer,kernel,initial_dimention, decreasing):
    
    model = Sequential()
    func = None
    if decreasing:
        func = operator.floordiv
    else:
        func = operator.mul
        
    model = Sequential()
    model.add(Conv2D(initial_dimention, kernel_size=kernel, activation='relu',input_shape=shape))
    model.add(BatchNormalization())
    next_dimention = func(initial_dimention,2)
    model.add(Conv2D(next_dimention, kernel_size=kernel,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    next_dimention = func(next_dimention,2)
    model.add(Conv2D(next_dimention, kernel_size=kernel, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='linear'))
    
    model = multi_gpu_model(model)
    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mse', metrics =  ['mse','accuracy',ccc,'mae'])
    
    return model

def trainRegressor(model,x,y,epochs,batches,verb,calls,val_data):
    history = model.fit(x, y, batch_size = batches, epochs = epochs, verbose=verb, callbacks=calls,validation_data=val_data, shuffle=False)
    return model, history

def getDeepFeatures(featureDetector,x):
    getLastLayer = K.function([featureDetector.layers[0].input], [featureDetector.layers[-2].output])
    return getLastLayer([x])[0]

def RegressorPrediction(model,x):
    return model.predict(x)

def graphTrainingData(history, imagePath='train_graph.png', show = False):
    """
    This function cretes a graph where the x axis is the epoch or training iteration, and 
    in the y axis we represent the metric speficied (by default, accuracy) of the model
    at that point in the training process on the training and validation dataset.
    
    Params:
    history: training history object returned by Keras after training the model
    imagePath: path and name of the graph image to create
    metric: metric to graph or plot
    """
    fig = plt.figure()
    metrics = []
    for metric in history.history:
        if not metric.startswith("val_"):
            metrics.append(metric)
            print(metric)
    nrows = max(1, len(metrics)//3)
    ncols = min(3, len(metrics))
    print('Number of rows: ', nrows)
    print('Number of cols: ', ncols)
    for i in range(len(metrics)):
        print('Plotting metric ' + metrics[i])
        sbplt = fig.add_subplot(nrows, ncols, i+1)
        sbplt.plot(history.history[metrics[i]])
        sbplt.plot(history.history['val_' + metrics[i]])
        sbplt.set_title(metrics[i] + ' Training Graph')
        sbplt.set_ylabel(metrics[i])
        sbplt.set_xlabel('epoch')
    fig.legend(['train', 'validation'], loc = 'upper left')
    if (show):
        plt.show()
    else:
        mpl.use('PDF')
        plt.savefig(imagePath)
