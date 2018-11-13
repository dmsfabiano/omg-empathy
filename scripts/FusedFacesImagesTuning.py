import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import operator
from keras.optimizers import RMSprop, adam,adamax, Nadam
from keras import backend as K


def ccc(y_true, y_pred):
    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = 1.0 / (N - 1.0 + K.epsilon()) * K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)
    
    # condordance correlation coefficient
    c = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return c

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

    model.add(Dense(units = output_neurons, kernel_initializer = 'uniform',activation = 'relu'))
    model.add(Dense(1,activation='linear'))

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = 'mean_squared_error', metrics = ['accuracy'])
            
    return model


def CreateConv2DRegressor(shape, output_neurons,learning_rate, optimizer,kernel,initial_dimention, decreasing):
    
    func = None
    if decreasing:
        func = operator.floordiv
    else:
        func = operator.mul
        
    model = Sequential()
    model.add(Conv2D(initial_dimention, kernel_size=kernel, activation='relu',input_shape=shape))
    model.add(Dropout(0.2))
    next_dimention = func(initial_dimention,2)
    model.add(Conv2D(next_dimention, kernel_size=kernel,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    next_dimention = func(next_dimention,2)
    model.add(Conv2D(next_dimention, kernel, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Flatten())
    model.add(Dense(output_neurons, activation='relu'))
    #model.add(Dropout(0.5))
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
		train_landmarks_subject.append(np.asarray(frame[0:136]))
		train_landmarks_actor.append(np.asarray(frame[136:272]))
		train_audio.append(np.asarray(frame[272:5272]))

for video in x_validation:
	for frame in video:
		validation_landmarks_subject.append(np.asarray(frame[0:136]))
		validation_landmarks_actor.append(np.asarray(frame[136:272]))
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
train_images = fp.read_landmark_images('../data/Images/Training/')
validation_images = fp.read_landmark_images('../data/Images/Validation/')

kernels = [3,5,7,9, 11]
output_dim = [16,32,64]
dec = [True,False]
optimizers = ['adam','RMSprop']
rates = [0.01,0.001,0.0001,0.00001]
output_neurons = [10,100,1000]

for ker in kernels:
    for out_dim in output_dim:
        for mode in dec:
            for outNeurons in output_neurons:
                for opt in optimizers:
                    for rate in rates:
                
                        ConvReg = CreateConv2DRegressor(shape=(128,128,1), output_neurons=outNeurons,learning_rate=rate, optimizer=opt,kernel=ker,initial_dimention=out_dim, decreasing=mode)
                        ConvReg = trainRegressor(ConvReg,np.asarray(train_images),y_train,epochs=100,batches=250)

                        metric = ConvReg.evaluate(np.asarray(validation_images),y_validation, verbose=1)
                        mse = metric[1]
                        accuracy = metric[2] * 100
                        CCC = metric[3]
                    
                        dim_reduct = 'increasing'
                        if mode:
                            dim_reduct = 'decreasing'
                            
                        print('Faces images statistics: {0} mse, {1}% accuracy, and {2} ccc \n with parameters: {3} optimizer, rate: {4}, {5} kernel size, {6} output dimentions, {7} dim neurons, and mode: {8}'.format(
                                mse,accuracy,CCC,opt,rate,ker,out_dim,outNeurons.dim_reduct))
