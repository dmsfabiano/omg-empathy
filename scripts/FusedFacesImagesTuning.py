import file_operations as fp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import operator
from keras.optimizers import RMSprop, adam,adamax, Nadam
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

    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = ccc_loss, metrics = ['mse','accuracy',ccc, 'mae', pearsonr,ccc_loss])
            
    return model


def CreateConv2DRegressor(shape, output_neurons,learning_rate, optimizer,kernel,initial_dimention, decreasing):
    
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
    model.compile(optimizer = getOptimizer(optimizer,learning_rate), loss = ccc_loss, metrics =  ['mse','accuracy',ccc, 'mae',ccc_loss])

    return model

def trainRegressor(model,x,y,epochs,batches,verb,calls,val_data):
    history = model.fit(x, y, batch_size = batches, epochs = epochs, verbose=verb, callbacks=calls,validation_data=val_data, shuffle=False)
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

_,y_train = getData(train_sbj_list,train_story_list,'../data/results/Training/')
_,y_validation = getData(train_sbj_list,validation_story_list,'../data/results/Validation/')

temp = []
for video in y_train:
	temp.extend(video)
y_train = temp.copy()

temp = []
for video in y_validation:
	temp.extend(video)
y_test = temp.copy()

# DDNet Structure
train_images = fp.read_landmark_images('../data/Images/Training/')
test_images = fp.read_landmark_images('../data/Images/Validation/')

train_images = np.reshape(train_images, (train_images.shape[0],128,128,1))
test_images = np.reshape(validation_images, (validation_images.shape[0],128,128,1))

train_images,x_validation,y_train,y_validation = train_test_split(train_images,y_train,test_size=0.2,random_state=1)

output_dim = [16,32,64,128]
dec = [True,False]
output_neurons = [10,100,1000,2500,5000,10000]

for out_dim in output_dim:
    for mode in dec:
        for outNeurons in output_neurons:
            
            ConvReg = CreateConv2DRegressor(shape=(128,128,1), output_neurons=outNeurons,learning_rate=0.00001, optimizer='adam',kernel=3,initial_dimention=out_dim, decreasing=mode)

            reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,verbose=1,mode='min',min_lr=0.0000001,min_delta=0.001)
            earlyStop = callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=10)
            ConvReg,history = trainRegressor(ConvReg,np.asarray(train_images),np.asarray(y_train),epochs=100,batches=250,calls=[reduceLR,earlyStop],verb=2,val_data=(np.asarray(x_validation),np.asarray(y_validation)))

            loss_values = history.history['loss']
            ccc_values = history.history['ccc']
            
            print('Average loss: {0} +/- {1}, ccc: {2} +/- {3}'.format(np.mean(loss_values),np.std(loss_values),np.mean(ccc_values),np.std(ccc_values)))

            metric = ConvReg.evaluate(np.asarray(test_images),y_test, verbose=0)
            mse = metric[1]
            accuracy = metric[2] * 100
            CCC = metric[3]
            mae = metric[4]
            r = metric[5]
        
            print('Faces images statistics on testing: {0} mse, {1}% accuracy, and {2} ccc \n with parameters: {3} output dimentions, {4} dim neurons, and mode: {5}, MAE: {6}, R:{7}'.format(
                    mse,accuracy,CCC,out_dim,outNeurons,mode,mae,r))
