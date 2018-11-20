import file_operations as fp
import numpy as np
import networks as net
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import callbacks

            
def getData(subject_list,story_list, directory):
    x_separated,y_separated = fp.read_all_data(subject_list,story_list, directory)

    x = np.asarray([story for subject in x_separated for story in subject])
    y = np.asarray([story for subject in y_separated for story in subject])
    
    return x,y

# --------------------------------------- DATA PRE-PROCESSING -----------------------------------------------------
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
test_images = np.reshape(test_images, (test_images.shape[0],128,128,1))

train_images,x_validation,y_train,y_validation = train_test_split(train_images,y_train,test_size=0.2)
test_images, y_test = shuffle(test_images,y_test)

# Define parameters
outNeurons = 100
out_dim = 16
mode = False # This increases or decreases the out_dim by */ 2 every lager


# ---------------------------------------- NETWORK -------------------------------------------------------

ConvReg = net.CreateConv2DRegressor(shape=(128,128,1), output_neurons=outNeurons,learning_rate=0.00001, optimizer='adam',kernel=3,initial_dimention=out_dim, decreasing=mode)

reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,verbose=1,mode='min',min_lr=0.0000001,min_delta=0.001)
earlyStop = callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=10)
ConvReg,history = net.trainRegressor(ConvReg,np.asarray(train_images),np.asarray(y_train),epochs=5,batches=25,calls=[reduceLR,earlyStop],verb=2,val_data=(np.asarray(x_validation),np.asarray(y_validation)))

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
