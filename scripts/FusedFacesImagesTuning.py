import sys
import os
import file_operations as fp
import numpy as np
import networks as net
from keras import callbacks
from sklearn.preprocessing import StandardScaler

            
def getData(subject_list,story_list, directory):
    x_separated,y_separated = fp.read_all_data(subject_list,story_list, directory)

    x = np.asarray([story for subject in x_separated for story in subject])
    y = np.asarray([story for subject in y_separated for story in subject])
    
    return x,y

# --------------------------------------- DATA PRE-PROCESSING -----------------------------------------------------
train_sbj_list = [1,2,3,4,5,6,7,8,9,10]
train_story_list = [2,4,5,8]
validation_story_list = [1]
test_story_list =  [3,6,7,9,10]

print('Starting!')
sys.stdout.flush()

train_subject_x, train_actor_x, train_y = fp.read_raw_images_by_story(data_directory='../data/OriginalImages/Training/',
                                               subject_list=train_sbj_list,
                                               story_list = train_story_list,
                                               y_directory='../data/Training/Annotations/',
                                               subjectActorBoth=2,
                                               reduce_factor=0.01)

train_audio_x, _ = fp.read_raw_images_by_story(data_directory='../data/spectrograms-1-sec/Training/',
                                               subject_list=train_sbj_list,
                                               story_list = train_story_list,
                                               y_directory='../data/Training/Annotations/',
                                               subjectActorBoth=0,
                                               reduce_factor=0.01)

print('Training Loaded!')
sys.stdout.flush()

val_subject_x, val_actor_x, val_y = fp.read_raw_images_by_story(data_directory='../data/OriginalImages/Validation/',
                                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                                             story_list = validation_story_list,
                                             y_directory='../data/Validation/Annotations/',
                                             subjectActorBoth=2,
                                             reduce_factor=0.01)

val_audio_x, _ = fp.read_raw_images_by_story(data_directory='../data/spectrograms-1-sec/Validation/',
                                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                                             story_list = validation_story_list,
                                             y_directory='../data/Validation/Annotations/',
                                             subjectActorBoth=0,
                                             reduce_factor=0.01)

print('Validation Loaded!')
sys.stdout.flush()

test_subject_x, test_actor_x, _ = fp.read_raw_images_by_story(data_directory='../data/OriginalImages/Testing/',
                                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                                             story_list = test_story_list,
                                             y_directory=None,
                                             subjectActorBoth=2,
                                             reduce_factor=1)

test_audio_x, _ = fp.read_raw_images_by_story(data_directory='../data/spectrograms-1-sec/Testing/',
                                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                                             story_list = test_story_list,
                                             y_directory=None,
                                             subjectActorBoth=0,
                                             reduce_factor=1)

print('Testing Loaded!')
sys.stdout.flush()

train_x = train_subject_x
val_x = val_subject_x
test_x = test_subject_x
'''
train_x = []
for i in range(len(train_y)):
	train_x.append(fp.fuseImages(train_subject_x[i], train_actor_x[i], (5,5), (5,5)))
train_x = np.asarray(train_x)
train_subject_x = None
train_actor_x = None

print('Fusion 1!')
sys.stdout.flush()

val_x = []
for i in range(len(val_y)):
	val_x.append(fp.fuseImages(val_subject_x[i], val_actor_x[i], (5,5), (5,5)))
val_x = np.asarray(val_x)
val_subject_x = None
val_actor_x = None

print('Fusion 2!')
sys.stdout.flush()

test_x = []
for i in range(len(test_subject_x)):
	test_x.append(fp.fuseImages(test_subject_x[i], test_actor_x[i], (5,5), (5,5)))
test_x = np.asarray(test_x)
test_subject_x = None
test_actor_x = None

print('Fusion Done!')
sys.stdout.flush()
'''

for lr in [0.0001]:
	for out_neuron in [128]:
		for kernel in [(3,3)]:
			for opt in ['adam']:
				for dim in [32]:
					for time in [25]:
						
						regressor = net.CreateConv2DRegressor((256,256,3),100,0.0001,'adam',3,64,False)

						for subject in range(len(train_x)):
							for story in range(len(train_story_list)):
								regressor.fit(	x = np.asarray(train_x[subject][story]),
										y = np.asarray(train_y[subject][story]),
										epochs=5,
										batch_size=128,
										validation_data=(np.asarray(val_x[subject][0]), np.asarray(val_y[subject][0])))
						
						for subject in range(len(train_sbj_list)):
							for story in range(len(test_story_list)):
								args = [str(item) for item in [lr,out_neuron,kernel[0],opt,dim,time]]
								directory = os.path.dirname('../data/ModelOutput/clstm_{0}_{1}_{2}_{3}_{4}_{5}/'.format(*args))
								if not os.path.exists(directory):
									os.makedirs(directory)
								fp.writeModelOutput(directory,
											regressor.predict(np.asarray(test_x[subject][story]), batch_size=128),
											train_sbj_list[subject], test_story_list[story])



'''
						def create_time_series(x, time, image=True):
							y = [[] for _ in range((len(x)//time) + 1)]
							j = 0
							for i in range(0,len(x),time):
								if i + time < len(x):
									y[j] = np.asarray(x[i:i+time])
									j += 1
								else:
									y[len(y)-1] = y[len(y)-2]
							if image:
								return np.reshape(np.asarray(y), (len(y),time,256,256,3))
							else:
								return np.asarray(y)

						regressor = net.finalNetwork(8, 32, time)
						regressor.compile(optimizer=net.getOptimizer(opt,lr), loss='mse')

						for subject in range(len(train_x)):
							for story in range(len(train_story_list)):
								train_y[subject][story] = create_time_series(train_y[subject][story], time, False)
								val_y[subject][story] = create_time_series(val_y[subject][story], time, False)
								regressor.fit(	x = [create_time_series(np.asarray(train_x[subject][story]),time), np.asarray(train_audio_x[subject][story])],
										y = [np.asarray(train_y[subject][story]), np.asarray(train_y[subject][story])],
										epochs=5,
										batch_size=16,
										validation_data=([create_time_series(np.asarray(val_x[subject][0]), time), np.asarray(val_audio_x[subject][story])], [val_y, val_y]))

						for subject in range(len(train_sbj_list)):
							for story in range(len(test_story_list)):
								args = [str(item) for item in [lr,out_neuron,kernel[0],opt,dim,time]]
								directory = os.path.dirname('../data/ModelOutput/final_{0}_{1}_{2}_{3}_{4}_{5}/'.format(*args))
								if not os.path.exists(directory):
									os.makedirs(directory)
								fp.writeModelOutput(directory,
											regressor.predict([create_time_series(np.asarray(test_x[subject][story]),time), np.asarray(test_audio_x[subject][story])], batch_size=16),
											train_sbj_list[subject], test_story_list[story])

train_split = 0.9

# DDNet Structure
train_subject_x, train_actor_x, train_y = fp.read_raw_images(subject_list=train_sbj_list,subjectActorBoth=2, reduce_factor=1.)

#shuffled_data = net.unison_shuffle([train_subject_x,train_actor_x,train_y])
#train_subject_x, train_actor_x, train_y = shuffled_data[0], shuffled_data[1], shuffled_data[2]

test_subject_x, test_actor_x, test_y= fp.read_raw_images(data_directory='C:/Users/Diego Fabiano/Research/Data/OMG_RAW/Validation/',
                                                 subject_list=train_sbj_list, 
                                                 story_list = validation_story_list, 
                                                 y_directory='C:/Users/Diego Fabiano/Documents/OMG-FG-Challenge/data/Validation/Annotations/',reduce_factor=1.,subjectActorBoth=2)
#shuffled_data = net.unison_shuffle([test_subject_x,test_actor_x,test_y])
#test_subject_x, test_actor_x, test_y = shuffled_data[0], shuffled_data[1], shuffled_data[2]

# Get Subject validation data
val_subj_x = train_subject_x[int(len(train_subject_x)*train_split):-1]
train_subject_x = train_subject_x[0:int(len(train_subject_x)*train_split)]

# Get validation ground truth
val_y = train_y[int(len(train_y)*train_split):-1]
train_y = train_y[0:int(len(train_y)*train_split)]

# Get validation actor data
val_actor_x = train_actor_x[int(len(train_actor_x)*train_split):-1]
train_actor_x = train_actor_x[0:int(len(train_subject_x)*train_split)]


# Define parameters
outNeurons = 100
out_dim = 32
mode = False # This increases or decreases the out_dim by */ 2 every layer


# ---------------------------------------- NETWORK -------------------------------------------------------

ConvReg = net.CreateConv2DRegressor(shape=(128,128,3), output_neurons=outNeurons,learning_rate=0.0001, optimizer='sgd',kernel=3,initial_dimention=out_dim, decreasing=mode)

reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,mode='min',min_lr=0.0000001,min_delta=0.001)
ConvReg, history = net.trainRegressor(ConvReg,train_subject_x,train_y,10,250,1,[reduceLR],(val_subj_x,val_y))

loss = history.history['loss']
ccc = history.history['ccc']
mse = history.history['mean_squared_error']
mae = history.history['mean_absolute_error']

print('Training statistics')
print('Average loss: {0} +/- {1}, ccc: {2} +/- {3}'.format(np.mean(loss),np.std(loss),np.mean(ccc),np.std(ccc)))
print('Average mse: {0} +/- {1}, mae: {2} +/- {3}'.format(np.mean(mse),np.std(mse),np.mean(mae),np.std(mae)))


metric = ConvReg.evaluate(test_subject_x,test_y)

mse = metric[1]
accuracy = metric[2] * 100
CCC = metric[3]
mae = metric[4]

print('Testing statistics')

print('Faces images statistics on testing: {0} mse, {1}% accuracy, and {2} ccc \n with parameters: {3} starting dimentions, {4} pre last layer neurons, and mode: {5}, MAE: {6}'.format(
        mse,accuracy,CCC,out_dim,outNeurons,mode,mae))

values = ConvReg.predict(test_subject_x)
'''
