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
validation_story_list =  [1]

train_x, train_y = fp.read_raw_images_by_story(data_directory='../data/Spectrograms/Training/',
                                               subject_list=train_sbj_list,
                                               story_list = train_story_list,
                                               y_directory='../data/Training/Annotations/',
                                               subjectActorBoth=0,
                                               reduce_factor=1)

print('Training Loaded!')

test_x, test_y = fp.read_raw_images_by_story(data_directory='../data/Spectrograms/Validation/',
                                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                                             story_list = validation_story_list,
                                             y_directory='../data/Validation/Annotations/',
                                             subjectActorBoth=0,
                                             reduce_factor=1)

print('Testing Loaded!')

for lr in [0.0001]:
	for out_neuron in [128]:
		for kernel in [(3,3)]:
			for opt in ['adam']:
				for dim in [32]:
					for time in [5]:

						def create_time_series(x, time):
							y = [[] for _ in x]
							for i in range(1,len(x)+1):
								count = 1
								while count <= time:
									if i - count >= 0:
										y[i-1].append(x[i-count])
									else:
										y[i-1].append(x[0])
									count += 1
							return np.reshape(np.asarray(y), (len(x),time,256,256,3))

						ConvReg = net.CreateLSTMConv2DRegressor(shape=(time,256,256,3), output_neurons=out_neuron,
											learning_rate=lr, optimizer=opt,kernel=kernel,
											initial_dimension=dim, decreasing=False)

						for subject in range(len(train_x)):
							for story in range(len(train_story_list)):
								ConvReg, history = net.trainRegressor(ConvReg,
												create_time_series(np.asarray(train_x[subject][story]),time),
											     	np.asarray(train_y[subject][story]),
											     	batches=32, epochs=5,
											     	verb=1, calls=None, val_data=None)

						for subject in range(len(train_sbj_list)):
							story = 0
							args = [str(item) for item in [lr,out_neuron,kernel[0],opt,dim,time]]
							directory = os.path.dirname('../data/ModelOutput/clstm_{0}_{1}_{2}_{3}_{4}_{5}/'.format(*args))
							if not os.path.exists(directory):
								os.makedirs(directory)
							fp.writeModelOutput(output_directory,
									ConvReg.predict(create_time_series(np.asarray(new_test_x[subject][story]),time), batch_size=32),
									train_sbj_list[subject], validation_story_list[story])

'''
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
