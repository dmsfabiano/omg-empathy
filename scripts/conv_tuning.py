import file_operations as operations
from keras.optimizers import RMSprop, adam,adamax, Nadam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def getOptimizer(name,rate):
    if name is 'adamax':
        return adamax(lr=rate)
    elif name is 'adam':
        return adam(lr=rate)
    elif name is 'nadam':
        return Nadam(lr=rate)
    else:
        return RMSprop(lr=rate)
    
if __name__ == '__main__':
    
	kernels = [3,5,7,9, 11]
	optimizers = ['adam','RMSprop']
	rates = [0.01,0.001,0.0001,0.00001]
    
    
	x_train,y_train = operations.read_images_together()
	offset = int(len(x_train) * 0.3)
	x_train = x_train[0:offset]
	y_train = y_train[0:len(x_train)]
	x_test,y_test = operations.read_images_together(images_dir= '../data/faces/Validation/',y_dir='../data/Validation/Annotations/')

	rows, columns = x_train[0][0].shape[0], x_train[0][0].shape[0]
    
	
	for kernel in kernels:
		print('\t kernel size: ', str(kernel), '\n')
					
		model = Sequential()
		model.add(Conv2D(32, kernel_size=kernel, activation='relu',input_shape=(rows,columns,3)))
		model.add(Dropout(0.2))
		model.add(Conv2D(32, kernel_size=kernel,activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(Conv2D(64, kernel, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1,activation='linear'))

    
		for opt in optimizers:
			print('\t\t optimizer: ', opt, '\n')
			for rate in rates:
				print('\t\t\t learning rate:' , str(rate), '\n')
				
				OPT = getOptimizer(opt,rate)
				
				model.compile(loss='mean_squared_error', optimizer=OPT,metrics=['mae','accuracy'])
				model.fit(x_train, y_train, batch_size = 50, epochs = 100, verbose=0)
				scores = model.evaluate(x_test,y_test, verbose=0)[1]*100

				print('\t\t\t\t Accuracy with validation data:  {0} %'.format(scores))

