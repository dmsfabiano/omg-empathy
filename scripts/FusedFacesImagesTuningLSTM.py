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
train_sbj_list = [2]
train_story_list = [2]
validation_story_list =  [1]
timesteps = 20
train_split = 0.9

# DDNet Structure
train_subject_x, train_actor_x, train_y = fp.read_raw_images_timesteps(data_directory='/home/raulestrada/Projects/omg-empathy/data/Images/Training/',
        y_directory='/home/raulestrada/Projects/omg-empathy/data/Training/Annotations/',
        subject_list=train_sbj_list,subjectActorBoth=2, reduce_factor=.25,
        story_list = train_story_list,
        timesteps = timesteps)
print("Train images loaded")


test_subject_x, test_actor_x, test_y= fp.read_raw_images_timesteps(data_directory='/home/raulestrada/Projects/omg-empathy/data/Images/Validation/',
                                                 subject_list=train_sbj_list, 
                                                 story_list = validation_story_list, 
                                                 y_directory='/home/raulestrada/Projects/omg-empathy/data/Validation/Annotations/',reduce_factor=.25,subjectActorBoth=2,
                                                 timesteps = timesteps)
print("Test images loaded")
print("Train subject: ", train_subject_x.shape)
print("Train actor: ", train_actor_x.shape)
print("Train y: ", train_y.shape)
print("Test subject: ", test_subject_x.shape)
print("Test actor: ", test_actor_x.shape)
print("Test y: ", test_y.shape)
print("\n\n\n\n")


# Get Subject validation data
val_subj_x = train_subject_x[int(len(train_subject_x)*train_split):-1]
train_subject_x = train_subject_x[0:int(len(train_subject_x)*train_split)]

# Get validation ground truth
val_y = train_y[int(len(train_y)*train_split):-1]
train_y = train_y[0:int(len(train_y)*train_split)]

# Get validation actor data
val_actor_x = train_actor_x[int(len(train_actor_x)*train_split):-1]
train_actor_x = train_actor_x[0:int(len(train_subject_x)*train_split)]


print("Test images loaded")
print("Train subject: ", train_subject_x.shape)
print("Train actor: ", train_actor_x.shape)
print("Train y: ", train_y.shape)
print("Validation subject: ", val_subj_x.shape)
print("Validation actor: ", val_actor_x.shape)
print("Validation y: ", val_y.shape)
print("Test subject: ", test_subject_x.shape)
print("Test actor: ", test_actor_x.shape)
print("Test y: ", test_y.shape)
print("\n\n\n\n")


# Define parameters
outNeurons = 100
out_dim = 32
mode = False


# ---------------------------------------- NETWORK -------------------------------------------------------

model = net.CreateLSTMConv2DRegressor(shape=(timesteps,128,128,3), output_neurons=outNeurons,learning_rate=0.0001, optimizer='adam',kernel=3,initial_dimension=out_dim, decreasing=mode)

reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,mode='min',min_lr=0.0000001,min_delta=0.001)
model, history = net.trainRegressor(model,train_subject_x,train_y,10,16,1,[reduceLR],(val_subj_x,val_y))

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


