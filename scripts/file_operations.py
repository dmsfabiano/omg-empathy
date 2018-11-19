import os
import pandas as pd
import numpy as np
import cv2
from scipy.io.wavfile import read
from multiprocessing import Pool

def read_images_together(images_dir= '../data/faces/Training/',
                y_dir='../data/Training/Annotations/'):
    
    experiments = list(os.walk(images_dir))[0][2]

    x, y = [],[]
    for experiment in experiments:
        if experiment.endswith('.png'):
            destination = images_dir + experiment
        
            y_path = y_dir + experiment.split('.')[0] + '.csv'
        
            y_values = pd.read_csv(y_path, index_col=None)['valence'].values
            
            frame_number = int(experiment.split('.')[1].split('e')[-1]) - 1
            valence = y_values[frame_number]
                
            x.append(cv2.imread(destination,1))
            y.append(valence)
                
    return np.asarray(x), np.asarray(y)
                
def read_images_subject(subject_list, story_list,
                        images_dir= '../data/faces/Training/',
                        y_dir='../data/Training/Annotations/'):
    
    experiments = list(os.walk(images_dir))[0][2]
    
    x= [[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]
    y= [[[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]]
    
    for experiment in experiments:
        if experiment.endswith('.png'):
            destination = images_dir + experiment
        
            y_path = y_dir + experiment.split('.')[0] + '.csv'
        
            subject = int(experiment.split('_')[1])
            story = int(experiment.split('_')[-1].split('.')[0])
        
            y_values = pd.read_csv(y_path, index_col=None)['valence'].values
        
            frame_number = int(experiment.split('.')[1].split('e')[-1]) - 1
            valence = y_values[frame_number]
                
            x[subject][story].append(cv2.imread(destination,1))
            y[subject][story].append(valence)
                
    return np.asarray(x), np.asarray(y)
          
def read_all_data(subject_list,story_list, data_directory = '../data/results/'):
    
    import os

    
    file_list = [file for file in os.listdir(data_directory) if file.endswith('.out')]
    
    data_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    y_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    
    for file in file_list:
       f = data_directory + file
       data = pd.read_csv(f,header=None,index_col=None)
       
       subject = int(f.split('/')[-1].split('_')[1])
       story = int(f.split('/')[-1].split('_')[-1][0])
       
       data_container[subject_list.index(subject)][story_list.index(story)] = data.iloc[:,0:5272].values
       y_container[subject_list.index(subject)][story_list.index(story)] = data.iloc[:,-1].values
      
    return np.asarray(data_container),np.asarray(y_container)
	
def readImage(path):
	return (cv2.imread(path[0],cv2.IMREAD_GRAYSCALE) / 255, path[1])

def readImage_rgb(path):
	return (cv2.imread(path[0]), path[1])
        
def read_landmark_images(data_directory='../data/Images/'):
	file_list = []
	for root, path, files in os.walk(data_directory):
		for file in files:
			if file.endswith('.png'):
				frame = file.split('/')[-1].split('_')[1]
				file_list.append((os.path.join(root,file),frame))
	
	with Pool() as p:
		images = p.map(readImage, file_list)
		return np.asarray([items[0] for items in sorted(images, key=lambda x: x[1])])

def read_raw_images(data_directory,subject_list, story_list, y_directory):
    data_directory = ''
    subject_list = [1,2,3,4,5,6,7,8,9,10]
    story_list = [2,4,5,8]
    
    file_list = [file for file in os.listdir(data_directory) if file.endswith('.png')]
    y_file_list = [file for file in os.listdir(y_directory) if file.endswith('.png')]
    
    subject_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    actor_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    
    # Get Images
    for file in file_list:
        f = data_directory + file
        image = cv2.imread(f)
        subject = int(f.split('/')[-1].split('_')[1])
        story = int(f.split('/')[-1].split('_')[3])
        frame = int(f.split('/')[-1].split('_')[5])
        
        person = f.split('/')[-1].split('_')[-1].split('.')[0]
        
        if person is 'actor':    
            actor_container[subject][story].append((image,frame))
        else:
            subject_container[subject][story].append((image,frame))

    # Arrange images in right order
    for subject in range(0,len(subject_list)):
        for story in range(0,len(story)):
            subject_container = sorted(subject_container[subject][story], key=lambda x: x[1])
            actor_container = sorted(actor_container[subject][story], key=lambda x: x[1])
    
    
    # Get valence levels
    y_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]    
    for file in y_file_list:
        f = y_directory + file
        subject = int(f.split('/')[-1].split('_')[1])
        story = int(f.split('/')[-1].split('_')[-1].split('.')[0])
        y_container[subject][story] = pd.read_csv(f, index_col=None)['valence'].values
        
    subject_x,actor_x,y = [], [], []
    # Order everything in one big container
    for subject in range(0,len(subject_list)):
        for story in range(0,len(story)):
            for index in range(0,len(actor_container[subject][story])):
                subject_x.append(subject_container[subject][story][index][0])
                actor_x.append(actor_container[subject][story][index][0])
                y.append(y_container[subject][story][index])                
                
    # Scale Images
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy =False)
    subject_x = scaler.fit_transform(subject_x)
    actor_x = scaler.fit_transform(actor_x)
    
    return subject_x,actor_x, y