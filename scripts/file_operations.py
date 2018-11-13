import os
import pandas as pd
import numpy as np
from cv2 import imread, resize
from scipy.io.wavfile import read

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
                
            x.append(imread(destination,1))
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
                
            x[subject][story].append(imread(destination,1))
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
        
def read_landmark_images(data_directory='../data/Images/'):
	frameNumber = {}
	for root, path, files in os.walk(data_directory):
		for file in files:
			if file.endswith('.png'):
				frame = file.split('/')[-1].split('_')[1]
				frameNumber[frame] = imread(os.path.join(root,file))
	orderedByFrame = sorted(frameNumber.items(), key=lambda k: k[0])
	return np.asarray(orderedByFrame.values())
