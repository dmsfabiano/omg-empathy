import os
import pandas as pd
import numpy as np
from cv2 import imread

def read_images_together(images_dir= 'C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\faces\\Training\\',
                y_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\Training\\Annotations\\'):
    
    experiments = list(os.walk(images_dir))[0][1]
    
    x, y = [],[]
    for experiment in experiments:
        destination = images_dir + '\\' + experiment + '\\Subject\\'
        
        y_path = y_dir   + experiment.split('\\')[-1].split('.')[0] + '.csv'
        
        y_values = pd.read_csv(y_path, index_col=None)['valence'].values
        
        for file in os.listdir(destination):
            if file.endswith('.png'):
                frame_number = int(file.split('.')[0])
                
                valence = y_values[frame_number]
                
                x.append(imread(destination + file,1))
                y.append(valence)
                
    return np.asarray(x), np.asarray(y)
                
def read_images_subject(subject_list, story_list,
                        images_dir= 'C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\faces\\Training\\',
                        y_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\Training\\Annotations\\'):
    
    experiments = list(os.walk(images_dir))[0][1]
    
    x= [[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]
    y= [[[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]]
    
    for experiment in experiments:
        destination = images_dir + '\\' + experiment + '\\Subject\\'
        
        y_path = y_dir   + experiment.split('\\')[-1].split('.')[0] + '.csv'
        
        subject = int(experiment.split('\\')[-1].split('_')[1])
        story = int(experiment.split('\\')[-1].split('_')[-1].split('.')[0])
        
        y_values = pd.read_csv(y_path, index_col=None)['valence'].values
        
        for file in os.listdir(destination):
            if file.endswith('.png'):
                frame_number = int(file.split('.')[0])
                
                valence = y_values[frame_number]
                
                x[subject][story].append(imread(destination + file,1))
                y[subject][story].append(valence)
                
    return np.asarray(x), np.asarray(y)

x,y = read_images_together()