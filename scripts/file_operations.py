import os
import pandas as pd
import numpy as np
from cv2 import imread
from scipy.io.wavfile import read

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

def read_audio_together(audio_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\audio\\Training\\',
                        y_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\Training\\Annotations\\'):
    
    x,y,z = [],[],[]
    
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            
            audio = np.asarray(read(audio_dir + file)[1])
            normalized_audio = []
            
            y_path = y_dir + file.split('.')[0] + '.csv'
            
            y_values = pd.read_csv(y_path, index_col=None)['valence'].values
            
            normalization_ratio = int(len(audio)/len(y_values))
            
            for i in range(0,len(audio), normalization_ratio):
                normalized_audio.append(int(np.average(audio[i:i+normalization_ratio])))
                
            
            # fix offset if wished
            offset = len(normalized_audio) - len(y_values)
            average = np.average(y_values)
            y_values = np.append(y_values,np.full((1,offset),average)[0])
            
            x.append(normalized_audio)
            y.append(y_values)
            z.append(file.split('.')[0])
            
    return np.asarray(x), np.asarray(y), np.asarray(z)

# NOT FUNCTIONAL YET, HAS TO MATCH REQUIREMENTS OF CLUSTERED CONTAINER
def read_audio_subject(subject_list, story_list, audio_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\audio\\Training\\',
                        y_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\Training\\Annotations\\'):
    
    x= [[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]
    y= [[[ [] for j in range(0,len(story_list))]for i in range(0,len(subject_list))]]
    
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            
            audio = np.asarray(read(audio_dir + file)[1])
            
            y_path = y_dir + file.split('.')[0] + '.csv'
            
            y_values = pd.read_csv(y_path, index_col=None)['valence'].values
            
            subject = int(file.split('.')[0].split('_')[1])
            story = int(file.split('.')[0].split('_')[-1])
            
            x[subject][story].append(audio)
            y[subject][story].append(y_values)
            
    return np.asarray(x), np.asarray(y)


def normalize_audio(size=None,container=None, 
                    audio_dir='C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\audio\\Training\\'):
    
    if size is None and container is None:
        x,y,z = read_audio_together()
        
        min_frame_count = 10000000
        for story in x:
            if len(story) < min_frame_count:
                min_frame_count = len(story)
                
                
        audio_destination = '\\'.join(audio_dir.split('\\')[0:-2]) + '\\Normalized\\' + audio_dir.split('\\')[-2] + '\\'
        
        valence_destination = 'C:\\Users\\Diego Fabiano\\Documents\\OMG-FG-Challenge\\data\\' + audio_dir.split('\\')[-2] + '\\Annotations\\Normalized\\' 
        
        for i in range(0,len(x)):
        
            audio = x[i]
            valence = y[i]
            file_name = z[i]
            
            normalized_audio, normalized_valence = [],[]
            
            normalization_ratio = int(len(audio)/min_frame_count)
            
            for index in range(0,len(audio),normalization_ratio):
                
                normalized_audio.append(int(np.average(audio[index:index+normalization_ratio])))
                normalized_valence.append(np.average(valence[index:index+normalization_ratio]))

            normalized_audio = np.asarray(normalized_audio)
            normalized_valence = np.asarray(normalized_valence)
            
            np.savetxt(audio_destination + file_name + '.csv',normalized_audio, delimiter='\n')
            np.savetxt(valence_destination + file_name + '.csv',normalized_valence, delimiter='\n')
          

            


                
            
                


                
                
        

            
            
    
    