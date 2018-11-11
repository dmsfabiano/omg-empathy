import os
import numpy as np
import sys
from scipy.io import wavfile

def video_to_all_relevant_files():
    relevant_files = {}
    end_values = ['.txt', '.txt.', '.wav', '.csv']

    for i in range(0,4):
        if i == 1:
            continue
        for root, dirs, files in os.walk(sys.argv[i+1 if i < 1 else i]):
            for f in files:
                if f.endswith(end_values[i]):
                    path = os.path.join(root,f)
                    if i is 0:
                        subject = True if path.split('/')[-1].split('.')[2] == 'landmarks' else False
                    video = path.split('/')[-1].split('.')[0]
                    try:
                        if subject:
                            relevant_files[video][i].append(path)
                        else:
                            relevant_files[video][i+1 if i < 1 else i].append(path)
                    except KeyError:
                        relevant_files[video] = ([],[],[],[])
                        if subject:
                            relevant_files[video][i].append(path)
                        else:
                            relevant_files[video][i+1 if i < 1 else i].append(path)
    return relevant_files

def getLandmarks(paths):
    path = paths[0]
    of = open(path, 'r')
    landmark_lines = [','.join(item.strip().split(' ')) for item in of.readlines()]
    return landmark_lines

def getAudio(paths, frames):
    audio = []
    path = paths[0]
    fs, data = wavfile.read(path)
    for i in range(0,frames):
        audio.append(','.join([str(item) for item in data[i*5000:(i*5000)+5000]]))
    return audio

def getValence(paths):
    path = paths[0]
    of = open(path, 'r')
    valence_lines = [item.strip() for item in of.readlines()]
    return valence_lines[1:]

def file_for_each_video(videoToPathsDict):
    for key,value in videoToPathsDict.items():
        video_file = open(sys.argv[4] + key + '.out', 'w')
        landmarks_subject = getLandmarks(value[0])
        landmarks_actor = getLandmarks(value[1])
        valence = getValence(value[3])
        audio = getAudio(value[2], len(valence))
        for data in zip(landmarks_subject, landmarks_actor, audio, valence):
            video_file.write(data[0])
            video_file.write(',')
            video_file.write(data[1])
            video_file.write(',')
            video_file.write(data[2])
            video_file.write(',')
            video_file.write(data[3])
            video_file.write('\n')

if(len(sys.argv) == 5):
    data = video_to_all_relevant_files()
    file_for_each_video(data)
else:
    print('Error: Needs 4 command line argument!')
