# write a graph for each subject to compare the original and normalized audio, along with valence, in each story

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

original_training_audio_path = '../data/audio/CSV-Training'
normalized_training_audio_path = '../data/audio/CSV-Training-Normalized'
training_valence = '../data/Training/Annotations'

original_validation_audio_path = '../data/audio/CSV-Validation'
normalized_validation_audio_path = '../data/audio/CSV-Validation-Normalized'
validation_valence = '../data/Validation/Annotations'

def read_csv(csv_path):
    return pandas.read_csv(csv_path, index_col=False).values

def subjectToStories(path1, path2, path3):
    subjectToStoryPath = dict()
    
    for index, path in enumerate((path1, path2, path3)):
        for (paths, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    subject_number = '_'.join(filename.split('_')[0:2])
                    if subject_number in subjectToStoryPath:
                        subjectToStoryPath[subject_number][index].append(os.path.join(paths, filename)) 
                        subjectToStoryPath[subject_number][index] = sorted(subjectToStoryPath[subject_number][index], key=lambda x: int(x.split('_')[3][0]))
                    else:
                        subjectToStoryPath[subject_number] = [[],[],[]]
                        subjectToStoryPath[subject_number][index].append(os.path.join(paths, filename))
    return sorted(subjectToStoryPath.items(), key=lambda x: x[0])

def graphSubjects(original_path, normalized_path, valence_path, output_path):
    fig, axes = plt.subplots(10, 12, figsize=(150,150), sharey='col')
    for index, (subject, storyList) in enumerate(subjectToStories(original_path, normalized_path, valence_path)):
        graphSubject(subject, storyList, index, fig, axes)
    fig.savefig(output_path)

def graphSubject(subject, storyList, index, fig, axes):
    colors = ['b', 'g', 'r', 'c']
    for data_index, stories in enumerate(storyList):
        for story_index, story in enumerate(stories):
            data = read_csv(story)
            column = (len(storyList[0]) * data_index) + story_index
            if column <= 3:
                ylabel = 'Audio Original'
            elif column <= 7:
                ylabel = 'Audio Normalized'
            else:
                ylabel = 'Valence'
            axes[index][column].set_title(subject, fontsize=32)
            axes[index][column].set_xlabel('Time', fontsize=16)
            axes[index][column].set_ylabel(ylabel, fontsize=16)
            axes[index][column].tick_params(labelsize=16, labelbottom=True, labelleft=True)
            axes[index][column].plot(data, colors[story_index])

if __name__ == '__main__':
   graphSubjects(original_training_audio_path, normalized_training_audio_path, training_valence, '../data/Visualization/training_audio_comparison2')
   #graphSubjects(original_validation_audio_path, normalized_validation_audio_path, validation_valence, '../data/Visualization/validation_audio_comparison')
