# write a graph for each subject with color for valence level in each story

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

training_annotations_path = '../data/Training/Annotations'
validation_annotations_path = '../data/Validation/Annotations'

def read_csv(csv_path):
    return pandas.read_csv(csv_path, index_col=False)['valence'].values

def subjectToStories(path):
    subjectToStoryPath = dict()
    for (paths, dirnames, filenames) in os.walk(path):
        for filename in filenames:
		if filename.endswith('.csv'):
			subject_number = '_'.join(filename.split('_')[0:2])
			if subject_number in subjectToStoryPath:
				subjectToStoryPath[subject_number].append(os.path.join(paths, filename)) 
				subjectToStoryPath[subject_number] = sorted(subjectToStoryPath[subject_number], key=lambda x: int(x.split('_')[3][0]))
			else:
				subjectToStoryPath[subject_number] = []
				subjectToStoryPath[subject_number].append(os.path.join(paths, filename))
    return sorted(subjectToStoryPath.items(), key=lambda x: int(x[0].split('_')[1]))

def graphSubjects(path, output_path):
    fig, axes = plt.subplots(10, 4, figsize=(150,100), sharey='col')
    for index, (subject, storyList) in enumerate(subjectToStories(path)):
        graphSubject(subject, storyList, index, fig, axes)
    fig.savefig(output_path)

def graphSubject(subject, storyList, index, fig, axes):
    colors = ['b', 'g', 'r', 'c']
    for story_index, story in enumerate(storyList):
        data = read_csv(story)
        axes[index][story_index].set_title(subject, fontsize=32)
        axes[index][story_index].set_xlabel('Story' + storyList[story_index].split('_')[3] + ' Time', fontsize=16)
        axes[index][story_index].set_ylabel('Valence', fontsize=16)
        axes[index][story_index].tick_params(labelsize=16, labelbottom=True, labelleft=True)
        axes[index][story_index].plot(data, colors[story_index])

if __name__ == '__main__':
   graphSubjects(training_annotations_path, '../data/Visualization/training_output') 
   graphSubjects(validation_annotations_path, '../data/Visualization/validation_output') 
