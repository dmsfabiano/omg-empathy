import os
import pandas as pd
import numpy as np
import cv2
from scipy.io.wavfile import read
from multiprocessing import Pool
import slidingwindow

def writeModelOutput(path, y_pred, y_true, subject, story):
    file_path = path + 'Subject_' + str(subject) + '_Story_' + '1' + '.csv'
    with open(file_path, 'w') as file:
        file.write('valence\n')
        for val in y_pred:
            file.write(str(val[0]) + '\n')

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

def read_raw_image_paths(data_directory='C:/Users/Diego Fabiano/Research/Data/OMG_RAW/Training/',
                         subject_list=[1,2,3,4,5,6,7,8,9,10], 
                         story_list = [2,4,5,8], 
                         y_directory='C:/Users/Diego Fabiano/Documents/OMG-FG-Challenge/data/Training/Annotations/'):
    
    file_list = [file for file in os.listdir(data_directory) if file.endswith('.png')]
    y_file_list = [file for file in os.listdir(y_directory) if file.endswith('.csv')]
    
    subject_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    actor_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    
    # Get in order paths
    for file in file_list:
        f = data_directory + file
        # image = cv2.imread(f)
        subject = int(f.split('/')[-1].split('_')[1])
        story = int(f.split('/')[-1].split('_')[3])
        
        
        person = f.split('/')[-1].split('_')[-1].split('.')[0]
        
        if person == 'actor':    
            frame = int(f.split('/')[-1].split('_')[5])
            actor_container[subject_list.index(subject)][story_list.index(story)].append((f,frame))
        else:
            frame = int(f.split('/')[-1].split('_')[-1].split('.')[0])
            subject_container[subject_list.index(subject)][story_list.index(story)].append((f,frame))

    # Arrange paths in right order
    for subject in range(0,len(subject_list)):
        for story in range(0,len(story_list)):
            subject_container[subject][story] = sorted(subject_container[subject][story], key=lambda x: x[1])
            actor_container[subject][story] = sorted(actor_container[subject][story], key=lambda x: x[1])
    
    # Get valence levels
    y_container = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]    
    for file in y_file_list:
        f = y_directory + file
        subject = int(f.split('/')[-1].split('_')[1])
        story = int(f.split('/')[-1].split('_')[-1].split('.')[0])
        y_container[subject_list.index(subject)][story_list.index(story)] = pd.read_csv(f, index_col=None)['valence'].values
    return subject_container,actor_container,y_container

def read_raw_images(data_directory='C:/Users/Diego Fabiano/Research/Data/OMG_RAW/Training/',
                    subject_list=[1,2,3,4,5,6,7,8,9,10],
                    story_list = [2,4,5,8],
                    y_directory='C:/Users/Diego Fabiano/Documents/OMG-FG-Challenge/data/Training/Annotations/',
                    subjectActorBoth=0,
                    reduce_factor=0.5):
    
	# subjectActorBoth: 0 for subject only, 1 for actor only, 2 for both
	subject_container,actor_container,y_container = read_raw_image_paths(data_directory,subject_list, story_list,y_directory)
    
	sbj_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
	actor_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]

	for subject in range(0,len(subject_list)):
		for story in range(0,len(story_list)): 
			subject_container[subject][story] = subject_container[subject][story][0:int(len(subject_container[subject][story])*reduce_factor)]
			actor_container[subject][story] = actor_container[subject][story][0:int(len(actor_container[subject][story])*reduce_factor)]
           
			for k in range(0,len(subject_container[subject][story])):
				if subjectActorBoth >= 1:
					actor_images[subject][story].append(cv2.resize(cv2.imread(actor_container[subject][story][k][0]),(128,128)))
				if (subjectActorBoth == 0 or subjectActorBoth == 2):    
					sbj_images[subject][story].append(cv2.resize(cv2.imread(subject_container[subject][story][k][0]),(128,128)))

    # can easily be changed here to return proper container that can be used for temporal features
	subject_x,actor_x,y = [], [], []
    # Order everything in one big container
	for subject in range(0,len(subject_list)):
		for story in range(0,len(story_list)):
			for index in range(0,len(actor_images[subject][story])):
				if subjectActorBoth == 0 or subjectActorBoth == 2:
					subject_x.append(sbj_images[subject][story][index])
				if subjectActorBoth >= 1:
					actor_x.append(actor_images[subject][story][index])
				y.append(y_container[subject][story][index])                
    
	subject_container, actor_container, y_container = None,None,None
	if subjectActorBoth == 2:
		return np.asarray(subject_x), np.asarray(actor_x), np.asarray(y)
	if subjectActorBoth == 0:
		return np.asarray(subject_x), np.asarray(y)
	if subjectActorBoth == 1:
		return np.asarray(actor_x), np.asarray(y)

def read_raw_images_by_story(data_directory='C:/Users/Diego Fabiano/Research/Data/OMG_RAW/Training/',
                             subject_list=[1,2,3,4,5,6,7,8,9,10],
                             story_list = [2,4,5,8],
                             y_directory='C:/Users/Diego Fabiano/Documents/OMG-FG-Challenge/data/Training/Annotations/',
                             subjectActorBoth=0,
                             reduce_factor=0.5):
    
	# subjectActorBoth: 0 for subject only, 1 for actor only, 2 for both
    subject_container,actor_container,y_container = read_raw_image_paths(data_directory,subject_list, story_list,y_directory)
    
    sbj_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
    actor_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]

    for subject in range(0,len(subject_list)):
        for story in range(0,len(story_list)): 
            subject_container[subject][story] = subject_container[subject][story][0:int(len(subject_container[subject][story])*reduce_factor)]
            actor_container[subject][story] = actor_container[subject][story][0:int(len(actor_container[subject][story])*reduce_factor)]
            y_container[subject][story] = y_container[subject][story][0:int(len(y_container[subject][story])*reduce_factor)]
            
            for k in range(0,len(subject_container[subject][story])):
                if subjectActorBoth >= 1:
                    actor_images[subject][story].append(cv2.imread(actor_container[subject][story][k][0]))
                if (subjectActorBoth == 0 or subjectActorBoth == 2):    
                    sbj_images[subject][story].append(cv2.imread(subject_container[subject][story][k][0]))
                        
    if subjectActorBoth == 2:
        return np.asarray(sbj_images), np.asarray(actor_images), np.asarray(y_container)
    if subjectActorBoth == 0:
        return np.asarray(sbj_images), np.asarray(y_container)
    if subjectActorBoth == 1:
        return np.asarray(actor_images), np.asarray(y_container)

# for images
def fusion(data_container):
   
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0.5,1))
    variance_container = np.array([np.var(signal) for signal in data_container])
    variance_container = scaler.fit_transform(np.reshape(variance_container, newshape=(-1,1)))
    
    weighted_container = np.array([0.0 for i in range(0,len(data_container[0]))])
    
    for index,signal in enumerate(data_container):
        signal = np.multiply(np.asarray(signal),variance_container[index])
        weighted_container = np.add(weighted_container,signal)
    weighted_container = np.asarray([value if value <= 255 else 255 for value in weighted_container ])
    return weighted_container

def fuseImages(image_1,image_2, kernel_size=(5,5), stride=(1,1)):
    
    shape = image_1.shape
    colum_image = np.zeros(shape, dtype='uint8')
    row_image = np.zeros(shape, dtype='uint8')
    avg_image = np.zeros(shape, dtype='uint8')

    w1 = slidingwindow.generate(image_1,slidingwindow.DimOrder.HeightWidthChannel,kernel_size[0],(kernel_size[0]-stride[0])/kernel_size[0])
    w2 = slidingwindow.generate(image_2,slidingwindow.DimOrder.HeightWidthChannel,kernel_size[0],(kernel_size[0]-stride[0])/kernel_size[0])
    
    for i in range(0,len(w1)):
        s1 = image_1[w1[i].indices()]
        s2 = image_2[w2[i].indices()]
        
        for k in range(0,len(s1)):
                # Column part
                b1,g1,r1 = s1[:,k][:,0], s1[:,k][:,1], s1[:,k][:,2]
                b2,g2,r2 = s2[:,k][:,0], s2[:,k][:,1], s2[:,k][:,2]
                f1,f2,f3 = fusion([b1,b2]), fusion([g1,g2]), fusion([r1,r2])
                
                                
                colum_image[w1[i].indices()][:,k][:,0] = f1
                colum_image[w1[i].indices()][:,k][:,1] = f2
                colum_image[w1[i].indices()][:,k][:,2] = f3
                
                # Row part
                b1,g1,r1 = s1[k,:][:,0], s1[k,:][:,1], s1[k,:][:,2]
                b2,g2,r2 = s2[k,:][:,0], s2[k,:][:,1], s2[k,:][:,2]
                f1,f2,f3 = fusion([b1,b2]), fusion([g1,g2]), fusion([r1,r2])
                
                row_image[w1[i].indices()][k,:][:,0] = f1
                row_image[w1[i].indices()][k,:][:,1] = f2
                row_image[w1[i].indices()][k,:][:,2] = f3
                
    for i in range(0,len(avg_image)):
        for j in range(0,len(avg_image)):
            avg_image[i][j][0] = np.mean([colum_image[i][j][0],row_image[i][j][0]])
            avg_image[i][j][1] = np.mean([colum_image[i][j][1],row_image[i][j][1]])
            avg_image[i][j][2] = np.mean([colum_image[i][j][2],row_image[i][j][2]])
            
    return avg_image

def read_raw_images_timesteps(data_directory='C:/Users/Diego Fabiano/Research/Data/OMG_RAW/Training/',
                    subject_list=[1,2,3,4,5,6,7,8,9,10],
                    story_list = [2,4,5,8],
                    y_directory='C:/Users/Diego Fabiano/Documents/OMG-FG-Challenge/data/Training/Annotations/',
                    subjectActorBoth=0,
                    reduce_factor=0.5,
                    timesteps = 15):

        # subjectActorBoth: 0 for subject only, 1 for actor only, 2 for both
        subject_container,actor_container,y_container = read_raw_image_paths(data_directory,subject_list, story_list,y_directory)

        sbj_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]
        actor_images = [[[] for story in range(0,len(story_list))] for subject in range(0,len(subject_list))]

        for subject in range(0,len(subject_list)):
                print("Read subject ", subject)
                for story in range(0,len(story_list)):
                        print("Read story ", story)
                        subject_container[subject][story] = subject_container[subject][story][0:int(len(subject_container[subject][story])*reduce_factor)]
                        actor_container[subject][story] = actor_container[subject][story][0:int(len(actor_container[subject][story])*reduce_factor)]
                        y_container[subject][story] = y_container[subject][story][0:int(len(y_container[subject][story])*reduce_factor)]

                        for k in range(0,len(subject_container[subject][story])):
                                if subjectActorBoth >= 1:
                                        actor_images[subject][story].append(cv2.resize(cv2.imread(actor_container[subject][story][k][0]),(128,128)))
                                if (subjectActorBoth == 0 or subjectActorBoth == 2):
                                        sbj_images[subject][story].append(cv2.resize(cv2.imread(subject_container[subject][story][k][0]),(128,128)))
        subject_ts = []
        actor_ts = []
        y = []
        for subject in range(0,len(subject_list)):
                for story in range(0,len(story_list)):
                        for indx in range(0, len(actor_images[subject][story]) - timesteps):
                            endSeq = indx + timesteps
                            if subjectActorBoth == 0 or subjectActorBoth == 2:
                                    subject_ts.append(sbj_images[subject][story][indx:endSeq])
                            if subjectActorBoth >= 1:
                                    actor_ts.append(actor_images[subject][story][indx:endSeq])
                            y.append(y_container[subject][story][indx:endSeq])

        if subjectActorBoth == 2:
                return np.asarray(subject_ts), np.asarray(actor_ts), np.asarray(y)
        if subjectActorBoth == 0:
                return np.asarray(subject_ts), np.asarray(y)
        if subjectActorBoth == 1:
                return np.asarray(actor_ts), np.asarray(y)

