import numpy as np
import cv2
import os
from multiprocessing import Pool

class Face:
	face = None
	actor_face = None
	count = 0
	actor_count = 0

class BoolWrapper:
	isDone = False

def face_detect(frame, cascade, previous_face, actor):
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = cascade.detectMultiScale(    
			gray,
			scaleFactor=1.5,
			minNeighbors=3,
			minSize=(90, 90),
		)
	if not np.any(faces):
		if actor:
			return previous_face.actor_face	
		else:
			return previous_face.face
	for (x,y,w,h) in faces:
		frame = frame[y:y+h, x:x+w]
		if actor:
			previous_face.actor_face = cv2.resize(frame,(256,256))
		else:
			previous_face.face = cv2.resize(frame,(256,256))
		break
	if actor:
		return previous_face.actor_face
	else:		
		return previous_face.face


def read_frames(filename, batch_size=256, cObj=None, previous_face_wrapper=None):
	cascade = cv2.CascadeClassifier('./c++-scripts/haarcascade_frontalface_default.xml')
	cap = cv2.VideoCapture(filename) 
	frames = []
	count = 0
	actor_count = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			y,x,_ = frame.shape
			actor = face_detect(frame[0:y, 0:(x//2)],cascade,previous_face_wrapper,True)
			subject = face_detect(frame[0:y, (x//2):x],cascade,previous_face_wrapper,False)
			if actor is None:
				actor_count += 1
			elif subject is None:
				count += 1
			frames.append((actor,subject))
			while actor is not None and actor_count > 0:
				for frame in frames:
					if frame[0] == None:
						frame = (actor,frame[1])
						actor_count -= 1
						break
			while subject is not None and count > 0:
				for frame in frames:
					if frame[1] == None:
						frame = (frame[0],subject)
						count -= 1
						break
		else:
			cObj.isDone = True
			break
	return (frames, filename)

def write_frames(data, training=True):
	frames,path = data[0],data[1]
	file_name_write = path.split('/')[-1].split('.')[0]
	for number, frame in enumerate(frames):
		cv2.imwrite('../data/OriginalImages/Testing/'+file_name_write+'_frame_'+str(number)+'.png' if training else '../data/OriginalImages/Validation/'+file_name_write+'_frame_'+str(number)+'.png', frame[1])
		cv2.imwrite('../data/OriginalImages/Testing/'+file_name_write+'_frame_'+str(number)+'_actor.png' if training else '../data/OriginalImages/Validation/'+file_name_write+'_frame_'+str(number)+'_actor.png', frame[0])

def write_frames_wrapper(args):
	write_frames(*args)

def run():
	arglist_training = []
	for (root,path,files) in os.walk('../data/Testing/Videos/'):
		for file in files:
			if file.endswith('.mp4'):
				path = os.path.join(root,file)
				arglist_training.append((path, 256, BoolWrapper(), Face()))

	for args in arglist_training:
		frames_to_write = read_frames(*args)
		write_frames(frames_to_write)

	#arglist_validation = []
	#for (root,path,files) in os.walk('../data/Validation/Videos/'):
	#	for file in files:
	#		if file.endswith('.mp4'):
	#			path = os.path.join(root,file)
	#			arglist_validation.append((path, 256, BoolWrapper(), Face()))
	#
	#for args in arglist_validation:
	#	frames_to_write = read_frames(*args)
	#	write_frames(frames_to_write,False)

run()
