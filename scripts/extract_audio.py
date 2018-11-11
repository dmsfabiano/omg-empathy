
import os
from shutil import copyfile

import subprocess



def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l


def extractAudio(path, savePath):
    videos = os.listdir(path)

    for video in videos:
        videoPath = path + "/" + video

        if not os.path.exists(savePath):
            print ("- Processing Video:", videoPath + " ...")
            os.makedirs(savePath)

        #copyfile(videoPath, copyTargetVideo)
        print ("--- Extracting audio:", savePath + "/" + video + ".wav" + " ...")

        command1 = 'ffmpeg -v quiet -i {0} -ar 125000 -ac 1 -vn {1}'.format(videoPath, savePath + "/" + video + ".wav")
        subprocess.call(command1, shell=True)




if __name__ == "__main__":

    print('Starting extracting Training audio')
    # Path where the videos are
    #path = "/data/Validation/Videos/"
    #path = "D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\data\\Training\\Videos\\"
    #path = "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Training/Videos/"
    path = "/media/neil/30CA58BFCA588350/data/Training/Videos/"
    #path = "../data/Validation/Videos"

    # Path where the audios will be saved
    #savePath ="/data/audio/Validation/"
    #savePath = "D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\data\\audio-reduced\\Training\\"
    #savePath = "/mnt/d/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/audio-reduced/Training/"
    savePath = "/media/neil/30CA58BFCA588350/data/audio-reduced/Training/"
    #savePath = "../data/audio/Validation"

    extractAudio(path,savePath)

    print('Starting extracting Validation audio')
    path = "/media/neil/30CA58BFCA588350/data/Validation/Videos/"
    savePath = "/media/neil/30CA58BFCA588350/data/audio-reduced/Validation/"
    extractAudio(path,savePath)