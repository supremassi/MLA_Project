from os import listdir
import os
import sys
import pandas as pd 

import keras
from keras.layers import Sequential, Bidirectional, Embedding, SeqSelfAttention
from keras.layers import Dense, Flatten, LSTM 
## PARAMETERS
TL = 30  # and 15 for UCF101-24 and J-HMDB
C = 4096 # as the dimension of the fc7
P = 8   # number of heads 
dr= 0.1 # drop out rate
##

folderPath = str(os.path.abspath(''))
print(folderPath)
trainFolder = folderPath+"//Data//UCF101"
sys.path.insert(1, trainFolder)

csv_filename = "//Data//RGB//UCF101//data_RGB_train.csv"
name_file = folderPath+"//"+csv_filename
train_data = pd.read_csv(name_file, sep=";")

videosFolder = listdir(folderPath+"//Data//RGB//UCF101//TRAIN")
print(videosFolder)

def get_vid_infos(videosFolder):
    videos = []     # videos' path
    lg_videos = []  # nb of frames for each video
    idxvideos=[]    # Index of videos' frames to use on the train_data table
    for video in videosFolder:
        videopath = trainFolder+video
        nb_frame=len(listdir(videopath))
        if lg_videos==[]:
            indvid=range(nb_frame)
        else:
            indvid=range(indvid[-1][-1],nb_frame)
        videos.append(videopath)
        lg_videos.append(nb_frame)
        idxvideos.append(indvid)
    return videos, lg_videos, idxvideos

## For one video
def get_vid_unit_size(lgvid,TL):
    return lgvid//TL
    
def video_seq(video,lgvid,TL):
    C = get_vid_unit_size(lgvid,TL)
    video_seq=[]
    frames=[]
    for i, frame in listdir(video):
        frames.append(frame)
        if i%C==0 and i!=0:     # 1 unit
            video_seq.append(frames)
            frames=[]
    return C,video_seq

def dot(vi,vk):
    prod=[]
    for i in range(len(vi)):
        prod.append(vi[i]*vk[i])
    return sum(prod)

def pair_function(vi,videos_seq):
    prod=[]
    for k in range(videos_seq):
        prod.append(dot(vi,videos_seq[k]))
    return sum(prod)

def self_attention_model(input_size,vi,video_seq):
    model = Sequential()
    model.add(Embedding(input_size, input_size))
    return model 

