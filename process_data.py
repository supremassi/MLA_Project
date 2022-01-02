# RUN THE IDE AS ADMINISTRATOR

import os
import sys

from pathlib import Path
import pickle
import shutil
import pandas as pd

# IMPORT RAW DATASET FOLDER TO USE THE DATA

# if IDE
# file_path = __file__
# parentFolder = str(Path(filePath).parent)   # Folder of this file --> "Project"

# if interactive
parentFolder = str(os.path.abspath(''))

for type in ["rgb", "brox"]:
    print()
    print("######### "+type.upper()+" IMAGES #########")
    print()
    print("--------- IMPORT DATASETS FOLDERS ------------")
    datasetFolderName = "datasets"      # Name of the folder which has the datasets
    # Absolute path of the dataset folder
    dataFolderPath = Path(r''+parentFolder+"//" + datasetFolderName)

    # UCF101 DATASET
    dataFolderPath_UCF101 = str(
        Path(r''+parentFolder+"//" + datasetFolderName + "//UCF101_v2"))
    print("Absolute path of the UCF101 dataset folder " + dataFolderPath_UCF101)

    # Import the path of the UCF101 dataset folder to the file so we can have acces to it
    sys.path.insert(1, dataFolderPath_UCF101)

    # JHMDB DATASET
    dataFolderPath_JHMDB = str(
        Path(r''+parentFolder+"//" + datasetFolderName + "//JHMDB"))
    print("Absolute path of the JHMDB dataset folder " + dataFolderPath_JHMDB)

    # Import the path of the JHMDB dataset folder to the file so we can have acces to it
    sys.path.insert(1, dataFolderPath_JHMDB)

    print()

    # CREATE NEW DATA FOLDER TO SORT THE DATA
    print("--------- CREATE OUR DATA FOLDERS ------------")
    # Create train and test data folders for the UCF101 in the folder "Project/Data"
    try:
        os.makedirs(parentFolder + "//Data//"+type.upper()+"//UCF101//TRAIN")
        print("Directory ", parentFolder +
              "//Data//"+type.upper()+"//UCF101//TRAIN",  " Created ")
    except FileExistsError:
        print("Directory ", parentFolder +
              "//Data//"+type.upper()+"//UCF101//TRAIN",  " already exists")
    try:
        os.makedirs(parentFolder + "//Data//"+type.upper()+"//UCF101//TEST")
        print("Directory ", parentFolder + "//Data//" +
              type.upper()+"//UCF101//TEST",  " Created ")
    except FileExistsError:
        print("Directory ", parentFolder +
              "//Data//"+type.upper()+"//UCF101//TEST",  " already exists")

        # Create train and test  data folders for the JHMDB in the folder "Project/Data"
    try:
        os.makedirs(parentFolder+"//Data//"+type.upper()+"//JHMDB//TRAIN")
        print("Directory ", parentFolder+"//Data//" +
              type.upper()+"//JHMDB//TRAIN",  " Created ")
    except FileExistsError:
        print("Directory ", parentFolder +
              "//Data//"+type.upper()+"//JHMDB//TRAIN",  " already exists")
    try:
        os.makedirs(parentFolder + "//Data//"+type.upper()+"//JHMDB//TEST")
        print("Directory ", parentFolder+"//Data//" +
              type.upper()+"//JHMDB//TEST",  " Created ")
    except FileExistsError:
        print("Directory ", parentFolder +
              "//Data//"+type.upper()+"//JHMDB//TEST",  " already exists")

    print()


    # PROCESS DATA
    print("--------- PROCESS DATA ------------")
    print("Process data ...")

    # # UCF101
    # print("UCF101's data")
    # with open(dataFolderPath_UCF101+'//UCF101v2-GT.pkl', 'rb') as f:
    #     classifier = pickle.load(f, encoding="latin1")  # Dictionnary

    # keys_tab = []
    # for keys in classifier:
    #     # ['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution']
    #     keys_tab.append(keys)

    # # 2284 VIDEOS FOR TRAINING (contain the folder and video name : Basketball/v_Basketball_g08_c02 )
    # train_videos = classifier["train_videos"][0]
    # test_videos = classifier["test_videos"][0]    # 910 VIDEOS FOR TESTING

    # labels_names = classifier['labels']  # 24 classes (0à23)

    # # GROUND TRUTH'S BOUNDINGS BOXES TUBES FOR EACH VIDEO
    # tubes = classifier['gttubes']
    # # VIDEOS' NAMES Basketball/v_Basketball_g08_c02 (3194 VIDEOS)
    # file_names = list(classifier['gttubes'].keys())

    # data = []
    # for item in tubes.values():
    #     data.append(list(item.values())[0][0])
    # # data: 3196 videos * ((nb_frames/video)*5)   : n° frame  x1 y1 x2 y2 (coordinates of the bounding box)

    # labels = []
    # for e in tubes.keys():
    #     nb = list(tubes[e].keys())  # NUMBER OF THE CORRESPONDING LABEL 0-23
    #     labels.append(nb[0])

    # # CREATE A TRAIN AND A TEST TABLE CONTAINING THE FRAMES FILE, GT COORDINATES, LABEL NB AND LABEL NAME
    # data_train = []
    # data_test = []
    # for i in range(len(data)):  # every video
    #     video = []
    #     file_name = file_names[i]
    #     label_number = labels[i]
    #     for idx, e in enumerate(data[i]):
    #         image = [file_name+"/"+str(idx+1).zfill(5)+".jpg", e[1],
    #                  e[2], e[3], e[4], label_number, labels_names[label_number]]
    #         # if idx == 0 and i == 0:
    #         #     print(file_name+"/"+str(idx+1).zfill(5)+".jpg")
    #         video.append(image)
    #     if file_name in test_videos:
    #         data_test.append(video)
    #     else:
    #         data_train.append(video)

    # print("NUMBERS OF VIDEOS FOR TRAINNING", len(data_train))  # 2284
    # print("NUMBERS OF VIDEOS FOR TESTING", len(data_test))  # 910

    # print()

    # # TRAIN DATA
    # print("Sorting train data ...")
    # for idx, video in enumerate(data_train):
    #     video_name = (video[0][0]).split('/')[0] + \
    #         "//"+(video[0][0]).split('/')[1]
    #     video_folder_path = parentFolder + \
    #         "//Data//"+type.upper()+"//UCF101//TRAIN//"+str(video_name)

    #     try:
    #         os.makedirs(video_folder_path)
    #     except FileExistsError:
    #         pass

    #     for frame in video:
    #         nameVideo = frame[0]
    #         file = parentFolder+"/datasets/UCF101_v2/"+type+"-images/"+nameVideo
    #         destination = parentFolder + "//Data//" + \
    #             type.upper()+"//UCF101//TRAIN//" + nameVideo
    #         shutil.copyfile(file, destination)

    # # EXPORTING TRAINING DATA
    # print("Exporting train data ...")
    # df_train = pd.DataFrame(data_train[0], columns=[
    #                         "file_name", "x1", "y1", "x2", "y2", "label", "label_name"])
    # print(df_train.head())

    # export_csv_train = df_train.to_csv(parentFolder + "//Data//"+type.upper()+"//UCF101//" +
    #                                    "data_"+type.upper()+"_train.csv", index=None, header=True, encoding='utf-8', sep=';')

    # print()

    # # TEST DATA
    # print("Sorting test data ...")
    # for idx, video in enumerate(data_test):
    #     video_name = (video[0][0]).split('/')[0] + \
    #         "//"+(video[0][0]).split('/')[1]
    #     video_folder_path = parentFolder + \
    #         "//Data//"+type.upper()+"//UCF101//TEST//"+str(video_name)
    #     try:
    #         os.makedirs(video_folder_path)
    #     except FileExistsError:
    #         pass
    #     # label_name=e[0].split('/')
    #     for frame in video:
    #         nameVideo = frame[0]
    #         file = parentFolder+"/datasets/UCF101_v2/"+type+"-images/"+nameVideo
    #         destination = parentFolder + "//Data//" + \
    #             type.upper()+"//UCF101//TEST//" + nameVideo
    #         shutil.copyfile(file, destination)

    # # EXPORTING TRAINING DATA
    # print("Exporting test data ...")
    # df_test = pd.DataFrame(data_test[0], columns=[
    #     "file_name", "x1", "y1", "x2", "y2", "label", "label_name"])
    # print(df_test.head())

    # export_csv_test = df_test.to_csv(parentFolder + "//Data//"+type.upper()+"//UCF101//" +
    #                                  "data_"+type.upper()+"_test.csv", index=None, header=True, encoding='utf-8', sep=';')

    # print()
    # print()

# JHMDB
    print("JHMDB's data")
    with open(dataFolderPath_JHMDB+'//JHMDB-GT.pkl', 'rb') as f:
        classifier = pickle.load(f, encoding="latin1")  # Dictionnary

    keys_tab = []
    for keys in classifier:
        # ['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution']
        keys_tab.append(keys)

    # 660 VIDEOS FOR TRAINING (contain the folder and video name : 
    train_videos = classifier["train_videos"][0]
    test_videos = classifier["test_videos"][0]    # 268 VIDEOS FOR TESTING

    labels_names = classifier['labels']  # 24 classes (0à23)
    print("nb labels ",len(labels_names))
    # GROUND TRUTH'S BOUNDINGS BOXES TUBES FOR EACH VIDEO
    tubes = classifier['gttubes']
    # VIDEOS' NAMES Basketball/v_Basketball_g08_c02 (3194 VIDEOS)
    file_names = list(classifier['gttubes'].keys())

    data = []
    for item in tubes.values():
        data.append(list(item.values())[0][0])
    # data: n videos * ((nb_frames/video)*5)   : n° frame  x1 y1 x2 y2 (coordinates of the bounding box)

    labels = []
    for e in tubes.keys():
        nb = list(tubes[e].keys())  # NUMBER OF THE CORRESPONDING LABEL 0-23
        labels.append(nb[0])

    # CREATE A TRAIN AND A TEST TABLE CONTAINING THE FRAMES FILE, GT COORDINATES, LABEL NB AND LABEL NAME
    data_train = []
    data_test = []
    for i in range(len(data)):  # every video
        video = []
        file_name = file_names[i]
        label_number = labels[i]
        for idx, e in enumerate(data[i]):
            if type=="rgb":
                image = [file_name+"/"+str(idx+1).zfill(5)+".png", e[1], e[2], e[3], e[4], label_number, labels_names[label_number]]
            else:
                image = [file_name+"/"+str(idx+1).zfill(5)+".jpg", e[1], e[2], e[3], e[4], label_number, labels_names[label_number]]
            # if idx == 0 and i == 0:
            #     print(file_name+"/"+str(idx+1).zfill(5)+".png")
            video.append(image)
        if file_name in test_videos:
            data_test.append(video)
        else:
            data_train.append(video)

    print("NUMBERS OF VIDEOS FOR TRAINNING", len(data_train))  # 660
    print("NUMBERS OF VIDEOS FOR TESTING", len(data_test))  # 910

    print()

    # TRAIN DATA
    print("Sorting train data ...")
    for idx, video in enumerate(data_train):
        video_name = (video[0][0]).split('/')[0] + \
            "//"+(video[0][0]).split('/')[1]
        video_folder_path = parentFolder + \
            "//Data//"+type.upper()+"//JHMDB//TRAIN//"+str(video_name)

        try:
            os.makedirs(video_folder_path)
        except FileExistsError:
            pass

        for frame in video:
            nameVideo = frame[0]
            if type=="rgb":
                file = parentFolder+"/datasets/JHMDB/Frames/"+nameVideo
            else:
                file = parentFolder+"/datasets/JHMDB/FlowBrox04/"+nameVideo
            destination = parentFolder + "//Data//" + \
                type.upper()+"//JHMDB//TRAIN//" + nameVideo
            shutil.copyfile(file, destination)

    # EXPORTING TRAINING DATA
    print("Exporting train data ...")
    df_train = pd.DataFrame(data_train[0], columns=[
                            "file_name", "x1", "y1", "x2", "y2", "label", "label_name"])
    print(df_train.head())

    export_csv_train = df_train.to_csv(parentFolder + "//Data//"+type.upper()+"//JHMDB//" +
                                       "data_"+type.upper()+"_train.csv", index=None, header=True, encoding='utf-8', sep=';')

    print()

    # TEST DATA
    print("Sorting test data ...")
    for idx, video in enumerate(data_test):
        video_name = (video[0][0]).split('/')[0] + \
            "//"+(video[0][0]).split('/')[1]
        video_folder_path = parentFolder + \
            "//Data//"+type.upper()+"//JHMDB//TEST//"+str(video_name)
        try:
            os.makedirs(video_folder_path)
        except FileExistsError:
            pass
        # label_name=e[0].split('/')
        for frame in video:
            nameVideo = frame[0]
            if type=="rgb":
                file = parentFolder+"/datasets/JHMDB/Frames/"+nameVideo
            else:
                file = parentFolder+"/datasets/JHMDB/FlowBrox04/"+nameVideo
            destination = parentFolder + "//Data//" + \
                type.upper()+"//JHMDB//TEST//" + nameVideo
            shutil.copyfile(file, destination)

    # EXPORTING TRAINING DATA
    print("Exporting test data ...")
    df_test = pd.DataFrame(data_test[0], columns=[
        "file_name", "x1", "y1", "x2", "y2", "label", "label_name"])
    print(df_test.head())

    export_csv_test = df_test.to_csv(parentFolder + "//Data//"+type.upper()+"//JHMDB//" +
                                     "data_"+type.upper()+"_test.csv", index=None, header=True, encoding='utf-8', sep=';')
